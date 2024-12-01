import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_ops import pointnet2_utils
from timm.models.layers import trunc_normal_
import sys
from timm.models.layers import DropPath

from checkpoint import get_missing_parameters_message, get_unexpected_parameters_message, fps

sys.path.append("../utils")
sys.path.append("..")
sys.path.append("./")

from pointnet2_utils import PointNetFeaturePropagation_
from utils.logger import *
from modules import *


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base  # torch.Size([1, 64, 20])

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, in_dim=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, -1)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LAM(nn.Module):
    def __init__(self, dim=512, downrate=8, gcn_k=20):
        super(LAM, self).__init__()
        self.k = gcn_k
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(int(self.dim // downrate))
        self.bn2 = nn.BatchNorm1d(self.dim)

        act_mod = nn.LeakyReLU
        act_mod_args = {'negative_slope': 0.2}

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim * 2, int(self.dim // downrate), kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args),
                                   )
        self.conv2 = nn.Sequential(nn.Conv1d(int(self.dim // downrate), self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))

    def forward(self, x, center):
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k, idx=center)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = self.conv2(x1)

        x = x1.permute(0, 2, 1)

        return x



class BlockCPR(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downrate=8, gcn_k=20):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim / downrate), act_layer=act_layer, drop=drop)
        self.attn = LAM(dim, downrate=downrate, gcn_k=gcn_k)

    def forward(self, x, idx):
        x = x + self.drop_path(self.attn(self.norm1(x), idx))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class EncoderCPR(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., downrate=8, gcn_k=20):
        super().__init__()

        self.blocks = nn.ModuleList([
            BlockCPR(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                downrate=downrate, gcn_k=gcn_k
            )
            for i in range(depth)])
        self.gcn_k = gcn_k

    def forward(self, x, pos, idx):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for _, block in enumerate(self.blocks):
            x = block(x + pos, idx)
            if _ in fetch_idx:
                feature_list.append(x)
        return feature_list

# finetune model
class Point_CLM_SEG(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = cls_dim
        self.num_heads = 6

        self.group_size = 32
        self.num_group = 128
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(self.trans_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.downrate = 4
        self.gcn_k = 5
        self.seg_depth = 3
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = EncoderCPR(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            downrate=self.downrate,
            gcn_k=self.gcn_k,
        )

        self.encoder_norms = nn.ModuleList()
        for i in range(self.seg_depth):
            self.encoder_norms.append(nn.LayerNorm(self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.propagations = nn.ModuleList()
        for i in range(3):
            self.propagations.append(PointNetFeaturePropagation_(in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, 1024]))

        self.convs1 = nn.Conv1d(6208, 1024, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(1024, 512, 1)
        self.convs3 = nn.Conv1d(512, 256, 1)
        self.convs4 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(512)
        self.bns3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def load_model_from_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

        for k in list(base_ckpt.keys()):
            if k.startswith('MAE_encoder'):
                base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='Transformer')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, cls_label, test=True):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)

        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)
        x = group_input_tokens
        idx = knn(center.permute(0, 2, 1), k=self.gcn_k)
        # transformer
        x_vis_list = self.blocks(x, pos, idx)

        for i in range(self.seg_depth):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i]).transpose(-1, -2).contiguous()

        for i in range(self.seg_depth):
            x_vis_list[i] = self.propagations[i](pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x_vis_list[i])
            
        x = torch.cat((x_vis_list[0], x_vis_list[1], x_vis_list[2]), dim=1)  # 96 + 192 + 384
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature + x_avg_feature, cls_label_feature), 1) # 672 * 2 + 64

        x = torch.cat((x_global_feature, x), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.relu(self.bns3(self.convs3(x)))
        x = self.convs4(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss