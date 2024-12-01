import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mamba_ssm import Mamba



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
        center = misc.fps(xyz, self.num_group)  # B G 3
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


## Transformers
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



class LAL(nn.Module):
    def __init__(self, dim=512, downrate=8, gcn_k=20):
        super(LAL, self).__init__()
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



class BlockLCM(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downrate=8, gcn_k=20):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim / downrate), act_layer=act_layer, drop=drop)
        self.attn = LAL(dim, downrate=downrate, gcn_k=gcn_k)

    def forward(self, x, idx):
        x = x + self.drop_path(self.attn(self.norm1(x), idx))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class EncoderLCM(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., downrate=8, gcn_k=20):
        super().__init__()

        self.blocks = nn.ModuleList([
            BlockLCM(
                dim=embed_dim, 
                drop=drop_rate,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                downrate=downrate, gcn_k=gcn_k
            )
            for i in range(depth)])
        self.gcn_k = gcn_k

    def forward(self, x, pos, idx):
        for _, block in enumerate(self.blocks):
            x = block(x + pos, idx)
        return x


class LCFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., downrate=8,
                 gcn_k=20):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features * 2, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.k = gcn_k

    def forward(self, x, idx):
        x = get_graph_feature(x.permute(0, 2, 1), k=self.k, idx=idx).permute(0,2,3,1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.max(dim=-2, keepdim=False)[0]
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BlockMamba(nn.Module):
    def __init__(self, dim, mlp_ratio=1.,
                 drop_path=0., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downrate=8, gcn_k=5):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attn = Mamba(d_model=dim)
        self.mlp = LCFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                         downrate=downrate, gcn_k=gcn_k)

    def forward(self, x, idx):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), idx))
        return x

class MambaDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, mlp_ratio=1.,
                 drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, downrate=8, gcn_k=20):
        super().__init__()
        self.blocks = nn.ModuleList([
            BlockMamba(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                downrate=downrate, gcn_k=gcn_k
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.gcn_k = gcn_k

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num, idx):
        for _, block in enumerate(self.blocks):
            x = block(x + pos, idx)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x

class YOrder(nn.Module):
    def __init__(self):
        super(YOrder, self).__init__()

    def forward(self, x):
        return x[:, :, 1].argsort(dim=-1)[:, :]

@MODELS.register_module()
class Point_LCM_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_LCM_MAE] ', logger='Point_LCM_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.mask_ratio = config.transformer_config.mask_ratio
        self.mask_type = config.transformer_config.mask_type
        self.depth = config.transformer_config.depth
        self.num_heads = config.transformer_config.num_heads

        print_log(f'[Point_LCM_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_LCM_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.downrate = config.transformer_config.downrate
        self.gcn_k = config.transformer_config.gcn_k
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = EncoderLCM(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            downrate=self.downrate,
            gcn_k=self.gcn_k,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self.order = YOrder()

        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            downrate=self.downrate,
            gcn_k=self.gcn_k
        )

        print_log(f'[Point_LCM_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_LCM_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def build_loss_func(self, loss_type):
        self.loss_l1 = nn.SmoothL1Loss()
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def forward(self, pts, vis=False, eval=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        # generate mask
        bool_masked = self._mask_center_rand(center, noaug=eval)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        B, _, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked].reshape(B, -1, C)
        center_vis = center[~bool_masked].reshape(B, -1, 3)
        center_unvis = center[bool_masked].reshape(B, -1, 3)
        nei_unvis = neighborhood[bool_masked].reshape(B,-1,self.group_size,3)
        pos_vis = self.pos_embed(center_vis)
        idx = knn(center_vis.permute(0, 2, 1), k=self.gcn_k)

        # transformer
        x_vis = self.blocks(x_vis, pos_vis, idx)
        x_vis = self.norm(x_vis)

        # decoder reorder
        vis_order = self.order(center_vis)
        unvis_order = self.order(center_unvis)

        center_vis = center_vis.gather(dim=1, index=torch.tile(vis_order.unsqueeze(2), (1, 1, center_vis.shape[-1])))
        x_vis = x_vis.gather(dim=1, index=torch.tile(vis_order.unsqueeze(2), (1, 1, x_vis.shape[-1])))
        center_unvis = center_unvis.gather(dim=1, index=torch.tile(unvis_order.unsqueeze(2), (1, 1, center_unvis.shape[-1])))
        nei_unvis = nei_unvis.gather(dim=1, index=torch.tile(unvis_order.unsqueeze(2).unsqueeze(3), (1, 1, nei_unvis.shape[-2], neighborhood.shape[-1])))


        B,_,C = x_vis.shape # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center_vis).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center_unvis).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        center_full = torch.cat([center_vis, center_unvis], dim=1)
        idx = knn(center_full.permute(0, 2, 1), k=self.gcn_k)


        x_rec = self.MAE_decoder(x_full, pos_full, N, idx)
        

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = nei_unvis.reshape(B*M,-1,3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        return loss1



# finetune model
@MODELS.register_module()
class PointLCM(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.downrate = config.downrate
        self.gcn_k = config.gcn_k
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = EncoderLCM(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            downrate=self.downrate,
            gcn_k=self.gcn_k,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()


    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc_raw(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
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

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

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

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        pos = self.pos_embed(center)
        idx = knn(center.permute(0, 2, 1), k=self.gcn_k)

        x = group_input_tokens
        # transformer
        x = self.blocks(x, pos, idx)
        x = self.norm(x)
        concat_f = torch.cat([x.mean(1), x.max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret