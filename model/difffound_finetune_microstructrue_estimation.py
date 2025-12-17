from functools import partial
import torch.nn as nn
from vits_q import VIT, Patchify
from timm.models.layers import to_3tuple
from monai.networks.blocks import UnetrPrUpBlock, UnetrUpBlock
from dynunet_block import UnetResBlock
from monai.networks.blocks.dynunet_block import get_conv_layer


def _build_mlp(in_dim, out_dim, hidden_dim=4096, num_layers=1, norm=nn.LayerNorm, out_norm=False):
    if num_layers == 0:
        return None
    projector = nn.Sequential()
    for l in range(num_layers):
        dim1 = in_dim if l == 0 else hidden_dim
        dim2 = out_dim if l == num_layers - 1 else hidden_dim
        projector.add_module(f"linear{l}", nn.Linear(dim1, dim2, bias=False))
        if out_norm or l < num_layers - 1:
            projector.add_module(f"norm{l}", norm(dim2))
        if l < num_layers - 1:
            projector.add_module(f"act{l}", nn.GELU())
    return projector

class Ps_Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv3d_down1 = nn.Conv3d(in_channels=in_channels,
                                      out_channels=in_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.relu1 = nn.ReLU()
        self.conv3d_down2 = nn.Conv3d(in_channels=in_channels,
                                      out_channels=in_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.relu2 = nn.ReLU()
        self.conv3d_down3 = nn.Conv3d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.output_channels = out_channels

    def forward(self, x):
        indice = self.conv3d_down1(x)
        indice = self.relu1(indice)
        indice = self.conv3d_down2(indice)
        indice = self.relu2(indice)
        indice = self.conv3d_down3(indice)

        return indice

class DiffFound(nn.Module):
    """ Latent Masked Image Modeling with VisionTransformer backbone
    """

    def __init__(self, grid_size=14, patch_size=16, patch_gap=0, in_chans=30,
                 embed_dim=1024, num_heads=16, depth=24,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 drop=0., attn_drop=0., drop_path=0.,
                 tau=0.2, num_vis=20,
                 avg_sim_coeff=0., gradient_directions=30, mask_ratio=0.75, mask_target=True,
                 loss='infonce_patches', freeze_pe=None, proj_cfg=None, norm_name="instance", conv_block=False,
                 res_block=True):
        super().__init__()

        self.loss = loss
        self.patch_gap = patch_gap
        self.tau = tau
        self.num_vis = num_vis
        self.avg_sim_coeff = avg_sim_coeff

        self.grid_size = to_3tuple(grid_size)
        self.gradient_directions = gradient_directions
        self.mask_ratio = mask_ratio
        self.patch_size = to_3tuple(patch_size)

        self.patchify = Patchify(patch_size=patch_size, grid_size=grid_size, gradient_directions=gradient_directions)
        self.mask_target = mask_target
        self.embed_dim = embed_dim

        down_radio = 32
        hidden_size = (embed_dim // down_radio) * gradient_directions
        self.hidden_size = hidden_size
        # --------------------------------------------------------------------------
        # Encoder
        head = _build_mlp(embed_dim, embed_dim, proj_cfg.mlp_dim, proj_cfg.mlp_depth, out_norm=True)
        self.encoder = VIT(patchify=self.patchify,
                           grid_size=grid_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, head=head, mask=False, cls=False,
                           drop=drop, attn_drop=attn_drop, drop_path=drop_path, freeze_pe=freeze_pe)

        # --------------------------------------------------------------------------
        # Feature Decoder and Predictor
        self.q_agg1 = nn.Sequential(
        nn.Linear(embed_dim, embed_dim // down_radio // 2),
        nn.GELU(),
        nn.Linear(embed_dim // down_radio // 2, embed_dim // down_radio),
        nn.GELU()
        )
        self.q_agg2 = nn.Sequential(
        nn.Linear(embed_dim, embed_dim // down_radio // 2),
        nn.GELU(),
        nn.Linear(embed_dim // down_radio // 2, embed_dim // down_radio),
        nn.GELU()
        )
        self.q_agg3 = nn.Sequential(
        nn.Linear(embed_dim, embed_dim // down_radio // 2),
        nn.GELU(),
        nn.Linear(embed_dim // down_radio // 2, embed_dim // down_radio),
        nn.GELU()
        )
        self.q_agg4 = nn.Sequential(
        nn.Linear(embed_dim, embed_dim // down_radio // 2),
        nn.GELU(),
        nn.Linear(embed_dim // down_radio // 2, embed_dim // down_radio),
        nn.GELU()
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=64,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=128,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=256,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=384,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=384,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name
        )

        self.transConv = get_conv_layer(
            spatial_dims=3,
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
            conv_only=True,
            is_transposed=True,
        )
        self.ps_conv = Ps_Conv(in_channels=64, out_channels=1)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, imgs):
        B, C, H, W, L = imgs.shape
        patch_pix = self.patchify(imgs, self.patch_gap)
        x_vis, hidden_states_out = self.encoder(patch_pix)
        x_vis = self.q_agg1(x_vis)
        x_vis = x_vis.view(B, C, 216, -1).permute(0, 2, 1, 3).flatten(2)

        convBlock = self.encoder1(imgs)

        x2 = hidden_states_out[0]
        x2 = self.q_agg2(x2)
        x2 =x2.view(B, C, 216, -1).permute(0, 2, 1, 3).flatten(2)

        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.grid_size))
        x3 = hidden_states_out[1]
        x3 = self.q_agg3(x3)
        x3 = x3.view(B, C, 216, -1).permute(0, 2, 1, 3).flatten(2)

        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.grid_size))
        x4 = hidden_states_out[2]
        x4 = self.q_agg4(x4)
        x4 = x4.view(B, C, 216, -1).permute(0, 2, 1, 3).flatten(2)

        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.grid_size))

        dec4 = self.proj_feat(x_vis, self.hidden_size, self.grid_size)
        dec3 = self.decoder4(dec4, enc4)
        dec2 = self.decoder3(dec3, enc3)
        dec1 = self.decoder2(dec2, enc2)

        dec0 = self.transConv(dec1)
        out = dec0 + convBlock
        logit = self.ps_conv(out).permute(0, 2, 3, 4, 1).contiguous()

        return logit

CFG = {
    'vit_tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4, 'patch_size': 16},
    'vit_huge': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'mlp_ratio': 4, 'patch_size': 14},
}


def build_difffound(backbone, **kwargs):
    cfg = CFG[backbone]
    model = DiffFound(
        patch_size=cfg['patch_size'], embed_dim=cfg['embed_dim'], depth=cfg['depth'], num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
