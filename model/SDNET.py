import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .PCR import PCRUnit3D,  PCRBlock3D, PCRBlock3DUP,Light_PCR_ASPP
import math

class CBAM(nn.Module):
    def __init__(self, c_in, c_out, units=1, kernel=3, stride=1, dilation=1,
                 pool=False, batch_norm=True, inst_norm=False, reduce=4):
        super(CBAM, self).__init__()

        self.rb1 = PCRBlock3D(c_in, c_out // reduce, c_out, units=units, dilation=dilation,
                              pool=pool, residual=False, batch_norm=batch_norm,
                              inst_norm=inst_norm)

        self.channel_gate = nn.Sequential(
            nn.Linear(c_out, c_out // reduce),
            nn.ReLU(inplace=True),
            nn.Linear(c_out // reduce, c_out),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

    def forward(self, input):
        x = input
        x = self.rb1(x)

        avg_out = self.channel_gate(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.channel_gate(self.max_pool(x).view(x.size(0), -1))
        channel_out = (avg_out + max_out).view(x.size(0), x.size(1), 1, 1, 1)
        x = x * channel_out.expand_as(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_gate(spatial_out)
        x = x * spatial_out.expand_as(x)

        out = x + input
        return out


class fusion_module(nn.Module):
    def __init__(self, dim, bias=True, batch_norm=True):
        super(fusion_module, self).__init__()
        self.dim = dim
        self.bias=bias

        self.dw = nn.Conv3d(dim*2, dim*2, kernel_size=3, padding=1,
                                   groups=dim*2, bias=bias)
        self.pw = nn.Conv3d(dim*2, dim, kernel_size=1, bias=bias)

        if batch_norm:
            self.norm = nn.BatchNorm3d(dim)
        else:
            self.norm = nn.Identity()

        self.relu = nn.ReLU()

        self.avg_pool_wh = nn.AdaptiveAvgPool3d((None, None, 1))
        self.avg_pool_wd = nn.AdaptiveAvgPool3d((None, 1, None))
        self.avg_pool_hd = nn.AdaptiveAvgPool3d((1, None, None))

        self.max_pool_wh = nn.AdaptiveMaxPool3d((None, None, 1))
        self.max_pool_wd = nn.AdaptiveMaxPool3d((None, 1, None))
        self.max_pool_hd = nn.AdaptiveMaxPool3d((1, None, None))

        self.mlp = nn.Sequential(
            nn.Conv3d(dim * 4, dim*2, 1),
            nn.ReLU(),
            nn.Conv3d(dim*2, dim*2, 1),
            nn.Sigmoid()
        )
    def forward(self, input1, input2):
        B, C, W, H, D = input1.shape
        x = torch.cat([input1, input2], dim=1)
        x=self.dw(x)

        # W-H
        avg_wh = self.avg_pool_wh(x)  # [B,2C,W,H,1]
        max_wh = self.max_pool_wh(x)  # [B,2C,W,H,1]
        corr_wh = torch.cat([avg_wh, max_wh], dim=1)  # [B,4C,W,H,1]
        corr_wh = corr_wh.reshape((B, 4 * C, W * H, 1, 1))
        # W-D
        avg_wd = self.avg_pool_wd(x)  # [B,2C,W,1,D]
        max_wd = self.max_pool_wd(x)  # [B,2C,W,1,D]
        corr_wd = torch.cat([avg_wd, max_wd], dim=1)  # [B,4C,W,1,D]
        corr_wd = corr_wd.permute(0, 1, 2, 4, 3).reshape((B, 4 * C, W * D, 1, 1))

        # H-D
        avg_hd = self.avg_pool_hd(x)  # [B,2C,1,H,D]
        max_hd = self.max_pool_hd(x)  # [B,2C,1,H,D]
        corr_hd = torch.cat([avg_hd, max_hd], dim=1)  # [B,4C,1,H,D]
        corr_hd = corr_hd.permute(0, 1, 3, 4, 2).reshape((B, 4 * C, H * D, 1, 1))

        corr_plane=torch.cat([corr_wh,corr_wd,corr_hd],dim=2)
        attn_plane=self.mlp(corr_plane)
        attn_wh, attn_wd, attn_hd=torch.split(attn_plane,[W*H,W*D,H*D],dim=2)
        attn_wh=attn_wh.reshape((B, C*2, W , H, 1))
        attn_wd = attn_wd.reshape((B,C*2, W, D, 1)).permute(0, 1, 2,  4,3)
        attn_hd = attn_hd.reshape((B, C*2, H, D, 1)).permute(0, 1, 4,2, 3)

        out=x*(attn_wh*attn_wd*attn_hd)+x

        out = self.relu(self.norm(self.pw(out)))
        return out



class SDNET(nn.Module):
    def __init__(self, residual=True, batch_norm=True, inst_norm=False, priors=True):
        super(SDNET, self).__init__()

        self.priors = priors
        # tsdf branch
        self.t0 = nn.Conv3d(1, 16, 3, stride=1, bias=True, padding=1)
        self.t1 = PCRBlock3D(16, 16, 16, units=1, pool=True, residual=residual, batch_norm=batch_norm,
                             inst_norm=inst_norm)
        self.t2 = PCRBlock3D(16, 16, 32, units=1, pool=True, residual=residual, batch_norm=batch_norm,
                                inst_norm=inst_norm)
        # depth branch
        if self.priors:
            self.d0 = nn.Conv3d(12, 16, 3, stride=1, bias=True, padding=1)
            self.d1 = PCRBlock3D(16, 16, 16, units=1, pool=False, residual=residual, batch_norm=batch_norm,
                                 inst_norm=inst_norm)
            self.d2 = PCRBlock3D(16, 16, 32, units=1, pool=False, residual=residual, batch_norm=batch_norm,
                                 inst_norm=inst_norm)

            self.fusion_0 = fusion_module(32,  bias=True, batch_norm=batch_norm)

        # final
        self.fd0 = Light_PCR_ASPP(32, 32, 32, residual=residual, batch_norm=batch_norm)

        self.enc1_ss = PCRBlock3D(32, 32, 32, units=1,  pool=False, residual=residual, batch_norm=batch_norm,
                                  inst_norm=inst_norm)

        self.enc2_ss = PCRBlock3D(32, 32, 32, units=1,  pool=True, residual=residual, batch_norm=batch_norm,
                                  inst_norm=inst_norm)

        self.enc3_ss = PCRBlock3D(32, 32, 32, units=1,  pool=True, residual=residual,
                                  batch_norm=batch_norm,
                                  inst_norm=inst_norm)
        # auxiliary
        self.aux1_ss = nn.Sequential(nn.Conv3d(32, 12, 1),
                                     # nn.Softmax(dim=1),
                                     )
        self.aux2_ss = nn.Sequential(nn.Conv3d(32, 12, 1),
                                     # nn.Softmax(dim=1),
                                     )
        self.aux3_ss = nn.Sequential(nn.Conv3d(32, 12, 1),
                                     # nn.Softmax(dim=1),
                                     )


        self.enc1_main = PCRBlock3D(32, 32, 32, units=1,  pool=False, residual=residual,
                                          batch_norm=batch_norm,
                                          inst_norm=inst_norm)
        self.enc2_main = PCRBlock3D(32, 32, 32, units=1,  pool=True, residual=residual,
                                          batch_norm=batch_norm,
                                          inst_norm=inst_norm)
        self.enc3_main = PCRBlock3D(32, 32, 32, units=1,  pool=True, residual=residual,
                                          batch_norm=batch_norm,
                                          inst_norm=inst_norm)
        self.fusion_enc1 = fusion_module(32,  bias=True, batch_norm=batch_norm)
        self.fusion_enc2 = fusion_module(32,  bias=True, batch_norm=batch_norm)
        self.fusion_enc3 = fusion_module(32,  bias=True, batch_norm=batch_norm)

        self.P3 = CBAM(32, 32, units=1)
        self.dec3 = nn.Sequential(
            PCRBlock3DUP(32, 32, 32, units=1, residual=residual, batch_norm=batch_norm,
                         inst_norm=inst_norm),
            Light_PCR_ASPP(32, 16, 32, residual=residual, batch_norm=batch_norm),
        )
        self.dec2 = nn.Sequential(
            PCRBlock3DUP(32*2, 32, 32, units=1,  residual=residual, batch_norm=batch_norm,
                         inst_norm=inst_norm),
            Light_PCR_ASPP(32, 16, 32, residual=residual, batch_norm=batch_norm),
        )
        self.dec1 = nn.Sequential(
            PCRBlock3D(32*2, 32, 32, units=1, pool=False, residual=residual, batch_norm=batch_norm,
                       inst_norm=inst_norm),
            Light_PCR_ASPP(32, 16, 32, residual=residual, batch_norm=batch_norm),
        )

        self.ssc_head = nn.Sequential(nn.Conv3d(32, 16, 1),
                                      nn.ReLU(),
                                      nn.Conv3d(16, 12, 1),
                                      )

    def forward(self, tsdf, depth=None):
        feature_t_0 = self.t0(tsdf)
        # print("feature_t_0",feature_t_0.shape)

        prior = False
        if (self.priors and (depth is not None)):
            # print("feature_d_3d",feature_d_3d.shape)
            feature_d_0 = self.d0(depth)
            # print("feature_d_0", feature_d_0.shape)
            feature_d_1 = self.d1(feature_d_0)
            # # print("feature_d_1",feature_d_1.shape)
            feature_d_2 = self.d2(feature_d_1)
            # print("feature_d_2",feature_d_2.shape)
            prior = True

        feature_t_1 = self.t1(feature_t_0)
        # # print("feature_t_1", feature_t_1.shape)
        feature_t_2 = self.t2(feature_t_1)
        # print("feature_t_2", feature_t_2.shape)

        feature3d = self.fusion_0(feature_t_2, feature_d_2 )if prior else feature_t_2
        # print("feature3d", feature3d.shape)

        # final
        f0 = self.fd0(feature3d)
        # print("f1", f1.shape)

        f_enc1_ss = self.enc1_ss(f0)
        # print("f_enc1_ss", f_enc1_ss.shape)
        f_enc2_ss = self.enc2_ss(f_enc1_ss)
        # print("f_enc2_ss", f_enc2_ss.shape)
        f_enc3_ss = self.enc3_ss(f_enc2_ss)
        # print("f_enc3_ss", f_enc3_ss.shape)

        aux_ss = {'1': self.aux1_ss(f_enc1_ss),
                  '2': self.aux2_ss(f_enc2_ss),
                  '4': self.aux3_ss(f_enc3_ss)}

        f_enc1 = self.fusion_enc1(self.enc1_main(f0), f_enc1_ss)
        # print("f_enc1", f_enc1.shape)
        f_enc2 = self.fusion_enc2(self.enc2_main(f_enc1), f_enc2_ss)
        # print("f_enc2", f_enc2.shape)
        f_enc3 = self.fusion_enc3(self.enc3_main(f_enc2), f_enc3_ss)
        # print("f_enc3", f_enc3.shape)

        f_enc3_p3 = self.P3(f_enc3)
        # print("f_enc3_se3", f_enc3_se3.shape)
        f_dec3 = self.dec3(f_enc3_p3)
        # print("f_dec3", f_dec3.shape)
        f_dec2 = self.dec2(torch.cat([f_enc2, f_dec3], dim=1))
        # print("f_dec2", f_dec2.shape)
        f_dec1 = self.dec1(torch.cat([f_enc1, f_dec2], dim=1))
        # print("f_dec1", f_dec1.shape)
        pred_ssc = self.ssc_head(f_dec1)
        return pred_ssc, aux_ss

