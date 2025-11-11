import torch
import torch.nn as nn
import torch.nn.functional as F

class PCRUnit3D(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True,
                 batch_norm=False, inst_norm=False):
        super(PCRUnit3D, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d
        self.batch_norm = batch_norm
        self.conv_in = nn.Conv3d(c_in, c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c) if batch_norm else None
        self.conv1x3x3 = nn.Conv3d(c, c, (1, k, k), stride=s, padding=(0, p, p), bias=True, dilation=(1, d, d))
        self.conv3x3x1 = nn.Conv3d(c, c, (k, k, 1), stride=s, padding=(p, p, 0), bias=True, dilation=(d, d, 1))
        self.conv3x1x3 = nn.Conv3d(c, c, (k, 1, k), stride=s, padding=(p, 0, p), bias=True, dilation=(d, 1, d))
        self.bn4 = nn.BatchNorm3d(c) if batch_norm else None
        self.conv_out = nn.Conv3d(c, c_out, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm3d(c_out) if batch_norm else None
        self.residual = residual
        self.conv_resid = nn.Conv3d(c_in, c_out, kernel_size=1, bias=False) if residual and c_in != c_out else None
        self.inst_norm = nn.InstanceNorm3d(c_out) if inst_norm else None

    def forward(self, x):
        y0 = self.conv_in(x)
        if self.batch_norm:
            y0 = self.bn1(y0)
        y0 = F.relu(y0, inplace=True)

        y1 = self.conv1x3x3(y0)
        y1 = F.relu(y1, inplace=True)

        y2 = self.conv3x3x1(y1) + y1
        y2 = F.relu(y2, inplace=True)

        y3 = self.conv3x1x3(y2) + y2 + y1
        if self.batch_norm:
            y3 = self.bn4(y3)
        y3 = F.relu(y3, inplace=True)

        y = self.conv_out(y3)
        if self.batch_norm:
            y = self.bn5(y)

        x_squip = x if self.conv_resid is None else self.conv_resid(x)

        y = y + x_squip if self.residual else y

        y = self.inst_norm(y) if self.inst_norm else y

        y = F.relu(y, inplace=True)

        return y

class PCRBlock3D(nn.Module):
    def __init__(self, c_in, c, c_out, units=2, kernel=3, stride=1, dilation=1,
                 pool=True, residual=True, batch_norm=False, inst_norm=False):
        super(PCRBlock3D, self).__init__()
        self.pool = nn.MaxPool3d(2, stride=2) if pool else None
        self.units = nn.ModuleList()
        for i in range(units):
            if i == 0:
                self.units.append(PCRUnit3D(c_in, c, c_out, kernel, stride, dilation, residual, batch_norm, inst_norm))
            else:
                self.units.append(PCRUnit3D(c_out, c, c_out, kernel, stride, dilation, residual, batch_norm, inst_norm))

    def forward(self, x):
        y = self.pool(x) if self.pool is not None else x
        for unit in self.units:
            y = unit(y)
        return y



class Light_PCR_ASPP(nn.Module):
    def __init__(self, c_in, c, c_out, residual=False, batch_norm=False, ):
        super(Light_PCR_ASPP, self).__init__()
        print('Light_PCR_ASPP3d: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))

        self.conv_in=nn.Sequential(nn.Conv3d(c_in, c, 1, 1, 0),
                                   nn.ReLU())
        self.aspp0 = nn.Sequential(nn.Conv3d(c, c, 1, 1, 0),
                                   nn.ReLU())

        self.aspp1 = PCRUnit3D(c, c, c, dilation=3, residual=residual, batch_norm=batch_norm)

        self.aspp2 = PCRUnit3D(c, c, c, dilation=5, residual=residual, batch_norm=batch_norm)

        self.aspp3 = PCRUnit3D(c, c, c, dilation=7, residual=residual, batch_norm=batch_norm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                             nn.Conv3d(c, c, 1, 1, 0),
                                             nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv3d(c * 5, c_out, 1, 1, 0),
                                   nn.ReLU())

    def forward(self, x):
        x_in=self.conv_in(x)
        x0 = self.aspp0(x_in)
        x1 = self.aspp1(x_in)
        x2 = self.aspp2(x_in)
        x3 = self.aspp3(x_in)
        x_ = self.global_avg_pool(x_in)

        # x_ = F.upsample(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x_ = F.interpolate(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x_.shape)

        x = torch.cat((x0, x1, x2, x3, x_), dim=1)
        x = self.conv1(x)
        return x

class PCRBlock3DUP(nn.Module):
    def __init__(self, c_in, c, c_out, units=2, kernel=3, stride=1, dilation=1, residual=True,
                 batch_norm=False, inst_norm=False):
        super(PCRBlock3DUP, self).__init__()
        self.transp = nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2)
        self.units = nn.ModuleList()
        for _ in range(units):
            self.units.append(PCRUnit3D(c_out, c, c_out, kernel, stride, dilation, residual, batch_norm, inst_norm))

    def forward(self, x):
        y = self.transp(x)
        for ddr_unit in self.units:
            y = ddr_unit(y)
        return y