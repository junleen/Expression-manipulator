import torch.nn as nn
from .networks import NetworkBase
import torch
import torch.nn.utils.spectral_norm as spectral_norm
from networks.normalize import get_norm_layer


def downconv_norm_lrelu(input_dim, output_dim, kernel=4, stride=2, padding=1, bias=False, normtype='batchnorm', sn=False):
    conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
    norm = get_norm_layer(output_dim, normtype)
    lrelu = nn.LeakyReLU()
    if sn:
        conv = spectral_norm(conv)
    if norm is not None:
        layer = [conv, norm, lrelu]
    else:
        layer = [conv, lrelu]
    return nn.Sequential(*layer)

def upconv_norm_lrelu(input_dim, output_dim, kernel=4, stride=2, padding=1, bias=False, normtype='batchnorm', sn=False):
    conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
    norm = get_norm_layer(output_dim, normtype)
    lrelu = nn.LeakyReLU()
    if sn:
        conv = spectral_norm(conv)
    if norm is not None:
        layer = [conv, norm, lrelu]
    else:
        layer = [conv, lrelu]
    return nn.Sequential(*layer)

def upconv_norm_relu(input_dim, output_dim, kernel=4, stride=2, padding=1, bias=False, normtype='batchnorm', sn=False):
    conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
    norm = get_norm_layer(output_dim, normtype)
    lrelu = nn.ReLU()
    if sn:
        conv = spectral_norm(conv)
    if norm is not None:
        layer = [conv, norm, lrelu]
    else:
        layer = [conv, lrelu]
    return nn.Sequential(*layer)

def conv_norm_relu(input_dim, output_dim, kernel=3, stride=1, padding=1, bias=False, normtype='batchnorm', sn=False):
    conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
    norm = get_norm_layer(output_dim, normtype)
    lrelu = nn.ReLU()
    if sn:
        conv = spectral_norm(conv)
    if norm is not None:
        layer = [conv, norm, lrelu]
    else:
        layer = [conv, lrelu]
    return nn.Sequential(*layer)


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, normtype='batchnorm', sn=False):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            conv_norm_relu(dim_in, dim_out, kernel=3, stride=1, padding=1, bias=False, normtype=normtype, sn=sn),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))
        self.downsample = None
        if dim_out != dim_in:
            self.downsample = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.main(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        return residual + out

class MultiScaleModule(nn.Module):
    def __init__(self, input_dim, dconv_dim, output_dim, c_dim=17, normtype='batchnorm', sn=False):
        super(MultiScaleModule, self).__init__()
        self.in_conv = conv_norm_relu(input_dim + c_dim, input_dim, kernel=3, stride=1, padding=1,
                                       bias=False, normtype=normtype, sn=sn)
        downconv_dim = min(input_dim, dconv_dim)
        self.down_conv = downconv_norm_lrelu(input_dim + c_dim, downconv_dim, kernel=4, stride=2, padding=1,
                                             bias=False, normtype=normtype, sn=sn)
        transconv_dim = min(dconv_dim, output_dim)
        self.transpose_conv = upconv_norm_lrelu(downconv_dim + dconv_dim, transconv_dim, kernel=4, stride=2,
                                               padding=1, bias=False, normtype=normtype, sn=sn)

        self.out_conv1x1 = conv_norm_relu(transconv_dim + input_dim, output_dim, kernel=1, stride=1, padding=0,
                                          bias=False, normtype=normtype, sn=sn)
        self.out_conv3x3 = conv_norm_relu(transconv_dim + input_dim, output_dim, kernel=3, stride=1, padding=1,
                                          bias=False, normtype=normtype, sn=sn)
        self.out_conv = conv_norm_relu(output_dim * 2, output_dim, kernel=3, stride=1, padding=1,
                                       bias=False, normtype=normtype, sn=sn)

    def forward(self, x, dconv_x, c):
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        # higher resolution
        alpha = self.in_conv(x)
        # lower resolution
        encoded_feature = self.down_conv(x)
        low_resolution = torch.cat([encoded_feature, dconv_x], dim=1)
        beta = self.transpose_conv(low_resolution)

        out = torch.cat([alpha, beta], dim=1)
        out1 = self.out_conv1x1(out)
        out2 = self.out_conv3x3(out)
        out = torch.cat([out1, out2], dim=1)
        out = self.out_conv(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, input_hr_dim, input_lr_dim, output_hr_dim, output_lr_dim, normtype='instancenorm', sn=False):
        super(FusionBlock, self).__init__()
        self.hr_to_hr = conv_norm_relu(input_hr_dim, output_hr_dim//2, kernel=3, stride=1, padding=1,
                                       bias=False, normtype=normtype, sn=sn)
        self.hr_to_lr = downconv_norm_lrelu(input_hr_dim, output_lr_dim//2, kernel=4, stride=2, padding=1,
                                            bias=False, normtype=normtype, sn=sn)
        self.lr_to_hr = upconv_norm_lrelu(input_lr_dim, output_hr_dim-output_hr_dim//2, kernel=4, stride=2, padding=1,
                                          bias=False, normtype=normtype, sn=sn)
        self.lr_to_lr = conv_norm_relu(input_lr_dim, output_lr_dim-output_lr_dim//2, kernel=3, stride=1, padding=1,
                                       bias=False, normtype=normtype, sn=sn)

    def forward(self, hr_x, lr_x):
        out_hr1 = self.hr_to_hr(hr_x)
        out_hr2 = self.lr_to_hr(lr_x)
        out_hr_x = torch.cat([out_hr1, out_hr2], dim=1)

        out_lr1 = self.hr_to_lr(hr_x)
        out_lr2 = self.lr_to_lr(lr_x)
        out_lr_x = torch.cat([out_lr1, out_lr2], dim=1)
        return out_hr_x, out_lr_x

class MultiScaleFusion(nn.Module):
    def __init__(self, input_dim, dconv_dim, output_dim, c_dim=17, normtype='instancenorm', sn=False):
        super(MultiScaleFusion, self).__init__()
        self.in_conv = conv_norm_relu(input_dim + c_dim, input_dim, kernel=5, stride=1, padding=2,
                                       bias=False, normtype=normtype, sn=sn)
        # Block1
        self.block1 = FusionBlock(input_hr_dim=input_dim, input_lr_dim=dconv_dim,
                                  output_hr_dim=input_dim, output_lr_dim=dconv_dim, normtype=normtype, sn=sn)
        # self.block2 = FusionBlock(input_hr_dim=input_dim, input_lr_dim=dconv_dim, output_hr_dim=input_dim, output_lr_dim=dconv_dim, normtype=normtype, sn=sn)

        transconv_dim = min(dconv_dim, output_dim)
        self.transpose_conv = upconv_norm_lrelu(dconv_dim, transconv_dim, kernel=4, stride=2, padding=1,
                                                bias=False, normtype=normtype, sn=sn)
        self.out_conv1 = conv_norm_relu(transconv_dim + input_dim, output_dim//2, kernel=3, stride=1, padding=1,
                                       bias=False, normtype=normtype, sn=sn)
        self.out_conv2 = conv_norm_relu(transconv_dim + input_dim, output_dim-output_dim//2, kernel=5, stride=1, padding=2,
                                       bias=False, normtype=normtype, sn=sn)

    def forward(self, x, dconv_x, c):
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)    # input_dim+c_dim x HR
        # higher resolution
        high_resolution = self.in_conv(x)     # input_dim x HR
        # lower resolution
        high_resolution, low_resolution = self.block1(high_resolution, dconv_x)  # input_dim, dconv_dim
        # high_resolution, low_resolution = self.block2(high_resolution, low_resolution)  # input_dim, dconv_dim

        out = self.transpose_conv(low_resolution)
        out = torch.cat([high_resolution, out], dim=1)
        out1 = self.out_conv1(out)
        out2 = self.out_conv2(out)
        return torch.cat([out1, out2], dim=1)


class UNet_MSF_Generator(NetworkBase):
    def __init__(self, conv_dim=64, c_dim=17, normtype='instancenorm', sn=False):
        super().__init__()
        self._name = 'unet_msf_generator'
        bottle_neck_dim = min(512, conv_dim * 16)
        # Conv0
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.LeakyReLU(inplace=True),
        )       # 128 x 128

        # Down sampling
        self.downconv1 = downconv_norm_lrelu(conv_dim, conv_dim * 2, normtype=normtype, sn=sn)     # 64 x 64
        self.downconv2 = downconv_norm_lrelu(conv_dim * 2, conv_dim * 4, normtype=normtype, sn=sn)  # 32 x 32
        self.downconv3 = downconv_norm_lrelu(conv_dim * 4, conv_dim * 8, normtype=normtype, sn=sn)  # 16 x 16
        self.downconv4 = downconv_norm_lrelu(conv_dim * 8, bottle_neck_dim, normtype=normtype, sn=sn)  # 8 x 8
        self.resblock = nn.Sequential(
            conv_norm_relu(bottle_neck_dim + c_dim, bottle_neck_dim, normtype=normtype, sn=sn),
            # ResidualBlock(bottle_neck_dim, bottle_neck_dim, normtype=normtype, sn=sn)
        )     # 8 x 8

        # Multi-scale Unit
        self.skipu1 = MultiScaleFusion(input_dim=conv_dim*2, dconv_dim=conv_dim*4, output_dim=conv_dim*2,
                                       c_dim=c_dim, normtype=normtype, sn=sn)   # 64 x 64
        self.skipu2 = MultiScaleFusion(input_dim=conv_dim*4, dconv_dim=conv_dim*8, output_dim=conv_dim*4,
                                       c_dim=c_dim, normtype=normtype, sn=sn)   # 32 x 32
        self.skipu3 = MultiScaleFusion(input_dim=conv_dim*8, dconv_dim=bottle_neck_dim, output_dim=conv_dim*8,
                                       c_dim=c_dim, normtype=normtype, sn=sn)   # 16 x 16
        # Up sampling
        self.upconv4 = upconv_norm_relu(bottle_neck_dim, conv_dim*8, normtype=normtype, sn=sn)    # conv_dim*8 x 16 x 16
        self.upconv3 = upconv_norm_relu(conv_dim*8 + conv_dim*8, conv_dim*4, normtype=normtype, sn=sn)    # 32 x 32
        self.upconv2 = upconv_norm_relu(conv_dim*4 + conv_dim*4, conv_dim*2, normtype=normtype, sn=sn)     # conv_dim*4 x 64 x 64
        self.upconv1 = upconv_norm_relu(conv_dim*2 + conv_dim*2, conv_dim*1, normtype=normtype, sn=sn)     # conv_dim*2 x 128 x 128

        self.out_conv = nn.Sequential(
            nn.Conv2d(conv_dim*1, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, c):
        features = self.conv0(x)
        down1 = self.downconv1(features)    # conv_dim*2 x 64 x 64
        down2 = self.downconv2(down1)       # conv_dim*4 x 32 x 32
        down3 = self.downconv3(down2)       # conv_dim*8 x 16 x 16
        down4 = self.downconv4(down3)       # bottle_neck_dim x 8 x 8
        cmap = c.unsqueeze(2).unsqueeze(3)
        cmap = cmap.expand(cmap.size(0), cmap.size(1), down4.size(2), down4.size(3))
        blockin = torch.cat([down4, cmap], dim=1)  # conv_dim*12+c_dim x 8 x 8
        blockout = self.resblock(blockin)       # conv_dim*12 x 8 x 8

        skip3 = self.skipu3(down3, blockout, c)    # bottle_neck_dim x 16 x 16
        skip2 = self.skipu2(down2, skip3, c)    # conv_dim*8 x 32 x 32
        skip1 = self.skipu1(down1, skip2, c)    # conv_dim*4 x 64 x 64

        out = self.upconv4(blockout)            # conv_dim*8 x 16 x 16
        out = torch.cat([out, skip3], dim=1)    # conv_dim*8+bottle_neck_dim x 16 x 16
        out = self.upconv3(out)                 # conv_dim*8 x 32 x 32
        out = torch.cat([out, skip2], dim=1)    # conv_dim*8+conv_dim*8 x 32 x 32
        out = self.upconv2(out)                 # conv_dim*4 x 64 x 64
        out = torch.cat([out, skip1], dim=1)    # conv_dim*4+conv_dim*4 x 64 x 64
        out = self.upconv1(out)                 # conv_dim*2 x 128 x 128

        return self.out_conv(out)


# from networks.generator import UNet_MSF_Generator
# import torch
# x = torch.randn(2, 3, 128, 128)
# x = torch.randn(2, 3, 160, 160)
# c = torch.randn(2, 13)
# net = UNet_MSF_Generator(c_dim=13)
# y = net(x, c)