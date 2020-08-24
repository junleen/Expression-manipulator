import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch.nn.utils.spectral_norm as spectral_norm
from networks.normalize import get_norm_layer


def downconv_norm_lrelu(input_dim, output_dim, kernel=4, stride=2, padding=1, bias=False, normtype='batchnorm', sn=False):
    conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
    norm = get_norm_layer(output_dim, normtype)
    lrelu = nn.LeakyReLU(inplace=True)
    if sn:
        conv = spectral_norm(conv)
    if norm is not None:
        layer = [conv, norm, lrelu]
    else:
        layer = [conv, lrelu]
    return nn.Sequential(*layer)


class Discriminator_Plus(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=17, repeat_num=6, normtype='None', sn=False):
        super(Discriminator_Plus, self).__init__()
        self._name = 'discriminator_wgan_plus'

        layers = []
        layers.append(downconv_norm_lrelu(3, conv_dim, kernel=5, stride=2, padding=2, bias=False,
                                          normtype=normtype, sn=sn))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(downconv_norm_lrelu(curr_dim, curr_dim*2, kernel=4, stride=2, padding=1,
                                              bias=False, normtype=normtype, sn=sn))
            curr_dim = curr_dim * 2
        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv1 = nn.Sequential(*layers)

        layers = []
        k_size = int(image_size / np.power(2, repeat_num))
        layers.append(nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False))
        self.conv2 = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()

