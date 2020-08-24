import torch.nn as nn
import functools

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, opt):
        image_size = opt.image_size
        ngf = opt.ngf
        ndf = opt.ndf
        normtype_D = opt.normtype_D
        normtype_G = opt.normtype_G
        sn_G = opt.use_sn_G
        sn_D = opt.use_sn_D
        if network_name == 'unet_msf_generator':
            from .generator import UNet_MSF_Generator
            network = UNet_MSF_Generator(conv_dim=ngf, c_dim=opt.cond_nc, normtype=normtype_G, sn=sn_G)
        elif network_name == 'discriminator_wgan_plus':
            from .discriminator import Discriminator_Plus
            network = Discriminator_Plus(image_size=image_size, conv_dim=ndf, c_dim=opt.cond_nc,
                                       normtype=normtype_D, sn=sn_D)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print("Network %s was created" % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer
