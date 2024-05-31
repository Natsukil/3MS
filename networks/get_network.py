from networks.UNet import UNet
from networks.S_UNet import S_UNet


def get_network(network, concat_method):
    if network == 'UNet':
        if concat_method == 'plane':
            return UNet(in_channels=1, out_channels=1, bilinear=True)
        elif concat_method == 'channels':
            return UNet(in_channels=4, out_channels=4, bilinear=True)
    elif network == 'S_UNet':
        return S_UNet(in_channels=1, base_filters=16, bias=True)
    return UNet(in_channels=1, out_channels=1, bilinear=True)
