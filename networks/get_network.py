from networks.UNet import UNet
from networks.S_UNet import S_UNet


def get_network(network):
    if network == 'UNet':
        return UNet(in_channels=1, out_channels=1, bilinear=True)
    elif network == 'S_UNet':
        return S_UNet(in_channels=1, base_filters=16, bias=True)
    return UNet(in_channels=1, out_channels=1, bilinear=True)
