import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

save_path = "static/vgg19.pth"


class PerceptualLoss(nn.Module):
    def __init__(self, layers=None, normalize_inputs: bool = True, model='vgg19', alpha=0.01, scale=100):
        super(PerceptualLoss, self).__init__()
        if layers is None:
            layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
        self.layers = layers
        self.vgg = models.vgg19()
        self.vgg.load_state_dict(torch.load(save_path))

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        self.alpha = alpha
        self.scale = scale

        vgg_avg_pooling = []
        self.select_layers = []
        layer_id = 0

        for module in self.vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
                self.select_layers.append(layer_id - 1)
                layer_id += 1
            else:
                vgg_avg_pooling.append(module)
                layer_id += 1

        self.vgg = nn.Sequential(*vgg_avg_pooling)

        if len(self.select_layers) == 5:
            self.W_init = [100., 1.6, 2.3, 1.8, 2.8, 100.]
        else:
            self.W_init = [1 for _ in range(len(self.select_layers) + 1)]

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        # check_and_warn_input_range(target, 0, 1, 'PerceptualLoss target in partial_losses')

        # we expect input and target to be in [0, 1] range
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target

        loss = F.mse_loss(features_input, features_target, reduction='none')
        if mask is not None:
            loss = loss * mask
        loss_id = 0
        loss = loss.mean()
        self.W_init[loss_id] = self.W_init[loss_id] + self.alpha * (loss.item() - self.W_init[loss_id])
        losses.append(loss / self.W_init[loss_id] * self.scale)
        for id_layer, layer in enumerate(self.vgg):

            features_input = layer(features_input)
            features_target = layer(features_target)

            if id_layer in self.select_layers:
                loss_id += 1
                loss = F.mse_loss(features_input, features_target, reduction='none')

                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:],
                                             mode='bilinear', align_corners=False)
                    loss = loss * cur_mask

                loss = loss.mean()
                self.W_init[loss_id] = self.W_init[loss_id] + self.alpha * (loss.item() - self.W_init[loss_id])
                losses.append(loss / self.W_init[loss_id] * self.scale)

        return losses

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        input_features = self.get_features(input)
        target_features = self.get_features(target)


if __name__ == '__main__':
    vgg19 = models.vgg19(weights='VGG19_Weights.DEFAULT')
    save_path = "static/vgg19.pth"
    torch.save(vgg19.state_dict(), save_path)
