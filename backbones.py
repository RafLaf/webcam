import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import random

def linear(indim, outdim):
    return nn.Linear(indim, outdim)


class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return out
    
class ResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(ResNet12, self).__init__()
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps))
        self.layers = nn.Sequential(*layers)
        self.linear = linear(10 * feature_maps, num_classes)
        self.rotations = rotations
        self.linear_rot = linear(10 * feature_maps, 4)
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, index_mixup = None, lam = -1):
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = self.mp(F.leaky_relu(out, negative_slope = 0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features
        return out, features

class Clip(nn.Module):
    def __init__(self, name, device, return_tokens = False):
        super(Clip, self).__init__()
        self.backbone = clip.load(name, device=device)[0]
        self.return_tokens = return_tokens
    def forward(self, x, mixup = None, lbda = None, perm = None):
        if self.return_tokens:
            x = self.backbone.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.backbone.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.backbone.visual.positional_embedding.to(x.dtype)
            x = self.backbone.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.backbone.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            out = self.backbone.visual.ln_post(x)
        else:
            out = self.backbone.encode_image(x)
        return out

def load_model_weights(model, path, device):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if 'bn' in k:
                new_dict[k] = v
            else:
                new_dict[k] = v.half()
    model_dict.update(new_dict) 
    model.load_state_dict(model_dict)
    print('Model loaded!')

# Get the model
def get_model(model_name, model_path, device):
    if model_name == 'resnet12':
        model = ResNet12(64, [3, 84, 84], 351, True, False).to(device)
        load_model_weights(model, model_path, device)
    elif model_name == 'clip':
        model, _ = Clip('ViT-B/32', device, return_tokens=False)
    else:
        raise NotImplementedError
    return model