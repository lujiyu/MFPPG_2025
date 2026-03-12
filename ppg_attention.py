import random
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys
import os

from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
torch.cuda.device(cfg.cuda)


class MultiChannelFusion(nn.Module):
    def __init__(self, in_channels=5, mid_channels=64, out_channels=1, mode='all'):  # attn conv all avg
        super().__init__()
        self.mode = mode

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                             # [B, C, 1, 1]
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

        self.cross_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feats):  # feats: [B, C=5, T, F]
        if self.mode == 'attn':
            attn_weights = self.channel_attn(feats)          # [B, C, 1, 1]
            fused = feats * attn_weights                     # Broadcast: [B, C, T, F]
            fused = torch.sum(fused, dim=1, keepdim=True)    # [B, 1, T, F]

        elif self.mode == 'conv':
            fused = self.cross_conv(feats)                   # [B, 1, T, F]

        elif self.mode == 'all':
            #plt_logmel = feats[0,0,:,:].squeeze(0).cpu().detach().numpy()
            #plt.imshow(plt_logmel, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()
            attn_weights = self.channel_attn(feats)
            attn_fused = feats * attn_weights
            attn_fused = torch.sum(attn_fused, dim=1, keepdim=True)

            conv_fused = self.cross_conv(feats)              # [B, 1, T, F]
            fused = attn_fused + conv_fused
            #plt_logmel = fused[0,0,:,:].squeeze(0).cpu().detach().numpy()
            #plt.imshow(plt_logmel, origin='lower', aspect='auto', cmap='jet'), plt.show(), plt.close()

        elif self.mode == 'avg':
            fused = torch.mean(feats, dim=1, keepdim=True)

        else:
            fused = feats

        return fused  # [B, 1, T, F]


class AnomalyGen(nn.Module):
    def __init__(self, noise_std=0.1, patch_ratio=0.10, mode="gaussian", s=4, adaptive_noise=True, min_noise_std=0.02, size_threshold=32):
        super(AnomalyGen, self).__init__()
        self.noise_std = noise_std
        self.patch_ratio = patch_ratio
        self.mode = mode.lower()
        self.base_s = s
        self.adaptive_noise = adaptive_noise
        self.min_noise_std = min_noise_std
        self.size_threshold = size_threshold

    def forward(self, x):
        B, C, T, F = x.shape
        anomaly_x = x.clone()

        # noise_std
        noise_std = self.noise_std
        if self.adaptive_noise:
            min_dim = min(T, F)
            if min_dim < self.size_threshold:
                ratio = min_dim / self.size_threshold
                noise_std = max(self.min_noise_std, self.noise_std * ratio)

        # patch
        patch_area = int(T * F * self.patch_ratio)
        patch_h = max(1, min(int(patch_area ** 0.5), T // 2))
        patch_w = max(1, min(patch_area // patch_h, F // 2))

        # s
        s = min(self.base_s, max(1, min(T // (2 * patch_h), F // (2 * patch_w))))

        if self.mode == "gaussian":
            gauss_noise = torch.randn_like(x) * noise_std
            for i in range(B):
                if T - s * patch_h - 1 <= s or F - s * patch_w - 1 <= s:
                    anomaly_x[i] += gauss_noise[i]
                    continue

                t_center = random.randint(s, T - s * patch_h - 1)
                f_center = random.randint(s, F - s * patch_w - 1)

                for dt in range(-s, s + 1):
                    for df in range(-s, s + 1):
                        t_start = t_center + dt * patch_h
                        f_start = f_center + df * patch_w

                        if 0 <= t_start < T - patch_h and 0 <= f_start < F - patch_w:
                            anomaly_x[i, :, t_start:t_start + patch_h, f_start:f_start + patch_w] += \
                                gauss_noise[i, :, t_start:t_start + patch_h, f_start:f_start + patch_w]

        elif self.mode == "mask":
            for i in range(B):
                t_start = random.randint(0, T - patch_h)
                f_start = random.randint(0, F - patch_w)
                anomaly_x[i, :, t_start:t_start + patch_h, f_start:f_start + patch_w] = 0

        elif self.mode == "permute":
            for i in range(B):
                t_start = random.randint(0, T - patch_h)
                f_start = random.randint(0, F - patch_w)
                src_t = random.randint(0, T - patch_h)
                src_f = random.randint(0, F - patch_w)
                anomaly_x[i, :, t_start:t_start + patch_h, f_start:f_start + patch_w] = \
                    x[i, :, src_t:src_t + patch_h, src_f:src_f + patch_w]

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return anomaly_x


### mobilenetv2
#__all__ = ['mobilenetv2']
def _make_divisible(v, divisor, min_value=None):  # 32 8
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):  # model size 8.515MB
    def __init__(self, input_dim=1, num_classes=cfg.class_num, width_mult=1.,learn_method=cfg.learn_method,aug_method=''):
        super(MobileNetV2, self).__init__()
        self.learn_method = learn_method
        self.aug_method = aug_method
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(input_dim, input_channel, 2)]  # layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

        self.anomalygen = AnomalyGen()

    def forward(self, x):
        x = self.features(x)
        if (self.learn_method == 'Contrastive') & (self.aug_method == 'anomalygen'):
            x_anomaly = self.anomalygen(x.clone())
            x = torch.cat([x, x_anomaly], dim=0)
            #feat_x = x.clone()
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if (self.learn_method == 'Contrastive') or (cfg.premodel_ext == 'True'):
            x1 = self.classifier(x)
            return x1,x
            #return feat_x, x
        else:
            x = self.classifier(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 models
    """
    return MobileNetV2(**kwargs)



class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_1, z_2):
        batch_size, device = z_1.size(0), z_1.device

        # compute similarities between all the 2N views
        z = torch.cat([z_1, z_2], dim=0)  # (2 * bs, dim_emb)
        similarity = F.cosine_similarity(z[:, None], z[None, :], dim=2) / self.temperature  # (2 * bs, 2 * bs)
        sim_ij = torch.diag(similarity, batch_size)  # (bs,)
        sim_ji = torch.diag(similarity, -batch_size)  # (bs,)

        # positive contains the 2N similarities between two views of the same sample
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2 * bs,)

        # negative contains the (2N, 2N - 1) similarities between the view of a sample and all the other views that are
        # not from that same sample
        mask = ~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)  # (2 * bs, 2 * bs)
        negatives = similarity[mask].view(2 * batch_size, 2 * batch_size - 1)  # (2 * bs, 2 * bs - 1)

        # the loss can be rewritten as the sum of the alignement loss making the two representations of the same
        # sample closer, and the distribution loss making the representations of different samples farther
        loss_alignement = -torch.mean(positives)
        loss_distribution = torch.mean(torch.logsumexp(negatives, dim=1))
        loss = loss_alignement + loss_distribution

        return loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batchsize = 8
    x = torch.randn(size=(batchsize, 1, 128, 128))
    model = MobileNetV2() # MobileNetV2()
    output_z = model(x)  # , output_recon
    print(f'input dim:{x.shape}. output_z dim:{output_z[1].shape}.')  #  output_recon dim:{output_recon.shape}
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the models: {num_params}")
    num_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Number of parameters in bytes: {num_params}")
    import MLP_head as mlp
    model_head = mlp.MLPHead(in_channels=1280, mlp_hidden_size=512)
    y=model_head(output_z[1])
    print(y)
    num_params = sum(p.numel() for p in model_head.parameters())
    print(f"Number of parameters in the models: {num_params}")
    num_params = sum(p.numel() * p.element_size() for p in model_head.parameters())

    print(f"Number of parameters in bytes: {num_params}")