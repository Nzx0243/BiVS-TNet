import torch
import torch.nn as nn

class NDCSE(nn.Module):
    def __init__(self, inc, channel, entropy_lambda=0.1, gamma=2.0, beta=2.0, inhibit_lambda=0.5):
        super(NDCSE, self).__init__()
        C3, C4, C5 = inc
        self.entropy_lambda = entropy_lambda
        self.gamma = gamma
        self.beta = beta
        self.inhibit_lambda = inhibit_lambda

        self.conv3 = nn.Sequential(
            nn.Conv2d(C3, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(C4, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(C5, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
        )

        self.upsample_4_to_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_5_to_3 = nn.Upsample(scale_factor=4, mode='nearest')

    def compute_l2_energy(self, feat):
        N, C, H, W = feat.shape
        energy = torch.sqrt(torch.sum(feat ** 2, dim=(2, 3)) + 1e-6)
        return energy.mean(dim=1)

    def compute_entropy_loss(self, feat):
        N, C, H, W = feat.shape
        eps = 1e-6
        feat_flat = feat.view(N, C, -1)
        probs = feat_flat / (feat_flat.sum(dim=2, keepdim=True) + eps)
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=2)
        entropy_mean = entropy.mean(dim=1, keepdim=True)
        loss = ((entropy - entropy_mean) ** 2).mean()
        return self.entropy_lambda * loss

    def excite(self, x):
        return x.sign() * (1 - torch.exp(-self.gamma * x.abs()))

    def inhibit(self, x):
        return 2 / (1 + torch.exp(-self.beta * x)) - 1

    def forward(self, x, return_entropy_loss=False):
        p3, p4, p5 = x
        p3_m = self.conv3(p3)
        p4_m = self.conv4(p4)
        p5_m = self.conv5(p5)

        p4_up = self.upsample_4_to_3(p4_m)
        p5_up = self.upsample_5_to_3(p5_m)

        N = p3_m.shape[0]

        h3 = self.compute_l2_energy(p3_m)
        h4 = self.compute_l2_energy(p4_up)
        h5 = self.compute_l2_energy(p5_up)
        entropy_stack = torch.stack([h3, h4, h5], dim=0)
        weight = torch.softmax(-2.0 * entropy_stack, dim=0)

        w3 = weight[0].view(N, 1, 1, 1)
        w4 = weight[1].view(N, 1, 1, 1)
        w5 = weight[2].view(N, 1, 1, 1)

        fused = w3 * p3_m + w4 * p4_up + w5 * p5_up

        F_enhanced = self.excite(fused) + self.inhibit_lambda * self.inhibit(p3_m)

        if return_entropy_loss:
            loss = (
                self.compute_entropy_loss(p3_m)
                + self.compute_entropy_loss(p4_up)
                + self.compute_entropy_loss(p5_up)
            ) / 3
            return F_enhanced, loss
        else:
            return F_enhanced