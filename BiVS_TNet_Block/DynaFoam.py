class DynaFoam(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # 不再固定倍率，留到 forward 中动态计算
        self.dysample_s = None  # 延迟初始化

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        l = F.adaptive_max_pool2d(l, m.shape[2:]) + F.adaptive_avg_pool2d(l, m.shape[2:])
        
        # 动态计算倍率（例如 m.height / s.height）
        scale_factor = m.shape[2] // s.shape[2]  # 假设高度方向比例
        if self.dysample_s is None or self.dysample_s.scale != scale_factor:
            self.dysample_s = DySample(s.shape[1], scale_factor, 'lp')  # 按需创建
        
        s = self.dysample_s(s)
        lms = torch.cat([l, m, s], dim=1)
        return lms