class DSTSM(nn.Module):
    def __init__(
        self, channel, q_size, n_heads=8, n_groups=4,
        attn_drop=0.0, proj_drop=0.0, stride=1,
        offset_range_factor=4, use_pe=True, dwc_pe=True,
        no_off=False, fixed_pe=False, ksize=3, log_cpb=False, kv_size=None,
        target_center=(0.5, 0.22),
        target_std=(0.04, 0.4),
    ):
        super().__init__()

        n_head_channels = channel // n_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h = q_size
        self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups

        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride

        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.bias_proj = nn.Conv2d(1, self.n_group_channels, kernel_size=1)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.n_group_channels,
                      self.n_group_channels, kk, stride, pad_size,
                      groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 3, 1, 1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

        tc = torch.tensor(target_center, dtype=torch.float32)
        ts = torch.tensor(target_std, dtype=torch.float32)
        self.target_center = nn.Parameter(tc)
        self.target_log_std = nn.Parameter(torch.log(ts))

        init_bias = torch.ones(n_heads) * 0.05
        self.bias_strength = nn.Parameter(init_bias)

        init_theta = torch.tensor(0.0)
        self.theta = nn.Parameter(init_theta)

        self.lambda1 = nn.Parameter(torch.tensor(0.5))
        self.lambda2 = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(5.0))

        self.dir_convs = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        ])

        self.gate_proj = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def _create_spatial_bias(self, H, W, dtype, device):
        y, x = torch.meshgrid(
            torch.linspace(0, 1, H, dtype=dtype, device=device),
            torch.linspace(0, 1, W, dtype=dtype, device=device),
            indexing='ij'
        )
        pos = torch.stack((y, x), dim=-1)
        delta = pos - self.target_center.view(1, 1, 2)
        std = torch.exp(self.target_log_std)
        sigma_y, sigma_x = std[0], std[1]
        theta = self.theta
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=device, dtype=dtype)
        rotated = torch.einsum('ij,hwj->hwi', R, delta)
        x_prime = rotated[..., 1]
        y_prime = rotated[..., 0]
        exp_term = -0.5 * ((y_prime / sigma_y) ** 2 + (x_prime / sigma_x) ** 2)
        dist = torch.exp(exp_term)
        min_val = dist.min()
        max_val = dist.max()
        if max_val - min_val > 1e-6:
            norm_dist = (dist - min_val) / (max_val - min_val)
        else:
            norm_dist = dist - min_val
        mean_val = norm_dist.mean()
        centered = norm_dist - mean_val
        bias_maps = centered.unsqueeze(0).repeat(self.n_heads, 1, 1)
        bs = self.bias_strength.view(self.n_heads, 1, 1)
        bias_maps = bs * bias_maps
        bias_maps = bias_maps.view(self.n_heads, H * W)
        return bias_maps

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        Hk, Wk = H // self.stride, W // self.stride
        bias_full = self._create_spatial_bias(H, W, dtype, device)
        bias_full_map = bias_full.view(self.n_heads, H, W).unsqueeze(1)
        bias_off = F.interpolate(bias_full_map, size=(Hk, Wk), mode='bilinear', align_corners=True)
        bias_off_guide = bias_off.mean(dim=0, keepdim=True).expand(B * self.n_groups, 1, Hk, Wk)

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)

        q_off_down = F.avg_pool2d(q_off, self.stride) if self.stride > 1 else q_off
        bias_mapped = self.bias_proj(bias_off_guide)
        conv_in = torch.cat((q_off_down, bias_mapped), dim=1)
        offset = self.conv_offset(conv_in)

        phi1 = 2.0 * torch.sigmoid(offset) - 1.0
        phi2 = torch.sign(offset) * (1.0 - torch.exp(-self.gamma * torch.abs(offset)))
        offset_modulated = self.lambda1 * phi1 + self.lambda2 * phi2

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).view(1, 2, 1, 1)
            offset = offset_modulated.mul(offset_range).mul(self.offset_range_factor)
        else:
            offset = offset_modulated

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        offset = offset.fill_(0.0) if self.no_off else offset
        pos = offset + reference if self.offset_range_factor >= 0 else (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        else:
            pos = pos.type(x.dtype)
            x_g = x.reshape(B * self.n_groups, self.n_group_channels, H, W)
            x_sampled = F.grid_sample(x_g, grid=pos[..., (1, 0)], mode='bilinear', align_corners=True)
        x_sampled = x_sampled.reshape(B, C, 1, Hk * Wk)

        q_flat = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_flat = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, Hk * Wk)
        v_flat = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, Hk * Wk)

        attn = torch.einsum('b c m, b c n -> b m n', q_flat, k_flat) * self.scale
        bias_full = bias_full.unsqueeze(0).expand(B, -1, -1).reshape(B * self.n_heads, H * W)
        attn = attn + bias_full.unsqueeze(-1)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                attn_bias = self.rpe_table[None, ...].expand(B, -1, -1, -1)
                attn += attn_bias.reshape(B * self.n_heads, H * W, Hk * Wk)

        attn_centered = attn - attn.mean(dim=-1, keepdim=True)
        topk = 8
        threshold = torch.topk(attn_centered, k=topk, dim=-1)[0][:, :, -1:]
        attn_sparse = attn_centered.masked_fill(attn_centered < threshold, float('-inf'))
        attn = F.softmax(attn_sparse, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v_flat)
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        if hasattr(self, 'dir_convs') and hasattr(self, 'gate_proj'):
            y_res = 0
            for conv in self.dir_convs:
                y_res += conv(out)
            y_res /= len(self.dir_convs)
            gate = torch.sigmoid(self.gate_proj(F.adaptive_avg_pool2d(out, 1)))
            out = out + gate * y_res

        y = self.proj_drop(self.proj_out(out))
        return y
