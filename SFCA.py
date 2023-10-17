import torch
import torch.nn as nn

# Scale Feature Coordinated Attention Block
class SFCA(nn.Module):
    def __init__(self, in_channels):
        super(SFCA, self).__init__()
        self.conv_avg = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_max = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_re = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H, W = x.size()[2], x.size()[3]
        # [B, C, H, 1]
        t_attn_max = torch.max(x, dim=3, keepdim=True)[0]
        # [B, C, 1, H]
        t_attn_max = t_attn_max.permute(0, 1, 3, 2).contiguous()
        # [B, C, 1, W]
        f_attn_max = torch.max(x, dim=2, keepdim=True)[0]

        t_attn_avg = torch.mean(x, dim=3, keepdim=True)
        t_attn_avg = t_attn_avg.permute(0, 1, 3, 2).contiguous()
        f_attn_avg = torch.mean(x, dim=2, keepdim=True)
        # [B, C, 1, H+W]
        max = self.conv_max(torch.cat([t_attn_max, f_attn_max], dim=3))
        # [B, C, 1, H+W]
        avg = self.conv_avg(torch.cat([t_attn_avg, f_attn_avg], dim=3))
        attn = self.conv_re(torch.cat([max, avg], dim=1))
        h_split, w_split = attn.split([H, W], 3)
        print(h_split.size(), w_split.size())
        attn = self.sigmoid(h_split.permute(0, 1, 3, 2).contiguous()*w_split)

        return attn*x


