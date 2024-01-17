import torch
from einops import rearrange
from torch import nn

class CascadedSpatialCrossAttention(torch.nn.Module):
    r""" CascadedSpatialCrossAttention."""

    def __init__(self, channels, group_num):
        super(CascadedSpatialCrossAttention, self).__init__()
        self.group_num = group_num
        self.attn = nn.ModuleList(SpatialCrossAttention(channels//group_num) for i in range(self.group_num))

    def forward(self, x):  # x (B,C,H,W)
        feats_in = x.chunk(self.group_num, dim=1)
        feats_out = []
        feat = feats_in[0]
        for i in range(self.group_num):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = self.attn[i](feat)
            feats_out.append(feat)
        x = torch.cat(feats_out, 1)
        return x

class SpatialCrossAttention(nn.Module):
    r""" SpatialCrossAttention."""

    def __init__(self, channels):
        super(SpatialCrossAttention, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels, channels )
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b , -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        x11 = self.softmax(self.agp(x1).reshape(b, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b, c, -1)  # b*g, c//g, hw
        x2 = self.conv3x3(group_x)

        x21 = self.gn(group_x)
        x21 = x21.sigmoid()
        x12 = x2 + self.agp(self.gn(group_x)).sigmoid() * x21
        x21 = x12
        x12 = x12.reshape(b, c, -1)
        x21 = self.softmax(self.agp(x21).reshape(b, -1, 1).permute(0, 2, 1))
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b, 1, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

BATCH_SIZE_TRAIN = 64

NUM_CLASS = 16
class CSCANet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64):
        super(CSCANet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv1d_features = nn.Sequential(
            nn.Conv1d(in_channels=81, out_channels=81, kernel_size=(4,), stride=4),
            nn.BatchNorm1d(81),
            nn.ReLU(),
        )
        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2, 1], stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.CSCA = CascadedSpatialCrossAttention(224,4)

        # Tokenization x
        self.token_wAx = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wAx)
        self.token_wVx = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wVx)

        # Tokenization y
        self.token_wAy = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wAy)
        self.token_wVy = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wVy)

        self.nn1 = nn.Linear(dim * self.L, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        self.nn2 = nn.Linear(dim * self.L, num_classes)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)

    def forward(self, x):

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')

        x = self.CSCA(x) + x

        x = self.conv2d_features(x)

        y = rearrange(x, 'b c h w -> b c (h w)')
        x = rearrange(x, 'b c h w -> b (h w) c')

        wax = rearrange(self.token_wAx, 'b h w -> b w h')  # Transpose
        Ax = torch.einsum('bij,bjk->bik', x, wax)
        Ax = rearrange(Ax, 'b h w -> b w h')  # Transpose
        Ax = Ax.softmax(dim=-1)
        VVx = torch.einsum('bij,bjk->bik', x, self.token_wVx)
        Tx = torch.einsum('bij,bjk->bik', Ax, VVx)
        x = rearrange(Tx, 'b h w -> b (h w)')
        x = self.nn1(x)
        x = x.unsqueeze(1)

        way = self.token_wAy
        Ay = torch.einsum('bij,bjk->bik', way, y)
        Ay = Ay.softmax(dim=-1)
        VVy = torch.einsum('bij,bjk->bik', self.token_wVy, y)
        VVy = rearrange(VVy, 'b h w -> b w h')
        Ty = torch.einsum('bij,bjk->bik', Ay, VVy)
        y = rearrange(Ty, 'b h w -> b (h w)')
        y = self.nn2(y)
        y = y.unsqueeze(1)

        x = torch.cat((x, y), dim=1)
        x = x.unsqueeze(1)
        x = self.conv2d_features1(x).cuda()
        x = x.squeeze(1)
        x = x.squeeze(1)

        return x


if __name__ == '__main__':
    model = CSCANet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())

