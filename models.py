import torch
import torch.nn as nn


class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=dropout),
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ELU(inplace=True),
        )
        self.residual = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False) \
                        if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x   = self.temporal(x)
        x   = self.spatial(x)
        return x + res


class PhysNet(nn.Module):
    def __init__(self, T=128):
        super().__init__()
        self.T = T

        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(16),
            nn.ELU(inplace=True),
        )
        # FIX: reduced channels 32/64 -> 16/32 to reduce overfitting
        self.enc1  = STBlock(16, 32, dropout=0.2)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2  = STBlock(32, 32, dropout=0.2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.enc3  = STBlock(32, 32, dropout=0.3)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.enc4  = STBlock(32, 32, dropout=0.3)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.upsample     = nn.Upsample(size=(T, 1, 1), mode='trilinear', align_corners=False)
        self.head = nn.Sequential(
            nn.InstanceNorm3d(32, affine=True),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.pool4(self.enc4(x))
        x = self.spatial_pool(x)
        x = self.upsample(x)
        x = self.head(x)
        return x.view(x.size(0), -1)


class Nasir6(nn.Module):
    def __init__(self):
        super(Nasir6, self).__init__()
        self.conv1      = nn.Sequential(nn.Conv3d(3, 16, kernel_size=(3,3,3), stride=1, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.Dropout3d(p=0.1))
        self.maxpool1   = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv2      = nn.Sequential(nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.Dropout3d(p=0.2))
        self.maxpool2   = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv3      = nn.Sequential(nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.Dropout3d(p=0.2))
        self.maxpool3   = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.avgpool    = nn.AdaptiveAvgPool3d((128, 1, 1))
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.avgpool(x)
        x = self.final_conv(x)
        return x.view(x.size(0), -1)
