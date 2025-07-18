import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPDNet(nn.Module):
    def __init__(self, num_classes: int, bias=True, depth=6):
        super(DeepPDNet, self).__init__()
        self.depth = depth

        self.conv1 = nn.Conv3d(1, 16, (5,5,5), stride=1, padding=(2,2,2))
        self.bn1   = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d((2,2,2), stride=(2,2,2))

        layers = []
        in_channels = 16
        out_channels = 0
        for i in range(1, depth-1):
            if out_channels != 512:
                out_channels = 2**(depth-1+i)
            layers.append(nn.Conv3d(in_channels, out_channels, (3,3,3), stride=1, padding=(1,1,1)))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True)) 
            layers.append(nn.MaxPool3d((2,2,2), stride=(2,2,2)))
            in_channels = out_channels

        self.conv_block = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512, num_classes, bias=bias)


    def forward(self, x):
        latent = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv_block(x)
        x = self.gap(x)
        latent.append(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, latent

    def _forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DeepPDNet(num_classes=1)
    x = torch.randn(2, 1, 64, 64, 96)
    output, latent = model(x)
    print("Output shape:", output.shape)