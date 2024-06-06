from unet import *

class NestedUp(nn.Module):
    """Upscaling and then double conv with a nested skip connection."""
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """Extension of UNet by adding nested skip connections and deep supervision."""
    def __init__(self, n_channels, n_classes, bilinear=False, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))


        # Re-define the up blocks to include nested connections
        self.up1 = NestedUp(1024, 512, 512, bilinear)
        self.up2 = NestedUp(512, 256, 256, bilinear)
        self.up3 = NestedUp(256, 128, 128, bilinear)
        self.up4 = NestedUp(128, 64, 64, bilinear)
        # Deep supervision outputs if needed
        self.output1 = OutConv(512, n_classes)
        self.output2 = OutConv(256, n_classes)
        self.output3 = OutConv(128, n_classes)
        self.output4 = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        output1 = self.output1(x)
        x = self.up2(x, x3)
        output2 = self.output2(x)
        x = self.up3(x, x2)
        output3 = self.output3(x)
        x = self.up4(x, x1)
        output4 = self.output4(x)
        if self.deep_supervision:
            output1 = F.interpolate(output1, x.size()[2:], mode='bilinear', align_corners=True)
            output2 = F.interpolate(output2, x.size()[2:], mode='bilinear', align_corners=True)
            output3 = F.interpolate(output3, x.size()[2:], mode='bilinear', align_corners=True)
            return output4, output3, output2, output1
        else:
            return (output4,)

