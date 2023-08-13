import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from circular_pad import CircularPad2d
import matplotlib.pyplot as plt

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class CircDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CircDoubleConv, self).__init__()
        kwargs = {"kernel_size": 3, "stride": 1, "bias": False}    
        self.conv = nn.Sequential(
            CircularPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels, out_channels, padding=0, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            CircularPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(out_channels, out_channels, padding=0, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[16, 32, 64, 128], useCircular=False
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if(useCircular):
            conv_module = CircDoubleConv
        else:
            conv_module = DoubleConv

        # Down part of UNET
        for feature in features:
            self.downs.append(conv_module(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(conv_module(feature*2, feature))

        self.bottleneck = conv_module(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) #COLOCAR CIRC_CONV AQUI?

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True) #mudança de defaut da função, anteriormente None

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1, useCircular=True)
    preds = model(x)
    assert preds.shape == x.shape
    
    
def test2():
    #input = torch.randn((3, 1, 161, 161))
    input = torch.ones(1, 1, 16, 16).requires_grad_()
    kwargs = {"kernel_size": 3, "stride": 1, "bias": False}
    
    conv1 = DoubleConv(1,1)
    # Normal convolution
    conv1(input).sum().backward()
    init_weights(conv1)
    plt.figure()
    plt.imshow(input.grad.cpu().numpy()[0][0])
    plt.savefig('/mnt/data/matheusvirgilio/gigasistemica/UNET/imagens teste/normal_convolution.png')
    
    input.grad.zero_()
    
    conv2 = CircDoubleConv(1,1)
    init_weights(conv2)
    conv2(input).sum().backward()
    plt.figure()
    plt.imshow(input.grad.cpu().numpy()[0][0])
    plt.savefig('/mnt/data/matheusvirgilio/gigasistemica/UNET/imagens teste/circular_convolution.png')
    #input.grad.zero_()

if __name__ == "__main__":
    test2()