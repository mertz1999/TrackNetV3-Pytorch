import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary



################################################################################
######### ---------------------- Residual block ---------------------- #########
################################################################################

class ResNet_BottleNeck(nn.Module):
    """
        Using ResNet and TrackNetv2 together for better performance and speed
    """
    def __init__(self, in_channels, filters, downsample, decoder=False):
        super(ResNet_BottleNeck, self).__init__()
        #  Variables
        self.in_channels  = in_channels
        self.filters      = filters
        self.downsample   = downsample
        self.decoder      = decoder

        # Residual parts
        self.blocks = nn.Sequential(
                nn.Conv2d(self.in_channels, self.filters, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(self.filters),
                nn.ReLU(),

                nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=self.downsample, padding=1),
                nn.BatchNorm2d(self.filters),
                nn.ReLU(),

        )
        if not self.decoder:
            self.blocks.append(nn.Conv2d(self.filters, 2*self.filters, kernel_size=1, stride=1, padding='same'))
            self.blocks.append(nn.BatchNorm2d(self.filters*2))
        else:
            self.blocks.append(nn.Conv2d(self.filters, self.filters, kernel_size=1, stride=1, padding='same'))
            self.blocks.append(nn.BatchNorm2d(self.filters))

        self.blocks.append(nn.ReLU())


        # --- shortcut part
        if self.downsample == 2:
            self.shortcut = nn.Sequential(OrderedDict(
                {
                    'pool' : nn.AvgPool2d(kernel_size=2, stride=self.downsample),
                    'conv' : nn.Conv2d(self.in_channels, self.filters*2, kernel_size=1, stride=1, padding='same'),
                    'bn'   : nn.BatchNorm2d(self.filters*2)
                }
            ))
        elif not self.decoder:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv2d(self.in_channels, self.filters, kernel_size=1, stride=1, padding=0)
    
    # Forward path of model
    def forward(self, x):
        short_cut = self.shortcut(x) # Calc shortcut
        x = self.blocks(x)           # Calc resnet
        x += short_cut               # add shortcut to resnet output
        return x


# dummy = torch.ones((1, 3, 288, 512))
# model = ResNet_BottleNeck(in_channels=3, filters=16, downsample=2, decoder=False)
# print(model(dummy).shape)

# summary(model, (3, 288, 512))





##########################################################################################
######### ---------------------- Residual Transpose block ---------------------- #########
##########################################################################################

class ResNet_Transpose(nn.Module):
    """
        We use this class in decoder part as up-sampling
    """
    def __init__(self, in_channels, filters, upsample):
        super(ResNet_Transpose, self).__init__()
        # Variables
        self.in_channels = in_channels
        self.filters     = filters
        self.upsample    = upsample


        # base block
        self.blocks = nn.Sequential(
                nn.Conv2d(self.in_channels, self.filters, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(self.filters),
                nn.ReLU(),

                nn.ConvTranspose2d(self.filters, self.filters, kernel_size=3, stride=self.upsample, padding=1, output_padding=1),
                nn.BatchNorm2d(self.filters),
                nn.ReLU(),

                nn.Conv2d(self.filters, self.filters, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(self.filters),
                nn.ReLU(),

        )

        # ShortCut block
        self.shortcut = nn.Sequential(OrderedDict(
                {
                    'up_sample' : nn.UpsamplingBilinear2d(scale_factor=self.upsample),
                    'conv'      : nn.Conv2d(self.in_channels, self.filters, kernel_size=1, stride=1, padding='same'),
                    'bn'        : nn.BatchNorm2d(self.filters)
                }
        ))


    # Forward path of model
    def forward(self, x):
        short_cut = self.shortcut(x) # Calc shortcut
        x = self.blocks(x)           # Calc resnet
        x += short_cut               # add shortcut to resnet output
        return x

# dummy = torch.ones((1, 32, 144, 256))
# model = ResNet_Transpose(32, 16, 2)
# print(model(dummy).shape)

# summary(model, (3, 288, 512))