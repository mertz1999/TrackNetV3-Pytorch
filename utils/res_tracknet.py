from turtle import st
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


##########################################################################################
######### --------------------- Resnet + Tracknet full model --------------------- #######
##########################################################################################

class ResNet_Track(nn.Module):
    """
        This b block is combination of encoder and decoder parts of out model.
    """
    def __init__(self, in_channels=3, pre_channel=64, structure=[3,3,4,3], num_filters=[16,32,64,128]):
        super(ResNet_Track, self).__init__()
        self.in_channels = in_channels
        self.structure   = structure
        self.num_filters = num_filters 
        self.pre_channel = pre_channel

    
        # Initial block of model
        self.init = nn.Sequential(
                nn.Conv2d(self.in_channels, self.pre_channel, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(self.pre_channel),
                nn.ReLU(),

                nn.Conv2d(self.pre_channel, self.pre_channel, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(self.pre_channel),
                nn.ReLU()
        )

        # Encoder part of model
        self.block_1 = self.build_block(64                   , self.structure[0], self.num_filters[0], strides=2)
        self.block_2 = self.build_block(self.num_filters[0]*2, self.structure[1], self.num_filters[1], strides=2)
        self.block_3 = self.build_block(self.num_filters[1]*2, self.structure[2], self.num_filters[2], strides=2)
        self.block_4 = self.build_block(self.num_filters[2]*2, self.structure[3], self.num_filters[3], strides=2)

        # Decoder
        self.conv_t1 = ResNet_Transpose(self.num_filters[3]*2, self.num_filters[3], upsample=2)
        self.conv_d1 = self.build_block(self.num_filters[3]*2, (structure[2]-1), self.num_filters[3], strides=1, decoder=True)

        self.conv_t2 = ResNet_Transpose(self.num_filters[3], self.num_filters[2], upsample=2)
        self.conv_d2 = self.build_block(self.num_filters[2]*2, (structure[1]-1), self.num_filters[2], strides=1, decoder=True)

        self.conv_t3 = ResNet_Transpose(self.num_filters[2], self.num_filters[1], upsample=2)
        self.conv_d3 = self.build_block(self.num_filters[1]*2, (structure[0]-1), self.num_filters[1], strides=1, decoder=True)

        self.conv_t4 = ResNet_Transpose(self.num_filters[1], self.num_filters[0], upsample=2)


        # Initial block of model
        self.last = nn.Sequential(
                nn.Conv2d(self.num_filters[0], self.pre_channel, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(self.pre_channel),
                nn.ReLU(),

                nn.Conv2d(self.pre_channel, self.pre_channel, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(self.pre_channel),
                nn.ReLU(),

                nn.Conv2d(self.pre_channel, self.in_channels, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(self.in_channels),
                nn.Sigmoid()
        )

    # Building block function
    def build_block(self, input_channels, num_block, filters, strides, decoder=False):
        block = nn.Sequential()
        block.append(ResNet_BottleNeck(input_channels, filters, strides, decoder=decoder))
        for _ in range(num_block-1):
            if decoder == False:
                block.append(ResNet_BottleNeck(filters*2, filters, 1, decoder=decoder))
            else:
                block.append(ResNet_BottleNeck(filters, filters, 1, decoder=decoder))

        
        return block


    # Forward path of model
    def forward(self, x):
        # Init Con layers
        x = self.init(x)

        # Encoder blocks
        e1 = self.block_1(x)
        e2 = self.block_2(e1)
        e3 = self.block_3(e2)
        e4 = self.block_4(e3)

        # Decoder block + concatenation
        d_u3 = self.conv_t1(e4)
        d_u3 = torch.cat((d_u3, e3), dim=1)
        d_c3 = self.conv_d1(d_u3)

        d_u2 = self.conv_t2(d_c3)
        d_u2 = torch.cat((d_u2, e2), dim=1)
        d_c2 = self.conv_d2(d_u2)

        d_u1 = self.conv_t3(d_c2)
        d_u1 = torch.cat((d_u1, e1), dim=1)
        d_c1 = self.conv_d3(d_u1)

        # Last upsampling and last layers to create output
        output = self.conv_t4(d_c1)
        output = self.last(output)

        return output


# dummy = torch.ones((1, 3, 288, 512))
# model = ResNet_Track()
# print(model(dummy).shape)
# summary(model, (3, 288, 512))