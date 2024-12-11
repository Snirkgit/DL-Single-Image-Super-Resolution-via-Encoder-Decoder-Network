import torch
import torch.nn as nn

'''Modifeid encoder that is based on a VGG architecture and
   uses Residual Dense Blocks for feature extraction and SE blocks. 
'''


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = x.to(device)
        x_global_avg = self.squeeze(x).view(x.size(0), -1)
        excitation_weights = self.excitation(x_global_avg).view(x.size(0), x.size(1), 1, 1)
        return x * excitation_weights


class VGGSuperResEncoder(nn.Module):
    def __init__(self, in_channels=3, num_rdb_blocks=12, grow_rate0=64, grow_rate=64, n_conv_layers=8, scale_factor=2):
        super(VGGSuperResEncoder, self).__init__()

        self.vgg_encoder = VGGEncoder(in_channels=in_channels, num_features=grow_rate0)

        self.rdb_blocks = nn.ModuleList()
        for _ in range(num_rdb_blocks):
            self.rdb_blocks.append(RDB(grow_rate0, grow_rate, n_conv_layers))

        self.gff = nn.Sequential(
            nn.Conv2d(num_rdb_blocks * grow_rate0, grow_rate0, 1),
            nn.Conv2d(grow_rate0, grow_rate0, 3, padding=1),
            SqueezeExcitationBlock(grow_rate0)  # Adding SE Block
        )

        self.upscale = UpscaleBlock(grow_rate0, grow_rate0, scale_factor)

    def forward(self, x):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = x.to(device)
        x = self.vgg_encoder(x)

        rdb_outputs = []
        for rdb_block in self.rdb_blocks:
            x = rdb_block(x)
            rdb_outputs.append(x)

        x = self.gff(torch.cat(rdb_outputs, 1)) + x
        x = self.upscale(x)

        return x


class RDB(nn.Module):
    def __init__(self, grow_rate0, grow_rate, n_conv_layers, k_size=3):
        super(RDB, self).__init__()
        convs = [RDB_Conv(grow_rate0 + c * grow_rate, grow_rate, k_size) for c in range(n_conv_layers)]
        self.convs = nn.Sequential(*convs)
        self.lff = nn.Conv2d(grow_rate0 + n_conv_layers * grow_rate, grow_rate0, 1)
        self.se_block = SqueezeExcitationBlock(grow_rate0)  # Adding SE Block

    def forward(self, x):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = x.to(device)
        return self.se_block(self.lff(self.convs(x))) + x






# Define a VGG feature extractor
class VGGEncoder(nn.Module):
    def __init__(self, in_channels=3, num_features=64):
        super(VGGEncoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True)  # ReLU activation
        )

    def forward(self, x):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = x.to(device)
        x = self.features(x)  
        return x


# custom convolutional block for the Residual Dense Blocks
class RDB_Conv(nn.Module):
    def __init__(self, in_channels, grow_rate, k_size=3):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, grow_rate, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = x.to(device)
        out = self.conv(x)
        #print(x.size()) #debugging purpose
        #print(out.size()) #debugging purpose
        return torch.cat((x, out), 1)  # Concatenate input and output





# upscale block using pixel shuffle for increasing spatial resolution
class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpscaleBlock, self).__init__()
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = x.to(device)
        return self.upscale(x)


