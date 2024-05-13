import torch
import torch.nn as nn


class convolution_block(nn.Module):
    # conv -> skip & batch_norm -> activation -> conv -> batch_norm -> Add Skip -> activation 
    def __init__(self, in_channels, intermediate_channels ,out_channels, kernel_size, padding):
        super(convolution_block, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size = kernel_size, padding = padding, bias = True)
        self.conv2 = nn.Conv2d(intermediate_channels, out_channels,  kernel_size = kernel_size, padding = padding, bias = True)
        self.activation = nn.ReLU(inplace = True)
        # self.batch_norm1 = nn.BatchNorm2d(intermediate_channels)
        # self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        conv1_x = x # For skip connection
        # x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        # x = self.batch_norm2(x)
        x = self.activation(x + conv1_x) # From Skip connection
        return x
        
class EChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(EChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias = False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        output = avg_out + max_out
        return self.sigmoid(output)

class Siam_Ecam(nn.Module):
    # Siam_Ecam model
    def __init__(self, in_channels = 3, out_channels = 2):
        super(Siam_Ecam, self).__init__()
        torch.nn.Module.dump_patches = True
        
        channels = [32, 64, 128, 256, 512]
        self.pooling = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Convolution Blocks & Upsampling Blocks
        self.conv0_0 = convolution_block(in_channels, channels[0], channels[0], 3, 1) 
        self.conv1_0 = convolution_block(channels[0], channels[1], channels[1], 3, 1)
        self.conv2_0 = convolution_block(channels[1], channels[2], channels[2], 3, 1)
        self.conv3_0 = convolution_block(channels[2], channels[3], channels[3], 3, 1)
        self.conv4_0 = convolution_block(channels[3], channels[4], channels[4], 3, 1)
        self.up1_0 = nn.ConvTranspose2d(channels[1], channels[1], 2, stride = 2)
        self.up2_0 = nn.ConvTranspose2d(channels[2], channels[2], 2, stride = 2)
        self.up3_0 = nn.ConvTranspose2d(channels[3], channels[3], 2, stride = 2)
        self.up4_0 = nn.ConvTranspose2d(channels[4], channels[4], 2, stride = 2)
        
        self.conv0_1 = convolution_block(2*channels[0] + channels[1], channels[0], channels[0], 3, 1)
        self.conv1_1 = convolution_block(2*channels[1] + channels[2], channels[1], channels[1], 3, 1)
        self.conv2_1 = convolution_block(2*channels[2] + channels[3], channels[2], channels[2], 3, 1)
        self.conv3_1 = convolution_block(2*channels[3] + channels[4], channels[3], channels[3], 3, 1)
        self.up1_1 = nn.ConvTranspose2d(channels[1], channels[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(channels[2], channels[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(channels[3], channels[3], 2, stride = 2)
        
        self.conv0_2 = convolution_block(3*channels[0]+channels[1], channels[0], channels[0], 3, 1)
        self.conv1_2 = convolution_block(3*channels[1] + channels[2], channels[1], channels[1], 3, 1)
        self.conv2_2 = convolution_block(3*channels[2] + channels[3], channels[2], channels[2], 3, 1)
        self.up1_2 = nn.ConvTranspose2d(channels[1], channels[1], 2, stride= 2)
        self.up2_2 = nn.ConvTranspose2d(channels[2], channels[2], 2, stride = 2)
        
        self.conv0_3 = convolution_block(4*channels[0] + channels[1], channels[0], channels[0], 3, 1)
        self.conv1_3 = convolution_block(4*channels[1] + channels[2], channels[1], channels[1], 3, 1)
        self.up1_3 = nn.ConvTranspose2d(channels[1], channels[1], 2, stride = 2)
        
        self.conv0_4 = convolution_block(5*channels[0] + channels[1], channels[0], channels[0], 3, 1)
        
        self.conv_out = nn.Conv2d(4*channels[0], out_channels, kernel_size=1)
        
        self.ecam0 = EChannelAttentionModule(channels[0] * 4)
        self.ecam1 = EChannelAttentionModule(channels[0], ratio = 4)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x_A, x_B):
        # 2 input images (Siamese Network)
        # A
        x0_0_A = self.conv0_0(x_A)
        x1_0_A = self.conv1_0(self.pooling(x0_0_A))
        x2_0_A = self.conv2_0(self.pooling(x1_0_A))
        x3_0_A = self.conv3_0(self.pooling(x2_0_A))
        
        # B
        x0_0_B = self.conv0_0(x_B)
        x1_0_B = self.conv1_0(self.pooling(x0_0_B))
        x2_0_B = self.conv2_0(self.pooling(x1_0_B))
        x3_0_B = self.conv3_0(self.pooling(x2_0_B))
        x4_0_B = self.conv4_0(self.pooling(x3_0_B))
        
        x0_1 = self.conv0_1(torch.cat([x0_0_A, x0_0_B, self.up1_0(x1_0_B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0_A, x1_0_B, self.up2_0(x2_0_B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0_A, x0_0_B, x0_1, self.up1_1(x1_1)], 1))
        
        
        x2_1 = self.conv2_1(torch.cat([x2_0_A, x2_0_B, self.up3_0(x3_0_B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0_A, x1_0_B, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0_A, x0_0_B, x0_1, x0_2, self.up1_2(x1_2)], 1))
        
        x3_1 = self.conv3_1(torch.cat([x3_0_A, x3_0_B, self.up4_0(x4_0_B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0_A, x2_0_B, x2_1, self.up3_1(x3_1)], 1))
        
        x1_3 = self.conv1_3(torch.cat([x1_0_A, x1_0_B, x1_1, x1_2, self.up2_2(x2_2)], 1))
        
        x0_4 = self.conv0_4(torch.cat([x0_0_A, x0_0_B, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))
        
        output = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        
        stacking = torch.stack((x0_1, x0_2, x0_3, x0_4))
        
        intra = self.ecam1(torch.sum(stacking, dim = 0))
        inter = self.ecam0(output)
        
        output = inter * (output + intra.repeat(1, 4, 1, 1))
        output = self.conv_out(output)
        
        return (output, )
        
        
        
        