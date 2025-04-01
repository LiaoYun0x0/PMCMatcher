import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalFeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(LocalFeatureEnhancementModule, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
    
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
      
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #  (N, C, H, W)
        
       
        x_conv1 = self.conv1(x)
       
        weights = self.pointwise_conv(x_conv1)
        adaptive_weights = self.sigmoid(weights)  #  [0, 1]

        x_weighted = x * adaptive_weights
        
        x_conv2 = self.conv2(x_weighted)
        
        output = x + x_conv2
        
        return output


