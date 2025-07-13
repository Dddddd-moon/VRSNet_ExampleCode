import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

class FlattenLayer(ModuleWrapper):
    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)
    
def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    

class Bottleneck(nn.Module):
    def __init__(self,nChannels,growthRate,DB_No):
        super(Bottleneck,self).__init__()
    #通常1x1卷积的通道数为GrowthRate的4倍
        self.DB_No=DB_No
        self.input_shape=512//(2**(DB_No-1))
        interChannels=4*growthRate
        self.bn1=nn.BatchNorm2d(nChannels)
        self.conv1=nn.Conv2d(nChannels,interChannels,kernel_size=(3,1),padding='same',bias=False)
        self.bn2=nn.BatchNorm2d(interChannels)
        self.conv2=nn.Conv2d(interChannels,growthRate,kernel_size=(3,1),padding='same',bias=False)
        self.pool=nn.MaxPool2d(kernel_size=(2,1),stride=2)
        #Self-Attention
        self.attention=CBAM(growthRate)

    def forward(self,x):
        out=self.conv1(F.relu(self.bn1(x)))
        out=self.conv2(F.relu(self.bn2(out)))
        out=self.attention(out)
        out=torch.cat((x,out),1)
        out=self.pool(out)
        return out
    
class Denseblock(nn.Module):
    def __init__(self,nChannels,growthRate,nDenseBlocks):
        super(Denseblock,self).__init__()
        layers=[]
        for i in range(int(nDenseBlocks)):
            layers.append(Bottleneck(nChannels,growthRate,i+1))
            nChannels+=growthRate
        self.denseblock=nn.Sequential(*layers)
    def forward(self,x):
        return self.denseblock(x)
    
class Approximator(nn.Module):
    def __init__(self,  inputs):
        super(Approximator, self).__init__()
        self.db_num=8
        self.growthrate=13
        self.Encoder=Denseblock(inputs,self.growthrate,self.db_num)
        self.feature_shape=(inputs+self.db_num*self.growthrate)*2
        self.fl=FlattenLayer(self.feature_shape)
        self.fc_in=nn.Linear(1,512)
    def forward(self, x):
        x = self.fc_in(x)
        x = x.unsqueeze(dim=1).unsqueeze(dim=-1)
        x = self.Encoder(x)
        x = self.fl(x)
        return x