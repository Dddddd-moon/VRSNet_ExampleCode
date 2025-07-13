import torch.nn as nn
import torch
import torch.nn.functional as F

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
    
class Bottleneck(nn.Module):
    def __init__(self,nChannels,growthRate,DB_No):
        super(Bottleneck,self).__init__()
        self.DB_No=DB_No
        self.input_shape=16384//(2**(DB_No-1))
        interChannels=4*growthRate
        self.bn1=nn.BatchNorm2d(nChannels)
        self.conv1=nn.Conv2d(nChannels,interChannels,kernel_size=(3,1),padding='same',bias=False)
        self.bn2=nn.BatchNorm2d(interChannels)
        self.conv2=nn.Conv2d(interChannels,growthRate,kernel_size=(3,1),padding='same',bias=False)
        self.pool=nn.MaxPool2d(kernel_size=(2,1),stride=2)
        #Self-Attention
        # self.attention=CBAM(growthRate)

    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.attention(out) # No Attention
        out = torch.cat((x,out),1)
        out = self.pool(out)
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
    
class Extractor(nn.Module):
    def __init__(self,  inputs,outputs):
        super(Extractor, self).__init__()
        self.db_num=13
        self.growthrate=8
        self.Encoder=Denseblock(inputs,self.growthrate,self.db_num)
        self.feature_shape=(inputs+self.db_num*self.growthrate)*2
        self.fl=FlattenLayer(self.feature_shape)
        self.projector=nn.Sequential(*[
            nn.Linear(self.feature_shape,self.feature_shape*2),
            nn.ReLU(),
            nn.Linear(self.feature_shape*2,outputs)
        ])
    def forward(self, x):
        x = self.Encoder(x)
        x = self.fl(x)
        feature_pro = self.projector(x)
        return x,feature_pro
    

if __name__=='__main__':
    model=Extractor(1,64).cuda()
    random_input=torch.ones((1,1,8192*2,1)).cuda()
    model(random_input)
    print(model(random_input))