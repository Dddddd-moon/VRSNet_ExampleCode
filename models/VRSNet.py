import torch.nn as nn
import torch
import torch.nn.functional as F
from .Extractor_VRSNet  import Extractor
from .Approximator_VRSNet import Approximator
import torch.nn.init as init
    
class Classifier(nn.Module):
    def __init__(self,inputs,hidden,feature_dim):
        super(Classifier, self).__init__()
        self.input=inputs
        self.hidden=hidden
        self.out=feature_dim
        self.drop=nn.Dropout(p=0.3)
        self.fc1=nn.Linear(inputs,hidden)
        self.fc2=nn.Linear(self.hidden,self.out)
        self.ac=nn.ReLU()
    def forward(self, x):
        x=self.fc1(x)
        x=self.ac(x)
        x=self.drop(x)
        x=self.fc2(x)
        return x
    
class Supportor(nn.Module):
    def __init__(self,extractor,approximator,classifier):
        super(Supportor, self).__init__()
        self.Extractor=extractor
        self.Classifier=classifier
        self.Approximator=approximator
        # self.Spyder.eval()
    def forward(self, x,speed):
        Extractor_x,_=self.Extractor(x)
        Spyder_x=self.Approximator(speed)
        pre=self.Classifier(Extractor_x-Spyder_x)
        return pre,Extractor_x,Spyder_x
    
 #-------------------------------------------------------------------------------
 # This model is for paper: 
 # Vibration Representation and Speed-Joint Network for Machinery Fault Diagnosis under Time-varying Conditions with Sparse Fault Data

class VRSNet(nn.Module):
    def __init__(self,inputs,Projector_nodes,Approximator_hidden,Clf_hidden,Classes):
        super(VRSNet, self).__init__()
        self.Extractor_in=inputs
        self.Approximator_hidden=Approximator_hidden
        self.Clf_hidden=Clf_hidden
        self.Extractor=Extractor(inputs,Projector_nodes)
        self.Approximator_in=1
        self.Approximator_out=self.Extractor.feature_shape
        self.Approximator=Approximator(1)
        self.Approximator_clf=Approximator(1)
        self.Classifier=Classifier(self.Approximator_out,self.Clf_hidden,Classes)


    def forward(self, x, speed):
        # Extractor extracts feature space representations of vibration signals at the current rotational speed.
        self.extractor_feature,_=self.Extractor(x)
        # Approximator computes the regression representation corresponding to the current rotational speed.
        self.regression_feature=self.Approximator(speed)
        # The ApproximatorC computes the fine-tuned representation corresponding to the current rotational speed.
        self.spyder_feature=self.Approximator_clf(speed)
        # Compute the difference vector between the vibration branch features and the speed branch features.
        self.mid_feature=self.extractor_feature-self.spyder_feature
        # Fuse the features and input them into the classifier to compute the prediction results.
        Pre_vector=self.Classifier(self.mid_feature)
        return Pre_vector

if __name__=='__main__':
    random_vibration = torch.ones((1,1,16384,1)).cuda()
    random_speed = torch.randn((1,1)).cuda()
    vrsnet=VRSNet(inputs=1,
                  Projector_nodes=64,
                  Approximator_hidden=512,
                  Clf_hidden=1024,
                  Classes=6).cuda()
    pre=vrsnet(random_vibration,random_speed)
    print(pre)

    