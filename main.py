##Outer import
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler,RMSprop,Adamax, AdamW
from sklearn.model_selection import KFold
import pandas as pd
from tqdm import tqdm
import logging
from matplotlib import pyplot as plt
import datetime
import time
import torch.functional as F
import metrics
from models import *
from metrics import *
import copy
import random



# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def run_InfoNCE_WTH(args):
    # Hyper Parameter settings
    n_epochs = args.epochs
    lr = args.lr
    num_workers = args.num_workers
    batch_size = args.batch_size
    net_type=args.net_type
    dataset='WTH'
    datapath='./Data/Train'

    # Using Sample Augmentation !!!!!!!!!!!!!!!!!!!!!!
    trainset=Self_Dataset_WTH(datapath,aug_flag=True) 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
            num_workers=num_workers,shuffle=True, drop_last=False)
    model_foler=rf".\history\{net_type}\{datetime.datetime.today().strftime('%Y%m%d-%H%M')}【InfoNCE】"
    os.makedirs(model_foler) if not os.path.exists(model_foler) else print()
    net = VRSNet(args.inc,args.P_shape,args.S_hidc,args.C_hidc,args.classes)
    logging.basicConfig(filename=os.path.join(model_foler, 'training.log'), level=logging.INFO)
    extractor=net.Extractor.cuda()
    logging.info(f"Start [{net_type}] training for dataset [{dataset}] with [{n_epochs}] epochs.")
    logging.info(f"CUDA acceleration available: [{torch.cuda.is_available()}]")
    logging.info(f"Model info:\tFeature_shape:[{extractor.feature_shape}];\tProjector_shape:[{args.P_shape}]\tFinetune:[{args.Finetune}]")

    # Contrastive Learning for Trend Self-extraction
    optimizer_extractor = Adam(extractor.parameters(), lr=lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer_extractor, patience=5,mode='min', verbose=True)
    criterion = InfoNCE(temperature=args.temperature)
    best_loss=np.inf
    epoch_loss=0
    Acc=0
    for epoch in range(n_epochs):
        extractor.train()
        self_loss_all=[]
        acc=[]
        extractor.requires_grad=True
        iteration=tqdm(train_loader)
        iteration.set_description_str(f"Extractor:{epoch+1}|{n_epochs}-Loss:{epoch_loss:.4f};-Acc:{np.mean(Acc):.4f}")
        for (i,slice) in enumerate(iteration,0): 
            # Contrastive Learning Gradient Calculation
            images = torch.cat(slice[0], dim=0).cuda()
            optimizer_extractor.zero_grad()
            _,pro_features=extractor(images)
            self_loss=criterion(pro_features[:images.size(0)//2],pro_features[images.size(0)//2:])
            acc.append(metrics.accuracy(criterion.logits,criterion.labels)[0]\
                       .detach().cpu().numpy().tolist()[0])
            self_loss.backward()
            optimizer_extractor.step()
            self_loss_all+=[self_loss.item()]
        epoch_loss = np.mean(self_loss_all)
        Acc=np.mean(acc)
        logging.info(f"【Extractor】Epoch(lr-{optimizer_extractor.state_dict()['param_groups'][0]['lr']}):{epoch+1}\tLoss: {epoch_loss:.4f}\tself_acc:{np.mean(acc):.4f}")
        lr_sched.step(epoch_loss)
        if epoch_loss<best_loss:
            best_loss=epoch_loss
            model_folder_path=model_foler+'/models'
            os.makedirs(model_folder_path) if not os.path.exists(model_folder_path) else None
            model_save_path=model_folder_path+f"\Extractor_InfoNCE.pt"
            torch.save(extractor.state_dict(),model_save_path)
            model_save_path='./history/VRSNet/'+f"/Extractor_InfoNCE.pt"
            torch.save(extractor.state_dict(),model_save_path)

def run_approximator_WTH(args):
    # Hyper Parameter settings
    n_epochs = args.epochs
    lr = args.lr
    num_workers = args.num_workers
    batch_size = args.batch_size
    net_type=args.net_type
    TT='InfoNCE'
    dataset='WTH'
    SPDpath='./Data/Train'

    # Without Sample Augmentation
    SPDset=Self_Dataset_WTH(SPDpath,aug_flag=False)
    SPD_loader=torch.utils.data.DataLoader(SPDset, batch_size=batch_size, 
            num_workers=num_workers,shuffle=True, drop_last=False)
    model_foler=rf".\history\{net_type}\{datetime.datetime.today().strftime('%Y%m%d-%H%M')}_【App_{TT}】"
    os.makedirs(model_foler) if not os.path.exists(model_foler) else print()
    net = VRSNet(args.inc,args.P_shape,args.S_hidc,args.C_hidc,args.classes)
    net.Extractor.load_state_dict(torch.load(rf'./history/VRSNet/Extractor_{TT}.pt'))
    logging.warning(f'Warning message: Loading Extractor Success!')
    logging.basicConfig(filename=os.path.join(model_foler, 'training.log'), level=logging.INFO)
    extractor=net.Extractor.cuda()
    approximator=net.Approximator.cuda()
    logging.info(f"Start [{net_type}] training for dataset [{dataset}] with [{n_epochs}] epochs.")
    logging.info(f"CUDA acceleration available: [{torch.cuda.is_available()}]")
    logging.info(f"Model info:\tFeature_shape:[{extractor.feature_shape}];\tProjector_shape:[{args.P_shape}]\tFinetune:[{args.Finetune}]")
    model_folder_path=model_foler+'/models'
    print('Extractor Pre-training Loading Success!')
    
    ##Phase2 Train Spyder
    criterion_SPD=nn.MSELoss()
    optimizer_spyder= Adam(approximator.parameters(), lr=lr)
    lr_sched_SPD = lr_scheduler.ReduceLROnPlateau(optimizer_spyder, patience=5,mode='min', verbose=True)
    best_loss=np.inf
    for epoch in range(int(n_epochs)):
        SPD_loss=0
        approximator.train()
        extractor.eval()
        iteration=tqdm(SPD_loader)
        iteration.set_description_str(f'Approximator_{TT}:{epoch+1}|{int(n_epochs)}-Loss:{best_loss:.4f}')
        for images,speed,target in iteration:
            images = torch.cat(images, dim=0).cuda()
            speed= torch.cat(speed, dim=0).to(device).unsqueeze(dim=1)
            optimizer_spyder.zero_grad()
            approximator_out=approximator(speed)
            extractor_feature,_=extractor(images)
            loss=criterion_SPD(approximator_out.cuda(),extractor_feature.cuda())
            loss.backward()
            optimizer_spyder.step()
            SPD_loss+=loss.item()*images.size(0)
        logging.info(f"【Approximator】Epoch(lr-{optimizer_spyder.state_dict()['param_groups'][0]['lr']}): {epoch+1}\tSPD_loss:{SPD_loss:.4f}")
        lr_sched_SPD.step(SPD_loss)
        if SPD_loss<best_loss:
            best_loss=SPD_loss
            model_folder_path=model_foler+'/models'
            os.makedirs(model_folder_path) if not os.path.exists(model_folder_path) else print()
            model_save_path=model_folder_path+f"\Approximator_{TT}.pt"
            torch.save(approximator.state_dict(),model_save_path)
            model_save_path=f'./history/{net_type}'+f"/Approximator_{TT}.pt"
            torch.save(approximator.state_dict(),model_save_path)       

def  run_classifier_WTH(args):
    # Hyper Parameter settings
    n_epochs =   args.epochs # // 5
    lr = args.lr
    num_workers = args.num_workers
    batch_size = {'1':1,'5':4,'10':8,'20':16}
    sample_sizes = ['1', '5', '10', '20'] 
    net_type=args.net_type
    TT='InfoNCE'
    dataset='WTH'
    model_foler=rf".\history\{net_type}\{datetime.datetime.today().strftime('%Y%m%d-%H%M')}_【Clf_{TT}】"
    os.makedirs(model_foler) if not os.path.exists(model_foler) else print()
    for shot in sample_sizes:
        Finetune_path=rf"./Data/Finetune/{shot}points"
        # Withouth Sample Augmentation
        Finetune_dataset=Self_Dataset_WTH(Finetune_path,aug_flag=False) 
        Anchoring_loader=torch.utils.data.DataLoader(Finetune_dataset, batch_size=batch_size[shot],
            num_workers=num_workers,shuffle=True, drop_last=False)

        net = VRSNet(args.inc,args.P_shape,args.S_hidc,args.C_hidc,args.classes)
        net.Extractor.load_state_dict(torch.load(rf'./history/VRSNet/Extractor_{TT}.pt'))
        net.Approximator.load_state_dict(torch.load(rf'./history/VRSNet/Approximator_{TT}.pt'))
        net.Approximator_clf.load_state_dict(torch.load(rf'./history/VRSNet/Approximator_{TT}.pt'))
        logging.basicConfig(filename=os.path.join(model_foler, 'training.log'), level=logging.INFO)

        extractor=net.Extractor.cuda()
        approximator=net.Approximator.cuda()
        approximator_clf=net.Approximator_clf.cuda()
        classifier=net.Classifier.cuda()

        supportor=Supportor(
            copy.deepcopy(extractor),
            copy.deepcopy(approximator_clf),
            copy.deepcopy(classifier),
        )
        logging.info(f"Start [{net_type}] training for dataset [{dataset}] with [{n_epochs}] epochs.")
        logging.info(f"CUDA acceleration available: [{torch.cuda.is_available()}]")
        logging.info(f"Model info:\tFeature_shape:[{extractor.feature_shape}];\tProjector_shape:[{args.P_shape}]\tAnchoring:[{shot}]")
        model_folder_path=model_foler+'/models'
        os.makedirs(model_folder_path) if not os.path.exists(model_folder_path) else None
        model_save_path=model_folder_path+f"\Extractor_{TT}_SP.pt"
        torch.save(extractor.state_dict(),model_save_path)
        model_save_path=model_folder_path+f"\Approximator_{TT}.pt"
        torch.save(approximator.state_dict(),model_save_path)
        print('Extractor/Approximator: Pre-trained Model Loading Success!')
        logging.warning(f'Warning message: Loading Extractor Success!')
        criterion_Finetune = nn.CrossEntropyLoss()
        optimizer_Finetune = Adamax([{'params': approximator_clf.parameters()},
                                    {'params': classifier.parameters()},
                                    {'params': extractor.parameters()}], lr=lr)
        lr_sched_Finetune= lr_scheduler.ReduceLROnPlateau(optimizer_Finetune, patience=5,mode='min', verbose=True)
        best_loss=np.inf
        best_acc=0
        Classifier_loss=0
        epoch_loss=0
        acc=[]
        validate_acc=0
        '''
        Phase 3: In the updated training procedure, freeze the Extractor and unfreeze the Spyder module for Classifier training.'''
        for epoch in range(int(n_epochs)):
            # Iterate over all submodules of the extractor
            for name, module in extractor.named_modules():
                # If it is a BatchNorm layer, freeze its parameters.
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
            for name, module in approximator_clf.named_modules():
                # If it is a BatchNorm layer, freeze its parameters.
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False  

            '''Switch to EVAL mode to preserve the pre-trained knowledge.'''
            extractor.eval()
            approximator_clf.eval()
            classifier.train()

            iteration=tqdm(Anchoring_loader)
            iteration.set_description_str(f'Classifier-{TT}-{shot}:{epoch+1}|{int(n_epochs)}-Best_acc:{best_acc*100:.4f};Epoch_loss:{epoch_loss:.4f};-Epoch_acc:{validate_acc*100:.4f};')
            Classifier_loss=0
            epoch_loss=0
            acc=[]
            for images,speed,target in iteration:
                images = torch.cat(images, dim=0).cuda()
                speed= torch.cat(speed,dim=0).cuda().unsqueeze(dim=-1)
                target= torch.cat(target,dim=0).cuda()
                optimizer_Finetune.zero_grad()
                representation,_=extractor(images)
                approximation=approximator_clf(speed)
                #Core step: generate fusion feature
                residual_features=representation-approximation
                pre=classifier(residual_features)
                loss=criterion_Finetune(pre.cuda(),target.cuda())
                loss.backward()
                optimizer_Finetune.step()
                epoch_loss+=loss.item()*images.size(0)
                acc.append(metrics.acc(pre.detach(), target.detach()))
            lr_sched_Finetune.step(epoch_loss)

            ''' Embed the three submodules into a cloned network for easier testing.'''

            with torch.no_grad():
                supportor.Extractor=extractor
                supportor.Approximator=approximator_clf
                supportor.Classifier=classifier
                validate_acc=validate_model_WTH(supportor)
            
            '''
            During the review process, the reviewers requested the inclusion of a validation set for testing.
            This indeed helps all models avoid overfitting, including both VRSNet and the other comparative models.
            However, in real-world industrial scenarios with sparse data, obtaining a validation set is often tough. 
            Readers may decide whether to include a validation set based on their specific needs.
            '''
            if validate_acc>best_acc:
                best_loss=epoch_loss
                best_acc=validate_acc
                model_folder_path=model_foler+'/models'
                os.makedirs(model_folder_path) if not os.path.exists(model_folder_path) else print()
                # Save to training path
                model_save_path=model_folder_path+f"\Extractor_{TT}_{shot}.pt"
                torch.save(extractor.state_dict(),model_save_path)
                model_save_path=model_folder_path+f"\Classifier_{TT}_{shot}.pt"
                torch.save(classifier.state_dict(),model_save_path)
                model_save_path=model_folder_path+f"\Approximator_clf_{TT}_{shot}.pt"
                torch.save(approximator_clf.state_dict(),model_save_path)
            logging.info(f"【Classifier】Epoch(lr-{optimizer_Finetune.state_dict()['param_groups'][0]['lr']}): Best_acc:{best_acc*100:.4f};Epoch_loss:{epoch_loss:.4f};-Epoch_acc:{validate_acc*100:.4f};")

        # Integrate network 
        net = VRSNet(args.inc,args.P_shape,args.S_hidc,args.C_hidc,args.classes)
        net.Extractor.load_state_dict(torch.load(model_folder_path+f"\Extractor_{TT}_{shot}.pt"))
        net.Approximator.load_state_dict(torch.load(model_folder_path+f"\Approximator_{TT}.pt"))
        net.Approximator_clf.load_state_dict(torch.load(model_folder_path+f"\Approximator_clf_{TT}_{shot}.pt")) 
        net.Classifier.load_state_dict(torch.load(model_folder_path+f"\Classifier_{TT}_{shot}.pt")) 
        
        # Test accurancy
        test_acc=test_model_WTH(net)

        model_folder_path=model_foler+'/models'
        model_save_path=model_folder_path+f'\{net_type}_{TT}_{shot}_{best_acc*100:.4f}_{test_acc*100:.4f}.pt'
        torch.save(net.state_dict(),model_save_path)



'''
If GPU memory is insufficient, CPU can be used for validation and testing (One-by-one approch). 
The example code utilizes matrix operations to enhance computational efficiency.
'''
def validate_model_WTH(supportor):
    supportor.eval()
    datapath='./Data/Valid'
    dataset=Self_Dataset_WTH(datapath,aug_flag=False)
    validation_dataloader=torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
        num_workers=0,shuffle=True, drop_last=False)
    for images,speed,target in validation_dataloader:
        images = torch.cat(images, dim=0).cuda()
        speed= torch.cat(speed,dim=0).cuda().unsqueeze(dim=-1)
        target= torch.cat(target,dim=0).cuda()
        Classifier_out,_,_=supportor(images,speed)
        Acc=metrics.acc(Classifier_out.detach(), target.detach())
    return Acc

'''
If GPU memory is insufficient, CPU can be used for validation and testing (One-by-one approch). 
The example code utilizes matrix operations to enhance computational efficiency.
'''
def test_model_WTH(model):
    model.eval()
    datapath='./Data/Test'
    dataset=Self_Dataset_WTH(datapath,aug_flag=False)
    validation_dataloader=torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
        num_workers=0,shuffle=True, drop_last=False)
    model.cuda()
    for images,speed,target in validation_dataloader:
        images = torch.cat(images, dim=0).cuda()
        speed= torch.cat(speed,dim=0).cuda().unsqueeze(dim=-1)
        target= torch.cat(target,dim=0).cuda()
        Classifier_out=model(images,speed)
        Acc=metrics.acc(Classifier_out.detach(), target.detach())
    return Acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Frequentist Model Training")
    parser.add_argument('--net_type', default='VRSNet', type=str, help='model')
    parser.add_argument('--dataset', default='WTH', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int, 
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--inc', default=1, type=int, help='inchannel')
    parser.add_argument('--P_shape', default=64, type=int, help='inchannel')
    parser.add_argument('--S_hidc', default=512, type=int, help='inchannel')
    parser.add_argument('--C_hidc', default=1024, type=int, help='inchannel')
    parser.add_argument('--Finetune', default='1', type=str, help='1/5/10/20')
    parser.add_argument('--classes', default=6, type=int, help='inchannel')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--num_workers', default=0, type=int,
                         help='num_workers')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--temperature', default=0.05, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--device', default=torch.device('cuda'),  help='CUDA.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    global args
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Train network using benchmark dataset from Ottawa University

    '''Step#1 Contrastive Training'''
    run_InfoNCE_WTH(args) 

    '''Step#2 Regression Training'''
    run_approximator_WTH(args)

    '''Step#3 BatchNorm freezing-based Vibration-speed Joint Fine-tuning'''
    run_classifier_WTH(args)