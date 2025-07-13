import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())
def self_acc(outputs, targets):
    return np.mean(np.argmax(outputs,axis=1) == np.array(targets))
def unloss(outputs, targets):
    return F.mse_loss(outputs, targets)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
    def info_nce(self,query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ transpose(positive_key)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)
        self.logits=logits
        self.labels=labels
        return F.cross_entropy(logits / temperature, labels, reduction=reduction)  

    def forward(self, query, positive_key, negative_keys=None):
        return self.info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

'''

This code example is trained using the publicly available benchmark dataset from Ottawa University;
Please cite the following reference when using it:
[1]. A Method for Tachometer-free and Resampling-free Bearing Fault Diagnostics under Time-varying Speed Conditions
[2]. Vibration Representation and Speed-Joint Network for Machinery Fault Diagnosis under Time-varying Conditions with Sparse Fault Data

'''
class Self_Dataset_WTH():
    def __init__(self,imgs_dir,aug_flag=False):
        self.root=imgs_dir
        self.files=[i for i in os.listdir(imgs_dir) if i[-3:]=='txt']
        self.files.sort()
        self.aug_flag=aug_flag
        self.negative_transoform=self.get_simclr_transform()[0]
        self.inverse_transoform=self.get_simclr_transform()[1]
        self.raw_transform=self.get_simclr_transform()[2]
    def __len__(self,):
        return len(self.files)
    
    def __getitem__(self,i):
        file_i = self.root+'/'+str(self.files[i])
        with open(file_i,'r+',encoding='utf-8') as f:
            data=[[float(eval(j)) for j in i[:-2].split(',')] for i in f.readlines()]
        speed=float(str(self.files[i].split('_')[-1])[:-4])
        target=int(str(self.files[i].split('_')[0]))
        if self.aug_flag:
            tensor = ([self.inverse_transoform(data),
                       self.negative_transoform(data)],
                        [torch.tensor(speed)],
                        [torch.tensor(target)])
        else:
            tensor=([self.raw_transform(data)],
                    [torch.tensor(speed)],
                    [torch.tensor(target)])
        return tensor
    
    def get_simclr_transform(self):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        negative_transform=lambda x: torch.tensor([[-i for i in imf] for imf in x]).unsqueeze(dim=-1).to(torch.float32)
        inverse_transform=lambda x: torch.tensor([imf[::-1] for imf in x]).unsqueeze(dim=-1).to(torch.float32)
        raw_transoforms=lambda x :torch.tensor(x).unsqueeze(dim=-1).to(torch.float32)
        return negative_transform,inverse_transform,raw_transoforms
