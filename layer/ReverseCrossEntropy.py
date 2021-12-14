import torch
import torch.nn as nn
import torch.nn.functional as F

class RCE(nn.Module):
    def __init__(self,num_cls,ratio=1):
        super().__init__()
        self.num_cls=num_cls
        self.ratio=ratio

    def forward(self,input,target):
        reverse_logits=-input
        reverse_logits=torch.exp(reverse_logits)/(
            (torch.sum(torch.exp(reverse_logits),dim=-1)).view(-1,1)
        )
        reverse_logits=reverse_logits.unsqueeze(-1)
        batch_size=input.size(0)
        reverse_lable=torch.ones(size=(batch_size,self.num_cls),dtype=torch.float)*(self.ratio/(self.num_cls-1))
        reverse_lable=reverse_lable.cuda()

        for idx,label in enumerate(target):
            reverse_lable[idx,label]=0.
        reverse_lable=reverse_lable.unsqueeze(1)
        entropy=-torch.bmm(reverse_lable,torch.log(reverse_logits))
        return torch.sum(
            entropy
        )

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss