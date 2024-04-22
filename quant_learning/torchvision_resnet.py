from torchvision.models.resnet import resnet18
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import timm
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx,convert_to_reference_fx
from torch.ao.quantization import QConfigMapping
import torchvision.transforms as transforms
import torch.nn as nn
from torch.ao.quantization.observer import *
import os
# Set up warnings
import warnings
from tqdm import tqdm, trange
import time
from torchvision import models


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def load_model(model_file):
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)   #5
        batch_size = target.size(0)      #50

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test


def evaluate(model, criterion, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
   
    cnt = 0
    pbar = tqdm(data_loader)
    with torch.no_grad():
        for image, target in pbar:
            pbar.set_description('Processing ')
            time.sleep(0.2)
           
            output = model(image)     #[50,1000]
            # loss = criterion(output, target)
            cnt += 1
            if cnt ==20:
                import pdb
                pdb.set_trace()
            acc1 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc1[1], image.size(0))
    print('')

    return top1, top5

criterion = nn.CrossEntropyLoss()

data_path = '/data1/data/imagenet2012/'
train_batch_size = 30
eval_batch_size = 50
data_loader, data_loader_test = prepare_data_loaders(data_path)

resnet50 = models.resnet50(pretrained=True) #pretrained=True 加载模型以及训练过的参数


top1, top5 = evaluate(resnet50, criterion, data_loader_test)



