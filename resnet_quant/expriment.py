import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import timm
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx
from torch.ao.quantization import QConfigMapping
import torch.ao as ao
import torchvision.transforms as transforms
import torch.nn as nn
from torch.ao.quantization.observer import *
import os
# Set up warnings
import warnings
from tqdm import tqdm, trange
import time
import pdb
from pdb import set_trace as bp
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
    # get_evas_backend_config,
    get_tensorrt_backend_config_dict,
    get_qnnpack_backend_config
)
from torch.onnx import OperatorExportTypes    
import copy
from torchviz import make_dot
from torch.fx.passes.graph_drawer import FxGraphDrawer
import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler


# from torchvision.utils import save_image
# save_image(x,"imagenetdemo0.png")


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)   #5
        batch_size = target.size(0)      #50

        _, pred = output.topk(maxk, 1, True, True)  #取出maxk个索引  [b,maxk]
        pred = pred.t()           #[100,5]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))    # 100是为了把表示准确率的小数最后以整数形式展现
        return res

def evaluate(model,data_loader,device):
 
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.to(device)
    cnt = 0
    pbar = tqdm(data_loader)
    with torch.no_grad():
        for image, target in pbar:
            pbar.set_description('Processing ')
            time.sleep(0.2)
            image = image.to(device)
            target = target.to(device)
            output = model(image)     #[50,1000]
            # loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))    # accuracy： ％
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5


def load_model(device):
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    # model = replace_linear_with_conv(model)
    # checkpoint = torch.load("/home/wrq/checkpoint/resnet18_a1_0-d63eafa0.pth")

    # state_dict =  checkpoint["model"].state_dict()
    # model.load_state_dict(checkpoint,strict=True)
    # print(model.default_cfg)

    fp_model = copy.deepcopy(model)
    
    model.to(device)
    model.eval()
    return fp_model, model


def prepare_data_loaders(data_path,train_batch_size,val_batch_size):
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
        sampler=train_sampler,num_workers=32)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=val_batch_size,
        sampler=test_sampler,num_workers=32)

    return data_loader, data_loader_test




class CFG:    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    data_path = '/data1/data/imagenet2012/'
    # train: [30,3,224,224]
    # val: [100,3,224,224]
    data_loader,data_loader_test = prepare_data_loaders(data_path=data_path,train_batch_size=30 ,val_batch_size = 100)
    # get an example input (1,3,224,224)
    example_inputs = (next(iter(data_loader_test))[0])[0].view(1,3,224,224)


    
q_backend = "qnnpack"  # qnnpack  or fbgemm
torch.backends.quantized.engine = q_backend
fp_model, model = load_model(CFG.device)

# print("evaluate fp32")

top1, top5 = evaluate(model, CFG.data_loader_test,CFG.device)
print(top1)