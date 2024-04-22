import tempfile
import os
# Set up warnings
import warnings
from tqdm import tqdm, trange
import random
import time
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import copy
from copy import deepcopy
import re
from typing import Dict
import sys
import argparse

import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import timm
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization import QConfigMapping
import torch.ao as ao
import torchvision.transforms as transforms
import torch.nn as nn
from torch.ao.quantization.observer import *
from torch.onnx import OperatorExportTypes    
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx,prepare_qat_fx, convert_fx,_convert_fx,convert_to_reference_fx,_prepare_fx
import torchvision.transforms as transforms
import torch.nn as nn
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
    # get_evas_backend_config,
    get_tensorrt_backend_config_dict,
    get_qnnpack_backend_config
)
from torch.ao.quantization.observer import *
import torch.fx
import torch.fx.wrap
import torch.nn.functional as F
import torch
import torch.fx
from torch.fx.node import Node

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.ao.quantization import (
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
from torch.ao.quantization.qconfig import default_symmetric_qnnpack_qat_qconfig,get_default_qat_qconfig,default_per_channel_symmetric_qnnpack_qat_qconfig
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import timm
import numpy as np
from tqdm import tqdm
import logging
from vit import *


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)   # 对不同设备之间的value求和
        if average:  # 如果需要求平均，获得多块GPU计算loss的均值
            value /= world_size

    return value

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def cleanup():
    dist.destroy_process_group()
    
    
    
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


def evaluate(model, val_loader, num_batches=-1):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0

    with torch.no_grad():
        for image, target in tqdm(val_loader):
            output = model(image)
            # loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            # neval_batches set -1 to eval all test_set  
            if num_batches >= 0 and cnt > num_batches:
                return top1, top5
    return top1, top5

def init_distributed_mode(args):
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    if'RANK'in os.environ and'WORLD_SIZE'in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        # LOCAL_RANK代表某个机器上第几块GPU
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif'SLURM_PROCID'in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)  # 对当前进程指定使用的GPU
    args.dist_backend = 'nccl'# 通信后端，nvidia GPU推荐使用NCCL
    dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续
    
    
    
    
def prepare_imagenet_loaders(data_path, train_batch_size, eval_batch_size):
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
        #给每个rank对应的进程分配训练的样本索引
    train_sampler=torch.utils.data.distributed.DistributedSampler(dataset)
    val_sampler=torch.utils.data.distributed.DistributedSampler(dataset_test)
    
    #将样本索引每batch_size个元素组成一个list
    train_batch_sampler=torch.utils.data.BatchSampler(train_sampler,train_batch_size,drop_last=True)
    nw = min([os.cpu_count(), train_batch_size if train_batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(dataset,
                                            batch_sampler=train_batch_sampler,
                                            pin_memory=True,   # 直接加载到显存中，达到加速效果
                                            num_workers=nw,
                                            collate_fn=dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=eval_batch_size,
                                                sampler=val_sampler,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=dataset_test.collate_fn)

    return train_loader, val_loader,train_sampler,val_sampler

    
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




    
def main(args):
    
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    # 初始化分布式环境，主要用来帮助进程间通信
    init_distributed_mode(args=args)
    
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""
    
    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")
            
            
    DataDir = "/data1/data/imagenet2012/"
    train_batch_size=64
    eval_batch_size=100
    epoch=2
    train_loader, val_loader,train_sampler,val_sampler = prepare_imagenet_loaders(DataDir,train_batch_size,eval_batch_size)

    # 实例化模型
    model = vit_base_patch16_224()

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                                if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        
        
    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer使用SGD+余弦淬火策略
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    
    qconfig = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.default_per_channel_weight_observer
    )

    qconfig_conv = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.default_per_channel_weight_observer
    )

    qconfig_linear = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)   # qnnpakc not support perchannel for linear oop now
    )

    qconfig_add = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.PlaceholderObserver,
        weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)   # qnnpakc not support perchannel for linear oop now
    )

    qconfig_softmax = torch.ao.quantization.qconfig.QConfig(activation=FixedQParamsObserver.with_args(scale=0.00390625, zero_point=0, dtype=torch.qint8, quant_min=0, quant_max=255),
                                    weight=FixedQParamsObserver.with_args(scale=0.00390625, zero_point=0, dtype=torch.qint8, quant_min=0, quant_max=255))

    qconfig_layernorm = torch.ao.quantization.qconfig.QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float16),
                                    weight=PlaceholderObserver.with_args(dtype=torch.float16))
    
    
    qconfig_mapping = (QConfigMapping()
        .set_global(qconfig)  
        # .set_object_type('transpose', qconfig) 
        # .set_object_type('permute', qconfig)
        # .set_object_type('reshape', qconfig)
        # .set_object_type(torch.add,qconfig) 
        .set_object_type(torch.mul, None) 
        .set_object_type(torch.add,None) 
        .set_object_type(torch.nn.Linear, qconfig_linear)     #  quantized::linear (xnnpack): Unsupported config for dtype KQInt8
        # .set_object_type(torch.nn.Conv2d, qconfig_linear)
        # .set_object_type(torch.nn.Linear, None)    
        # .set_module_name("blocks_0_norm1", None) 
        # .set_object_type(torch.nn.Softmax, qconfig_softmax)   # softmax需要setting才能被量化
        .set_object_type(torch.nn.Softmax, None)
        # .set_object_type(torch.nn.LayerNorm, qconfig_layernorm)
        # .set_object_type(torch.nn.Dropout, None)
        .set_object_type(torch.nn.LayerNorm, None)
    )
        
    prepared = prepare_qat_fx(model, qconfig_mapping,
                                    example_inputs=(torch.randn(1, 3, 224, 224),),
                                    backend_config= get_tensorrt_backend_config_dict() 
    )
    
    
    
    

    optimizer = torch.optim.SGD(prepared.parameters(), lr = 0.0001)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  

        mean_loss = train_one_epoch(model=prepared,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device=device,
                                        epoch=epoch)



        temp_model = copy.deepcopy(prepared)
        temp_model.to("cpu")
        #nnqat.op --> nnq.op 
        
        int8_model = _convert_fx(temp_model, is_reference=True,  # 选择reference模式
                                qconfig_mapping=qconfig_mapping,backend_config=get_tensorrt_backend_config_dict())
        
        
        prepare_imagenet_loaders()
        sum_num = evaluate(int8_model,
                        val_loader,
                        device="cpu")
        acc = sum_num / val_sampler.total_size
        
        
        
        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
            
            # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    
    cleanup()

        
        
        
        
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        loss = reduce_value(loss, average=True)  #  在单GPU中不起作用，多GPU时，获得所有GPU的loss的均值。
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="/home/wz/data_set/flower_data/flower_photos")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='/home/wrq/checkpoint/vit_base_patch16_224.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
	
