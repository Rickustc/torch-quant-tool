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

import operator

# from torchvision.utils import save_image
# save_image(x,"imagenetdemo0.png")


def _parent_name(target):
    """
    Turn 'foo.bar' into ['foo', 'bar']
    """
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

def replace_linear_with_conv(model):
    """
    modified linear module in model.__init__
    to use this function you should define sub_module in init function of class
    """
    
    modules = dict(model.named_modules(remove_duplicate=False))

    for name, module in modules.items():
        # if "qkv" in name or "proj" in name or "fc" in name:
        if isinstance(module, nn.Linear):
                
                in_features = module.in_features
                out_features = module.out_features
                conv_layer = nn.Conv2d(in_features, out_features, kernel_size=1)
                conv_layer.weight.data = module.weight.data.view(out_features, in_features,1,1)
                conv_layer.bias.data = module.bias.data
                
    
                parent_name, module_name = _parent_name(name)
                setattr(modules[parent_name], module_name, conv_layer)
    
    return model


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

    model = timm.create_model('regnety_320.tv2_in1k', pretrained=False)
    
    # model = replace_linear_with_conv(model)
    checkpoint = torch.load("/home/wrq/resnet_quant/regnet_y_32gf-8db6d4b5.pth")

    # state_dict =  checkpoint["model"].state_dict()
    model.load_state_dict(checkpoint,strict=False)
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


def calibrate(model, calib_loader,device):
    # model should be eval during infer
    model.eval()
    
    total=len(calib_loader)
    cnt = 0
    pbar = tqdm(calib_loader, desc="PTQ Cali",total=total)
    with torch.no_grad():
        for image, target in pbar:
            image = image.to(device)
            model(image)
            cnt +=1
            if cnt>1:
                break
            
            
            
            

        
def calibrate_single_image(model, data):
    cnt = 0
    with torch.no_grad():
        model(data)
        print("single image cali DONE")
        

def get_average_err(x, y):  
    return (torch.abs(x - y).sum()).detach().numpy() / len(x)


class CFG:    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    data_path = '/data1/data/imagenet2012/'
    # train: [30,3,224,224]
    # val: [100,3,224,224]
    data_loader,data_loader_test = prepare_data_loaders(data_path=data_path,train_batch_size=30 ,val_batch_size = 50)
    # get an example input (1,3,224,224)
    example_inputs = (next(iter(data_loader_test))[0])[0].view(1,3,224,224)


    
q_backend = "qnnpack"  # qnnpack  or fbgemm
torch.backends.quantized.engine = q_backend
fp_model, model = load_model(CFG.device)


# fp_model.to("cpu")
# fp_path = "regnety_fp32.onnx"
# x = torch.randn(1, 3, 224, 224)
# torch.onnx.export(fp_model, 
#                   x,
#                   fp_path ,
#                 #   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
#                   opset_version=13)

# print("evaluate fp32")

# top1, top5 = evaluate(model, CFG.data_loader_test,CFG.device)

# print("fp32")
# print(top1)
# print(top5)


# check fp32 model
# o1 = fp_model(CFG.example_inputs)

# set qconfig
# qconfig is used to specify ob type for op
# such as Histogramobserver/minmaxobserver   per_tensor_sym/per_channel_sym  int8/uint8 
qconfig = torch.ao.quantization.qconfig.QConfig(
    activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
    # weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
    weight= torch.ao.quantization.observer.default_per_channel_weight_observer
)
qconfig_conv = torch.ao.quantization.qconfig.QConfig(
    activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
    weight= torch.ao.quantization.observer.default_per_channel_weight_observer
)
# qnnpack not support perchannel for linear oop now
qconfig_linear = torch.ao.quantization.qconfig.QConfig(
    activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
    weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)   
)
qconfig_mapping = (QConfigMapping()
    .set_global(qconfig)  
    .set_object_type(torch.nn.Linear, qconfig_linear)
    .set_object_type(operator.add, None)
    # .set_object_type(torch.nn.Conv2d, qconfig_conv)    
)

# backend_config is used to specify 
# op dtype/ob scle/zp tpe  such as shared_op weight_op non_weight_op
prepared = prepare_fx(model, qconfig_mapping,
                                example_inputs=(torch.randn(1, 3, 224, 224),),
                                backend_config= get_tensorrt_backend_config_dict() 
)

observed_graph_module_attrs = prepared.meta["_observed_graph_module_attrs"]

qconfig_mapping = observed_graph_module_attrs.qconfig_mapping
node_name_to_qconfig = observed_graph_module_attrs.node_name_to_qconfig
observed_node_names = observed_graph_module_attrs.observed_node_names
is_observed_standalone_module = observed_graph_module_attrs.is_observed_standalone_module
node_name_to_scope = observed_graph_module_attrs.node_name_to_scope



calibrate(prepared, CFG.data_loader_test,CFG.device)
# torch.cuda.empty_cache()
prepared.to("cpu")    # lower model should get a cpu tensor not cuda tentor
# print(prepared)

# g = FxGraphDrawer(prepared, "vit_prepared")
# g.get_main_dot_graph().write_svg("vit_prepared.svg")
# print(prepared)

quantized_fx = _convert_fx(prepared, 
            is_reference = False,  # 选择reference模式
            qconfig_mapping = qconfig_mapping,
            backend_config =  get_tensorrt_backend_config_dict()    #get_qnnpack_backend_config()  
            )


# print fx table
# quantized_fx.graph.print_tabular()

# print(quantized_fx.graph)
# print(quantized_fx.code)


# save pytorch-fx graph
# g = FxGraphDrawer(quantized_fx, "quantized_fx")
# g.get_main_dot_graph().write_svg("quantized_fx.svg")
# print(quantized_fx)
# import pdb
# pdb.set_trace()


#TODO profile quantized model
# with profiler.profile(with_stack=True, profile_memory=True) as prof:
#     o1 = fp_model(CFG.example_inputs)
    
# with profiler.profile(with_stack=True, profile_memory=True) as prof:
#     o1 = resnet_model(CFG.example_inputs)

# print(prof.key_averages(group_by_stack_n=1).table(sort_by='self_cpu_time_total', row_limit=5))


# o1 = fp_model(CFG.example_inputs)

o2 = quantized_fx(CFG.example_inputs)


################################################# PTQ Evaluate############################################


# top1, top5 = evaluate(fp_model, CFG.data_loader_test,CFG.device)
# print("fp32")
# print(top1)
# print(top5)

# cuda tensor can not be 
# top1, top5 = evaluate(quantized_fx, CFG.data_loader_test,'cpu')
# print("quant")
# print(top1)
# print(top5)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


# similarity = torch.cosine_similarity(o1.flatten(), o2.flatten(),0)
# err = get_average_err(o1.flatten(), o2.flatten())
# print('similarity', similarity)
# print("avg erro", err)
# print(o1.flatten()[:10])
# print(o2.flatten()[:10])



exported_model_path = 'quant.onnx'




torch.onnx.export(quantized_fx, 
                  CFG.example_inputs,
                  exported_model_path,
                #   training=TrainingMode.TRAINING
                #   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                  opset_version=15)
# image_tensor = torch.rand([1,3,224,224])










#operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
# if reference = :





    
    
    
    
    