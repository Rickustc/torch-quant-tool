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
from copy import deepcopy
import re
from typing import Dict

import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx,prepare_qat_fx, convert_fx,_convert_fx,convert_to_reference_fx,_prepare_fx
from torch.ao.quantization import QConfigMapping
import torch.ao as ao
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
from torch.ao.quantization.qconfig import default_symmetric_qnnpack_qat_qconfig,get_default_qat_qconfig,default_per_channel_symmetric_qnnpack_qat_qconfig
import torch.nn.functional as F
import torch
import torch.fx
from torch.fx.node import Node
import os
import random
import time
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.ao.quantization import (
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx,convert_to_reference_fx,prepare_qat_fx
from torch.ao.quantization.qconfig import default_symmetric_qnnpack_qat_qconfig,get_default_qat_qconfig,default_per_channel_symmetric_qnnpack_qat_qconfig
from accelerate import Accelerator
import numpy as np
from tqdm import tqdm
import logging
from vit import *


def get_log(log_name):
    logger = logging.getLogger(log_name)  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level

    fh = logging.FileHandler(log_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever



    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger

train_logger = get_log("train_resnet50_QAT")
test_logger = get_log("test_resnet50_QAT")

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
    model.to("cpu")
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    print("eval: ")
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

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)


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

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=32, pin_memory=True,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=32, pin_memory=True,      #  os.cpu_count()

        )

    return data_loader, data_loader_test


 

def train_one_epoch(model_prepared, train_loader,criterion, optimizer, accelerator,nepoch,ntrain_batches=20):
    model_prepared.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    
    
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps =  num_update_steps_per_epoch  #epoch=1
    # accelerator for multi gpu train
    model_prepared, train_loader, optimizer = \
            accelerator.prepare(model_prepared, train_loader, optimizer)
            
    # https://github.com/huggingface/transformers/blob/2788f8d8d5f9cee2fe33a9292b0f3570bd566a6d/examples/research_projects/luke/run_luke_ner_no_trainer.py#L656
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Train step")
    #accelerator do not need specify a device
    for i,(image, target) in enumerate(train_loader):
       
        output = model_prepared(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        progress_bar.update(1)
        #every 20 step
        if accelerator.is_local_main_process:
            if i % ntrain_batches==0:
                train_logger.info("Training Epoch:{} step:{}, Loss :{}, Training metric: {} , {}".format(
                                    nepoch,
                                    i,
                                    avgloss.avg.item(),
                                    top1,
                                    top5
                                )
                                )
        # if debug: return one step        
        return     

    
       
            
        
    train_logger.info('Full imagenet train set one epoch:  * Acc@1 {} Acc@5 {}'
          .format(top1, top5))
    return



def train_model(qat_model,eval_temp_model,train_loader,val_loader, accelerator, epoch,qconfig_mapping):
    
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    
    # define training setting
    
    
    train_logger.info(accelerator.print(f'device {str(accelerator.device)} is used!'))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    nepochs = epoch
    num_eval_batches=-1

            
   

    for nepoch in range(nepochs):
        if accelerator.is_main_process:
            train_logger.info("Training epoch: [{}/{}]\n".format(nepoch,nepochs))
        train_one_epoch(qat_model, train_loader,criterion,optimizer,accelerator,nepoch)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("Main_process is evaluating")
            save_model_path = "qat_epoch{}_model".format(nepoch)
            unwrapped_model = accelerator.unwrap_model(qat_model)
            # save model
            accelerator.save(unwrapped_model.state_dict(), save_model_path)
                          
            # unwrapped_model.load_state_dict(torch.load(save_model_path))
            # convert is a inplace operation ,so a temp_model is used to save model which be trained
            eval_temp_model.load_state_dict(torch.load(save_model_path,map_location="cpu")) 
            # # save multi GPU trained model to eval
            # torch.save(qat_model.state_dict(),path)
            # Check the accuracy after each epoch
            # convert_fx:  # convert fakequant_model to quantize model   
           
            #nnqat.op --> nnq.op 
            
            int8_model = _convert_fx(eval_temp_model, is_reference=True,  # 选择reference模式
                                    qconfig_mapping=qconfig_mapping,backend_config=get_tensorrt_backend_config_dict())


            test_logger.info("Convert QAT_model to int_model Done...\n")
            test_logger.info("Epoch {} Evaluation\n".format(nepoch))
            
            # check int8 model acc on CPU
            
            top1, top5 = evaluate(int8_model, val_loader,num_batches=20)
            
            test_logger.info('Epoch {} :Quantized model: Evaluation top1 accuracy : {} / top5 accuracy : {}'.format(
                nepoch, top1.avg, top5.avg
                )
            )
            import pdb
            pdb.set_trace()
            best_acc = 0
            if top1.avg>best_acc:
                best_acc = top1.avg
                best_model = int8_model
                test_logger.info("Best acc is {}".format(best_acc))
                torch.save(
                    qat_model.state_dict(),
                    os.path.join(os.getcwd(),"ckpt", "qat_checkpoint.ckpt"),
                )
                torch.save(
                    int8_model.state_dict(),
                    os.path.join(os.getcwd(), "ckpt","qat_checkpoint.ckpt"),
                )
        else:
            accelerator.wait_for_everyone()
            print("every one is here")
        return best_model



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

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def calibrate(model, calib_loader,device):
    # model should be eval during infer
    model.eval()
    
    total=len(calib_loader)
    pbar = tqdm(calib_loader, desc="PTQ Cali",total=total)
    with torch.no_grad():
        for image, target in pbar:
            image = image.to(device)
            model(image)
            
            
def load_model(device):
    # ViT-Base model (ViT-B/16) ImageNet-1k weights @ 224x224
    model = vit_base_patch16_224()
    checkpoint = torch.load("/home/wrq/checkpoint/vit_base_patch16_224.pth")
    # state_dict =  checkpoint["model"].state_dict()
    model.load_state_dict(checkpoint,strict=False)
    # model = replace_linear_with_conv(model)
    # model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=False,num_classes=1000,checkpoint_path = "/home/wrq/checkpoint/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz")
    # print(model.default_cfg)
    # imagenet1k trianed VIT
    # model = torchvision.models.vision_transformer.vit_b_16(pretrained=False)

    fp_model = copy.deepcopy(model)
    
    model.to(device)
    model.eval()
    return fp_model, model
            


class CFG:    
    

    # device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    data_path = '/data1/data/imagenet2012/'
    data_loader,data_loader_test = prepare_data_loaders(data_path=data_path,train_batch_size=30 ,val_batch_size = 100)
    example_inputs = (next(iter(data_loader_test))[0]) # get an example input

    
    
def main():
    q_backend = "qnnpack"  # qnnpack  or fbgemm
    torch.backends.quantized.engine = q_backend

    # set random seed
    random_seed = 0
    set_random_seeds(random_seed=random_seed)    
    fp_model, model = load_model(CFG.device)
    
    DataDir = "/data1/data/imagenet2012/"
    train_batch_size=64
    eval_batch_size=100
    epoch=1
    #operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK



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
    
    train_loader, val_loader = prepare_imagenet_loaders(DataDir,train_batch_size,eval_batch_size)


    # backend_config is used to specify 
    # op dtype/ob scle/zp tpe  such as shared_op weight_op non_weight_op

    prepared = prepare_qat_fx(model, qconfig_mapping,
                                    example_inputs=(torch.randn(1, 3, 224, 224),),
                                    backend_config= get_tensorrt_backend_config_dict() 
    )
    import pdb
    pdb.set_trace()
    eval_temp_model = copy.deepcopy(prepared)
    eval_temp_model.to("cpu")
    accelerator = Accelerator()
    # calibrate(prepared, CFG.data_loader_test,CFG.device)
    if accelerator.is_main_process:
    # Evaluation loop
        train_logger.info("Calibrate Model Done...")
        train_logger.info("Prepare QAT Model Done...")
        # train_logger.info(print(prepared))
        train_logger.info("Training QAT Model...")
    int8_model = train_model(prepared,eval_temp_model,train_loader, val_loader, accelerator,epoch,qconfig_mapping)

    

    # quantized_fx = _convert_fx(prepared, 
    #             is_reference=True,  # 选择reference模式
    #             qconfig_mapping = qconfig_mapping,
    #             backend_config =  get_tensorrt_backend_config_dict()    #get_qnnpack_backend_config()  
    #             )
    # import pdb
    # pdb.set_trace()
    # # print(quantized_fx)exit
    
    # # export int8model to onnx
    # image_tensor = torch.rand([100,3,224,224])
    # save_path = "./onnx_model/qat_int8_resnet.onnx"
    # onnx_quant_model = torch_to_int8_onnx(int8_model,example_input=image_tensor,save_path=save_path)
    # #比较 fx_int8_model 与 int8_onnx_model之间的相似度
    # cos_similarity = cos_similarity_bettween_model(int8_model,save_path,image_tensor)
    # print("cos_similarity is : {}".format(cos_similarity))
    
    # onnx_infer_result = eval_onnxmodel(onnx_quant_model)
    # print("onnx_infer_acc: {}".format(onnx_infer_result))
    
    # print('export done')
    
    # exported_model_path = 'vit_quant.onnx'
    # image_tensor = torch.rand([1,3,224,224])

    torch.onnx.export(quantized_fx, CFG.example_inputs,exported_model_path,opset_version=17)
    
    
    
if __name__=="__main__":
    main()




    
    
    
    
    