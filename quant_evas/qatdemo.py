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

from quant_api import create_model,quant,torch_to_int8_onnx,eval_onnxmodel,cos_similarity_bettween_model

def is_int8_weight(model):
    """Check whether the input model is a int8 model.

    Args:
        model (torch.nn.Module): input model

    Returns:
        result(bool): Return True if the input model is a int8 model.
    """
    def _is_int8_value(value):
        """Check whether the input tensor is a qint8 tensor."""
        if hasattr(value, 'dtype') and 'int8'  in str(value.dtype):
            return True
        else:
            return False

    stat_dict = dict(model.state_dict())
    for name, value in stat_dict.items():
        if _is_int8_value(value):
            return True
        # value maybe a tuple, such as 'linear._packed_params._packed_params'
        if isinstance(value, tuple):
            for v in value:
                if _is_int8_value(v):
                    return True
    return False


def is_int8_activation(model):
    #check activation int8

    input_data = torch.randn(1, 3, 256, 256)
    # hook for getting the output shape ofeach layer
    def print_layer_output(module, input, output):
        # print(f"Output of {module.__class__.__name__}: {output.dtype}")
        assert output.dtype==torch.qint8, "activation should be int8"
        
    hooks = []
    for layer in model.named_children():
    
        hook = layer[1].register_forward_hook(print_layer_output)
        hooks.append(hook)
        if isinstance(layer, nn.Module):
            hook = layer.register_forward_hook(print_layer_output)
            hooks.append(hook)
    output = model(input_data)
    for hook in hooks:
        hook.remove()
    return True



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


def evaluate(model, criterion, val_loader, neval_batches=-1):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0

    with torch.no_grad():
        for image, target in tqdm(val_loader):
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            # neval_batches set -1 to eval all test_set  
            if neval_batches >= 0 and cnt > neval_batches:
                return top1, top5
    return top1, top5

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
        sampler=train_sampler, num_workers=8, pin_memory=True,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=8, pin_memory=True,
        )

    return data_loader, data_loader_test

def create_model():
    model = timm.create_model("resnet50", pretrained=True)
    return model
 

def train_one_epoch(model_prepared, train_loader,criterion, optimizer, accelerator,nepoch,ntrain_batches=20):
    model_prepared.train()
    
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    
    
    # accelerator for multi gpu train
    model_prepared, train_loader, optimizer = \
            accelerator.prepare(model_prepared, train_loader, optimizer)

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
        #every 20 step
        if i % ntrain_batches==0:
            train_logger.info("Training Epoch:{} step:{}, Loss :{}, Training metric: {} , {}".format(
                                  nepoch,
                                  i,
                                  avgloss.avg.item(),
                                  top1,
                                  top5
                              )
                            )
        return 
       
            
        
    train_logger.info('Full imagenet train set one epoch:  * Acc@1 {} Acc@5 {}'
          .format(top1, top5))
    return



def train_model(qat_model,train_loader,val_loader,  epoch,qconfig_mapping,backend_config):
    
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    
    # define training setting
    accelerator = Accelerator()
    
    train_logger.info(accelerator.print(f'device {str(accelerator.device)} is used!'))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    nepochs = epoch
    num_eval_batches=-1

            
    backend_config=torch.ao.quantization.backend_config.qnnpack.get_qnnpack_backend_config()

    for nepoch in range(nepochs):
        train_logger.info("Training epoch: [{}/{}]\n".format(nepoch,nepochs))
        train_one_epoch(qat_model, train_loader,criterion,optimizer,accelerator,nepoch)
        # Check the accuracy after each epoch
        # convert_fx:  # convert fakequant_model to quantize model   
        temp_model = copy.deepcopy(qat_model)
        #nnqat.op --> nnq.op 
        int8_model = convert_fx(temp_model.to("cpu"),qconfig_mapping=qconfig_mapping,backend_config=backend_config)
        # check wether weight is int8
        assert is_int8_weight(int8_model)==True, "The exported model is not INT8 weight, "\
        "please reset 'dtype' to 'FP32' or check your model."
        # check wether activation is int8
        assert is_int8_activation(int8_model)==True, "The exported model is not INT8 activation, "\
        "please reset 'dtype' to 'FP32' or check your model."
        test_logger.info("Convert QAT_model to int_model Done...\n")
        test_logger.info("Epoch {} Evaluation\n".format(nepoch))
        
        # check int8 model acc
        top1, top5 = evaluate(int8_model,criterion, val_loader,num_eval_batches=20)
        
        test_logger.info('Epoch {} :Quantized model: Evaluation top1 accuracy : {} / top5 accuracy : {}'.format(
            nepoch, top1.avg, top5.avg
            )
        )
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
        return best_model

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

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

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model

def main():
    
    # create model
    model = create_model()
    # set random seed
    random_seed = 0
    set_random_seeds(random_seed=random_seed)
    

    DataDir = "/data1/data/imagenet2012/"
    train_batch_size=64
    eval_batch_size=16
    epoch=4
    
    '''
    accelerate config
    accelerate launch filename.py 
    '''
    
    
    # set backend_config for q_config support
    # set backend for int8 export
    # set qconfig  
    backend_config=torch.ao.quantization.backend_config.qnnpack.get_qnnpack_backend_config()
    torch.backends.quantized.engine = 'qnnpack'
    qconfig=default_per_channel_symmetric_qnnpack_qat_qconfig
    qconfig_linear = default_symmetric_qnnpack_qat_qconfig
    qconfig_mapping = (QConfigMapping()
        .set_global(qconfig)  
        .set_object_type(torch.nn.Linear, qconfig_linear)    
    )


    
    train_loader, val_loader = prepare_imagenet_loaders(DataDir,train_batch_size,eval_batch_size)
    ## prepare fakequant qatmodel
    #nn.op --> nnqat.op  
    qat_model = prepare_qat_fx(model,qconfig_mapping=qconfig_mapping,example_inputs=(torch.randn(1, 3, 256, 256),),backend_config=backend_config)
    train_logger.info("Prepare QAT Model Done...")
    train_logger.info(print(qat_model))
    # train qat model
    train_logger.info("Training QAT Model...")
    int8_model = train_model(qat_model,train_loader, val_loader, epoch,qconfig_mapping,backend_config)
    
    # export int8model to onnx
    image_tensor = torch.rand([100,3,224,224])
    save_path = "./onnx_model/qat_int8_resnet.onnx"
    onnx_quant_model = torch_to_int8_onnx(int8_model,example_input=image_tensor,save_path=save_path)
    #比较 fx_int8_model 与 int8_onnx_model之间的相似度
    cos_similarity = cos_similarity_bettween_model(int8_model,save_path,image_tensor)
    print("cos_similarity is : {}".format(cos_similarity))
    
    onnx_infer_result = eval_onnxmodel(onnx_quant_model)
    print("onnx_infer_acc: {}".format(onnx_infer_result))
    
    print('export done')
    
    

    # # Save quantized model.
    # save_torchscript_model(model=quantized_model,
    #                        model_dir=model_dir,
    #                        model_filename=quantized_model_filename)

    # # Load quantized model.
    # quantized_jit_model = load_torchscript_model(
    #     model_filepath=quantized_model_filepath, device=cpu_device)

    # _, fp32_eval_accuracy = evaluate_model(model=model,
    #                                        test_loader=test_loader,
    #                                        device=cpu_device,
    #                                        criterion=None)
    # _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model,
    #                                        test_loader=test_loader,
    #                                        device=cpu_device,
    #                                        criterion=None)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    # print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    # print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    # fp32_cpu_inference_latency = measure_inference_latency(model=model,
    #                                                        device=cpu_device,
    #                                                        input_size=(1, 3,
    #                                                                    32, 32),
    #                                                        num_samples=100)
    # int8_cpu_inference_latency = measure_inference_latency(
    #     model=quantized_model,
    #     device=cpu_device,
    #     input_size=(1, 3, 32, 32),
    #     num_samples=100)
    # int8_jit_cpu_inference_latency = measure_inference_latency(
    #     model=quantized_jit_model,
    #     device=cpu_device,
    #     input_size=(1, 3, 32, 32),
    #     num_samples=100)
    # fp32_gpu_inference_latency = measure_inference_latency(model=model,
    #                                                        device=cuda_device,
    #                                                        input_size=(1, 3,
    #                                                                    32, 32),
    #                                                        num_samples=100)

    # print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(
    #     fp32_cpu_inference_latency * 1000))
    # print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(
    #     fp32_gpu_inference_latency * 1000))
    # print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(
    #     int8_cpu_inference_latency * 1000))
    # print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(
    #     int8_jit_cpu_inference_latency * 1000))



if __name__ == "__main__":
    main()
