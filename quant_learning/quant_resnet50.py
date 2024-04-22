import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import timm
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx,convert_to_reference_fx
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


# import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
# from torch_tensorrt.fx import InputTensorSpec, TRTInterpreter, TRTModule
# from torch_tensorrt.fx.utils import LowerPrecision


warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)
from torchvision.models.resnet import resnet18

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

    
def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

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
            import pdb
            pdb.set_trace()
            # loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5


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



def calibrate(model, data_loader,device):
  
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            # image = image.to(device)
            if cnt>1:
                break
            print(cnt)
            cnt += 1
            # image = image.to(device)
            model(image)
            break




def check(model_type,CFG):
    if model_type == "int8_model":
        prepared_model = prepare_fx(CFG.model, CFG.qconfig_mapping, CFG.example_inputs) 
        calibrate(prepared_model, CFG.data_loader_test,CFG.device)
        prepared_model = _convert_fx(prepared_model,is_reference=True,)
        top1, top5 = evaluate(prepared_model, CFG.data_loader_test,CFG.device)
        print("int8acc:    "+'\n')
        print(top1)
        print(top5)

    elif model_type == "fp32_model":

        top1, top5 = evaluate(CFG.model, CFG.data_loader_test,CFG.device)
        print(top1)
        print(top5)
        
def lower_to_trt(model, inputs, shape_ranges):
    """Lower a quantized model to TensorRT"""
    assert len(inputs) == 1, "lower_to_trt only works for one input currently"
    model = acc_tracer.trace(model, inputs)  # type: ignore[attr-defined]
    # TODO: test multiple inputs setting and enable multiple inputs
    input_specs = [
        InputTensorSpec(
            torch.Size([-1, *inputs[0].shape[1:]]),
            torch.float,
            shape_ranges=shape_ranges,
            has_batch_dim=True,
        )
    ]

    interp = TRTInterpreter(
        model, input_specs, explicit_batch_dimension=True, explicit_precision=True
    )
    result = interp.run(lower_precision=LowerPrecision.INT8)
    trt_mod = TRTModule(result.engine, result.input_names, result.output_names)
    return trt_mod


def fx2onnx(CFG):
    prepared_model = prepare_fx(CFG.model, CFG.qconfig_mapping, CFG.example_inputs,backend_config=get_evas_backend_config()) 
   
 
    calibrate(prepared_model, CFG.data_loader_test,CFG.device)
    convert_model = _convert_fx(prepared_model,is_reference=True,backend_config=get_evas_backend_config())
    print(convert_model)
    # import pdb
    # pdb.set_trace()
    # state_dict = prepared_model.state_dict(keep_vars=True)

    # for k, v in state_dict.items():
    #     print(k)
        # print(v)
    exported_model_path = 'resnet50_int8_sym_0817.onnx'
    image_tensor = torch.rand([100,3,224,224])

    torch.onnx.export(convert_model, image_tensor, exported_model_path, verbose=False, input_names=["images"], output_names=["pred"], opset_version=13)

def resnet2onnx(CFG):
    exported_model_path = 'resnet500000000000000.onnx'

    image_tensor = torch.rand([100,3,224,224]).to(CFG.device)
    torch.onnx.export(CFG.model, image_tensor, exported_model_path, verbose=False, input_names=["images"], output_names=["pred"], opset_version=10
                      )


def resnetuint82onnx(CFG):
    # qconfig = get_default_qconfig("fbgemm")
    # qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    
    exported_model_path = '/home/wrq/00rrrrrrresnet50fp32_uint8_86-0825.onnx'
    image_tensor = torch.rand([100,3,224,224])       #.to(CFG.device)
    model = timm.create_model('resnet18', pretrained=True)
    model.eval()
    # model.to(CFG.device)
    prepared_model = prepare_fx(model, qconfig_mapping, CFG.example_inputs)
    prepared_model = _convert_fx(prepared_model,is_reference=False)
    print(prepared_model)
   
    # state_dict = prepared_model.state_dict(keep_vars=True)
    # for k, v in state_dict.items():
    #     print(k)
    print("######################################################################")
    torch.onnx.export(prepared_model, image_tensor, exported_model_path, verbose=False, input_names=["images"], output_names=["pred"], opset_version=16)
    
    

def is_int8_weight(model):
    """Check whether the input model is a int8 model.

    Args:
        model (torch.nn.Module): input model

    Returns:
        result(bool): Return True if the input model is a int8 model.
    """
    def _is_int8_value(value):
        """Check whether the input tensor is a int8 tensor."""
        if hasattr(value, 'dtype') and 'int8' in str(value.dtype):
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
        print(f"Output of {module.__class__.__name__}: {output.dtype}")
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

def fx2tr(CFG):
    
    q_backend = "qnnpack"  # qnnpack  or fbgemm
    torch.backends.quantized.engine = q_backend
    
    model = timm.create_model('resnet50', pretrained=True)
    
    model.eval()
    
    # model.forward = model.my_forward
  
    # torch.jit.trace(model)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}")
    qconfig = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.default_per_channel_weight_observer 
    )
    qconfig_linear = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)   # qnnpakc not support perchannel for linear oop now
    )
    qconfig_mapping = (QConfigMapping()
        .set_global(qconfig)  
        .set_object_type(torch.nn.Linear, qconfig_linear)  
    )


    prepared = prepare_fx(model, qconfig_mapping,
                                    example_inputs=(torch.randn(1, 3, 224, 224),),
                                    backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config()    
                                    )
    
    # print(prepared)
   
    calibrate(prepared, CFG.data_loader_test,CFG.device)
    

    print("###################################")
    
    quantized_fx = _convert_fx(prepared, 
                is_reference=False,  # 选择reference模式为False
                backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config() 
                )
    
    # from torch.fx.passes.graph_drawer import FxGraphDrawer
    # g = FxGraphDrawer(quantized_fx, "resnet50_fx_quantize")
    # g.get_main_dot_graph().write_svg("resnet50_fx_quantize.svg")
    # import pdb
    # pdb.set_trace()
    # check weight int8
    assert is_int8_weight(quantized_fx), "The exported model is not INT8 weight, "\
    "please reset 'dtype' to 'FP32' or check your model."
    # check activation int8
    assert is_int8_activation(quantized_fx), "The exported model is not INT8 activation, "\
    "please reset 'dtype' to 'FP32' or check your model."

    # print(quantized_fx.layer1.0.conv1.weight)
    # import pdb
    # pdb.set_trace()
    # quantized_fx.print_readable()
    # quantized_fx[""]
    # for name, param in quantized_fx.named_parameters():
    #     import pdb
    #     pdb.set_trace()
    #     print(f"Layer: {name}")
   
    # trt_mod = lower_to_trt(quantized_fx, inputs, shape_ranges) 
    # print(trt_mod)
 
    exported_model_path = 'quant_int8.onnx'
    image_tensor = torch.rand([100,3,224,224])

    torch.onnx.export(quantized_fx, image_tensor, exported_model_path,verbose=False,operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,input_names=["images"], output_names=["pred"], opset_version=16)
    
    
    


def vit_fx(CFG):
    q_backend = "qnnpack"  # qnnpack  or fbgemm
    torch.backends.quantized.engine = q_backend
    
    model = timm.create_model('vit_small_patch32_224.augreg_in21k_ft_in1k', pretrained=True)
    model.eval()
    exported_model_path = 'vit.onnx'
    image_tensor = torch.rand([1,3,224,224])
    torch.onnx.export(model, image_tensor, exported_model_path, input_names=["images"], output_names=["pred"], opset_version=13)
    import pdb
    pdb.set_trace()
    qconfig = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
        weight = torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8) 
    )
    
    qconfig_conv = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.default_per_channel_weight_observer
    )
    
    qconfig_linear = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8),
        weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8)   # qnnpakc not support perchannel for linear oop now
    )
    qconfig_mapping = (QConfigMapping()
        .set_global(qconfig)  
        .set_object_type(torch.nn.Linear, qconfig_linear) 
        .set_object_type(torch.nn.Conv2d, qconfig_conv)
    )
    prepared = prepare_fx(model, qconfig_mapping,
                                    example_inputs=(torch.randn(1, 3, 224, 224),),
                                    backend_config= get_tensorrt_backend_config_dict()  #get_qnnpack_backend_config() 
    )

    calibrate(prepared, CFG.data_loader_test,CFG.device)
    quantized_fx = _convert_fx(prepared, 
                is_reference=False,  # 选择reference模式
                backend_config=get_tensorrt_backend_config_dict()    #get_qnnpack_backend_config() 
                )

    # print(quantized_fx)
    exported_model_path = 'vit_int8_sym_0828.onnx'
    image_tensor = torch.rand([1,3,224,224])
    # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
    torch.onnx.export(quantized_fx, image_tensor, exported_model_path, input_names=["images"], output_names=["pred"], opset_version=13,export_params = False)
    

    
    
class CFG:    
    model = timm.create_model('resnet50', pretrained=True)

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.eval()

    data_path = '/data1/data/imagenet2012/'
  
    data_loader,data_loader_test = prepare_data_loaders(data_path=data_path,train_batch_size=30 ,val_batch_size = 100)

    example_inputs = (next(iter(data_loader_test))[0]) # get an example input

    
    qconfig = torch.ao.quantization.QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_channel_symmetric)
    )
                
    qconfig_mapping = (QConfigMapping()
        .set_global(qconfig)  # qconfig_opt is an optional qconfig, either a valid qconfig or None
        # .set_object_type(torch.nn.Conv2d, qconfig1) # can be a callable...
    )



    

if __name__=='__main__':
    cfg = CFG()
    check("fp32_model",cfg)    
    # fx2onnx(cfg)                 #in8quantize reference model2onnx
    # resnet2onnx(cfg)          #fp32model2onnx       
    # resnetuint82onnx(cfg)     #uint8quantizemodel2onnx
    # fx2tr(CFG)                      # fx--> tenssorRT
    # vit_fx(CFG)




    
    


