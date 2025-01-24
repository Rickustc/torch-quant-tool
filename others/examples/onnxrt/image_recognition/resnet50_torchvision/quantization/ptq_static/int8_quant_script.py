from qdq2qoperator import *
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
from neural_compressor.config import _Config, options
import pdb
from pdb import set_trace as bp
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
    # get_evas_backend_config,
    get_tensorrt_backend_config_dict,
    get_qnnpack_backend_config
)


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


def replace_linear_with_conv(module):
    for name, sub_module in module.named_children():
        
        if isinstance(sub_module, nn.Linear):
            
            in_features = sub_module.in_features
            out_features = sub_module.out_features

            conv_layer = nn.Conv2d(in_features, out_features, kernel_size=1)
            conv_layer.weight.data = sub_module.weight.data.view(out_features, in_features,1,1)
            conv_layer.bias.data = sub_module.bias.data
            # model.fc =  conv_laye
            setattr(module, "fc", conv_layer)
            
        else:
            # 如果不是线性层，则递归处理子模块
            if len(list(sub_module.children())) > 0:
                replace_linear_with_conv(sub_module)





###############################################################################################

q_backend = "qnnpack"  # qnnpack  or fbgemm
torch.backends.quantized.engine = q_backend
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

model = timm.create_model('resnet50', pretrained=True)
model.eval()



replace_linear_with_conv(model)



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
                                backend_config= get_tensorrt_backend_config_dict()     
                                )

data_path = '/data1/data/imagenet2012/'

data_loader,data_loader_test = prepare_data_loaders(data_path=data_path,train_batch_size=30 ,val_batch_size = 100)

# calibrate(prepared, data_loader_test,device)


quantized_fx = _convert_fx(prepared, 
            is_reference=False,  # 是否选择reference模式
            backend_config= get_tensorrt_backend_config_dict()   
            )


exported_model_path = 'fx_resnet_int8_0823.onnx'
image_tensor = torch.rand([100,3,224,224])

torch.onnx.export(quantized_fx, image_tensor, exported_model_path, verbose=False, input_names=["images"], output_names=["pred"], opset_version=16)


###############################################################################################
# qdqmodel = onnx.load(exported_model_path)

# user_config = PostTrainingQuantConfig(   
#         quant_format="QOperator"
# )

# wrapped_model = Model(qdqmodel, conf=user_config)   

# user_configs = _Config(quantization=user_config, benchmark=None, pruning=None, distillation=None, nas=None)

# config_strategy = ConfigTuneStrategy(wrapped_model, user_configs)

# q_config = config_strategy.config_generate()

# convertor = Convertor(model= wrapped_model ,q_config=q_config)

# convertor.convert_model()

# qlinearmodel = convertor.model.model

# onnx = LazyImport('onnx')
# onnx.save(qlinearmodel,"INC-resnet50-int8-0823.onnx")

# print("successs")