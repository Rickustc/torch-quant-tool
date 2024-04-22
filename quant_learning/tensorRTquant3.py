''''manually insert QDQ'''


import torch
import torchvision
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import utils as quant_nn_utils
def quant_state(module):
    for name,module in module.name_modules():
        if isinstance(module,quant_nn.tensorQuantier):
            print(name,module)
            
def transfer_torch_to_quantization(nninstance,quantmodule):
    
    #quant op 继承 原始op的 k,v
    quant_instance = quantmodule.__new__(quantmodule)
    for k,val in vars(nninstance).item():
        setattr(quant_instance,k,val)
        
    def __init__(self):
        if isinstance(self,quant_nn_utils.QuantInputMixin()):
            # 仅仅插入input的Descriptor，如relu
            quant_desc_input  =quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only = True)
            self.init_quantizer(quant_desc_input,quant_desc_weight)
            if instance(self.__input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            #插入weight和input的Descriptor
            quant_desc_input,quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input,quant_desc_weight)
            # 如果采用Histogramcalibrator，则设置_torch_hist为True
            if isinstance(self._input_quantizer._calibrator, calib.Histogramcalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True
        
        __init__(quant_instance)
        return quant_instance
        
            
# use this fuction to make a module to quantization module manually            
def replace_to_quantization_module(model):
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod,entry.mod_name)
        module_dict[id(module)] = entry.replace_mod       #replace conv1d of quant_conv.QuantConv1d
           
    # recursive traversal
    def recursive_and_replace_module(module,prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path = name if prefix == "" else prefix+'.' +name
            recursive_and_replace_module(submodule,path)
            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                module._modules[name] = transfer_torch_to_quantization(submodule,module_dict[submodule_id])
    recursive_and_replace_module(model)
        

# use initialize function to insert qdq automatically, you can set disable_quantization().apply()  to disable a specific op

# quant_modules.initialize()
model = torchvision.models.resnet50()
model.cuda()



#disable_quantization(model.conv1).apply()     关闭conv1算子的QDQ插入
#replace_to_quantization_module(model)


inputs = torch.randn(1,3,224,224,device="cuda")
quant_nn.TensorQuantizer.use_fd_fake_quant = True
torch.onnx.export(model,inputs,"quant_resnet_50.onnx",opset_version=13)