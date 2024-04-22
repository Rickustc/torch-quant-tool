import torch
import torch.nn as nn
from torch.ao.nn.quantized.modules.normalization import LayerNorm as QuantizedLayerNorm
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
    # get_evas_backend_config,
    get_tensorrt_backend_config_dict,
    get_qnnpack_backend_config
)
from torch.onnx import OperatorExportTypes   
# import torch.ao.nn.quantized.DeQuantize as DeQuantize
# import torch.ao.nn.quantized.Quantize as Quantize
import timm


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        # Functionals
        self.q_scaling_product = torch.ao.nn.quantized.FloatFunctional()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        self.float_fn = torch.nn.quantized.FXFloatFunctional()

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # x = self.dequant(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = self.quant(qkv)
        
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        
        # attn = self.float_fn.mul(q @ k.transpose(-2, -1),self.scale)
        # q = self.dequant(q)
        # k = self.dequant(k)
        attn = self.dequant(torch.matmul(q, k.transpose(-2, -1)))
        attn = self.q_scaling_product.mul_scalar(attn, self.scale)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dequant(attn)
        attn = attn.softmax(dim=-1)
        attn = self.quant(attn)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
    

model = Attention(768)
model.eval()
input_dummy = torch.rand([100, 197, 768])
#set backend for int infer
q_backend = "qnnpack"  # qnnpack  or fbgemm
torch.backends.quantized.engine = q_backend

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

qconfig_mapping = (QConfigMapping()
    # .set_global(qconfig)  
    .set_object_type('transpose', None) 
    .set_object_type('permute', None)
    .set_object_type('mul', None) 
    .set_object_type(torch.matmul, qconfig_linear) 
    .set_object_type(torch.nn.Linear, qconfig_linear) 
    .set_object_type(torch.nn.Conv2d, qconfig_conv)
    # .set_object_type(torch.nn.LayerNorm, qconfig_conv)
)


# prepare
prepared = prepare_fx(model, qconfig_mapping,
                                example_inputs=input_dummy,
                                backend_config= get_tensorrt_backend_config_dict() 
)

# calibrate
with torch.no_grad(): 
    prepared(input_dummy)

# convert
quantized_fx = _convert_fx(prepared, 
            is_reference=False,  # 选择reference模式
            backend_config=    get_tensorrt_backend_config_dict()    #get_qnnpack_backend_config()  
            )
import pdb
pdb.set_trace()
# export 
path = 'attention.onnx'
torch.onnx.export(quantized_fx, input_dummy,path, opset_version=13)