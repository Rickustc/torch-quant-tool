import torch

from collections import OrderedDict
import os
import contextlib
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.quantized.dynamic as nniqd
import torch.multiprocessing as mp
# from torch.ao.quantization import is_activation_post_process
# graph mode quantization based on fx
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
    convert_to_reference_fx,
    prepare_qat_fx,
    fuse_fx,
)
from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    QuantWrapper,
    default_qconfig,
    default_dynamic_qconfig,
    default_qat_qconfig,
    default_reuse_input_qconfig,
    per_channel_dynamic_qconfig,
    float16_dynamic_qconfig,
    float16_static_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
    get_default_qconfig,
    get_default_qat_qconfig,
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    fuse_modules,
    fuse_modules_qat,
    prepare,
    prepare_qat,
    convert,
    quantize_dynamic,
    default_placeholder_observer,
    default_weight_observer,
    PerChannelMinMaxObserver,
    FixedQParamsFakeQuantize,
    FixedQParamsObserver,
    FusedMovingAvgObsFakeQuantize,
    FakeQuantize,
    MovingAverageMinMaxObserver,
    HistogramObserver,
    QConfig,
    default_embedding_qat_qconfig,
)

from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
)
from torch.testing._internal.common_quantization import NodeSpec as ns


class M(torch.nn.Module):
    def forward(self, x, y):
        z = torch.matmul(x, y)
        return z

m = M().eval()
example_inputs = (torch.randn(2, 2), torch.randn(2, 2))
qconfig_dict = get_default_qconfig_mapping("fbgemm")

mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
mp(*example_inputs)
mq = convert_fx(mp)


expected_occurrence = {
    ns.call_function(torch.matmul): 0,
    ns.call_function(torch.ops.quantized.matmul): 1,
}

# verify no crash
res = mq(*example_inputs)
print(res)