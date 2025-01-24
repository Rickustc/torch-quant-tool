import numpy as np

import torch
from torchvision.models.quantization import mobilenet as qmobilenet
from torchvision.models import mobilenet
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from PIL import Image

def get_transform():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_real_image(im_height, im_width):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    return Image.open(img_path).resize((im_height, im_width))

def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)

def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(target, 0)))

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).numpy(), runtime

def quantize_model(model, inp):
    model.fuse_model()  #用于 fuse 符合特定模式的算子序列为一个算子
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack") #fbgemm
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    # [convert]quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
    torch.quantization.convert(model, inplace=True)


model = mobilenet.mobilenet_v2(pretrained=True).eval()

inp = np.random.rand(1,3, 224, 224).astype(np.float32)
qmodel = qmobilenet.mobilenet_v2(pretrained=True).eval()

##############################################################################
# Quantize, trace and run the PyTorch Mobilenet v2 model
# ------------------------------------------------------
# The details are out of scope for this tutorial. Please refer to the tutorials
# on the PyTorch website to learn about quantization and jit.
pt_inp = torch.from_numpy(inp)
quantize_model(qmodel, pt_inp)
qscript_module = torch.jit.trace(qmodel, pt_inp).eval()
script_module = torch.jit.trace(model, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()
    
    
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
print(mod) # comment in to see the QNN IR dump
import pdb
pdb.set_trace()

target = "llvm"
tvm_result, rt_mod = run_tvm_model(mod, params, input_name, inp, target=target)