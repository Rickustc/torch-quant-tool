from pytorch_quantization import tensor_quant

import torch
torch.manual_seed(1234)
x = torch.rand(10)
fake_x = tensor_quant.fake_tensor_quant(x,x.abs().max())  #/home/wrq/anaconda3/envs/quant/lib/python3.9/site-packages/pytorch_quantization/tensor_quant.py   quant+dequant

print(x)
print(fake_x)