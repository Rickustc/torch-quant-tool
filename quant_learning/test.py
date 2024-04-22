
import torch
import torch.fx

x = torch.rand(2,3, dtype=torch.float32)     #fp32 tensor([[0.0180, 0.6441, 0.2113],  #浮点数
       


import pdb
pdb.set_trace()
xq = torch.quantize_per_tensor(x, scale = 0.0036, zero_point = 0, dtype=torch.qint8)      #qint8  tensor([[0.0000, 0.5000, 0.0000]   # 反量化得到的qint8，形式上是浮点数，占用空间比fp32小

x_ = xq.int_repr()        #int8     tensor([[ 8,  9,  8],  取出定点数
 


xxx = xq.dequantize()     #fp32   tensor([[0.0000, 0.5000, 0.0000],  #浮点数  dequantize没做啥，就是把 qint 转换为 fp32

y = torch.quantize_per_tensor(xxx, scale = 0.0036, zero_point = 0, dtype=torch.quint8)
yy  = y.int_repr()

# yyy = x_.dequantize()     #fp32   tensor([[ 8.,  9.,  8.],    

# x

print(x)


