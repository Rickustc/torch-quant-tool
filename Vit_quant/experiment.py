import torch
import torch.nn as nn

q_backend = "qnnpack"  # qnnpack  or fbgemm
torch.backends.quantized.engine = q_backend


m = nn.quantized.Linear(10, 10)        # (内部初始化的scale为1)
params = m.state_dict()
for key, value in params.items():
    print(f"{key}: {value}")

import pdb
pdb.set_trace()
input = torch.randn(128, 10)

input = torch.quantize_per_tensor(input, 0.1, 0, torch.qint8)
'''
x -> q -> dq  = qint(x)
'''
# tensor([[ 0.1000, -0.9000, -0.6000,  ..., -0.9000,  0.2000,  0.3000],
#         [-1.5000,  0.1000, -0.4000,  ...,  0.3000,  1.8000,  1.7000],
#         [-0.7000, -1.1000,  0.5000,  ...,  2.2000, -0.2000, -0.3000],
#         ...,
#         [ 0.2000,  1.2000, -1.4000,  ...,  0.7000, -1.5000, -0.9000],
#         [ 1.3000,  0.9000, -0.5000,  ..., -1.0000, -2.3000,  1.6000],
#         [ 0.5000, -0.3000, -0.5000,  ...,  1.1000,  0.9000, -0.8000]],
#        size=(128, 20), dtype=torch.qint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.1, zero_point=0)
input.int_repr()
# tensor([[  1,  -9,  -6,  ...,  -9,   2,   3],
#         [-15,   1,  -4,  ...,   3,  18,  17],
#         [ -7, -11,   5,  ...,  22,  -2,  -3],
#         ...,
#         [  2,  12, -14,  ...,   7, -15,  -9],
#         [ 13,   9,  -5,  ..., -10, -23,  16],
#         [  5,  -3,  -5,  ...,  11,   9,  -8]], dtype=torch.int8)
output = m(input)
print(output.size())
