

import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16


import torch
import torch.onnx
import onnx

class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # 权重矩阵
        self.W_q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_v = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # 计算查询（query）、键（key）和值（value）向量
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(1, 2))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 使用注意力权重对值进行加权平均
        output = torch.matmul(attention_weights, value)

        return output

# 创建自注意力模型
hidden_dim = 64
model = SelfAttention(hidden_dim)

# 定义一个示例输入
input_data = torch.randn(1, 10, hidden_dim)  # 输入形状为(batch_size, sequence_length, hidden_dim)

# 导出模型为ONNX
onnx_filename = "self_attention.onnx"
dummy_input = input_data
torch.onnx.export(model, dummy_input, onnx_filename, verbose=True,input_names=["images"], output_names=["pred"])

# 验证导出的ONNX模型
onnx_model = onnx.load(onnx_filename)
onnx.checker.check_model(onnx_model)

print("ONNX模型已成功导出并验证。")





