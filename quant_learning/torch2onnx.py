# 创建 PyTorch ResNet50 模型实例
import torch
import torchvision


model = torchvision.models.resnet50(pretrained=True)


batch_size = 4  

input_shape = (batch_size, 3, 224, 224)
input_data = torch.randn(input_shape)

output_path = "rrrrrresnet50.onnx"
torch.onnx.export(model, input_data, output_path,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

# 使用 ONNX 运行时加载模型
# session = onnxruntime.InferenceSession(output_path)

# # 定义一个 ONNX 张量来模拟输入数据
# new_batch_size = 8  # 定义新的批处理大小
# new_input_shape = (new_batch_size, 3, 224, 224)
# new_input_data = torch.randn(new_input_shape)

# # 在 ONNX 运行时中运行模型
# outputs = session.run(["output"], {"input": new_input_data.numpy()})