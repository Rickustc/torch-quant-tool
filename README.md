基于torch.fx的量化方案

目前实现:
resnet yolo regnety ViT模型的对称int8量化(W8A8)),包含PTQ和QAT
整个workflow为:
torch_quant_model --> onnx_qdq_modell --> onnxqlinear_model

todo:
- [ ] 实现LSQ算法以及集成到框架中
- [ ] bencmark相关指标
- [ ] 量化友好的模型设计探索
