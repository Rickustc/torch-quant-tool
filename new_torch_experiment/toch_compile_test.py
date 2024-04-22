import torch
import time
import torch._dynamo
import torchvision.models as models

torch._dynamo.config.verbose=True
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

model = models.resnet50().cuda()

print("prepare model and input")
dummy_input = torch.randn(1,3,1024,1024).cuda()

NITER = 300
print("warm...")
for _ in range(10):
    res = model(dummy_input)
    torch.cuda.synchronize()
    
print("begin eval ...")
torch.cuda.synchronize()
s = time.time()
for _ in range(NITER):
    res = model(dummy_input)
    torch.cuda.synchronize()
    
print('benchmark time (CUDA normal) (ms/iter)', (time.time() - s) / NITER * 1000)

compiled_model = torch.compile(model)
print("warm...")
for _ in range(10):
    res_compiled = compiled_model(dummy_input)
    torch.cuda.synchronize()
    
print("begin eval ...")
torch.cuda.synchronize()
s = time.time()
for _ in range(NITER):
    res_compiled = compiled_model(dummy_input)
    torch.cuda.synchronize()
    
print('benchmark time (torch.compiled) (ms/iter)', (time.time() - s) / NITER * 1000)
print("check res cosine_similarity")
assert (
    torch.nn.functional.cosine_similarity(
        res.flatten(), res_compiled.flatten(), dim=0
    )
    > 0.9999
)