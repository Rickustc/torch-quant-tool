import tvm
import tvm.relay as relay
import onnx
import torchvision
import torchvision.transforms as transforms
import torch
import onnxruntime as ort

def prepare_data_loaders(data_path, train_batch_size, eval_batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=8, pin_memory=True,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=8, pin_memory=True,
        )

    return data_loader, data_loader_test


data_path = '/data1/data/imagenet2012/'
train_batch_size = 100
eval_batch_size = 1
data_loader, data_loader_test = prepare_data_loaders(data_path, train_batch_size, eval_batch_size)
example_inputs = (next(iter(data_loader_test))[0]) # get an example input # (100,3,224,224)   

# prepare model
onnx_model = onnx.load("/home/wrq/resnet50_fp32.onnx")

# get onnx input info
sess = ort.InferenceSession("/home/wrq/resnet50_fp32.onnx")
input_nodes = sess.get_inputs()[0]

# prepare input
target = 'llvm'

shape_dict = {input_nodes.name: input_nodes.shape}

# relay IR (Relay expression) 
sys, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# set opt level
with relay.build_config(opt_level=3):
     intrp = relay.build_module.create_executor('graph', 
sys, tvm.cpu(0), target)
import pdb
pdb.set_trace()
# opt graph
func = intrp.evaluate(sys)

# run
output = func(input)   



#TODO: script-->relay ir





