import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx,_convert_fx,convert_to_reference_fx
from torch.ao.quantization import QConfigMapping
import torchvision
import timm
import pprint
import onnx
import thop
import onnxruntime as rt
from scipy import spatial
import os,shutil
import torchvision.transforms as transforms
import torch.nn as nn
import warnings
import pdb

from torch.ao.quantization.observer import *



def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')
    


model = timm.create_model('resnet18', pretrained=True)
model.eval()


# Set up warnings

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    
def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5


def prepare_data_loaders(data_path):
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
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)


    return data_loader, data_loader_test


# qconfig = get_default_qconfig("x86")

# qconfig_mapping = QConfigMapping().set_global(qconfig)


qconfig = get_default_qconfig("x86")
# qconfig1 = torch.ao.quantization.QConfig(
#     activation=MinMaxObserver.with_args(dtype=torch.qint8),
#     weight=MinMaxObserver.with_args(dtype=torch.qint8))


# Qconfigmapping设置规则
qconfig_mapping = (QConfigMapping()
    .set_global(qconfig)  # q
    # .set_object_type(torch.nn.Conv2d, qconfig1) # can be a callable...
)

# qconfig_mapping = (QConfigMapping()
#     .set_global(qconfig_opt)  # qconfig_opt is an optional qconfig, either a valid qconfig or None
#     .set_object_type(torch.nn.Conv2d, qconfig_opt) # can be a callable...
#     .set_object_type("torch.nn.functional.add", qconfig_opt) # ...or a string of the class name
#     .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig_opt) # matched in order, first match takes precedence
#     .set_module_name("foo.bar", qconfig_opt)
#     .set_module_name_object_type_order()
# )

data_path = '/data1/data/imagenet2012/'
train_batch_size = 30
eval_batch_size = 50
data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
example_inputs = (next(iter(data_loader_test))[0]) # get an example input
example_inputs


prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)  # fuse modules and insert observer
import pdb
pdb.set_trace()



def calibrate(model, data_loader):
    model.eval()
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            if cnt > 50:
                break
            print(cnt)
            cnt += 1
            model(image)
calibrate(prepared_model, data_loader_test)



quantized_model = _convert_fx(prepared_model,is_reference=False,)

#计算scale zp等参数，
# quantized_model = convert_to_reference_fx(prepared_model)
print(quantized_model)

#print_size_of_model(quantized_model)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
