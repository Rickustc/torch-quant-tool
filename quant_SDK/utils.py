import torch
import os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import logging
from sklearn.metrics import accuracy_score
from neural_compressor.utils.utility import Statistics

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')
    
# op Staistic        
def _dump_model_op_stats(model):
    # list of op which can be quanted
    fp32_op_list = ['FusedConv', 'Conv', 'Gather', 'MatMul', 'Gemm', 'EmbedLayerNormalization', 
                    'Attention', 'Mul', 'Relu', 'Clip', 'LeakyRelu', 'Sigmoid', 'MaxPool', 'GlobalAveragePool', 
                    'Pad', 'Split', 'Add', 'Squeeze', 'Reshape', 'Concat', 'AveragePool', 'Unsqueeze', 
                    'Transpose', 'ArgMax', 'Resize','Softmax']

    qdq_ops = ["QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"]
    res = {}
    for op_type in fp32_op_list:
        res[op_type] = {'INT8':0, 'FP32':0}
    for op_type in qdq_ops:
        res[op_type] = {'INT8':0, 'FP32':0}

    for node in model.graph.node:
        if node.name.endswith('_quant'):
            if node.op_type.startswith('QLinear'):
                origin_op_type = node.op_type.split('QLinear')[-1]
            else:
                origin_op_type = node.op_type.split('Integer')[0]

            if origin_op_type in ["QAttention", "QGemm"]:
                origin_op_type = origin_op_type[1:]
            elif origin_op_type == "DynamicQuantizeLSTM":
                origin_op_type = "LSTM"
            elif origin_op_type == "QEmbedLayerNormalization":
                origin_op_type = "EmbedLayerNormalization"
            res[origin_op_type]['INT8'] += 1

        elif node.op_type in qdq_ops:
            res[node.op_type]['INT8'] += 1


        elif node.op_type in res:
            res[node.op_type]['FP32'] += 1

    field_names=["Op Type", "Total", "INT8",  "FP32"]
    output_data = [[
        op_type, sum(res[op_type].values()), 
        res[op_type]['INT8'], res[op_type]['FP32']]
    for op_type in res.keys()]

    Statistics(output_data, 
                header='Mixed Precision Statistics',
                field_names=field_names).print_stat()

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


def evaluate(model, criterion, data_loader, device):

    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in tqdm(data_loader):
            output = model(image.to(device))
            if len(output.shape) == 4:
                output = output.squeeze(-1).squeeze(-1)
            loss = criterion(output, target.to(device))
            cnt += 1
            acc1, acc5 = accuracy(output, target.to(device), topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print("acc top1: ", top1, "acc top5: ", top5)
    return top1, top5


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

def calibrate(model, data_loader, device, count=0):
    # count < 0 means run the whole dataset
    # count >=0 would early stop
    model.eval()
    cnt = 0
    with torch.no_grad():
        for image, target in tqdm(data_loader):
            if count >= 0 and cnt > count:
                break
            cnt += 1
            model(image.to(device))
            
            

            

