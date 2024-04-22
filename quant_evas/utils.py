import collections
import cv2
from PIL import Image
import re
import torch
import os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import logging
from sklearn.metrics import accuracy_score
from neural_compressor.utils.utility import Statistics
import numpy as np
from sklearn.metrics import accuracy_score
logger = logging.getLogger(__name__)

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
                    'Transpose', 'ArgMax', 'Resize']

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

def _topk_shape_validate(preds, labels):
    # preds shape can be Nxclass_num or class_num(N=1 by default)
    # it's more suitable for 'Accuracy' with preds shape Nx1(or 1) output from argmax
    if isinstance(preds, int):
        preds = [preds]
        preds = np.array(preds)
    elif isinstance(preds, np.ndarray):
        preds = np.array(preds)
    elif isinstance(preds, list):
        preds = np.array(preds)
        preds = preds.reshape((-1, preds.shape[-1]))

    # consider labels just int value 1x1
    if isinstance(labels, int):
        labels = [labels]
        labels = np.array(labels)
    elif isinstance(labels, tuple):
        labels = np.array([labels])
        labels = labels.reshape((labels.shape[-1], -1))
    elif isinstance(labels, list):
        if isinstance(labels[0], int):
            labels = np.array(labels)
            labels = labels.reshape((labels.shape[0], 1))
        elif isinstance(labels[0], tuple):
            labels = np.array(labels)
            labels = labels.reshape((labels.shape[-1], -1))
        else:
            labels = np.array(labels)
    # labels most have 2 axis, 2 cases: N(or Nx1 sparse) or Nxclass_num(one-hot)
    # only support 2 dimension one-shot labels
    # or 1 dimension one-hot class_num will confuse with N

    if len(preds.shape) == 1:
        N = 1
        class_num = preds.shape[0]
        preds = preds.reshape([-1, class_num])
    elif len(preds.shape) >= 2:
        N = preds.shape[0]
        preds = preds.reshape([N, -1])
        class_num = preds.shape[1]

    label_N = labels.shape[0]
    assert label_N == N, 'labels batch size should same with preds'
    labels = labels.reshape([N, -1])
    # one-hot labels will have 2 dimension not equal 1
    if labels.shape[1] != 1:
        labels = labels.argsort()[..., -1:]
    return preds, labels


class TopK:
    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):
        preds, labels = _topk_shape_validate(preds, labels)
        preds = preds.argsort()[..., -self.k:]
        if self.k == 1:
            correct = accuracy_score(preds, labels, normalize=False)
            self.num_correct += correct

        else:
            for p, l in zip(preds, labels):
                # get top-k labels with np.argpartition
                # p = np.argpartition(p, -self.k)[-self.k:]
                l = l.astype('int32')
                if l in p:
                    self.num_correct += 1

        self.num_sample += len(labels)

    def reset(self):
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        if self.num_sample == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        return self.num_correct / self.num_sample


#dataloader for ort infer
class Dataloader:
    def __init__(self, dataset_location, image_list, batch_size):
        self.batch_size = batch_size
        self.image_list = []
        self.label_list = []
        self.random_crop = False
        self.resize_side= 256
        self.mean_value = [0.485, 0.456, 0.406]
        self.std_value = [0.229, 0.224, 0.225]
        self.height = 224
        self.width = 224
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(dataset_location, image_name)
                if not os.path.exists(src):
                    continue

                self.image_list.append(src)
                self.label_list.append(int(label))

    def _preprpcess(self, src):
        with Image.open(src) as image:
            image = np.array(image.convert('RGB'))
            
            height, width = image.shape[0], image.shape[1]
            scale = self.resize_side / width if height > width else self.resize_side / height
            new_height = int(height*scale)
            new_width = int(width*scale)
            image = cv2.resize(image, (new_height, new_width))
            image = image / 255.
            shape = image.shape
            if self.random_crop:
                y0 = np.random.randint(low=0, high=(shape[0] - self.height +1))
                x0 = np.random.randint(low=0, high=(shape[1] - self.width +1))
            else:
                y0 = (shape[0] - self.height) // 2
                x0 = (shape[1] - self.width) // 2
            if len(image.shape) == 2:
                image = np.array([image])
                image = np.repeat(image, 3, axis=0)
                image = image.transpose(1, 2, 0)
            image = image[y0:y0+self.height, x0:x0+self.width, :]
            image = ((image - self.mean_value)/self.std_value).astype(np.float32)
            image = image.transpose((2, 0, 1))
        return image

    def __iter__(self):
        return self._generate_dataloader()

    def _generate_dataloader(self):
        sampler = iter(range(0, len(self.image_list), 1))

        def collate(batch):
            """Puts each data field into a pd frame with outer dimension batch size"""
            elem = batch[0]
            if isinstance(elem, collections.abc.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, collections.abc.Sequence):
                batch = zip(*batch)
                return [collate(samples) for samples in batch]
            elif isinstance(elem, np.ndarray):
                try:
                    return np.stack(batch)
                except:
                    return batch
            else:
                return batch

        def batch_sampler():
            batch = []
            for idx in sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch

        def fetcher(ids):
            data = [self._preprpcess(self.image_list[idx]) for idx in ids]
            label = [self.label_list[idx] for idx in ids]
            return collate(data), label

        for batched_indices in batch_sampler():
            try:
                data = fetcher(batched_indices)
                yield data
            except StopIteration:
                return
    
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
            
            
            
def _get_module_op_stats(self, model, tune_cfg, approach):
    """This is a function to get quantizable ops of model to user.
    Args:
        model (object): input model
        tune_cfg (dict): quantization config
        approach (str): quantization approach
    Returns:
        None
    """
    modules = dict(model.named_modules())
    res = dict()

    if approach == 'post_training_dynamic_quant':
        # fetch int8 and fp32 ops set by Neural Compressor from tune_cfg
        for key in tune_cfg['op']:
            op_type = key[1]
            #build initial dict
            if op_type not in res.keys(): # pragma: no cover
                res[op_type] = {'INT8': 0, 'BF16': 0, 'FP32': 0}
            value = tune_cfg['op'][key]
            # Special cases: QuantStub, Embedding
            if ('weight' in value and value['weight']['dtype'] == 'fp32') or \
                ('weight' not in value and value['activation']['dtype'] == 'fp32'):
                res[op_type]['FP32'] += 1
            elif value['activation']['dtype'] == 'bf16':  # pragma: no cover
                res[op_type]['BF16'] += 1
            else:
                res[op_type]['INT8'] += 1
    else:
        quantized_mode = False
        for node in model.graph.nodes:
            if node.op == 'call_module':
                if node.target not in modules:  # pragma: no cover
                    continue
                op_class = type(modules[node.target])
                op_type = str(op_class.__name__)
                if 'quantized' in str(op_class) \
                    or (quantized_mode and 'pooling' in str(op_class)):
                    if op_type not in res.keys():
                        res[op_type] = {'INT8': 0, 'BF16': 0, 'FP32': 0}
                    res[op_type]['INT8'] += 1
                elif op_class in self.white_list:
                    if op_type not in res.keys():
                        res[op_type] = {'INT8': 0, 'BF16': 0, 'FP32': 0}
                    res[op_type]['FP32'] += 1
                continue
            elif node.op == 'call_function':
                op_type = str(node.target.__name__)
            else:
                op_type = node.target
            # skip input and output
            if not "quantize_per" in op_type and not quantized_mode:
                continue
            # skip zero_pioint and scale
            if "zero_point" in op_type or "scale" in op_type:
                continue
            #build initial dict
            if op_type not in res.keys():
                res[op_type] = {'INT8': 0, 'BF16': 0, 'FP32': 0}

            if "quantize_per" in op_type and not quantized_mode:
                quantized_mode = True
            elif "dequantize" in op_type and quantized_mode:
                quantized_mode = False
            res[op_type]['INT8'] += 1
    return res
            
            
# def _dump_model_op_stats(self, model, tune_cfg, approach):
#         """This is a function to dump quantizable ops of model to user.
#         Args:
#             model (object): input model
#             tune_cfg (dict): quantization config
#             approach (str): quantization approach
#         Returns:
#             None
#         """
#         if self.sub_module_list is None or \
#           self.approach == 'post_training_dynamic_quant':
#             res = self._get_module_op_stats(model, tune_cfg, approach)
#         else:
#             res = dict()
#             self._get_sub_module_op_stats(model, tune_cfg, approach, res)

#             if self.use_bf16 and (self.version.release >= Version("1.11.0").release) and \
#                 (CpuInfo().bf16 or os.getenv('FORCE_BF16') == '1'): # pragma: no cover
#                 bf16_ops_list = tune_cfg['bf16_ops_list']
#                 if len(bf16_ops_list) > 0:
#                     for bf16_op in bf16_ops_list:
#                         op_type = bf16_op[1]
#                         if op_type in res.keys():
#                             res[op_type]['BF16'] += 1
#                             if res[op_type]['FP32'] > 0:
#                                 res[op_type]['FP32'] -= 1
#                         else:
#                             res[op_type] = {'INT8': 0, 'BF16': 1, 'FP32': 0}


#         output_data = [[
#             op_type,
#             sum(res[op_type].values()), res[op_type]['INT8'], res[op_type]['BF16'],
#             res[op_type]['FP32']
#         ] for op_type in res.keys()]

#         Statistics(output_data,
#                    header='Mixed Precision Statistics',
#                    field_names=["Op Type", "Total", "INT8", "BF16", "FP32"]).print_stat()
        
        

            

            

