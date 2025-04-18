import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from accelerate import Accelerator

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--tune', dest='tune', action='store_true',
                    help='tune best int8 model on calibration dataset')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-i', "--iter", default=0, type=int,
                    help='For accuracy measurement only.')
parser.add_argument('-w', "--warmup_iter", default=5, type=int,
                    help='For benchmark measurement only.')
parser.add_argument('--performance', dest='performance', action='store_true',
                    help='run benchmark')
parser.add_argument("--accuracy", dest='accuracy', action='store_true',
                    help='For accuracy measurement only.')
parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                    help='path to checkpoint tuned by Neural Compressor (default: ./)')
parser.add_argument('--int8', dest='int8', action='store_true',
                    help='run benchmark')
parser.add_argument("--no_cuda", action='store_true', help='use cpu for training.')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')

best_acc1 = 0


def main():
    args = parser.parse_args()
    accelerator = Accelerator(cpu=args.no_cuda)

    if 'mobilenet' in args.arch:
        import torchvision.models.quantization as models
    else:
        import torchvision.models as models

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, accelerator)
        return

    if args.tune:
        model, train_loader, val_loader, optimizer = \
            accelerator.prepare(model, train_loader, val_loader, optimizer)

        def train_func(model):
            epochs = 8
            iters = 30
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                for image, target in train_loader:
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    accelerator.backward(loss) # loss.backward()
                    optimizer.step()
                    if cnt >= iters:
                        break

                if nepoch > 3:
                    # Freeze quantizer parameters
                    model.apply(torch.quantization.disable_observer)
                if nepoch > 2:
                    # Freeze batch norm mean and variance estimates
                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)


        import copy
        from neural_compressor import QuantizationAwareTrainingConfig
        from neural_compressor.training import prepare_compression
        model = copy.deepcopy(model)
        conf = QuantizationAwareTrainingConfig()
        compression_manager = prepare_compression(model, conf)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model
        train_func(model)
        compression_manager.callbacks.on_train_end()
        model._model = accelerator.unwrap_model(model._model)
        compression_manager.save(args.tuned_checkpoint)
        return

    if args.performance or args.accuracy:
        model.eval()
        if args.int8:
            from neural_compressor.utils.pytorch import load
            new_model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)),
                             model,
                             dataloader=val_loader)
        else:
            new_model = model
        if args.performance:
            from neural_compressor.config import BenchmarkConfig
            from neural_compressor import benchmark
            b_conf = BenchmarkConfig(warmup=5,
                                     iteration=args.iter,
                                     cores_per_instance=4,
                                     num_of_instance=1)
            benchmark.fit(new_model, b_conf, b_dataloader=val_loader)
        if args.accuracy:
            validate(val_loader, new_model, criterion, args, accelerator)
        return


def train(train_loader, model, criterion, optimizer, epoch, args, accelerator):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        output = accelerator.gather(output)
        target = accelerator.gather(target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(accelerator.gather(loss).sum().data.item(), input.size(0)*accelerator.num_processes)
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        accelerator.backward(loss) # loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, args, accelerator):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if i >= args.warmup_iter:
                start = time.time()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            output = accelerator.gather(output)
            target = accelerator.gather(target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(accelerator.gather(loss).sum().data.item(), input.size(0)*accelerator.num_processes)
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            if i >= args.warmup_iter:
                batch_time.update(time.time() - start)

            if i % args.print_freq == 0:
                progress.print(i)

            if args.iter > 0 and i >= (args.warmup_iter + args.iter - 1):
                break

        print('Batch size = %d' % args.batch_size)
        print('Accuracy: {top1:.5f} Accuracy@5 {top5:.5f}'
              .format(top1=(top1.avg / 100), top5=(top5.avg / 100)))

    return top1.avg/100


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
