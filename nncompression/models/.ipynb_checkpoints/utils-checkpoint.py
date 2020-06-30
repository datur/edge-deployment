"""

"""
import torch
import torchvision.models as models

DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')  # nopep8 pyl disable=no-member

CIFAR10_MODELS = {}

IMAGENET_MODELS = {}


def get_top5(net, data):
    with torch.no_grad():
        out = net(data)
        return torch.topk(torch.nn.functional.softmax(out, dim=1), 5), out


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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
        self.max = float('-inf')
        self.min = float('inf')

    def update(self, val, n=1):
        self.max = val if val >= self.max else self.max
        self.min = val if val <= self.min and val > 0 else self.min
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



IMAGENET_MODELS = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "wide_resnet50": models.wide_resnet50_2,
        "wide_resnet101": models.wide_resnet101_2,
        "resnext50":models.resnext50_32x4d,
        "resnext101": models.resnext101_32x8d,
        "densenet121": models.densenet121,
        "densenet161": models.densenet161,
        "densenet169": models.densenet169,
        "densenet201": models.densenet201,
        "vgg11" : models.vgg11,
        "vgg11_bn":  models.vgg11_bn,
        "vgg13" : models.vgg13,
        "vgg13_bn":  models.vgg13_bn,
        "vgg16":  models.vgg16,
        "vgg16_bn":  models.vgg16_bn,
        "vgg19":  models.vgg19,
        "vgg19_bn":  models.vgg19_bn,
        "alexnet": models.alexnet,
        "googlenet": models.googlenet,
        "inceptionv3":models.inception_v3,
        "mnasnet_05":models.mnasnet0_5,
        "mnasnet_10":  models.mnasnet1_0,
        "mobilenetv2": models.mobilenet_v2,
        "shufflenetv2_05": models.shufflenet_v2_x0_5,
        "shufflenetv2_10": models.shufflenet_v2_x1_0,
        "squeezenetv1": models.squeezenet1_0,
        "squeezenetv1_1": models.squeezenet1_1
        }    
