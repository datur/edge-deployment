import torch
import torchvision.models as models
import torchvision.transforms as transforms
from nncompression.models.pytorch.utils import get_imagenet_val_loader
from openvino.inference_engine import IECore, IEPlugin
from nncompression.convert import to_openvino, to_onnx
from nncompression.utils import DEVICE
from nncompression.models.utils import IMAGENET_MODELS, AverageMeter, ProgressMeter
from nncompression.models.pytorch import utils as ptu
import nncompression.models.onnx.utils as onu
import nncompression.utils as nnu
import cv2
from PIL import Image
import numpy as np
import neptune
import os
import time
import json

# Global Vars
BATCH_SIZE = 1
print_freq = 1000

val_loader = get_imagenet_val_loader(
            '/media/linux/imagenet', batch_size=BATCH_SIZE)

dirs = sorted([d for d in [x for x in os.walk('models/openvino')][0][1] if not d.startswith('.') ])

neptune.init('davidturner94/edge-deployment-eval')

for model in dirs:
    


    model_xml = f"models/openvino/{model}/{model}.xml"
    model_bin = f"models/openvino/{model}/{model}.bin"

    ie = IECore()

    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="MYRIAD")

    device = ie.get_metric(metric_name="FULL_DEVICE_NAME", device_name="MYRIAD")
    precision = ie.get_metric(metric_name="OPTIMIZATION_CAPABILITIES", device_name="MYRIAD")

    PARAMS = {
                'device': device,
                'precision': precision[0],
    }

    tags = ['openvino', 'edge', 'pi']

    experiment = neptune.create_experiment(name=model, params=PARAMS)
    experiment.append_tags(tags)
    
    inputBlob = next(iter(net.inputs))
    (n, c, h, w) = net.inputs[inputBlob].shape

    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    batch_time = AverageMeter('Time', ':.5f')
    inference_time = AverageMeter('Inference Time', ':.5f')

    class_correct = list({'top1': 0., 'top5': 0., 'total': 0.} for _ in range(len(nnu.IMAGENET_LABELS)))

    progress = ProgressMeter(
                    len(val_loader),
                    [batch_time, top1, top5, inference_time],
                    prefix='Test: ')

    elapsed_time = 0

    for i, batch in enumerate(val_loader):
        

        end = time.time()
                
        img, lab = batch
        np_img = nnu.to_numpy(img)
                            
        end_infer = time.time()
        res = exec_net.infer({inputBlob:np_img})
        infer_time = time.time() - end_infer
                                            
        res_T5 = res[output_key][0].argsort()[-5:][::-1]
                                                    
                                                        
        batch_time.update(time.time() - end)
        elapsed_time += batch_time.val
       
        top1.update(100. if lab.item() == res_T5[0] else 0., lab.size(0))
        top5.update(100. if lab.item() in res_T5 else 0., lab.size(0))
                                                                                
        inference_time.update(infer_time)
                                                                                        
        class_correct[lab.item()]['total'] += 1
                                                                                                
        class_correct[lab.item()]['top1'] += 1 if lab.item() == res_T5[0] else 0
        class_correct[lab.item()]['top5'] += 1 if lab.item() in res_T5 else 0
                                                                                                            
                                                                                                                
        if i % print_freq == 0:
            progress.display(i)
                                                                                                                                    
        # neptune logging
        experiment.log_metric('inference_time_val', inference_time.val)
        experiment.log_metric('inference_time_avg', inference_time.avg)
        experiment.log_metric('top1_acc', top1.avg)
        experiment.log_metric('top5_acc', top5.avg)
        experiment.log_metric('batch', i)
        experiment.log_metric('total_time', elapsed_time)

    results = {
        model: {
            'batch_time': {
            'avg': batch_time.avg,
            'min': batch_time.min,
            'max': batch_time.max
            },
            'inference_time': {
            'avg': inference_time.avg,
            'min': infer_time.min,
            'max': infer_time.max
            },
            'top1_acc': top1.avg,
            'top5_acc': top5.avg
        }
    }
           
          
    class_accuracy = '\n'.join([json.dumps(dict(
        x, 
        **{'top1_accuracy': 100*x['top1']/x['total'], 
        'top5_accuracy': 100*x['top5']/x['total']}, 
        **{"class": nnu.IMAGENET_LABELS[i]})) for i, x in enumerate(class_correct)])
         
    experiment.log_text('class_accuracy', class_accuracy)
    experiment.log_text('results', results)
    experiment.stop()

    time.sleep(60)
