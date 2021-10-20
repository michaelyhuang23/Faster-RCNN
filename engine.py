import math
import sys
import time
import torch
import gc
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(writer, model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        assert(len(images) == len(targets))
        optimizer.zero_grad()
        # pseudo batch
        loss_sum = []
        for img, tar in zip(images, targets):
            imgs = [img]
            tars = [tar]
            imgs = list(image.to(device) for image in imgs)
            tars = [{k: v.to(device) for k, v in t.items()} for t in tars]
            loss_dict = model(imgs, tars)
            losses = sum(loss for loss in loss_dict.values())

            loss_val = losses.cpu().detach().item()
            loss_sum.append(loss_val)
            if not math.isfinite(loss_val):
                print("Loss is {}, stopping training".format(loss_val))
                print(loss_val)
                sys.exit(1)

            losses.backward()
        loss_val = sum(loss_sum)/len(loss_sum)
        writer.add_scalar('Loss/train', loss_val, epoch*len(data_loader)+i)
        metric_logger.update(loss=loss_val)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        optimizer.step()

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(writer, model, data_loader, device, print_freq = 100):
    with torch.no_grad():
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            images = list(img.to(device) for img in images)
            print(sum(image.element_size() * image.nelement() for image in images))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            del images
            gc.collect()
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator
