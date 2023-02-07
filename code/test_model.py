import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import torch.nn as nn
from weed_data_class_od import WeedDataOD
from weeds import Weeds
import os
import torch.nn as nn
import pandas
import json
from pprint import pprint
from metrics import DetectionMetrics
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
#padilla code need to be located in the same directory as the code calling it
from Evaluator import Evaluator
from utils import *
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from datetime import datetime
from torch_model_runner import get_num_examples_per_class_in_dataset



def trim_targets(targets):    
    #fix for extra dimension, when more than one bbox...
    if(targets["labels"].shape != torch.Size([1,1])):
        new_targets = {}            
        new_targets["boxes"] = np.squeeze(targets["boxes"])
        new_targets["labels"] = np.squeeze(targets["labels"])
        targets = new_targets
    else:
        new_targets = {}            
        new_targets["boxes"] = targets["boxes"][0]
        new_targets["labels"] = targets["labels"][0]
        targets = new_targets
    return targets


def move_targets_to_device(targets, device):
    #explicitly create tensors on the gpu for target data
    boxes_gpu = torch.as_tensor(targets["boxes"], dtype=torch.float32, device=device)
    labels_gpu = torch.as_tensor(targets["labels"], dtype=torch.int64, device=device)
    targets_gpu = {}
    targets_gpu["boxes"] = boxes_gpu
    targets_gpu["labels"] = labels_gpu
    return targets_gpu


def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]    
    targets = [item[1] for item in batch]
    img_paths = [item[2] for item in batch]
    return [data, targets, img_paths]


def move_targets_to_device(targets, device):
        #explicitly create tensors on the gpu for target data    
        targets_gpu = []
        for i in range(len(targets)):
            d = {}
            boxes_gpu = torch.as_tensor(targets[i]["boxes"], dtype=torch.float32, device=device)
            labels_gpu = torch.as_tensor(targets[i]["labels"], dtype=torch.int64, device=device)    
            d["boxes"] = boxes_gpu
            d["labels"] = labels_gpu
            targets_gpu.append(d)

        return targets_gpu


def validate_img_paths(pickle_file=None):
    df = pandas.read_pickle(pickle_file)
    for index, entry in df.iterrows():
        im_path = entry['img_path']
        if(not os.path.isfile(im_path)):
            org_path = im_path.split('auto_annotation/imgs')[-1]
            img = Image.open(org_path).convert("RGB")
            #assume img size 2k or 4k
            if(img.size != (1920, 1080)):
                side_crop = entry['side_crop']
                height_crop =entry['height_crop']
                #(left, upper, right, lower)
                crop_region = (side_crop*img.width, height_crop*img.height, img.width-img.width*side_crop, img.height)
                crop_img = img.crop(crop_region)            
                crop_img.save(im_path)
                print(im_path +  ' cropped and saved in auto annotation path')


def automatic_per_class_metric_test(model_path=None, pickle_file=None, save_imgs=False, 
                                    confidence_threshold=0.7, iou_threshold=0.5, class_map=None,
                                    confidence_threshold_save_img=0.7):
    SAVE_IMG = save_imgs
    if(SAVE_IMG):
        test_img_path = '/train/resnet18/test_imgs/final/'
        os.makedirs(name=test_img_path, mode=0o755, exist_ok=True)

    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    data_transform = transforms.Compose([transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])               

    model = torch.load(model_path)    
    model.to(device)
    model.eval()
    
    dataset_test = WeedDataOD(pandas_file=pickle_file, device=device, transform=data_transform, class_map=class_map)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=1, collate_fn=my_collate, drop_last=True)

    my_metrics = DetectionMetrics(class_map, confidence_thresholds=confidence_threshold, iou_thresholds=iou_threshold)
    #torch_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds= [0.50, 0.60, 0.70, 0.80, 0.90], class_metrics=True)
    torch_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    myBoundingBoxes = BoundingBoxes()
    
    for i, data in enumerate(data_loader_test, 0): 
        print(i)             
        inputs, targets, entry = data
        
        inputs_gpu = []
        for im in inputs:
            inputs_gpu.append(im.to(device))

        with torch.no_grad():
            preds = model(inputs_gpu)
        
        preds_boxes_np = []
        preds_labels_np = []
        preds_scores_np = []
        for pred in preds:
            preds_boxes_np.append(pred['boxes'].detach().cpu().numpy())
            preds_labels_np.append(pred['labels'].detach().cpu().numpy())
            preds_scores_np.append(pred['scores'].detach().cpu().numpy())

        targets_boxes_np = []
        targets_labels_np = []
        for target in targets:   
            targets_boxes_np.append(target['boxes'].detach().cpu().numpy())
            targets_labels_np.append(target['labels'].detach().cpu().numpy())

        #do not use this with array of thresholds, creates a lot of data
        if(SAVE_IMG):
            for z in range(len(entry)):
                img_path = entry[z]['img_path']
                img = dataset_test.load_img_with_path(i, img_path)
                draw = ImageDraw.Draw(img)
            
                #draw boxes        
                for j in range(len(preds_labels_np[z])):
                    print('score: ' + str(preds_scores_np[z][j]))
                    if(preds_scores_np[z][j] >= confidence_threshold_save_img): 
                        box = preds_boxes_np[z][j]
                        shape = [(box[0], box[1]), (box[2], box[3])]
                        draw.rectangle(shape, outline='red', width=3)

                for k in range(len(targets_labels_np[z])):
                    box = targets_boxes_np[z][k]
                    if(len(box) > 0):
                        shape = [(box[0], box[1]), (box[2], box[3])]
                        draw.rectangle(shape, outline='blue', width=3)

                img.save(test_img_path + img_path.replace('/', '_'))
                        

        #batch
        for r in range(len(entry)):                
            gt = {
                'boxes': targets_boxes_np[r],
                'labels': targets_labels_np[r]
            }

            #filter inside
            result_detection_metrics = {
                'boxes': preds_boxes_np[r],
                'labels': preds_labels_np[r],
                'scores': preds_scores_np[r]
            }

            preds_cpu = {}
            if(len(gt['boxes'][0]) > 0):
                preds_cpu['boxes'] = preds[r]['boxes'].detach().cpu()
                preds_cpu['labels'] = preds[r]['labels'].detach().cpu()
                preds_cpu['scores'] = preds[r]['scores'].detach().cpu()
                my_metrics.update(predictions=result_detection_metrics, gt=gt)
                torch_metric.update([preds_cpu], [targets[r]])

            for g in range(len(gt['labels'])):
                gt_boundingBox = BoundingBox(imageName='img_' + str(i) + '_' + str(r), classId=class_map[gt['labels'][g]], x=gt['boxes'][g][0], y=gt['boxes'][g][1], 
                                    w=gt['boxes'][g][2], h=gt['boxes'][g][3], typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2)
                
                myBoundingBoxes.addBoundingBox(gt_boundingBox)

            for d in range(len(preds_cpu['labels'])):
                detected_boundingBox = BoundingBox(imageName='img_' + str(i) + '_' + str(r), classId=class_map[preds_cpu['labels'][d]], classConfidence=preds_cpu['scores'][d], 
                                        x=preds_cpu['boxes'][d][0], y=preds_cpu['boxes'][d][1], w=preds_cpu['boxes'][d][2], h=preds_cpu['boxes'][d][3], 
                                        typeCoordinates=CoordinatesType.Absolute, bbType=BBType.Detected, format=BBFormat.XYX2Y2)
                            
                myBoundingBoxes.addBoundingBox(detected_boundingBox)


    my_metrics.calc_metrics(metrics_save_path=metrics_save_path)
    pprint(torch_metric.compute())
    evaluator = Evaluator()
    metricsPerClass = evaluator.GetPascalVOCMetrics(myBoundingBoxes, IOUThreshold=0.3)
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    AP = dict()
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        #print(str(precision))
        recall = mc['recall']
        #print(str(recall))
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        #print(str(ipre))
        irec = mc['interpolated recall']
        #print(str(irec))
        # Print AP per class
        if(average_precision.dtype != 'float64'):
            average_precision = 0.0
        print('%s: %f' % (c, average_precision))        
        AP[c] = average_precision

    t = datetime.now()
    datestr = t.strftime("%Y_%m_%d_%H_%M_%S")
    with open('/train/AveragePrecision_' + datestr + '.json', 'w') as ap_file:            
            json.dump(AP, ap_file)

    print('nice stopping point to print stats...')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the model and get metrics')
    parser.add_argument('-f', '--settings_file', type=str, default='/code/settings_file_gt_train_val.json', help='settings file for all settings regarding the networks to train', required=False)
    parser.add_argument('-m', '--model_path', type=str, default='/train/resnet18_model.pth', help='Pytorch model file path', required=False)
    parser.add_argument('-p', '--pickle_path', type=str, default='/train/pickled_weed/pd_val_full_hd.pkl', help='validation pickle file path', required=False)
    parser.add_argument('-s', '--save_path', type=str, default='/train', help='path to save metrics json file and graphs', required=False)
    parser.add_argument('-l', '--limit_metrics', type=bool, default=True, help='use limited metrics, if False use full version', required=False)
    parser.add_argument('-i', '--save_images', type=bool, default=False, help='save images for inspection, gt and pred boxes overlayed', required=False)
    
    args = parser.parse_args()

    settings = None
    with open(args.settings_file) as json_file:            
            settings = json.load(json_file)

    #argparse bool hack
    limit_metrics = args.limit_metrics == 'True'
    save_images = args.save_images == 'True'

    if(limit_metrics):
        metrics_confidence_range = settings['metrics_confidence_range_small']
        metrics_iou_range = settings ['metrics_iou_range_small']
    else:
        metrics_confidence_range = settings['metrics_confidence_range']
        metrics_iou_range = settings ['metrics_iou_range']
    
    metrics_save_path = args.save_path + '/class_metrics.json'        
    
    try:
        weeds = Weeds(port=int(os.environ["MONGODB_PORT"]))    
        class_map = weeds.get_object_classes_for_annotations_with_task_filter('gt_val')
    except:
        print('Error, no connection with MongoDB, this is ok if fast track mode is chosen.')
        
    get_num_examples_per_class_in_dataset(args.pickle_path)

    #override class map
    class_map = settings["default_class_map"]
    print(class_map)
    automatic_per_class_metric_test(model_path=args.model_path, pickle_file = args.pickle_path, save_imgs=save_images, 
                                    confidence_threshold=metrics_confidence_range,
                                    iou_threshold=metrics_iou_range, class_map=class_map,
                                    confidence_threshold_save_img=settings['confidence_threshold_save_img'])
    

    #do graphs as separate step here
    metrics = DetectionMetrics(class_map=class_map, confidence_thresholds=metrics_confidence_range, iou_thresholds=metrics_iou_range)    
    metrics.make_graphs(metrics_json_path=metrics_save_path, figure_save_folder=args.save_path + '/result_figures')
        