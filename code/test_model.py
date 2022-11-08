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


def automatic_per_class_metric_test(pickle_file=None, save_imgs=False, confidence_threshold=0.7, iou_threshold=0.5):
    SAVE_IMG = save_imgs
    if(SAVE_IMG):
        test_img_path = '/train/resnet18/test_imgs/final/'
        os.makedirs(name=test_img_path, mode=0o755, exist_ok=True)

    #variants = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    variants = ['resnet18']
    for variant in variants:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        data_transform = transforms.Compose([transforms.Normalize(
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])               

        model = torch.load('/train/' + variant + '_model.pth')        
        model.to(device)        
        model.eval()

        weeds = Weeds(port=27018)
        #verifiy that order of this array do not change
        #class_map = weeds.get_object_classes_for_all_annotations()
        #for testing without mongodb connection
        #class_map = ['Black bindweed', 'Cleavers', 'Common furnitory', 'Creeping thistle', 'Curled dock', 'Dandelion', 'Fat hen', 'Rape', 'Scentless Mayweed', 'Unknown weed']
        class_map = weeds.get_object_classes_for_annotations_with_task_filter('FieldData_20190604')
        print(class_map)        
        dataset_test = WeedDataOD(pandas_file=pickle_file, device=device, transform=data_transform, class_map=class_map)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=2, shuffle=False, num_workers=1, collate_fn=my_collate, drop_last=True)

        validate_img_paths(pickle_file=pickle_file)
        
        my_metrics = DetectionMetrics(class_map, confidence_thresholds=confidence_threshold, iou_thresholds=iou_threshold)
        
        for i, data in enumerate(data_loader_test, 0): 
            print(i)             
            inputs, targets, entry = data
            
            #targets = trim_targets(targets)    
            print(entry)
            inputs_gpu = []
            for im in inputs:
                inputs_gpu.append(im.to(device))

            with torch.no_grad():
                preds = model(inputs_gpu)
            
            preds_boxes_np = []
            preds_labels_np = []
            preds_scores_np = []
            for pred in preds:
                #print(pred['scores'].detach().cpu().numpy())
                preds_boxes_np.append(pred['boxes'].detach().cpu().numpy())
                preds_labels_np.append(pred['labels'].detach().cpu().numpy())
                preds_scores_np.append(pred['scores'].detach().cpu().numpy())

            targets_boxes_np = []
            targets_labels_np = []
            for target in targets:   
                targets_boxes_np.append(target['boxes'].detach().cpu().numpy())
                targets_labels_np.append(target['labels'].detach().cpu().numpy())

            #do not use this with array of thresholds
            if(SAVE_IMG):
                for z in range(len(entry)):
                    img_path = entry[z]['img_path']
                    img = dataset_test.load_img_with_path(i, img_path)
                    draw = ImageDraw.Draw(img)
                
                    #draw boxes        
                    for j in range(len(preds_labels_np[z])):
                        print('score: ' + str(preds_scores_np[z][j]))
                        if(preds_scores_np[z][j] >= confidence_threshold): 
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
                                
                if(len(gt['boxes'][0]) > 0):
                    my_metrics.update(predictions=result_detection_metrics, gt=gt)
                                                            
        my_metrics.calc_metrics(metrics_save_path=metrics_save_path)    





if __name__ == "__main__":

    confidence_range = [0.5, 0.6, 0.7, 0.8, 0.9]
    iou_range = [0.5, 0.6, 0.7, 0.8, 0.9]
    #confidence_range = [0.7]
    #iou_range = [0.5]
    
    metrics_save_path = '/train/class_metrics.json'
    #metrics_save_path = '/train/class_metrics_first_year.json'

    pickle_file = '/train/pickled_weed/pd_val.pkl'
    #pickle_file = '/train/pickled_weed/pd_val_first_year.pkl'
    automatic_per_class_metric_test(pickle_file = pickle_file, save_imgs=False, 
                            confidence_threshold=confidence_range,
                           iou_threshold=iou_range)
                      
    weeds = Weeds(port=27018)
    #class_map = weeds.get_object_classes_for_annotations_with_task_filter('FieldData_20190604')
    class_map = weeds.get_object_classes_for_annotations_with_task_filter('gt_val')
    
    #do graphs as separate step here
    metrics = DetectionMetrics(class_map=class_map, confidence_thresholds=confidence_range, iou_thresholds=iou_range)
    #metrics.make_graphs(metrics_json_path=metrics_save_path, figure_save_folder='/train/result_figures_first_year/')
    metrics.make_graphs(metrics_json_path=metrics_save_path, figure_save_folder='/train/result_figures/')
        