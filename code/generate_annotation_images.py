import torch
import numpy as np
from PIL import Image, ImageDraw
import torch.nn as nn
from weeds import Weeds
import os
import argparse


def crop_img(entry, img, height_crop, side_crop):
        #------------------------------
        #         height crop         |
        #        -------------        |
        #  side  |           |  side  |
        #  crop  |           |  crop  |
        #------------------------------
        crop_height = np.int(entry['img_height'] * height_crop)
        crop_width = np.int(entry['img_width'] * side_crop)
        crop_stop = np.int(entry['img_width'] - crop_width)
        full_height = np.int(entry['img_height'])            
        #img = img[crop_height: full_height, crop_width: crop_stop, :]
        img = img.crop((crop_width, entry['img_height'] - crop_height, crop_stop, entry['img_height']))
             
        
        return img


def purge_bboxes(entry, boxes, labels, shape_types, height_crop, side_crop):
        #sizes
        crop_height = np.int(entry['img_height'] * height_crop)
        crop_width = np.int(entry['img_width'] * side_crop)
        crop_stop = np.int(entry['img_width'] - crop_width)

        #indexes
        crop_img = {
            'xmin': crop_width - 1,
            'xmax': crop_stop - 1,
            'ymin': crop_height - 1,
            'ymax': np.int(entry['img_height'] - 1)
        }
        #bbox format: [xmin, ymin, xmax, ymax]
        remove_index = []
        #add usable pixel margin
        pixel_margin = 30
        for r in range(len(labels)):
            #fully above
            if(boxes[r][3] <= (crop_img['ymin'] + pixel_margin)):
                remove_index.append(r)
            #fully to the right
            if(boxes[r][0] >= (crop_img['xmax'] - pixel_margin)):
                remove_index.append(r)
            #fully to the left
            if(boxes[r][2] <= (crop_img['xmin'] + pixel_margin)):
                remove_index.append(r)                            
                
        
        purge_boxes = []
        purge_labels = []
        purge_shape_types = []
        for j in range(len(labels)):
            if(j not in remove_index):
                purge_boxes.append(boxes[j])
                purge_labels.append(labels[j])                
                purge_shape_types.append(shape_types[j])

        #adjust boxes, clamp to new border
        #bbox format: [xmin, ymin, xmax, ymax]
        new_width = entry['img_width'] - (2 * crop_width)
        new_height = entry['img_height'] - crop_height
        for p in range(len(purge_labels)):
            if(purge_boxes[p][0] < crop_width):
                purge_boxes[p][0] = 0
            else:
                purge_boxes[p][0] -= crop_width
            
            if(purge_boxes[p][1] < new_height):
                purge_boxes[p][1] = 0
            else:
                purge_boxes[p][1] -= crop_height
                        
            if(purge_boxes[p][2] >= crop_stop):
                purge_boxes[p][2] = new_width - 1
            else:
                purge_boxes[p][2] -= crop_width 
            
            purge_boxes[p][3] -= crop_height            
            if(purge_boxes[p][3] >= new_height):
                #should never happen...
                purge_boxes[p][3] = new_height - 1                
            #add condition to remove too small adjusted boxes on upper limit and sides. Later.
                
        return purge_boxes, purge_labels, purge_shape_types


def make_images(task_name, dataset_dir, height_crop, side_crop):
    work_folder = name='/annotation_imgs/' + args.task_name.replace(' ', '_')
    os.makedirs(work_folder, mode=0o755, exist_ok=True)

    weeds = Weeds()
    
    meta_cursor = weeds.get_meta_for_task(task_name)
    meta_data = {}
    for index, item in enumerate(meta_cursor):
        meta_data[item['task_name']] = item
    
    for meta in meta_data[task_name]['frames']:
        annotations = []
        annotations_cursor = weeds.get_annotations_for_task_and_frame(task_name, meta['name'])
        for index, item in enumerate(annotations_cursor):
            annotations.append(item)
                
        dataEntry = {}      
        dataEntry['img_path'] = dataset_dir + '/' + meta['name']
        dataEntry['img_width'] = meta['width']
        dataEntry['img_height'] = meta['height']
        dataEntry['task_name'] = task_name         
        dataEntry['height_crop'] = height_crop
        dataEntry['side_crop'] = side_crop
        bboxes = []
        labels = []
        shape_types = []
        for annotation in annotations:                                                     
            labels.append(annotation['object_class'])
            shape_types.append(annotation['shape_type'])
            num_points = len(annotation['points'])
            if(num_points == 4):
                #bbox format: [xmin, ymin, xmax, ymax]
                bboxes.append(np.array((annotation['points'][0],
                                        annotation['points'][1],
                                        annotation['points'][2],
                                        annotation['points'][3]), dtype=np.float32))                        
            elif(num_points > 4):
                #take extreme points from polygon and create a new box                                   
                points = annotation['points']                 
                x_points = points[0:len(points):2]
                y_points = points[1:len(points):2]

                xmin = x_points[np.argmin(x_points)]
                xmax = x_points[np.argmax(x_points)]
                ymin = y_points[np.argmin(y_points)]
                ymax = y_points[np.argmax(y_points)]
                bboxes.append(np.array((xmin, ymin, xmax, ymax), dtype=np.float32))
            else:
                print('less than 4 points in annotation!')
                print(annotation)

        #same annotations might fall outside crop, adjust
        purged_bboxes, purged_labels, purged_shape_types = purge_bboxes(dataEntry, bboxes, labels, shape_types, height_crop, side_crop)

        if(len(purged_labels) > 0):
            #save img with imprinted annotations
            img = Image.open(dataEntry['img_path']).convert("RGB")        
            img_cropped = crop_img(dataEntry, img, height_crop, side_crop)
            draw = ImageDraw.Draw(img_cropped)
                    
            #draw boxes                
            for k in range(len(purged_labels)):
                box = purged_bboxes[k]
                if(len(box) > 0):
                    shape = [(box[0], box[1]), (box[2], box[3])]
                    draw.rectangle(shape, outline='blue', width=3)         
                    draw.text(xy=[box[0], box[1]], text=purged_labels[k])
            
            img_save_path = work_folder + '/' + meta['name'].split('/')[-1]
            print('save img: ' + img_save_path)
            img_cropped.save(img_save_path)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='generate annotation images with imprinted annotations')
    parser.add_argument('-t', '--task_name', type=str, default='FieldData 20200520145736 1L GH020068', help='CVAT task name to make images from', required=False)
    args = parser.parse_args()    

    #constants
    dataset_dir = '/weed_data'
    height_crop = 0.50
    side_crop = 0.25
    print('run annotation image generation on: ' + args.task_name)
    make_images(args.task_name, dataset_dir, height_crop, side_crop)    
