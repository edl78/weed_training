import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class WeedDataOD(Dataset):
    def __init__(self, pandas_file, device, transform=None, augmentation=None, class_map=None):
        self.df = pd.read_pickle(pandas_file)
        self.device = device
        self.transform = transform
        self.class_map = class_map
        #self.height_crop = 0.5
        #self.side_crop = 0.25
        #self.use_crop = True
        self.augmentatation = augmentation


    def crop_img(self, entry, img, height_crop, side_crop):
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
        img = img[crop_height: full_height, crop_width: crop_stop, :]
        
        return img


    def __getitem__(self, idx):
        entry = self.df.iloc[idx]  
        boxes = entry["bboxes"]
        img_path = entry["img_path"]
        img = Image.open(img_path).convert("RGB")
        height_crop = entry['height_crop']
        side_crop = entry['side_crop']        
                   
        #translate class into global class list 
        num_objs = len(entry["labels"])        
        labels = []
        for i in range(num_objs):                        
            labels.append(self.class_map.index(entry['labels'][i]))

        # convert everything into a torch.Tensor        
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        
        img = np.array(img)/255.0
        if(img.shape == (1080, 1920, 3)):
            print('pre cropped img, no cropping!')
        elif(img.shape == (2160, 3840, 3)):
            #if img has been through auto annotation with uploading of cropped img
            #these properties are set in regard to the cropped img. We have pointed
            #the img path to the original file so this need to be reset
            img_height, img_width, channels = img.shape
            entry['img_height'] = img_height
            entry['img_width'] = img_width
            img = self.crop_img(entry, img, height_crop, side_crop)
        else:
            print('unsupport img size')
        
        img_tensor = torch.as_tensor(np.transpose(img, (2,0,1)), dtype=torch.float32)
        
        if(self.augmentatation is not None):
            img_tensor, target = self.augmentatation(img_tensor, target)

        if(self.transform is not None):
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, target, entry


    def __len__(self):
        return len(self.df.index)        

    
    def load_img(self, idx):
        entry = self.df.iloc[idx] 
        img = Image.open(entry["img_path"]).convert("RGB")
        height_crop = entry['height_crop']
        side_crop = entry['side_crop']
        img = np.array(img)
        if(img.shape == (1080, 1920, 3)):
            print('pre cropped img, no cropping!')
        elif(img.shape == (2160, 3840, 3)):
            #if img has been through auto annotation with uploading of cropped img
            #these properties are set in regard to the cropped img. We have pointed
            #the img path to the original file so this need to be reset
            img_height, img_width, channels = img.shape
            entry['img_height'] = img_height
            entry['img_width'] = img_width
            img = self.crop_img(entry, img, height_crop, side_crop)
        else:
            print('unsupport img size')         
        #img = self.crop_img(entry, np.array(img), height_crop, side_crop)   
        img = Image.fromarray(img)     
        return img
        
    
    def load_img_with_path(self, idx, path):
        entry = self.df.iloc[idx] 
        img = Image.open(path).convert("RGB")
        height_crop = entry['height_crop']
        side_crop = entry['side_crop']         
        img = np.array(img)
        if(img.shape == (1080, 1920, 3)):
            print('pre cropped img, no cropping!')
        elif(img.shape == (2160, 3840, 3)):
            #if img has been through auto annotation with uploading of cropped img
            #these properties are set in regard to the cropped img. We have pointed
            #the img path to the original file so this need to be reset
            img_height, img_width, channels = img.shape
            entry['img_height'] = img_height
            entry['img_width'] = img_width
            img = self.crop_img(entry, img, height_crop, side_crop)
        else:
            print('unsupport img size')
        #img = self.crop_img(entry, np.array(img), height_crop, side_crop)   
        img = Image.fromarray(img)     
        return img
