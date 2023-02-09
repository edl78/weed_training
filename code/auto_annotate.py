import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import torch.nn as nn
from weed_data_class_od import WeedDataOD
from weeds import Weeds
import os
import torch.nn as nn
from datetime import datetime
import argparse
from shapely.geometry import Polygon
import pandas
import requests
import json
import xmltodict
from numpyencoder import NumpyEncoder
import glob
import time


class AutoAnnotations():
    def __init__(self, username, password, cvat_base_url, backup=False):
        self.cvat = cvat_base_url
        self.upload_timeout_sec = 1800

        self.labels = None        
        with open('/code/labels.json') as json_file:            
            self.labels = json.load(json_file)
        
        self.username = username
        self.password = password
        self.auth_token = None
        self.cookies = None
        self.headers = None
        self.tasks = dict()
        self.login()
        try:
            self.weeds = Weeds(port=int(os.environ['MONGODB_PORT']))
            self.class_map = self.weeds.get_object_classes_for_annotations_with_task_filter(filter='gt_train')
        except:
            print('Run without MongoDB')


    def get_labels(self):
        return self.labels

    
    def get_class_map(self):
        return self.class_map


    def get_internal_task_list(self):
        return self.tasks


    def login(self):
        endpoint = 'auth/login'
        body = {'username': self.username, 'password': self.password}
        print('cvat auth in progress')
        r = requests.post(self.cvat+endpoint, data = body, timeout=30)
        if(r.status_code == 200):
            self.cookies = r.cookies
            self.headers = {'X-CSRFToken': r.cookies['csrftoken']}
            print('cvat auth done')
        else:
            print(r.reason)
            print('cvat auth failed')


    def create_task(self, task_name=None, labels=None):
        endpoint = 'tasks'
        
        task_labels = self.labels
        if(labels != None):
            task_labels = labels
        body = {'name': task_name, 'labels': task_labels}
        r = requests.post(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)
        if(r.status_code == 201):
            print('task creation started for task: ' + task_name)
            #ugly static wait since we cant query on the id before it exists
            time.sleep(5)
            #update internal task list
            self.get_tasks()         
            while(True):
                time.sleep(1)
                print('wait for task creation to complete')
                r = self.get_status(task_name=task_name)
                resp = r.json()                
                if(resp['state'] == 'Finished'):
                    print('task creation completed')
                    break                
        else:
            print(r.reason)
            print('task creation failed')


    def get_tasks(self):
        endpoint = 'tasks'
        #get first list of tasks
        r = requests.get(self.cvat+endpoint, cookies=self.cookies)
        if(r.status_code == 200):
            #tasks is a paginated list of tasks
            task_list = r.json()
            for task in task_list['results']:
                self.tasks[task['name']] = task
            #get all other lists of tasks
            while(task_list['next'] is not None):
                #get all pages, build the list
                r = requests.get(task_list['next'], cookies=self.cookies)
                task_list = r.json()
                if(r.status_code == 200):
                    for task in task_list['results']:
                        self.tasks[task['name']] = task
                else:
                    print(r.reason)                    
        else:
            print(r.reason)


    def get_annotations_for_task(self, task_name):
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/annotations'
        r = requests.get(self.cvat+endpoint, cookies=self.cookies, headers=self.headers)
        if(r.status_code == 200):
            resp = r.json()
        else:
            print(r.reason)
            print('get annotations for task failed')
        return r.json()


    def patch_annotation(self, task_name=None, annotation=None, auto_meta=None, task_meta=None, auto_labels=None):
        auto_frame_num = annotation['frame']
        auto_frame_path = auto_meta['frames'][auto_frame_num]['name']
        task_frame_num = None
        i = 0
        for frame in task_meta['frames']:
            if(frame['name'] in auto_frame_path):
                task_frame_num = i
                break
            i += 1
        
        task_labels = self.tasks[task_name]['labels']
        label_id = None
        #cross reference on id back to auto annotation task
        #to find the name, then look it up in the task labels
        #to find corresponding label_id
        for label in auto_labels:
            if(annotation['label_id'] == label['id']):
                for task_label in task_labels:
                    if(label['name'] == task_label['name']):
                        label_id = task_label['id']
                        break
        assert(label_id != None), 'Unable to find correct label!'


        gt_annotations = self.get_annotations_for_task(task_name=task_name)
        #see if patchs exists with iou score with annotation to add
        gt_annotations_for_current_frame = [gt for gt in gt_annotations['shapes'] if gt['frame'] == task_frame_num]
        iou_scores = [calculate_iou(annotation['points'], gt['points']) for gt in gt_annotations_for_current_frame]
        #handle missing gt annotations case for frame 
        if(len(iou_scores) == 0):
            iou_scores = [0.0]
        
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/annotations?action=create'           
        
        body = {
            'version': 1, 
            'tags': [],
            'shapes': [{
                        'type': 'rectangle',
                        'occluded': False,
                        'points': annotation['points'],
                        'frame': str(task_frame_num),
                        'label_id': str(label_id),
                        'group': 0,
                        'source': 'manual',
                        'attributes': []
            }],
            'tracks': []

        }
        #do not upload if any annotation matches the new            
        if(np.max(iou_scores) < 1.0):
            r = requests.patch(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)            
            if(r.status_code == 200):                
                print('annotation uploaded')
            else:
                print(r.reason)
                print('annotation upload failed')
        else:
            print('annotation already existing, abort upload')


    def set_task_to_status_complete(self, task_name, job_status):
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/jobs'               

        r = requests.get(self.cvat+endpoint, cookies=self.cookies, headers=self.headers)
        job_info = None
        if(r.status_code == 200):                
            job_info = r.json()    
        else:
            print(r.reason)
            print('get job info failed for task: ' + task_name)
            return
                
        for job in job_info:
            endpoint = 'jobs/' + str(job['id'])
            
            body = {
                'id': job['id'],
                'status': job_status
            }
            
            r = requests.patch(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)
            if(r.status_code == 200):                
                print('status updated for task: ' + task_name)
            else:
                print(r.reason)
                print('status update failed for task: ' + task_name)


    def update_annotations(self, task_match_pattern=None, auto_annotation_match_pattern=None):
        update_tasks = list()
        auto_annotation_tasks = list()
        for task in self.tasks:
            if(task_match_pattern in task):
                update_tasks.append(task)
            if(auto_annotation_match_pattern in task):
                auto_annotation_tasks.append(task)
        
        auto_annotation_tasks_purged = [task for task in auto_annotation_tasks if self.tasks[task]['status'] == 'completed']

        for task in auto_annotation_tasks_purged:
            print('auto annotations from task: ' + task)
            annotations = self.get_annotations_for_task(task_name=task)
            meta = self.get_meta_for_task(task_name=task)
            auto_labels = self.tasks[task]['labels']
            #match update task, find correct meta info img path name for correct frame index to update
            matches = list()            
            split_str = 'FieldData'
            for match_task in update_tasks:
                if(match_task.split(split_str)[-1] ==  task.split(split_str)[-1]):                    
                    matches.append(match_task)            
            assert(len(matches) == 1), 'There can be only one! Found more than one task to update with auto annotation.'
            match = matches[0]
            print('update gt annotations for task: ' + match)

            for shape in annotations['shapes']:                
                match_meta = self.get_meta_for_task(task_name=match)                
                self.patch_annotation(task_name=match, annotation=shape, auto_meta=meta, task_meta=match_meta, auto_labels=auto_labels)


    def upload_data_from_xml(self, task_name=None, image_list=None):
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/data'
        file_list = []
        for img_path in image_list:
            file_list.append('/' + img_path['@name'])
        
        body = {'image_quality':100,
                'original_chunk_type': 'imageset',
                'compressed_chunk_type': 'imageset',
                'server_files': file_list
                }

        r = requests.post(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)

        if(r.status_code == 202):
            print('data upload started')
            sleep_timer = 0
            #potentially long upload times with large amouts of data
            max_sleep = self.upload_timeout_sec
            while(True):
                print('waiting for upload to finish')
                time.sleep(1)
                r = self.get_status(task_name=task_name)
                if(r.status_code == 200):                    
                    resp = r.json()
                    if(resp['state'] == 'Failed'): 
                        print('upload data failed')
                        break
                    if(resp['state'] == 'Finished'):
                        print('done uploading data')
                        break
                    sleep_timer += 1
                    if(sleep_timer > max_sleep):
                        print('give up, taking too long...')
                        break
        else:
            print(r.reason)
            print('data upload failed')


    def upload_data(self, folder=None, task_name=None, original_task_name=None, annotation_pkl=None):
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/data'
        #use "server files" since the storage server is mapped to cvat  
        #file_list = glob.glob(folder + '/**/*.png', recursive=True)
        cvat_root = '/fielddata' + folder    
        annotations = annotation_pkl[original_task_name]
        annotations.dropna(inplace=True)
        file_list = [entry['img_path'] for entry in annotations]
        file_list_unique = np.unique(file_list)
        full_server_paths = [cvat_root + img_path for img_path in file_list_unique]
        upload_list = list()
        for row in full_server_paths:
            upload_list.append({
                'file': row
            })
        body = {'image_quality':100,
                'original_chunk_type': 'imageset',
                'compressed_chunk_type': 'imageset',
                'server_files': full_server_paths
                }
        r = requests.post(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)
        if(r.status_code == 202):
            print('data upload started')
            sleep_timer = 0
            #potentially long upload times with large amouts of data
            max_sleep = self.upload_timeout_sec
            while(True):
                print('waiting for upload to finish')
                time.sleep(1)
                r = self.get_status(task_name=task_name)
                if(r.status_code == 200):                    
                    resp = r.json()
                    if(resp['state'] == 'Failed'): 
                        print('upload data failed')
                        break
                    if(resp['state'] == 'Finished'):
                        print('done uploading data')
                        break
                    sleep_timer += 1
                    if(sleep_timer > max_sleep):
                        print('give up, taking too long...')
                        break
        else:
            print(r.reason)
            print('data upload failed')


    def upload_data_gt_format(self, task_name=None, original_task_name=None, annotation_dataframe=None):
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/data'
        #use "server files" since the storage server is mapped to cvat      
        #file_list = glob.glob(folder + '/**/*.png', recursive=True)
        #cvat_root = '/fielddata' + folder
        if(original_task_name is not None):
            annotations = annotation_dataframe[original_task_name]            
        else:
            annotations = annotation_dataframe

        annotations.dropna(inplace=True)
        file_list = []        
        if(original_task_name is not None):
            #this is a series so .iterrows do not work.
            for i in range(len(annotations.index)):
                for annotation in annotations[annotations.index[i]]:
                    file_list.append(annotation['img_path'])
        else:
            for row in annotations[task_name]:
                file_list.append(row['img_path'])

        file_list_unique = np.unique(file_list)
        #/fielddata will be the new root since the server has this mapped, /weed_data is a mapped
        #folder in the container corresponding to the host folder that contains the fielddata folder
        full_server_paths = [img_path.replace('/weed_data/', '/') for img_path in file_list_unique]

        body = {'image_quality':100,
                'original_chunk_type': 'imageset',
                'compressed_chunk_type': 'imageset',
                'server_files': full_server_paths
                }
        r = requests.post(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)
        if(r.status_code == 202):
            print('data upload started')
            sleep_timer = 0
            #potentially long upload times with large amouts of data
            max_sleep = self.upload_timeout_sec
            while(True):
                time.sleep(1)
                print('waiting for upload to finish')
                r = self.get_status(task_name=task_name)
                if(r.status_code == 200):                    
                    resp = r.json()
                    if(resp['state'] == 'Failed'): 
                        print('upload data failed')
                        break
                    if(resp['state'] == 'Finished'):
                        print('done uploading data')
                        break
                    sleep_timer += 1
                    if(sleep_timer > max_sleep):
                        print('give up, taking too long...')
                        break
        else:
            print(r.reason)
            print('data upload failed')


    def get_status(self, task_name=None):
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/status'        
        r = requests.get(self.cvat+endpoint, cookies=self.cookies, headers=self.headers)
        if(r.status_code == 200):
            resp = r.json()
            print(resp)
        else:
            print(r.reason)
            print('task status failed')
        return r
        

    def get_meta_for_task(self, task_name=None):
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/data/meta'
        r = requests.get(self.cvat+endpoint, cookies=self.cookies, headers=self.headers)
        if(r.status_code == 200):
            return r.json()
            
        else:
            print(r.reason)
            print('get task mest info failed')
            return r.json()

    def get_meta_index_of_path(self, meta, img_path):
        i=0
        for entry in meta['frames']:
            if(img_path in entry['name']):
                return i
            else:
                i+=1
        return -1
        

    def upload_annotations(self, task_name=None, original_task_name=None, annotation_pkl=None):
        endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/annotations?action=create'
        meta = self.get_meta_for_task(task_name=task_name)        
        annotations_one_task = annotation_pkl[original_task_name]
        annotations_one_task.dropna(inplace=True)
        annotations = [entry for entry in annotations_one_task]
        
        task_labels = self.tasks[task_name]['labels']
        db_labels = self.class_map
        for annotation in annotations:
            #get meta info on frame name to number conversion
            frame_num = self.get_meta_index_of_path(meta=meta, img_path=annotation['img_path'])
            if(frame_num < 0):
                print('could not find img path in meta, failing task ' + task_name +' on frame ' \
                    ' ' + annotation['img_path'])
                continue

            label_id = []
            for annotation_label in annotation['labels']:
                for label in task_labels:                
                    if(db_labels[annotation_label] == label['name']):
                        label_id.append(label['id'])
            if('bboxes' in annotation.keys()):
                bboxes = []
                for box in annotation['bboxes']:
                    bboxes.append([str(box[0]), str(box[1]), str(box[2]), str(box[3])])                
                for i in range(len(label_id)):
                    body = {
                        'version': 1, 
                        'tags': [],
                        'shapes': [{
                                    'type': 'rectangle',
                                    'occluded': False,
                                    'points': bboxes[i],
                                    'frame': str(frame_num),
                                    'label_id': label_id[i],
                                    'group': 0,
                                    'source': 'automatic',
                                    'attributes': []
                        }],
                        'tracks': []
                    }

                    r = requests.patch(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)            
                    if(r.status_code == 200):                
                        print('annotation uploaded')
                    else:
                        print(r.reason)
                        print('annotation upload failed')                

            if(annotation['shape_types'] == 'polygon'):
                points = []
                for pointlist in annotation['points']:                    
                    points.append([str(point) for point in pointlist])                
                for i in range(len(label_id)):
                    body = {
                        'version': 1, 
                        'tags': [],
                        'shapes': [{
                                    'type': 'polygon',
                                    'occluded': False,
                                    'points': points[i],
                                    'frame': str(frame_num),
                                    'label_id': label_id[i],
                                    'group': 0,
                                    'source': 'automatic',
                                    'attributes': []
                        }],
                        'tracks': []
                    }

                    r = requests.patch(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)            
                    if(r.status_code == 200):                
                        print('annotation uploaded')
                    else:
                        print(r.reason)
                        print('annotation upload failed')                

        print('annotation upload finished')

    def upload_annotations_xml_format(self, task_name=None, image_list=None):
            endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/annotations?action=create'
            meta = self.get_meta_for_task(task_name=task_name)                                

            task_labels = self.tasks[task_name]['labels']        
            for annotation in image_list:
                #get meta info on frame name to number conversion
                frame_num = self.get_meta_index_of_path(meta=meta, img_path=annotation['@name'])
                if(frame_num < 0):
                    print('could not find img path in meta, failing task ' + task_name +' on frame ' \
                        ' ' + annotation['img_path'])
                    continue

                label_id = []
                #box is dict if only one
                if(isinstance(annotation['box'], dict)):
                    annotation['box'] = [annotation['box']]
                for annotation_label in annotation['box']:
                    for label in task_labels:                
                        if(annotation_label['@label'] == label['name']):
                            label_id.append(label['id'])

                bboxes = []
                for box in annotation['box']:
                    bboxes.append([str(box['@xtl']), str(box['@ytl']), str(box['@xbr']), str(box['@ybr'])])                
                for i in range(len(label_id)):
                    body = {
                        'version': 1, 
                        'tags': [],
                        'shapes': [{
                                    'type': 'rectangle',
                                    'occluded': False,
                                    'points': bboxes[i],
                                    'frame': str(frame_num),
                                    'label_id': label_id[i],
                                    'group': 0,
                                    'source': 'automatic',
                                    'attributes': []
                        }],
                        'tracks': []

                    }
                    

                    r = requests.patch(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)            
                    if(r.status_code == 200):                
                        print('annotation uploaded')
                    else:
                        print(r.reason)
                        print('annotation upload failed') 

            print('annotation upload finished')



    def upload_annotations_gt_format(self, task_name=None, original_task_name=None, annotation_dataframe=None, class_map=None):
            endpoint = 'tasks/' + str(self.tasks[task_name]['id']) + '/annotations?action=create'
            meta = self.get_meta_for_task(task_name=task_name)
            if(original_task_name is not None):
                annotations_one_task = annotation_dataframe[original_task_name]
            else:
                annotations_one_task = annotation_dataframe[task_name]
            annotations_one_task.dropna(inplace=True)
            annotations = []
            #[entry for entry in annotations_one_task]
            #this is a series so .iterrows do not work.
            if(original_task_name is not None):
                for i in range(len(annotations_one_task.index)):
                    for annotation in annotations_one_task[annotations_one_task.index[i]]:
                        annotations.append(annotation)
            else:
                annotations = annotations_one_task
            
            task_labels = self.tasks[task_name]['labels']
            if(class_map is None):     
                db_labels = self.class_map
            else:
                db_labels = class_map

            for annotation in annotations:
                #get meta info on frame name to number conversion
                frame_num = self.get_meta_index_of_path(meta=meta, img_path=annotation['img_path'].split('/weed_data/')[-1])
                if(frame_num < 0):
                    print('could not find img path in meta, failing task ' + task_name +' on frame ' \
                        ' ' + annotation['img_path'])
                    continue

                label_id = []
                for annotation_label in annotation['labels']:
                    for label in task_labels:                
                        if(db_labels[annotation_label] == label['name']):
                            label_id.append(label['id'])

                if('bboxes' in annotation.keys()):
                    bboxes = []
                    for box in annotation['bboxes']:
                        bboxes.append([str(box[0]), str(box[1]), str(box[2]), str(box[3])])                
                    for i in range(len(label_id)):
                        body = {
                            'version': 1, 
                            'tags': [],
                            'shapes': [{
                                        'type': 'rectangle',
                                        'occluded': False,
                                        'points': bboxes[i],
                                        'frame': str(frame_num),
                                        'label_id': label_id[i],
                                        'group': 0,
                                        'source': 'automatic',
                                        'attributes': []
                            }],
                            'tracks': []
                        }

                        r = requests.patch(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)            
                        if(r.status_code == 200):                
                            print('annotation uploaded')
                        else:
                            print(r.reason)
                            print('annotation upload failed')                

                if('shape_types' in annotation.keys()):
                    if(annotation['shape_types'] == 'polygon'):                    
                        for i in range(len(label_id)):
                            body = {
                                'version': 1, 
                                'tags': [],
                                'shapes': [{
                                            'type': 'polygon',
                                            'occluded': False,
                                            'points': [str(point) for point in annotation['points'][i]],
                                            'frame': str(frame_num),
                                            'label_id': label_id[i],
                                            'group': 0,
                                            'source': 'automatic',
                                            'attributes': []
                                }],
                                'tracks': []
                            }

                            r = requests.patch(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)            
                            if(r.status_code == 200):                
                                print('annotation uploaded')
                            else:
                                print(r.reason)
                                print('annotation upload failed')
                    
                    if(annotation['shape_types'] == 'rectangle'):                    
                        for i in range(len(label_id)):
                            body = {
                                'version': 1, 
                                'tags': [],
                                'shapes': [{
                                            'type': 'rectangle',
                                            'occluded': False,
                                            'points': [str(point) for point in annotation['points'][i]],
                                            'frame': str(frame_num),
                                            'label_id': label_id[i],
                                            'group': 0,
                                            'source': 'automatic',
                                            'attributes': []
                                }],
                                'tracks': []
                            }

                            r = requests.patch(self.cvat+endpoint, json=body, cookies=self.cookies, headers=self.headers)            
                            if(r.status_code == 200):                
                                print('annotation uploaded')
                            else:
                                print(r.reason)
                                print('annotation upload failed')                

            print('annotation upload finished')


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


def calculate_iou(target_box, pred_box):
    box_1=((target_box[0], target_box[1]),
           (target_box[2], target_box[1]), 
           (target_box[2], target_box[3]),
           (target_box[0], target_box[3]))
    box_2=((pred_box[0], pred_box[1]),
           (pred_box[2], pred_box[1]),
           (pred_box[2], pred_box[3]),
           (pred_box[0], pred_box[3]))
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def generate_auto_annotation_from_folder(folder_path=None, threshold=0.5, iou_threshold=0.7, 
                                         model_path='/train/reset18_model.pth', img_file_ext='png', 
                                         settings=None):
    #construct pkl with paths and other info for dataclass? Then feed through normal flow and use
    #auto annotation function?        
    height_crop = 0.5
    side_crop = 0.25
    
    t = datetime.now()
    datestr = t.strftime("%Y_%m_%d_%H_%M_%S")

    class_map = settings['auto_annotation_class_map']
    auto_annotation_folder = '/auto_annotation/pre_annotation_pkl/'
    os.makedirs(name=auto_annotation_folder, mode=0o755, exist_ok=True)

    img_list = glob.glob(folder_path + '/*.' + img_file_ext)    
    task_name = 'auto_annotation_' + datestr
    df = pandas.DataFrame()
    for img_path in img_list:        
        data_entry = {"bboxes": [], "labels": [], "img_path": img_path, 
                      "height_crop": height_crop, "side_crop": side_crop,
                      "task_name": task_name}
        df = df.append(data_entry, ignore_index=True)
    
    pickle_path = auto_annotation_folder + datestr + '.pkl'
    df.to_pickle(pickle_path)
    print('run auto annotation from folder...') 
    datestr = generate_auto_annotations(pickle_file=pickle_path, threshold=threshold, 
                                        iou_auto_annotation_limit=iou_threshold, variant=None, model_path=model_path,
                                        class_map=class_map, save_img=False)
    print('auto annotation done, upload results to cvat from folder: ' + datestr)    
    upload_auto_annotation_task(folder=datestr, task_name=task_name, class_map=class_map)
    


def generate_auto_annotations(pickle_file='/train/pickled_weed/pd_val.pkl', threshold=0.5, 
                              iou_auto_annotation_limit=0.1, variant='resnet18', model_path=None,
                              class_map=None, save_img=True):
    t = datetime.now()
    datestr = t.strftime("%Y_%m_%d_%H_%M_%S")
    img_folder_path = '/auto_annotation/imgs'
    auto_annotation_folder = '/auto_annotation/'+datestr
    os.makedirs(name=auto_annotation_folder, mode=0o755, exist_ok=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #imagenet   
    data_transform = transforms.Compose([transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
    
    if(model_path is not None):
        model = torch.load(model_path)
    else:
        model = torch.load('/train/' + variant + '_model.pth')
    
    model.to(device)        
    model.eval()
    if(class_map is None):
        weeds = Weeds(port=int(os.environ['MONGODB_PORT']))
        #verifiy that order of this array do not change! Must save a class_map json with the model, db can evolve!
        class_map = weeds.get_object_classes_for_annotations_with_task_filter(filter='gt_train')    

    print(class_map)
    dataset_test = WeedDataOD(pandas_file=pickle_file, device=device, transform=data_transform, class_map=class_map)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=1, collate_fn=my_collate, drop_last=True)
    
    df = pandas.DataFrame()    
    for i, data in enumerate(data_loader_test, 0): 
        print(i)             
        inputs, targets, entry = data       
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
            preds_boxes_np.append(pred['boxes'].detach().cpu().numpy())
            preds_labels_np.append(pred['labels'].detach().cpu().numpy())
            preds_scores_np.append(pred['scores'].detach().cpu().numpy())

        targets_boxes_np = []
        targets_labels_np = []
        for target in targets:
            targets_boxes_np.append(target['boxes'].detach().cpu().numpy())
            targets_labels_np.append(target['labels'].detach().cpu().numpy())

        #match preds and targets            
        for z in range(len(entry)):
            data_entry = {}
            img_path =  entry[z]['img_path']               
            task_name = entry[z]['task_name']                    
            img_upper_path = img_path.split('/')[0:-1]
            img_folder_full_path = img_folder_path + '/' + '/'.join(img_upper_path)
            os.makedirs(name=img_folder_full_path, mode=0o755, exist_ok=True)
            img = dataset_test.load_img_with_path(i, img_path)                            
                            
            #check if we can find predictions not covered by targets
            #if so add img, and predictions for uploading
            add_annotation = False
            for j in range(len(preds_scores_np[z])):                
                if(preds_scores_np[z][j] > threshold):
                    add_pred = True
                    #check if targets are in preds, if so do not add them
                    for k in range(len(targets_labels_np[z])):
                        iou = calculate_iou(target_box=targets_boxes_np[z][k],
                                            pred_box=preds_boxes_np[z][j])
                        if(iou > iou_auto_annotation_limit):
                            #match metric is difficult to decide, can be small 
                            # overlap of two boxes
                            add_pred = False
                            break
                    if(add_pred):                            
                        if(task_name not in data_entry):
                            data_entry[task_name] = {
                                "bboxes": [],
                                "labels": [],
                                "img_path": img_path
                            }                            
                        data_entry[task_name]['bboxes'].append(preds_boxes_np[z][j])
                        data_entry[task_name]['labels'].append(preds_labels_np[z][j])
                        add_annotation = True
                        
                        #move img to upload folder                       
                        img_full_path = img_folder_path + '/' + img_path
                        #only save once:
                        if(save_img and not os.path.exists(img_full_path)):
                            img.save(img_full_path)

            #save annotations, if any, per image
            if(add_annotation):                
                df = df.append(data_entry, ignore_index=True)
    #now we have an upload folder and a pkl file of annotations to add,    
    df.to_pickle(auto_annotation_folder + '/' + 'auto_annotation.pkl')
    return auto_annotation_folder


def create_dataframe_with_task_as_keys(pickle_file=None, class_map=None):
    df = pandas.DataFrame()
    tasks = pickle_file.task_name.unique()
    for task in tasks:
        task_df = pickle_file[pickle_file['task_name'] == task]
        #task_df.dropna(inplace=True)        
        data_entry = {task: list()}
        int_labels = []
        for index, entry in task_df.iterrows():
            frame_entry = dict()
            #from mongodb:
            if('bboxes' in entry.keys()):
                frame_entry['points'] = entry['bboxes']
                frame_entry['shape_types'] = 'rectangle'
                #for compliance with auto annotation labels as int
                int_labels = [class_map.index(entry['labels'][i]) for i in range(len(entry['labels']))]    
            else:
                #pickle from cvat:
                if(entry['shape_types'] == 'rectange'):
                    frame_entry['points'] = entry['bboxes']
                    frame_entry['shape_types'] = 'rectangle'

            if(entry['shape_types'] == 'polygon'):
                frame_entry['points'] = entry['points']
                frame_entry['shape_types'] = 'polygon'

            if(len(int_labels) == 0):
                if(isinstance(entry['object_class'], list)):
                    int_labels = [class_map.index(entry['object_class'][i]) for i in range(len(entry['object_class']))]
                else:
                    int_labels.append(class_map.index(entry['object_class']))

            img_path = entry['img_path']
                        
            frame_entry['labels'] = int_labels
            frame_entry['img_path'] = img_path
            data_entry[task].append(frame_entry)
        df = df.append(data_entry, ignore_index=True)        
    
    return df


def set_all_cvat_tasks_to_complete():    
    uploader = AutoAnnotations(username=os.environ['CVAT_USERNAME'], password=os.environ['CVAT_PASSWORD'], 
                                    cvat_base_url=os.environ['CVAT_BASE_URL'])
    uploader.get_tasks()
    
    for task in uploader.tasks.keys():
        uploader.set_task_to_status_complete(task_name=task, job_status='completed')
    

def do_whatever():
    #placeholder for temporary stuff 
    uploader = AutoAnnotations(username=os.environ['CVAT_USERNAME'], password=os.environ['CVAT_PASSWORD'], 
                                    cvat_base_url=os.environ['CVAT_BASE_URL'])
    uploader.get_tasks()
    #implement temp function to set a list of tasks in a state
    settings = None        
    with open('/code/settings_file_gt_train_val.json') as json_file:            
        settings = json.load(json_file)
    
    for task in settings['first_year_tasks']:
        uploader.set_task_to_status_complete(task_name=task, job_status='completed')
    

def update_gt_with_auto_annotations(gt_match=None, auto_annotation_date=None):
    uploader = AutoAnnotations(username=os.environ['CVAT_USERNAME'], password=os.environ['CVAT_PASSWORD'], 
                                    cvat_base_url=os.environ['CVAT_BASE_URL'])
    uploader.get_tasks()    
    uploader.update_annotations(task_match_pattern=gt_match, auto_annotation_match_pattern=auto_annotation_date)
    

def validate_img_paths(dataframe=None, folder=None):
    #use frame based dataframe
    for i in range(len(dataframe.index)):
        entry = dataframe.iloc[i]
        side_crop = entry['side_crop']
        height_crop =entry['height_crop']
        img_full_path = folder + entry['img_path']
        if(not os.path.exists(img_full_path)):
            img = Image.open(entry['img_path']).convert("RGB")
            #(left, upper, right, lower)
            crop_region = (side_crop*img.width, height_crop*img.height, img.width-img.width*side_crop, img.height)
            crop_img = img.crop(crop_region)            
            crop_img.save(img_full_path)


def upload_ground_truths(pickle_file='/train/pickled_weed/pd_val.pkl', class_map=None):
    uploader = AutoAnnotations(username=os.environ['CVAT_USERNAME'], password=os.environ['CVAT_PASSWORD'], 
                                    cvat_base_url=os.environ['CVAT_BASE_URL'])
    if(class_map is None):
        class_map = uploader.get_class_map()

    gt_pkl = pandas.read_pickle(pickle_file)
    task_based_gt_dataframe = create_dataframe_with_task_as_keys(pickle_file=gt_pkl, class_map=class_map)
    
    for sub_task in task_based_gt_dataframe:        
        cvat_task_name = sub_task.replace(' ', '_')
        uploader.get_tasks()
        tasks = uploader.get_internal_task_list()
        if(cvat_task_name in tasks.keys()):
            #print('task already in cvat, check upload list and cvat')          
            #else:
            #uploader.create_task(task_name=cvat_task_name)
            uploader.get_tasks()
            #uploader.upload_data_gt_format(task_name=cvat_task_name, original_task_name=sub_task, annotation_dataframe=task_based_gt_dataframe)
            uploader.upload_annotations_gt_format(task_name=cvat_task_name, original_task_name=sub_task, 
                                                    annotation_dataframe=task_based_gt_dataframe, class_map=class_map)


def upload_annotations_xml(folder=None):        
    uploader = AutoAnnotations(username=os.environ['CVAT_USERNAME'], password=os.environ['CVAT_PASSWORD'],
                                cvat_base_url=os.environ['CVAT_BASE_URL'])

    xml_files = glob.glob(folder + "/*.xml")
    uploader.get_tasks()
    for xml_file_path in xml_files:
        with open(xml_file_path) as xml_file:     
            data_dict = xmltodict.parse(xml_file.read())
            xml_file.close()            
            json_str = json.dumps(data_dict)
            json_data = json.loads(json_str)
            meta = json_data['annotations']['meta']
            image_list = json_data['annotations']['image']
            task_name = meta['task']['name'].replace(' ', '_')
            print('start create task: ' + task_name)
            task_list = uploader.get_internal_task_list()            
            if(task_name in task_list.keys()):
                print('skipping, task already uploaded...')
            else:
                task_labels = meta['task']['labels']['label']
                for label in task_labels:
                    label['attributes'] = []                            
                uploader.create_task(task_name=task_name, labels=task_labels)
                uploader.get_tasks()
                uploader.upload_data_from_xml(task_name=task_name, image_list=image_list)
                uploader.upload_annotations_xml_format(task_name=task_name, image_list=image_list)
            

def upload_auto_annotation_task(folder=None, task_name=None, class_map=None):
    uploader = AutoAnnotations(username=os.environ['CVAT_USERNAME'], password=os.environ['CVAT_PASSWORD'], 
                                    cvat_base_url=os.environ['CVAT_BASE_URL'])

    annotation_dataframe = pandas.read_pickle(folder+'/auto_annotation.pkl')
    uploader.create_task(task_name=task_name)            
    uploader.get_tasks()
    uploader.upload_data_gt_format(task_name=task_name, original_task_name=None, annotation_dataframe=annotation_dataframe)
    uploader.upload_annotations_gt_format(task_name=task_name, original_task_name=None, annotation_dataframe=annotation_dataframe, class_map=class_map)

def upload_from_folder(folder=None, task_name=None):
    uploader = AutoAnnotations(username=os.environ['CVAT_USERNAME'], password=os.environ['CVAT_PASSWORD'], 
                                cvat_base_url=os.environ['CVAT_BASE_URL'])

    #break up task_creation and uploading of data and annotions per sub task_name
    #since it is problematic to upload all in one go.
    annotation_pkl = pandas.read_pickle(folder+'/auto_annotation.pkl')
    img_root = '/auto_annotation/imgs'
    for sub_task in annotation_pkl.keys():
        full_sub_task_name = task_name + '_' + sub_task.replace(' ', '_')
        uploader.create_task(task_name=full_sub_task_name)            
        uploader.get_tasks()
        uploader.upload_data(folder=img_root, task_name=full_sub_task_name, original_task_name=sub_task, annotation_pkl=annotation_pkl)
        uploader.upload_annotations(task_name=full_sub_task_name, original_task_name=sub_task, annotation_pkl=annotation_pkl)


def crop_all(src=None, dst=None, ext=None):
    #use glob to find image files with full path
    #crop and save with new root folder
    
    height_crop = 0.5
    side_crop = 0.25

    file_list = glob.glob(src + '/**/*' + ext, recursive=True)

    print('start cropping and saving')
    for file_path in file_list:
        landing_path = dst + file_path.split(src)[-1]
        if(not os.path.exists(landing_path)):
                img = Image.open(file_path).convert("RGB")
                #(left, upper, right, lower)
                crop_region = (side_crop*img.width, height_crop*img.height, img.width-img.width*side_crop, img.height)
                crop_img = img.crop(crop_region)
                tmp = landing_path.split('/')[0:-1]                
                folder_path = '/'.join(tmp)
                os.makedirs(folder_path, exist_ok=True)            
                crop_img.save(landing_path)
    
    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto annotation with pickle or movie path, run on servers with mapped storage server')
    parser.add_argument('-p', '--pickle_file', type=str, default='/train/pickled_weed/pd_val.pkl', help='pickle file for auto annotation', required=False)
    parser.add_argument('-f', '--folder_path', type=str, help='folder path to image folder', required=False)    
    parser.add_argument('-v', '--variant', help='model variant to run', required=False)
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='confidence threshold', required=False)
    parser.add_argument('-i', '--iou_threshold', type=float, default=0.1, help='iou auto annotation threshold', required=False)
    parser.add_argument('-u', '--update_annotations', type=bool, default=False, help='flag to update annotations in cvat', required=False)    
    parser.add_argument('--upload_pkl', type=bool, help='gt_folder_of_cropped_images', required=False)
    parser.add_argument('-r', '--rerun_task', help='task_name to rerun auto annotation', required=False)
    parser.add_argument('-m', '--match_pattern_gt', help='task_name match gt to auto annotation', required=False)
    parser.add_argument('-d', '--auto_annotation_folder', help='date based folder name of auto annotation', required=False)
    parser.add_argument('-w', '--whatever', type=bool, default=False, help='flag to enable an entrance to any function you like', required=False)
    parser.add_argument('-s', '--set_complete', type=bool, default=False, help='set all tasks in cvat to complete status', required=False)
    parser.add_argument('-x', '--xml_upload', type=str, help='upload xml annotations to CVAT from provided folder', required=False)    
    parser.add_argument('-c', '--crop_images', type=bool, help='flag to initiate images cropping', required=False)
    parser.add_argument('--ext', type=str, default='png', help='image file extension', required=False)
    parser.add_argument('--src', type=str, help='source folder to search for images', required=False)
    parser.add_argument('--dst', type=str, help='destination root folder for cropped images, paths will be preserved', required=False)
    parser.add_argument('--model_path', type=str, help='model path for auto annotation', required=False)
    parser.add_argument('--settings_file', type=str, default='/code/settings_file_gt_train_val.json', help='settings_file', required=False)
    parser.add_argument('--class_map', type=str, nargs='+', default='', help='class_map', required=False)

    args = parser.parse_args()

    settings = None
    with open(args.settings_file) as json_file:            
            settings = json.load(json_file)
    
    if(args.folder_path):        
        generate_auto_annotation_from_folder(folder_path=args.folder_path, threshold=args.threshold, iou_threshold=args.iou_threshold, 
                                             model_path=args.model_path, img_file_ext=args.ext, settings=settings)
    elif(args.crop_images):
        crop_all(src=args.src, dst=args.dst, ext=args.ext)
    elif(args.update_annotations):
        #update ground truth annotations with verified auto annotations
        #via rest api, download and then upload annotations if newer date on auto annotations        
        update_gt_with_auto_annotations(gt_match=args.match_pattern_gt, auto_annotation_date=args.auto_annotation_folder)
    elif(args.upload_pkl):                
        upload_ground_truths(pickle_file=args.pickle_file, class_map=settings['default_class_map'])
    elif(args.whatever):
        do_whatever()
    elif(args.set_complete):
        set_all_cvat_tasks_to_complete()
    elif(args.xml_upload):
        upload_annotations_xml(folder=args.xml_upload)
    elif(args.rerun_task):
        task_name = args.rerun_task
        #idea: make new pickle with only the specifik task and reuse code
        full_df = pandas.read_pickle(args.pickle_file)
        #filter on task name and make a new dataframe, save to new pickle
        #weeds = Weeds(port=int(os.environ['MONGODB_PORT']))
        #class_map = weeds.get_object_classes_for_annotations_with_task_filter(filter='FieldData')      
        #task_df = create_dataframe_with_task_as_keys(full_df, class_map)                
        task_df_one_task = full_df[full_df['task_name'] == task_name]
        task_df_one_task.dropna(inplace=True)

        #override
        threshold = 0.5
        if(args.threshold):
            threshold = args.threshold
        iou_threshold = 0.1
        if(args.iou_threshold):
            iou_threshold = args.iou_threshold
        variant='resnet18'
        if(args.variant):
            variant = args.variant
        
        #make folder and save pickle
        save_dir = '/auto_annotation/rerun_one_task/' + task_name
        os.makedirs(name=save_dir, mode=0o755, exist_ok=True)
        one_task_pkl = save_dir+ 'one_task.pkl'
        task_df_one_task.to_pickle(one_task_pkl)        

        datestr = generate_auto_annotations(pickle_file=one_task_pkl, threshold=threshold, 
                                             iou_auto_annotation_limit=iou_threshold, variant=variant)
        
        upload_from_folder(folder=save_dir + variant + '/' + datestr, 
                            task_name='auto_annotation_' + variant + '_' + datestr)
    else:
        print('run auto annotation...') 
        datestr = generate_auto_annotations(pickle_file=args.pickle_file, threshold=args.threshold, 
                                             iou_auto_annotation_limit=args.iou_threshold, variant=args.variant)
        print('auto annotation done, upload results to cvat from folder: auto_annotation/' + datestr)
        upload_from_folder(folder='/auto_annotation/' + datestr, 
                            task_name='auto_annotation_' + args.variant + '_' + datestr)
