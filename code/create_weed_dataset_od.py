from readline import replace_history_item
from weeds import Weeds
import pandas
import os
import numpy as np
from paramiko import SSHClient
from scp import SCPClient
import shutil


class PickledWeedOD():
    def __init__(self, task_name_list, save_dir, dataset_dir, mongo_port):
        self.task_name_list = task_name_list
        self.save_dir = save_dir
        self.dataset_dir = dataset_dir
        os.makedirs(name=self.save_dir, mode=0o755, exist_ok=True)
        self.weeds = Weeds(port=int(mongo_port))
        self.class_map = self.weeds.get_object_classes_for_all_annotations()
        self.height_crop = 0.5
        self.side_crop = 0.25


    def get_class_map(self, filter=None):
        if(filter is not None):
            self.class_map = self.weeds.get_object_classes_for_annotations_with_task_filter(filter=filter)        
        return self.class_map


    def purge_bboxes(self, entry, boxes, labels, shape_types):
        #sizes
        crop_height = np.int32(entry['img_height'] * self.height_crop)
        crop_width = np.int32(entry['img_width'] * self.side_crop)
        crop_stop = np.int32(entry['img_width'] - crop_width)

        #indexes
        crop_img = {
            'xmin': crop_width - 1,
            'xmax': crop_stop - 1,
            'ymin': crop_height - 1,
            'ymax': np.int32(entry['img_height'] - 1)
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


    def make_pandas_dataset_with_pre_split(self, full_hd=True, excluded_frames=[], train_pickle_path=None, val_pickle_path=None):
        #get annotations for each frame and build the pandas entry for od
        df_train = pandas.DataFrame()
        df_val = pandas.DataFrame()
        #pre split definition given in saved numpy files       
        if(full_hd):
            val_frames = np.load('/train/pickled_weed/val_frames_full_hd.npy', allow_pickle=True)
            train_frames = np.load('/train/pickled_weed/train_frames_full_hd.npy', allow_pickle=True)
        else:
            val_frames = np.load('/train/pickled_weed/val_frames_4k.npy', allow_pickle=True)
            train_frames = np.load('/train/pickled_weed/train_frames_4k.npy', allow_pickle=True)        

        #sanity check
        problems = list()
        for frame_path in val_frames:            
            if((len(np.argwhere(train_frames == frame_path)) > 0)):
                print(frame_path + ' found in train, should only be in val')
                problems.append(frame_path)        

        for task_name in self.task_name_list:
            print('extract annotations for: ' + task_name, flush=True)            
            #get frames with annotations from metadata
            meta_cursor = self.weeds.get_meta_for_task(task_name)
            meta_data = {}
            for index, item in enumerate(meta_cursor):
                meta_data[item['task_name']] = item
            
            for meta in meta_data[task_name]['frames']:
                if(meta['name'] in excluded_frames):
                    print('skipping problematic frame: ' + meta['name'])
                    continue

                annotations = []
                annotations_cursor = self.weeds.get_annotations_for_task_and_frame(task_name, meta['name'])
                for index, item in enumerate(annotations_cursor):
                    annotations.append(item)
                        
                dataEntry = {}      
                dataEntry['img_path'] = self.dataset_dir + '/' + meta['name']
                dataEntry['img_width'] = meta['width']
                dataEntry['img_height'] = meta['height']
                dataEntry['task_name'] = task_name 
                dataEntry['height_crop'] = self.height_crop
                dataEntry['side_crop'] = self.side_crop                
                
                pre_cropped_img = False
                if(meta['width'] == 1920 and meta['height'] == 1080):
                    pre_cropped_img = True
                
                bboxes = []
                labels = []
                shape_types = []
                points = []
                for annotation in annotations:                                                     
                    labels.append(annotation['object_class'])
                    shape_types.append(annotation['shape_types'])
                    num_points = len(annotation['points'])
                    points.append(annotation['points'])
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

                purged_bboxes, purged_labels, purged_shape_types = ([],[],[])                
                if(pre_cropped_img or not full_hd):
                    dataEntry['bboxes'] = bboxes
                    dataEntry['labels'] = labels
                    dataEntry['shape_types'] = shape_types
                    #ok since we only add points for upload via pkl
                    dataEntry['points'] = points
                else:
                    if(full_hd):
                        purged_bboxes, purged_labels, purged_shape_types = self.purge_bboxes(dataEntry, bboxes, labels, shape_types)
                        dataEntry['bboxes'] = purged_bboxes
                        dataEntry['labels'] = purged_labels
                        dataEntry['shape_types'] = purged_shape_types
                    else:
                        print('unhandled variant of annotations, please inspect!')
                        return

                if(len(dataEntry['labels']) > 0):
                    im_path = dataEntry['img_path']
                    validation_frame_match = None
                    #strip auto_annotation part from path string if we have a reference to the full hd cropped img
                    if('auto_annotation/imgs' in im_path):
                        #new format, common space for cropped frames
                        im_path_match = im_path.split('auto_annotation/imgs')[-1]
                        validation_frame_match = np.argwhere(val_frames == im_path_match)
                    else:
                        validation_frame_match = np.argwhere(val_frames == im_path)
                                        
                    #sneaky, df.append will return the new dataframe...
                    if(len(validation_frame_match) > 0):         
                        df_val = df_val.append(dataEntry, ignore_index=True)
                    else:                        
                        df_train = df_train.append(dataEntry, ignore_index=True)
        
        df_val.to_pickle(val_pickle_path)
        df_train.to_pickle(train_pickle_path)
        


    def make_pandas_dataset(self, pickle_name=None):                       
        #get annotations for each frame and build the pandas entry for od                
        df = pandas.DataFrame()
        for task_name in self.task_name_list:
            print('extract annotations for: ' + task_name)        
            #get frames with annotations from metadata
            meta_cursor = self.weeds.get_meta_for_task(task_name)
            meta_data = {}
            for index, item in enumerate(meta_cursor):
                meta_data[item['task_name']] = item
            
            for meta in meta_data[task_name]['frames']:
                annotations = []
                annotations_cursor = self.weeds.get_annotations_for_task_and_frame(task_name, meta['name'])
                for index, item in enumerate(annotations_cursor):
                    annotations.append(item)
                        
                dataEntry = {}      
                dataEntry['img_path'] = self.dataset_dir + '/' + meta['name']
                dataEntry['img_width'] = meta['width']
                dataEntry['img_height'] = meta['height']
                dataEntry['task_name'] = task_name 
                dataEntry['height_crop'] = self.height_crop
                dataEntry['side_crop'] = self.side_crop
                bboxes = []
                labels = []
                shape_types = []
                for annotation in annotations:                                                     
                    labels.append(annotation['object_class'])
                    shape_types.append(annotation['shape_types'])
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

                purged_bboxes, purged_labels, purged_shape_types = self.purge_bboxes(dataEntry, bboxes, labels, shape_types)
                dataEntry['bboxes'] = purged_bboxes
                dataEntry['labels'] = purged_labels
                dataEntry['shape_types'] = purged_shape_types

                #sneaky, df.append will return the new dataframe...
                if(len(purged_labels) > 0):    
                    df = df.append(dataEntry, ignore_index=True)
        if(pickle_name == None):
            df.to_pickle(self.save_dir + '/pd.pkl')
        else:
            df.to_pickle(self.save_dir + '/' + pickle_name)

    
    def make_simple_split(self, save_dir):
        #just to get started take every 10'th frame and its annotations for validation
        df = pandas.read_pickle(save_dir + '/pd.pkl')                
        df_val = df[df.index % 10 == 0]
        df_train = df[df.index % 10 != 0]
        df_val.to_pickle(self.save_dir + '/pd_val.pkl')
        df_train.to_pickle(self.save_dir + '/pd_train.pkl')
        print('simple strait 10/90 split done based on frames, no concern on how many annotations per frame we have!')


    def fetch_and_move(self, scp, image, bg=False):        
        server_base_path = '/fs/sefs1/obdb/'
        save_folder = None
        #remove docker img folder
        if(bg):
            img_path = server_base_path + '/' + image
            img_folder = image.split('/')[0:-1]
            save_folder = self.dataset_dir + '/' + '/'.join(img_folder)
            os.makedirs(name=save_folder, mode=0o755, exist_ok=True)
        else:            
            img_path = image.split('/')[2:]
            img_path = server_base_path + '/'.join(img_path)
            img_folder = image.split('/')[0:-1]
            save_folder = '/'.join(img_folder)
            os.makedirs(name=save_folder, mode=0o755, exist_ok=True)
        
        print('get: ' + img_path)
        scp.get(img_path, local_path=self.dataset_dir + '/' + image.split('/')[-1])

        
        #move to correct location in mapped container
        img_name = img_path.split('/')[-1]
        if(bg):
            shutil.move(self.dataset_dir + '/' + img_name, self.dataset_dir + '/' + image)            
        else:
            shutil.move(self.dataset_dir + '/' + img_name, image)


    #Idea: get files to local storage mimicking the fs
    #on training servers but get only the files needed
    def get_local_files(self):                
        #collect from pd_train/val since not all from pd are included
        df_train = pandas.read_pickle(self.save_dir + '/pd_train.pkl')        
        train_images = df_train.groupby('img_path').size()
        unique_images_train = [x for x in train_images.index]
        #val
        df_val = pandas.read_pickle(self.save_dir + '/pd_val.pkl')        
        val_images = df_val.groupby('img_path').size()
        unique_images_val = [x for x in val_images.index]


        with SSHClient() as ssh:
            ssh.load_system_host_keys(filename=os.environ['KNOWN_HOSTS_FILE_PATH'])
            #do not use password, use system with passwordless login over ssh
            ssh.connect(hostname=os.environ['SSH_HOST'], username=os.environ['SSH_USER'], port=os.environ['SSH_PORT'])

            with SCPClient(ssh.get_transport()) as scp:                                
                for image in unique_images_train:
                    self.fetch_and_move(scp, image)
                for image in unique_images_val:
                    self.fetch_and_move(scp, image)

#for isolated testing
def main():
    print('create pandas dataset of weeds')
    save_dir = '/pickled_weed'
    #annotations_list = ['FieldData 20200520145736 1L GH020068',
    #                    'FieldData 20200520145736 3R GH020072']    
    annotations_list = ['FieldData 20200520145736 1L GH020068']    
    pickledWeed = PickledWeedOD(task_name_list=annotations_list, save_dir=save_dir, dataset_dir='/weed_data', mongo_port=27018)
    pickledWeed.make_pandas_dataset()
    print('done, pickled pandas frame found at: ' + save_dir)
    pickledWeed.make_simple_split(save_dir)
    class_map = pickledWeed.get_class_map()
    print(class_map)
    


if __name__ == '__main__':
    main()
