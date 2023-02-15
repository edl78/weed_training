import time
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import vision.references.detection.transforms as VT
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import json

import numpy as np
from numpyencoder import NumpyEncoder

from weed_data_class_od import WeedDataOD


class ModelTrainer():
    def __init__(self, dataset_path=None, dataset_test_path=None,
                settings=None, variant=None, class_map=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.settings = settings
        self.variant = variant
        self.num_classes = len(class_map)
        self.class_map = class_map
        self.model = None
        self.best_loss = np.Infinity        
        self.stop_training = False
        self.plateau_cnt = 0
        self.optimal_settings = None            

        # Writer will output to ./runs/ directory for tensorboard by default
        #add path since Dockerfile ENTRYPOINT command will 
        folder = '/' + time.asctime().replace(' ', '_')
        self.writer = SummaryWriter(self.settings['writer_dir'] + folder)   
        
        self.transform = transforms.Compose([transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

        self.augmentation = VT.Compose([VT.RandomHorizontalFlip(0.5),
                                        VT.RandomPhotometricDistort()])
        
        #pickle file names by convention, change to parameter later
        self.dataset = WeedDataOD(pandas_file=dataset_path, device=self.device, transform=self.transform, 
                                    augmentation=self.augmentation, class_map=self.class_map)

        self.dataset_test = WeedDataOD(pandas_file=dataset_test_path, device=self.device, transform=self.transform, 
                                    augmentation=None, class_map=self.class_map)

        # define training and validation data loaders
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=self.my_collate, drop_last=True)
        
        self.data_loader_test = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=2, shuffle=False, num_workers=2, collate_fn=self.my_collate, drop_last=True)
       
        self.optimal_settings = self.settings[variant]['optimal_hpo_settings']


    def train_model(self, settings_file=None):
        if(self.variant == "retina_net"):
            self.model = self.get_retina_model_with_args(num_classes=self.num_classes)
        else:
            self.model = self.get_model_with_args(model_name=self.variant, num_classes=self.num_classes)

        # move model to the right device
        self.model.to(self.device)

        #override settings
        if(settings_file is not None):
            with open(settings_file) as json_file:
                print('override settings with: ' + settings_file)
                self.optimal_settings = json.load(json_file)

        #train again, longer with optimal settings!
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=self.optimal_settings['lr'],
                                momentum=self.optimal_settings['momentum'],
                                weight_decay=self.optimal_settings['weight_decay'])
        
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.optimal_settings['step_size'],
                                                    gamma=self.optimal_settings['gamma'])

        for epoch in range(self.settings['max_train_epochs']):
            self.train_run(epoch=epoch, optimizer=optimizer, variant=self.variant)
            self.eval_run(optimizer=optimizer, epoch=epoch)                
            if(self.stop_training == True):
                break
            if(lr_scheduler is not None):
                lr_scheduler.step()


    def train_run(self, epoch, optimizer, variant, trial_num=None):
        self.model.train()
        print('epoch: ' + str(epoch+1))
        torch.set_grad_enabled(True)
        running_loss = 0
        num_meas = int(len(self.data_loader))

        #do not update these, will destroy too much
        #https://keras.io/guides/transfer_learning/
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
               m.track_running_stats=False

        for i, data in enumerate(self.data_loader, 0):
            print(i)
            inputs, targets, img_path = data        
            targets = self.trim_targets(targets)        

            inputs_gpu = []
            for im in inputs:
                inputs_gpu.append(im.to(self.device))

            #inputs = inputs.to(device)            
            targets_gpu = self.move_targets_to_device(targets, self.device)            

            # zero the parameter gradients
            optimizer.zero_grad()

            #this model returns the losses as output during training!            
            loss_dict = self.model(inputs_gpu, targets_gpu)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
            
            losses.backward()
            optimizer.step()            

            if((i+1)%(num_meas)==0):
                running_loss /= num_meas
                self.writer.add_scalar(self.variant + '/training loss',
                                running_loss, epoch)
                print('train losses sum: ' + str(running_loss))
                running_loss = 0


    def eval_run(self, optimizer, epoch, study=None, trial=None, trial_num=None):
        #complicated with metrics for eval, so run in train
        #and look at loss with zero_grad to not effect weights
        #some layers (batch norm) may be different in eval but
        #hopefully not to much.
        running_loss_study = 0
        running_loss = 0 
        num_meas = int(len(self.data_loader_test))

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats=False
        
        for i, data in enumerate(self.data_loader_test, 0):
            optimizer.zero_grad()  
            inputs, targets, img_path = data                
            print(i)
            
            targets_gpu = self.move_targets_to_device(targets, self.device)
            inputs_gpu = []
            for im in inputs:
                inputs_gpu.append(im.to(self.device))                    

            loss_dict = self.model(inputs_gpu, targets_gpu)            
            losses = sum(loss for loss in loss_dict.values())
            
            running_loss += losses.item()
                        

            if((i+1)%(num_meas)==0):
                running_loss /= num_meas 
                print('validation loss: ', running_loss)
                self.writer.add_scalar(self.variant + '/validation loss',
                                running_loss, epoch)

                
                #only count plateau when in final training                
                self.plateau_cnt += 1
                print('plateau count: ' + str(self.plateau_cnt))
                #just save model after hyper parameter search and average loss over whole epoch
                #save model if loss is lowest so far
                if(running_loss < self.best_loss):
                    self.best_loss = running_loss
                    print('save best model with loss: ', running_loss)
                    self.model.train()
                    torch.save(self.model, '/train/' + self.variant + '_model.pth')
                    self.plateau_cnt = 0
            
                #stop?
                if(self.plateau_cnt == self.settings['max_plateau_count']):                    
                    self.stop_training = True
                    print('stopp training, no validation performance increase')
            
                running_loss = 0            


    def move_targets_to_device(self, targets, device):
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


    def trim_targets(self, targets):
        for i in range(len(targets)):
            #fix for extra dimension, when more than one bbox...
            if(len(targets[i]["boxes"].shape) == 1):
                new_targets = {}            
                new_targets["boxes"] = targets[i]["boxes"].resize_(1,4)
                new_targets["labels"] = targets[i]["labels"]
                targets[i] = new_targets
        return targets


    def my_collate(self, batch):
        # batch contains a list of tuples of structure (sequence, target)
        data = [item[0] for item in batch]    
        targets = [item[1] for item in batch]
        img_paths = [item[2] for item in batch]
        return [data, targets, img_paths]


    def get_model_with_args(self, model_name='resnet50', num_classes=3):
        #should get the same as below...
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

        print('Using %s as backbone...' % model_name)
        imagenet_pretrained_backbone = True
        #false on pretained (imagenet), will be replaced with coco
        backbone = resnet_fpn_backbone(model_name, pretrained=imagenet_pretrained_backbone, trainable_layers=5)

        #anchor_generator = AnchorGenerator(
        #    sizes=((16,), (32,), (64,), (128,), (256,)),
        #    aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

        #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
        #                                                output_size=7, sampling_ratio=2)
        if(imagenet_pretrained_backbone):
            pretrained_num_classes = 1000
        else:
            #coco
            pretrained_num_classes = 91

        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone, num_classes=pretrained_num_classes,
                            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

        #rpn_anchor_generator=anchor_generator,
        #                   box_roi_pool=roi_pooler
        #replace with dict with arc as keys
        if(not imagenet_pretrained_backbone):
            state_dict = load_state_dict_from_url(model_urls[model_name],
                                                    progress=True)
            model.load_state_dict(state_dict)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)    
        
        return model


    def get_retina_model_with_args(self, num_classes=3):
        from torchvision.models.detection.retinanet import RetinaNet

        backbone = torchvision.models.detection.retinanet_resnet50_fpn(
            pretrained=True, trainable_backbone_layers=5)
        
        model = RetinaNet(backbone.backbone, num_classes=num_classes,
                            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

        return model
