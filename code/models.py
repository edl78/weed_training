import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

def get_model_with_args(model_name='resnet50', num_classes=3):
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


def get_retina_model_with_args(num_classes=3):
    from torchvision.models.detection.retinanet import RetinaNet

    backbone = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=True, trainable_backbone_layers=5)
    
    model = RetinaNet(backbone.backbone, num_classes=num_classes, 
                        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    return model


def get_model_weeds_pretrained(model_name='resnet18', num_classes=3, 
                                model_path='/train/models/YOUR_PRETRAINED_MODEL.pth',
                                pretrained_num_classes=11):
    #should get the same as below...
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    print('Using %s as backbone...' % model_name)
    imagenet_pretrained_backbone = True
    #false on pretained (imagenet), will be replaced with coco
    #backbone = resnet_fpn_backbone(model_name, pretrained=imagenet_pretrained_backbone, trainable_layers=5)
    pretrained_model = torch.load(model_path)
    print(model_path)
    backbone = pretrained_model.backbone
    state_dict_pretrained = pretrained_model.state_dict()
    
    #adapt to new format
    #state_dict_pretrained["rpn.head.conv.0.0.bias"] = state_dict_pretrained.pop("rpn.head.conv.bias")
    #state_dict_pretrained["rpn.head.conv.0.0.weight"] = state_dict_pretrained.pop("rpn.head.conv.weight")
    
    #anchor_generator = AnchorGenerator(
    #    sizes=((16,), (32,), (64,), (128,), (256,)),
    #    aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

    #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
    #                                                output_size=7, sampling_ratio=2)
    
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone, num_classes=pretrained_num_classes, 
                        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    #rpn_anchor_generator=anchor_generator,
    #                   box_roi_pool=roi_pooler
    #replace with dict with arc as keys
    
    #torch.save({'model_state_dict': model.state_dict(),}, '/train/a100_state_dict.pt')

    model.load_state_dict(state_dict_pretrained)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)    
    
    return model