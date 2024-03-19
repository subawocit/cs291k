import torchvision

from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from PIL import Image
import open_clip
from collections import OrderedDict

class CustomCLIPModel(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPModel, self).__init__()
        self.clip_model = clip_model 

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features.reshape(-1, 3, 16, 16)
        return image_features
        

def create_model(num_classes, pretrained=True, coco_model=False):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')

    custom_weights_path = "/hdd/yuchen/satdata/weights/clip_epoch_20.pt" 
    custom_weights = torch.load(custom_weights_path, map_location="cuda:0")  
    state_dict = custom_weights['state_dict']
    
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = value
        
    model.load_state_dict(new_state_dict, strict=False) 

    backbone = CustomCLIPModel(model)

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 128 for this custom Mini DarkNet model.
    backbone.out_channels = 3

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

        
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)