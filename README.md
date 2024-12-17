# Application of Pre-training Strategies on Landslides Detection

*CS291K (Machine Learning and Data Mining) course project*

![plot](/figures/training_pipeline.png)

In this project, we investigated the application of pre-training strategies to improve landslides detection using deep learning. We employed the Faster R-CNN framework with transfer learning on satellite imagery datasets and explored the following pre-training strategies:
- Image Classification
- Knowledge Distillation
- Masked Autoencoder (MAE)

By utilizing various pre-trained models and techniques, we achieved improvements in landslides bouding box detection on satellite images.


:bulb: For environment setup and required weights and datasets, please see: [Implementation Details Section](#implementation-details)

:mag_right: For further reading, please see: [our final report](/final_report.pdf) 

# Introduction: Landslides Detection
Landslide has affected about 5 million people worldwide.

Existing deep-learning approaches either require training from scratch or use natural image pre-trained weights to initialize the model.

**Although the use of different pre-trainnig strategies (e.g., MAE) with satellite imagery has been extensively studied, their effectiveness in landslide detection remains unexplored**. 

# Methods
## Stage 1: Pre-training
- Objective: Pre-train image encoders using different strategies
- Datasets:
    - [ImageNet-1K](https://www.image-net.org/), [fMoW-RGB](https://github.com/fMoW/dataset), and [Landslide4Sense](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense)
- Architecture (image encoder):
    - CNN-based: ResNet-18 and EfficientNet-B0
    - Transformer-based:  ViT and Swin Transformer
- Image Encoder Pre-training Strategies
    - **Image Classification**: 
        - train an image encoder with the ImageNet dataset to predict the image category
    - **Knowledge Distillation**: "distilled" a more complex model into our smaller, task-specific image encoder:
        1) load the ImageNet-pre-trained weights onto a predetermined *teacher model*; 
        2) fine-tune the teacher model using the Landslide4Sense dataset on binary landslides image classification; 
        3) freeze the teacher model's weights; 
        4) train a smaller *student model* to predict the teacher model's soft target probabilities along with the ground-truth hard labels
    - **Maksed Autoencoder**: 
        - mask input image patches and train an encoder-decoder framework to reconstruct the original image

## Stage 2: Fine-tuning
- Objective: Landslide object bounding box detection
- Datasets: [Landslide4Sense](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense)
- Architecture:
    - Backbone: image encoder pretrained during Stage 1
    - Head: Faster R-CNN


## Evaluation
A detection is considered correct if Intersection Over Union (IoU) ≥ predefined threshold.

# Results

We first investigated the performance of various backbone architectures on landslide object detection. The pre-training strategy was fixed to be ImageNet-1K image classification for all architectures. **We found that the Swin-Base outperformed other architectures.**

![plot](/figures/Table1.jpg)

We then selected the three best-performing architectures (i.e., ViT-Large, Swin-Tiny, and Swin-Base) and investigated the effect of various pre-training strategies on further improving the models' performance for this landslide detection task. **Across all dataset-strategy-model combinations, the Swin-Base pre-trained using MAE yielded the best performance.**

![plot](/figures/Table2.jpg)

Sample prediction results from our best-performing model (Swin-Base pre-trained using MAE):

![plot](/figures/prediction.png)

# Conclusion
 We presented a comprehensive approach to landslide detection using deep-learning techniques, focusing on *using pre-trained image encoder architectures within the Faster R-CNN framework*. 
 
 We found that the Swin-Base architecture, pre-trained using Masked Autoencoder (MAE) yielded the best performance in detecting landslides within satellite imagery. 

  Our findings highlighted the importance of selecting appropriate pre-training strategies and backbone architectures for improving landslide detection performance. 

# About the Authors
This project was part of the CS291K: Machine Learning and Data Mining course. Study design and code implementation were done by me ([Yuchen Hou](https://github.com/subawocit)) and [Vihaan Akshaay Rajendiran](https://github.com/VihaanAkshaay).


# Implementation Details
## Environment Setup
```sh
cd 291k
conda env create -f environment.yml
conda activate 291k
```

## Model weights:
- CLIP image encoder weights: https://github.com/wangzhecheng/SkyScript
- vitdet (MAE) backbone model weights: https://github.com/sustainlab-group/SatMAE
- Other weights (pretrained for this project): https://drive.google.com/drive/folders/1kzUATRd5Rzyav2yyE-Bcl7hGmHzPDLgL?usp=sharing 

## Datasets:
- landslide4sense: https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense
- dataset used in paper "A novel Dynahead-Yolo neural network for the detection of landslides with variable proportions using remote sensing images": https://github.com/Abbott-max/dataset/tree/main


## Code adapted from:
- Swin-MAE: https://github.com/Zian-Xu/Swin-MAE
- General Faster-RCNN training pipeline: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
- Swin-Transformer FPN neck: https://github.com/oloooooo/faster_rcnn_swin_transformer_detection/tree/maste
- Knowledge distillation: https://huggingface.co/docs/transformers/en/tasks/knowledge_distillation_for_image_classification
- CLIP image encoder: https://github.com/mlfoundations/open_clip

