# Application of Pre-training Strategies on Landslides Detection

CS291K (Machine Learning and Data Mining) course project

![plot](/figures/training_pipeline.png)

In this project, we:
- utilized a Faster R-CNN framework and applied transfer learning on two landslides datasets
- explored various pre-training strategies (Image Classification, Knowledge Distillation, and Masked Autoencoder) to improve the model's landslides detection capability

For required weights and datasets, please see: [Implementation Details Section](#implementation-details)

For further reading, please see: [our final report](/final_report.pdf) 

# Introduction: Landslides Detection
Landslide has affected about five million people worldwide.

Existing deep-learning approaches either require training from scratch or use natural image pre-trained weights to initialize the model.

**Although the use of different pre-trainnig strategies (e.g., MAE) with satellite imagery has been extensively studied, their effectiveness in landslide detection remains unexplored**. 

# Methods
## Stage 1: Pre-training
- Objective: Pre-train image encoders using different stratgies
- Datasets:
    - [ImageNet-1K](https://www.image-net.org/), [fMoW-RGB](https://github.com/fMoW/dataset), and [Landslide4Sense](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense)
- Architecture (image encoder):
    - CNN: ResNet-18 and EfficientNet-B0
    - Transformer:  ViT and Swin Transformer
- Image Encoder Pre-training Strategies
    - **Image Classification**: train an image encoder with the ImageNet dataset to predict the corresponding image category given an input image. 
    - **Knowledge Distillation**: "distilled" a more complex model into our smaller, task-specific image encoder:
        1) load the ImageNet-pre-trained weights onto a predetermined teacher model; 
        2) fine-tune the teacher model using the Landslide4Sense dataset on binary landslides image classification; 
        3) freeze the teacher model's weights; 
        4) train a smaller student model to predict the teacher model's soft target probabilities along with the ground-truth hard labels
    - **Maksed Autoencoder**: train an encoder that encodes masked images into tokens and maps them to high-dimensional space, and a decoder that learns from the encoded latent image features and reconstructs the original image

## Stage 2: Fine-tuning
- Objective: Landslide object bounding box detection
- Datasets: [Landslide4Sense](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense)
- Architecture:
    - Backbone: image encoder pretrained during Stage 1
    - Head: Faster R-CNN


## Evaluation
The Intersection Over Union (IoU) is calculated based on the overlapped and union areas between the predicted and the ground-truth bounding boxes. The result is considered a correct detection if the detection with IoU is larger than or equal to a predefined threshold.

# Results

We first investigated the performance of various backbone architectures on landslide object detection. The pre-training strategy was fixed to be image classification and the pre-trained dataset was ImageNet-1K for all architectures. **We found the Swin-Base outperformed other architectures.**

![plot](/figures/Table1.jpg)

We then selected the three best-performing architectures (i.e., ViT-Large, Swin-Tiny, and Swin-Base) and investigated the effect of various pre-training strategies on further improving the models' performance for this landslide detection task. **Across all dataset-strategy-model combinations, the Swin-Base pre-trained using MAE yielded the best performance.**

![plot](/figures/Table2.jpg)

Example prediction results:
![plot](/figures/prediction.png)



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
  