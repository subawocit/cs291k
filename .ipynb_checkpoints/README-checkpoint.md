cs291k code backup

## Model weights
- CLIP image encoder weights: https://github.com/wangzhecheng/SkyScript
- vitdet (MAE) backbone model weights: https://github.com/sustainlab-group/SatMAE

## Datasets:
- landslides
- landslide4sense: https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense
- dataset used in paper "A novel Dynahead-Yolo neural network for the detection of landslides with variable proportions using remote sensing images": https://github.com/Abbott-max/dataset/tree/main
- - note: they didn't provide enough training details (e.g., metric, input image size, batch size, baseline model settings) so I couldn't replicate their results. I'm assuming the mAP in their paper is 0.5 mAP.

## Code adapted from
- Swin-MAE: https://github.com/Zian-Xu/Swin-MAE
- Faster-RCNN training pipeline: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
- Swin-Transformer FPN neck: https://github.com/oloooooo/faster_rcnn_swin_transformer_detection/tree/maste
- Knowledge distillation: https://huggingface.co/docs/transformers/en/tasks/knowledge_distillation_for_image_classification
- CLIP image encoder: https://github.com/mlfoundations/open_clip



  