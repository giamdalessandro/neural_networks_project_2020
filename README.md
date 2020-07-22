# neural_networks_project_2020

Repo for the project of NN 2020 based on the paper "Interpreting CNNs via Decision Trees"

## Requirements
Python programming analyzing the following CNNs (all of them pretrained on *ImageNet ILSVRC 2012*):
- AlexNet
- VGG-M, VGG-S, VGG-16

### Datasets
Just like in most part-localization studies, they used animal categories, which prevalently contain non-rigid shape deformation, for evaluation. I.e. we selected six animal categories—bird, cat, cow, dog, horse, and sheep—from the PASCAL Part Dataset. The CUB200-2011 dataset contains 11.8K images of 200 bird species. Like in, we ignored species labels and regarded all these images as a single bird category. The ILSVRC 2013 DET Animal-Part dataset consists of 30 animal categories among all the 200 categories for object detection in the ILSVRC 2013 DET dataset.

- PASCAL-Part Dataset
- CUB200-2011 dataset
- ILSVRC 2013 DET Animal-Part dataset