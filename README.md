# neural_networks_project_2020

Repo for the project of NN 2020 based on the paper "Interpreting CNNs via Decision Trees"

## Requirements
4 types of disentagled CNNs:
- ~AlexNet~   (**not suitable for multi-class**)
- ~VGG-M~
- ~VGG-S~
- VGG-16

All of them were pre-trained on *ImageNet ILSVRC 2012* with a loss for 1000-category classification.

They used 3 combined datasets to evaluate the new CNN:
- PASCAL VOC 2010 Part Dataset
    - 6 categorie: `bird`, `cat`, `cow`, `dog`, `horse`, `sheep`
    - http://roozbehm.info/pascal-parts/pascal-parts.html
- ~CUB200-2011~ (**not suitable for multi-class**)
    - 11.8K imgs (1.1GB) di 200 categorie di uccelli che hanno unificato in una sola: `bird`
- ILSVR-C 2013 DET Animal-Part
    - https://github.com/zqs1022/detanimalpart.git
    - 30 categorie
### Datasets
Just like in most part-localization studies, they used animal categories, which prevalently contain non-rigid shape deformation, for evaluation. I.e. we selected six animal categories—bird, cat, cow, dog, horse, and sheep—from the PASCAL Part Dataset. The CUB200-2011 dataset contains 11.8K images of 200 bird species. Like in, we ignored species labels and regarded all these images as a single bird category. The ILSVRC 2013 DET Animal-Part dataset consists of 30 animal categories among all the 200 categories for object detection in the ILSVRC 2013 DET dataset.


### Our implementation

- CNN: VGG16
- Benchmark dataset: PASCAL VOC 2010 Part Dataset + ILSVR-C 2013 DET

Environment set up:
```
cd ~
python3 -m venv --system-site-packages ./nn_venv
source ./nn_venv/bin/activate                       # to enter the virtual env
pip install --upgrade pip
pip install --upgrade tensorflow
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
deactivate                                              # to exit the virtual env
```