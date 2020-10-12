# neural_networks_project_2020

Repo for the project of NN 2020 based on the paper "Interpreting CNNs via Decision Trees"

### Implementation

- CNN: VGG16
- Benchmark dataset: PASCAL VOC 2010 Part Dataset + CUB200-2011

Environment set up:
```
cd ~
python3 -m venv --system-site-packages ./nn_venv
source ./nn_venv/bin/activate                       # to enter the virtual env
pip install --upgrade pip
pip install --upgrade tensorflow treelib opencv-python
deactivate                                              # to exit the virtual env
```

We also used:
    - https://github.com/caesar0301/treelib