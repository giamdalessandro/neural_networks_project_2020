# Decision Trees to understand CNNs

Repo for the project of NN 2020 based on the work [Interpreting CNNs via Decision Trees](https://arxiv.org/abs/1802.00121). 

Convolutional neural networks are nowadays widely used for different tasks in many fields, becomes thus important to understand what knowledge a CNN learns. In order to do so we have modified a VGG-16 network by adding a special mask layer at the end of the convolution, and trained it for an image classification task. Finally we have built a decision tree that could help us explain which object parts contributed the most to the final prediction and quantify these contributions. (Full report is available [here](docs/report.pdf))

## Implementation

- CNN: VGG16
- Benchmark dataset: PASCAL VOC 2010 Part Dataset + CUB200-2011

Virtual environment set up:
```
python3 -m venv --system-site-packages ./my_venv

# to enter the virtual env
source ./my_venv/bin/activate

# to exit the virtual env
deactivate                          
```

## Requirements
We developed this project with [tensorflow](https://www.tensorflow.org/) for python 3. 

- additional packages are indicated in the `requirements.txt` file, to install them using pip:
```
pip install --upgrade pip
pip install -r requirments.txt
```

- we exploited [treelib](https://github.com/caesar0301/treelib) as tree implementation.

## Authors
- [Giammarco D'Alessandro](https://github.com/giamdalessandro)
- [Maria Rosaria Fraraccio](https://gitlab.com/rooosyf)
- [Luca Gioffrè](https://github.com/balthier7997)
