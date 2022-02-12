
## Code Hierarchy

```
- config.py # good practice to centralize hyper parameters

- preprocess.py # Step 1, preprocess, store numpy/meta 'cache' at ./preprocess/

- train_segmentation.py 
- model_UNet.py # UNet model definition

- train_classificaion.py 
- model_Inception.py # Inception model definition
- model_ResNet.py # ResNet model definition
- model_DenseNet.py # DenseNet model definition

- generators.py # generator for segmentation & classificaiton models
- visual_utils.py # 3D visual tools

- dataset/ # dataset, changed in config.py
- preprocess/ # 'cache' preprocessed numpy/meta data, changed in config.py

- train_ipynbs # training process notebooks
```

## Preprocess

- use `SimpleITK` to read CT files, process, and store into cache with numpy arrays
- process with `scikit-image` lib, try lots of parameters for best cutting
    - binarized
    - clear-board
    - label
    - regions
    - closing
    - dilation
- collect all meta information(seriesuid, shape, file_path, origin, spacing, coordinates, cover_ratio, etc.) and store in **ONE** cache file for fast training init.
- see preprocessing in `/train_ipynbs/preprocess.ipynb` file

Distribution of the lung part takes on a whole CT.

Tumor size distribution

## Segmentation

- A **simplified and full UNet** both tested.
- **`dice_coef_loss`** as loss function.
- Periodically evaluate model with **lots of metrics**, which helps a lot to understand the model.
- 30% of negative sample, which has no tumor, for generalization.
- Due to memory limitation, 16 batch size used.

## Classification


### ResNet

- use `bottleneck` block instead of `basic_block` for implementation.
- A `bottleneck` **residual block** consists of:
    - (1, 1, 1) conv_bn_relu
    - (3, 3, 3) conv_bn_relu
    - (1, 1, 1) conv_bn_relu
    - (optional in config) kernel_size=(3, 3, 3), strides=(2, 2, 2) conv_bn_relu for compression.
    - **Add(not Concatenate)** with input
- Leave `RESNET_BLOCKS` as config to tune

### DenseNet
    - 3 dense\_block with 5 bn\_relu\_conv layers according to paper. 
    - transition\_block after every dense\_block, expcet the last one.
    - Optional config for **DenseNet-BC**(paper called it): **1\*1\*1 depth-size seperable conv**, and **transition_block compression**.

## Fine Tunning & Experience Got 

- Learning rate: `3e-5` works well for UNet, `1e-4` works well for classification models.
- Due to memory limitation, 16 batch size used.
- Data Augumentation: shift, rotate, etc.

