# FSRCNN_keras


A implementation of the original paper ['Accelerating the Super-Resolution Convolutional Neural Network'](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)



<center><img width = "800" src="https://user-images.githubusercontent.com/58276840/95052310-aba98800-0729-11eb-9c45-a520eb98b3ed.png"></center>



tensorflow-gpu 2.0.0, keras 2.3.1 based implementation on Python 3.6.9, using Jupyter Notebook.




### Implementation
-------------------------------------------------------
My implementation may have some differences with the original paper:


#### Networks)

Feature Extraction)
- Filter size: 5x5
- Channel maps: 56
- Activation function: PReLU
- Border mode: Same (Zero Padding)

Shrinking)
- Filter size: 1x1
- Channel mapls: 12
- Activation function: PReLU
- Border mode: Same (Zero Padding)

Non-linear mapping)
- Filter size : 3x3
- Channel maps: 12
- Activation function: PReLU
- Border mode: Same (Zero Padding)
- The number of Layers: 4

Expanding)
- Filter size: 1x1
- Channel maps: 56
- Activation function: PReLU
- Border mode: Same (Zero Padding)

Deconvolution)
- Filter size: 9x9
- Channel maps: 1
- Strides: 2x2 (for scale2), 3x3 (for scale3), 4x4 (for scale4)
- Border mode: Same (Zero Padding)


#### Training)

- Loss Function: MSE (Mean Squared Error)
- Optimizer: Adam
- Learning rate: 10e-3 (10e-4 for the last layer)
- Learning rate during fine-tuning: reduced by half for all layers
- Batch size: 128


#### Dataset)

##### Training)
- 91-image, General-100 (Training set), Another 20 images from the validation set of the BSD500 dataset (Validation set)
- Data augmentation (scaling:0.9, 0.8, 0.7, 0.6, Rotation: 90, 180, 270): 19 times more images for training
- Patch size: 10 (Input), 20 (Label)

##### Training Strategy from FSRCNN Paper)
A two-step training strategy. First, we train a network from scratch with the 91-image dataset. Then, when the training is saturated, we add the General-100 dataset for fine tuning

As we have obtained a well-trained model under the upsacling factor 3, we then train the network for x2 on the basis of that for x3. During training, we only fine-tune the deconvolution layer on the 91-image and General-100 Dataset of x2. We only train the networks from scratch fro x3, and fine-tune the corresponding deconvolution layers for x2 and x4


##### Test)
- Set5, Set14, BSD200, Urban100



### Use
-------------------------------------------------------

You can generate dataset (training sample, test sample) through matlab files in Dataset_Generating directory
- Excute for data augmentation: `FSRCNN_data_aug.m`
- Excute for making LR images: `Generate_LR.m`
- Excute for training sample: `FSRCNN_generate_train.m`
- Excute for test sample: `FSRCNN_generate_test.m`


I also uploaded the trained weight files.

With FSRCNN.ipynb files and weight files, you can test the network.
- Excute for x2 SRCNN: `FSRCNN_scale2.ipynb`, `/weights/FSRCNN_checkpoint_scale2_General100.h5`
- Excute for x3 SRCNN: `FSRCNN_scale3.ipynb`, `/weights/FSRCNN_checkpoint_scale3_General100.h5`
- Excute for x4 SRCNN: `FSRCNN_scale4.ipynb`, `/weights/FSRCNN_checkpoint_scale4_General100.h5`

the files without '_General100' tags are the weight files trained with 91 images
