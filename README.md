# Depth Estimation of the single images with Transfer Learning pretrained MobileNetV2

This project implements a deep learning neural network model to generate the depth image of a given image.
Model is a U-net model with MobileNetV2 as the encoder, and model has utilized skip connection from encoder to decoder.
Model generates a depth image of resolution 480x640 for input image of same size.


This project was implemented taking reference from the following paper: 

[High Quality Monocular Depth Estimation via Transfer Learning (arXiv 2018)](https://arxiv.org/abs/1812.11941)
**[Ibraheem Alhashim]** and **Peter Wonka** 

# Results 

| Original image                    | Original DenseDepth                | Predicted DenseDepth                   |
| ----------------------------------- | ----------------------------------- | ------------------------------------------- |
| ![Project](results/0_image.jpg) | ![Project](original_depth/depth0.png) | ![Project](results/0_depth.jpg)  |
| ![Project](results/1_image.jpg) | ![Project](original_depth/depth1.png) | ![Project](results/1_depth.jpg)  |
| ![Project](results/2_image.jpg) | ![Project](original_depth/depth2.png) | ![Project](results/2_depth.jpg)  |
| ![Project](results/3_image.jpg) | ![Project](original_depth/depth3.png) | ![Project](results/3_depth.jpg)  |


##### Structural Similarity Index (SSIM) between the above two images: 0.37290473188266604
# Data 
Downlaod the data from the link [click here](www.google.com) and put in the parent directory of the project. 

# Train the model. 

Run the command `python train.py` to train the model and save the model in `./models` directory. 

# Test the model.
Run the command `python test.py` the model the testing images is in the `./test_img_1` directory and produced DenseDepth saved in the './resutls' directory. 

# Documentation 
Documentation is present at the link : [documentation](https://vikash9899.github.io/depth_estimation/)


















































