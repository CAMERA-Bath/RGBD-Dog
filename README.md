# RGBD-Dog: Predicting Canine Pose from RGBD Sensors

![RGBD-Dog](/figs/firstIm_v2.png)

## Table of Contents
  * [Abstract](#abstract)
  * [Dataset](#dataset)
    * [Contents](#contents)
    * [Layout](#layout)
	* [Access](#access)
  * [Code](#code)
    * [Dependencies](#dependencies)
  * [Citation](#citation)
  * [Contact](#contact)
  
  
## Abstract
The automatic extraction of animal 3D pose from images without markers is of interest in a range of scientific fields. Most work to date predicts animal pose from RGB images, based on 2D labelling of joint positions. However, due to the difficult nature of obtaining training data, no ground truth dataset of 3D animal motion is available to quantitatively evaluate these approaches. In addition, a lack of 3D animal pose data also makes it difficult to train 3D pose-prediction methods in a similar manner to the popular field of body-pose prediction. In our work, we focus on the problem of 3D canine pose estimation from RGBD images, recording a diverse range of dog breeds with several Microsoft Kinect v2s, simultaneously obtaining the 3D ground truth skeleton via a motion capture system. We generate a dataset of synthetic RGBD images from this data. A stacked hourglass network is trained to predict 3D joint locations, which is then constrained using prior models of shape and pose. We evaluate our model on both synthetic and real RGBD images and compare our results to previously published work fitting canine models to images. Finally, despite our training set consisting only of dog data, visual inspection implies that our network can produce good predictions for images of other quadrupeds -- e.g. horses or cats -- when their pose is similar to that contained in our training set.

[Link to paper .pdf](https://arxiv.org/pdf/2004.07788.pdf)

[YouTube](https://www.youtube.com/watch?v=sRsjo-pE9hI)

## Dataset
Details on accessing the data will be posted in the next few days (as of June 8th 2020)

![RGBD-Dog](/figs/projAll_v2.png)


### Contents
Our dataset consists of five similar motions for five dogs: 

* walking in an approximately straight line
* trotting in an approximately straight line
* jump over poles
* walk over poles
* stepping/jumping on and off a table approximately 30cm in height.

For each sequence, the dog is accompanied by its handler.
This person is not wearing a motion capture suit and no skeleton data of the person is provided.

For each dog, this data is available in the form of:

* 3D marker locations
* the solved skeleton joint rotations
* the neutral mesh of the dog
* Linear Blend Skinning weights
* multi-view HD RGB footage recorded at 59.97 fps
* multi-view RGB and RGB-D images from the Microsoft Kinect recording at approximately 6 fps.

The HD RGB footage will be available in 4K resolution on request.
The number of cameras used per dog varied between eight to ten for the HD RGB cameras and five to six for the Kinects.

Note that the first frame of every .bvh file is the neutral pose of the dog. As such, frame F for camera C is frame F+1 in the .bvh/skeleton data.
### Layout
Data for each dog is located in its own folder. The structure of this folder is as follows:

- calibration
  - sony
    - calibFile\_CAMERA_ID
    - ...
  - kinect_rgb
    - calibFile\_CAMERA_ID
    - ...
  - kinect_depth
    - calibFile\_CAMERA_ID
    - ...
- meta
  - neutralMesh.obj
  - skinningWeights.mat
  - vskSticks.txt
- motion\_MOTION_NAME
  - kinect_depth
     - camera\_CAMERA_ID
         - images
         - masks
     - ...
  - kinect_rgb
     - camera\_CAMERA_ID
         - images
         - masks
     - ...
  - motion_capture
     - markers.json
     - skeleton.bvh
     - timecodes.json
  - sony
     - camera\_CAMERA_ID
         - masks
         - camera\_CAMERA\_ID_2K.mp4
     - ...
- motion\_MOTION_NAME
- ...

### Access
This data is available for academic use. Please have a staff faculty member complete the form Data_Release_Form_RGBDDog_CVPR_2020.pdf, listed on this github page, and email it to [Sinéad Kearney](s.kearney@bath.ac.uk). You will then receive details on how to access the data. Companies should instead contact [Prof. Darren Cosker](d.p.cosker@bath.ac.uk).

## Code
We provide code for visualising the data in both Python and Blender. This code is located in the "Source" folder. We also provide the shape model, structured to be similar to the Skinned Multi-Person Linear model (SMPL), Skinned Multi-Animal Linear model (SMAL), etc. We call this model the DynaDog model, and is located in "DynaDog_model".

### Dependenices
All code has been tested using Python3 on Windows 10. Blender is version 2.79.

Python libraries used:
* numpy
* scipy
* pylab
* cv2
* matplotlib
* pickle

## Citation
If you find this dataset useful, we would kindly ask you to cite:

```
@InProceedings{Kearney_2020_CVPR,
author = {Kearney, Sinead and Li, Wenbin and Parsons, Martin and Kim, Kwang In and Cosker, Darren},
title = {RGBD-Dog: Predicting Canine Pose from RGBD Sensors},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```


## Contact
This code is maintained by [Sinéad Kearney](s.kearney@bath.ac.uk)