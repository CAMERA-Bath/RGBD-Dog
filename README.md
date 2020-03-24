# RGBD-Dog: Predicting Canine Pose from RGBD Sensors

![V2V-PoseNet](/figs/firstIm_v2.png)

## Abstract
The automatic extraction of animal 3D pose from images without markers is of interest in a range of scientific fields.
Most work to date predicts animal pose from RGB images, based on 2D labelling of joint positions.
However, due to the difficult nature of obtaining training data, no ground truth dataset of 3D animal motion is available to quantitatively evaluate these approaches. 
In addition, a lack of 3D animal pose data also makes it difficult to train 3D pose-prediction methods in a similar manner to the popular field of body-pose prediction.
In our work, we focus on the problem of 3D canine pose estimation from RGBD images, recording a diverse range of dog breeds with several Microsoft Kinect v2s, simultaneously obtaining the 3D ground truth skeleton via a motion capture system.
We generate a dataset of synthetic RGBD images from this data.
A stacked hourglass network is trained to predict 3D joint locations, which is then constrained using prior models of shape and pose.
We evaluate our model on both synthetic and real RGBD images and compare our results to previously published work fitting canine models to images.
Finally, despite our training set consisting only of dog data, visual inspection implies that our network can produce good predictions for images of other quadrupeds -- e.g. horses or cats -- when their pose is similar to that contained in our training set.

## Data
[The dataset will be uploaded soon]

![V2V-PoseNet](/figs/projAll_v2.png)



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

## Citation
[to be added]

## License
[to be added]