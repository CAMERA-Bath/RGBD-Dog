'''
Apply the mask on the Sony or Kinect images
Uses matplotlib to display the final image

Edit the values in SetValues() before running the script
'''

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys, os
from utils import utils

def SetValues():
	pathToMainDataset = 'D:/DOG'
	dog = 'dog1'
	motion = 'walk'
	cam = '00'
	
	# cameraType = 'sony'
	cameraType = 'kinect_rgb'
	# cameraType = 'kinect_depth'
	
	cameraFrame = 13
	
	return pathToMainDataset, dog, motion, cam, cameraType, cameraFrame



def main(pathToMainDataset, dog, motion, cam, cameraType, cameraFrame):


	pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)

	# ----------------------------------- load the image -----------------------------
	frame = None
	reconPoints = None
	assert cameraType in ['sony', 'kinect_rgb', 'kinect_depth'], 'invalid cameraType (%s)'% cameraType

	lineThickness_forPlotting =3
	scaleAppliedAfterProjection = 1

	if cameraType == 'sony':
		scaleAppliedAfterProjection = 0.5 # scale down from 4K to 2K

		pathToVideo = os.path.join(pathToMotion, cameraType, 'camera%s'%cam, 'camera%s_2K.mp4'%cam)
		vidData = cv2.VideoCapture(pathToVideo)
		ret, vidData = utils.SetVideoToFrameNumber(vidData, cameraFrame)
		ret, frame = vidData.read()
		frame = frame[:,:,(2,1,0)].copy() # I get an error when plotting the skeleton if I don't have this
		# "TypeError: Layout of the output array img is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)"
		
	else:
		pathToImages = os.path.join(pathToMotion, cameraType, 'camera%s'%cam, 'images')
		startingImageName = '%s_%08d' % (cam, cameraFrame)
		imName = utils.GetFilesInFolder(pathToImages, contains=startingImageName)
		assert len(imName) > 0, 'no images found for frame %d' % cameraFrame

		if  cameraType == 'kinect_rgb':
			frame = cv2.imread(os.path.join(pathToImages, imName[0]))
			frame = frame[:,:,(2,1,0)].copy()
		else:
			lineThickness_forPlotting = 2
			
			frame = cv2.imread(os.path.join(pathToImages, imName[0]),-1)
			# make the image "human" viewable
			frame = frame / np.amax(frame) * 255 # scale from 0-255
			frame = frame.astype(np.uint8)
			# turn into 3 channels
			frame = np.stack((frame,)*3, axis=-1)


	pathToMasks = os.path.join(pathToMotion, cameraType, 'camera%s'%cam, 'masks')
	startingImageName = '%s_%08d' % (cam, cameraFrame)
	maskName = utils.GetFilesInFolder(pathToMasks, contains=startingImageName)
	assert len(maskName) > 0, 'no masks found for frame %d' % cameraFrame

	mask = cv2.imread(os.path.join(pathToMasks, maskName[0]), -1)
	mask = cv2.resize(mask, (0,0), fx=scaleAppliedAfterProjection, fy=scaleAppliedAfterProjection, interpolation=cv2.INTER_NEAREST ) 
	mask = mask/np.amax(mask)
	
	mask = np.stack((mask.astype(np.uint8),)*3, axis=-1)
	frame = frame * mask
	plt.imshow(frame); plt.show()
		
		

if __name__ == '__main__':
	pathToMainDataset, dog, motion, cam, cameraType, cameraFrame = SetValues()
	main(pathToMainDataset, dog, motion, cam, cameraType, cameraFrame)
	
	
	
	
	
	
	
	
	