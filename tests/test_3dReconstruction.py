'''
Reconstruct the points from the Kinect depth image, and plot in world space alongside the ground-truth 3D dog skeleton
Uses matplotlib to display the final image

Edit the values in SetValues() before running the script
'''

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import utils


def SetValues():
	pathToMainDataset = 'D:/DOG'
	dog = 'dog1'
	motion = 'walk'
	cam = '00'

	cameraFrame = 13
	
	return pathToMainDataset, dog, motion, cam, cameraFrame
	
	
def main(pathToMainDataset, dog, motion, cam, cameraFrame=0):


	pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)

	# ----------------------------------- load the calibration -----------------------------
	cameraType='kinect_depth'
	pathToCalib = os.path.join(pathToMainDataset, dog, 'calibration', cameraType)#, 'calibFile%s'%cam)
	calib = utils.GetCalibData(pathToCalib, cam)

	# ----------------------------------- load the image -----------------------------
	frame = None
	reconPoints = None
	scaleAppliedAfterProjection = 1
	pathToImages = os.path.join(pathToMotion, cameraType, 'camera%s'%cam, 'images')
	startingImageName = '%s_%08d' % (cam, cameraFrame)
	imName = utils.GetFilesInFolder(pathToImages, contains=startingImageName)
	assert len(imName) > 0, 'no images found for frame %d' % cameraFrame

	lineThickness_forPlotting = 2
	frame = cv2.imread(os.path.join(pathToImages, imName[0]),-1)

	bvhSkelFile = os.path.join(pathToMotion, 'motion_capture', 'skeleton.bvh')
	skelConnections_asNp = os.path.join(pathToMotion, 'motion_capture', 'skelConnections.npy')
	bvhSkelFile_asNp = bvhSkelFile[:bvhSkelFile.rfind('.')] + '.npy'

	# ----------------------------------- load the skeleton -----------------------------
	print('loading skeleton...')
	if os.path.isfile(bvhSkelFile_asNp):
		bvhJoints = np.load(bvhSkelFile_asNp)
		skelConnections = np.load(skelConnections_asNp)
	else:
		bvhJoints, skelConnections, nodes = utils.ReadBvhFile(bvhSkelFile, False)
		bvhJoints = utils.MovePointsOutOfMayaCoordSystem(bvhJoints, 1)
		np.save(bvhSkelFile_asNp, bvhJoints)
		np.save(skelConnections_asNp, skelConnections)

	bvhJoints = bvhJoints[:,:,cameraFrame+1]
	reconPoints = utils.ReconstructDepthImage(frame, calib['K'])
	sampleRate = 30 # for speed
	reconPoints = reconPoints[::sampleRate,:]
	# move from camera space to world space
	reconPoints = np.hstack((reconPoints,np.ones((reconPoints.shape[0],1))))
	# L moves 3D points from world space to camera space -> inv(L) moves points from camera space to world space
	
	L = np.array(calib['calib']['L'])
	invL = np.linalg.inv(L)
	numPointsRecon = reconPoints.shape[0]
	reconPoints_worldSpace = reconPoints.dot(invL.T)
	reconPoints_worldSpace = reconPoints_worldSpace[:,0:3]/ np.reshape(reconPoints_worldSpace[:,3], (numPointsRecon, 1)) # divide by the last column
	

	fig = plt.figure(); ax = fig.add_subplot(111, projection='3d') 
	ax.scatter(reconPoints_worldSpace[:,0], reconPoints_worldSpace[:,1], reconPoints_worldSpace[:,2], c="red")
	ax, fig = utils.Plot3d(bvhJoints, connections=skelConnections, style='bo-', ax=ax, differentColoursForSides=True)
	ax.set_xlabel('x');ax.set_ylabel('y');ax.set_aspect('auto');plt.show()
	reconPoints = reconPoints_worldSpace
	

if __name__ == '__main__':
	pathToMainDataset, dog, motion, cam, cameraFrame = SetValues()
	main(pathToMainDataset, dog, motion, cam, cameraFrame)
	
	
	
	
	
	
	
	