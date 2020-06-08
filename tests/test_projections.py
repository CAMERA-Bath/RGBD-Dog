'''
Show either the projected dog joints, or projected marker positions on the Sony or Kinect images
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
	
	# cameraType = 'sony'
	cameraType = 'kinect_rgb'
	# cameraType = 'kinect_depth'
	
	cameraFrame = 13
	
	action = 'plot_skeleton'
	# action = 'plot_markers'
	
	return pathToMainDataset, dog, motion, cam, cameraType, cameraFrame, action
	
def main(pathToMainDataset, dog, motion, cam, cameraType, cameraFrame, action):

	pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)

	# ----------------------------------- load the calibration -----------------------------
	pathToCalib = os.path.join(pathToMainDataset, dog, 'calibration', cameraType)#, 'calibFile%s'%cam)
	calib = utils.GetCalibData(pathToCalib, cam)

	# ----------------------------------- load the image -----------------------------
	frame = None
	reconPoints = None
	assert cameraType in ['sony', 'kinect_rgb', 'kinect_depth'], 'invalid cameraType (%s)'% cameraType
	assert action in ['plot_skeleton', 'plot_markers'], 'invalid action (%s)'% action

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



	if action == 'plot_skeleton':
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
		# ----------------------------------- project skeleton joints to 2d -----------------------------
		[bvhJoints_2d, j] = cv2.projectPoints(bvhJoints, calib['rot3x1'], calib['t'], calib['K'], calib['distCoeffs'])
		numJoints = bvhJoints_2d.shape[0]
		bvhJoints_2d = np.reshape(bvhJoints_2d, (numJoints, 2))
		bvhJoints_2d *= scaleAppliedAfterProjection
		
		# ----------------------------------- plot the skeleton on the image -----------------------------
		if frame is not None:
			imH, imW = frame.shape[0:2]
			bvhJoints_2d_inImage = np.ones((numJoints,))
			bvhJoints_2d_inImage[bvhJoints_2d[:,0] >= imW] = 0
			bvhJoints_2d_inImage[bvhJoints_2d[:,0] < 0] = 0
			bvhJoints_2d_inImage[bvhJoints_2d[:,1] >= imH] = 0
			bvhJoints_2d_inImage[bvhJoints_2d[:,1] < 0] = 0

			frame_withSkel = utils.Plot2d(frame, bvhJoints_2d, visibility=bvhJoints_2d_inImage, lineThickness=lineThickness_forPlotting, jointSpecificClrs=utils.GetDefaultColours())
			plt.imshow(frame_withSkel); plt.show()

		
		
		
	elif action == 'plot_markers':
		markerFile = os.path.join(pathToMotion, 'motion_capture', 'markers.json')
		markerFile_asNp = markerFile[:markerFile.rfind('.')] + '.npy'
		markerFile_namesText = markerFile[:markerFile.rfind('.')] + '_names.txt'

		# ----------------------------------- load the markers -----------------------------
		print('loading markers...')
		if os.path.isfile(markerFile_asNp):
			markers = np.load(markerFile_asNp)
			with open(markerFile_namesText, 'r') as f:
				markerNames_joined = f.read()
			markerNames = markerNames_joined.split(',')
		else:
			[markerNames, markers] = utils.GetPointsFromJsonFbx(markerFile)
			markers = utils.MovePointsOutOfMayaCoordSystem(markers)
			np.save(markerFile_asNp, markers)
			markerNames_joined = ','.join(markerNames)
			with open(markerFile_namesText, 'w') as f:
				f.write(markerNames_joined)
			
		# ----------------------------------- load the stick file that defines the relationship between the markers -----------------------------
		# use the vskSticks.txt file to get the connection between markers
		vskSticks = os.path.join(pathToMainDataset, dog, 'meta', 'vskSticks.txt')
		
		markerConn = []
		with open(vskSticks, 'r') as myFile:
			for f in myFile:
				res = [x.strip() for x in f.split(' ')]
				#convert from name to index
				markerConn.append([markerNames.index(res[0]), markerNames.index(res[1])])
						
							
		# ----------------------------------- project skeleton joints to 2d -----------------------------
		markers = markers[:,:,cameraFrame]
		[markers_2d, j] = cv2.projectPoints(markers, calib['rot3x1'], calib['t'], calib['K'], calib['distCoeffs'])
		numMarkers = markers_2d.shape[0]
		markers_2d = np.reshape(markers_2d, (numMarkers, 2))
		markers_2d *= scaleAppliedAfterProjection
		
		if frame is not None:
			imH, imW = frame.shape[0:2]
			markers_2d_inImage = np.ones((numMarkers,))
			markers_2d_inImage[markers_2d[:,0] >= imW] = 0
			markers_2d_inImage[markers_2d[:,0] < 0] = 0
			markers_2d_inImage[markers_2d[:,1] >= imH] = 0
			markers_2d_inImage[markers_2d[:,1] < 0] = 0

			frame_withMarkers = utils.Plot2d_markers(frame, markers_2d, connections=markerConn, lineThickness=lineThickness_forPlotting, radius=-1, markerNames=markerNames)
			plt.imshow(frame_withMarkers); plt.show()	


if __name__ == '__main__':
	pathToMainDataset, dog, motion, cam, cameraType, cameraFrame, action = SetValues()
	main(pathToMainDataset, dog, motion, cam, cameraType, cameraFrame, action)