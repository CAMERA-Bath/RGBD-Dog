'''
Plot the markers and skeleton in 3D space.
Markers are shown in black, the skeleton is given a different colour for each body part
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
	cameraFrame = 13
	
	return pathToMainDataset, dog, motion, cameraFrame
	
	
def main(pathToMainDataset, dog, motion, cameraFrame):


	pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)

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
	markers = markers[:,:,cameraFrame]	
	
	# ----------------------------------- load the stick file that defines the relationship between the markers -----------------------------
	# use the vskSticks.txt file to get the connection between markers
	vskSticks = os.path.join(pathToMainDataset, dog, 'meta', 'vskSticks.txt')
	
	markerConn = []
	with open(vskSticks, 'r') as myFile:
		for f in myFile:
			res = [x.strip() for x in f.split(' ')]
			#convert from name to index
			markerConn.append([markerNames.index(res[0]), markerNames.index(res[1])])
				
				
	fig = plt.figure(); ax = fig.add_subplot(111, projection='3d') 
	ax, fig = utils.Plot3d(bvhJoints, connections=skelConnections, style='bo-', ax=ax, jointSpecificClrs=utils.GetDefaultColours())
	# ax, fig = utils.Plot3d(markers, connections=markerConn, style='bo-', ax=ax, differentColoursForSides=True)
	ax, fig = utils.Plot3d(markers, connections=markerConn, style='ko--', ax=ax, differentColoursForSides=True)
	ax.set_xlabel('x');ax.set_ylabel('y');ax.set_aspect('auto');plt.show()
	

if __name__ == '__main__':
	pathToMainDataset, dog, motion, cameraFrame = SetValues()
	main(pathToMainDataset, dog, motion, cameraFrame)
	
	
	
	
	
	
	
	