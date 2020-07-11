'''
Given a mesh produced by the skinning function, we use barycentric coordindates to convert this mesh into the SMAL mesh
Using the SMAL joint regressor, we can obtain the 3D locations of the SMAL skeleton

Note that the SMAL model can be downloaded from: http://smal.is.tue.mpg.de/
'''
import os
import sys
from utils import utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np
import test_skinning


def SetValues():
	PATH_TO_SMAL = 'full/path/to/SMAL.pkl'
	pathToMainDataset = 'D:/DOG'
	dog = 'dog1'
	motion = 'trot'
	cameraFrame = 0
	
	return pathToMainDataset, dog, motion, cameraFrame, PATH_TO_SMAL

def main(pathToMainDataset, dog, motion, cameraFrame, PATH_TO_SMAL):

	assert dog not in ['dog6', 'dog7'], 'barycentric variables are not available for these dogs'
	
	bary_file = os.path.join(pathToMainDataset, dog, 'meta', 'SMAL_barycen.mat')
	assert os.path.isfile(bary_file), 'file could not be located. Please ensure you have the latest version of the dog\'s meta folder'
	
	barycen = utils.LoadMatFile(bary_file)
	barycen_neighVertIds = utils.LoadMatFile(os.path.join(pathToMainDataset, dog, 'meta', 'SMAL_barycen_neighVertIds.mat'))
	barycen_neighVertIds -= 1 # created in Matlab, so start index from 0
	numVertsInSmalMesh = barycen.shape[0]

	# get the mesh and joint locations for the specified frame
	verts, bvhJoints = test_skinning.main(pathToMainDataset, dog, motion, cameraFrame, plotMesh=False)

	# get SMAL vertex locations from the mesh we've generated
	verts_smal = np.zeros((numVertsInSmalMesh,3))
	for v in range(numVertsInSmalMesh):
		nn = barycen_neighVertIds[v]
		barycen_currFace = barycen[v]
		nn_location = np.transpose(verts[nn])
		verts_smal[v] = nn_location.dot(barycen_currFace)

	# use the SMAL joint regressor to get the 3D joint locations
	assert os.path.isfile(PATH_TO_SMAL), 'invalid location for the SMAL pickle file'
	with open(PATH_TO_SMAL, 'rb') as f: 
		smalData = pickle.load(f, encoding='latin1')
	J_regressor = smalData['J_regressor']
	skelConnections = smalData['kintree_table'][0]
	joint_locations = J_regressor.dot(verts_smal)

	# plot
	fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')	
	ax, fig = utils.Plot3d(joint_locations, connections=skelConnections, style='bo-', ax=ax, differentColoursForSides=False)
	ax.plot(verts_smal[:,0], verts_smal[:,1], verts_smal[:,2], 'go', markersize=2)
	plt.title('skeleton and mesh with skinning applied')
	ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('auto')
	plt.show()


if __name__ == '__main__':
	pathToMainDataset, dog, motion, cameraFrame, PATH_TO_SMAL = SetValues()
	main(pathToMainDataset, dog, motion, cameraFrame, PATH_TO_SMAL)


