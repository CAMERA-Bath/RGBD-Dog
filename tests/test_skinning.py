'''
Note that this test is for the skinning weights of the dog
It does not yet take the animation from the bvh file and apply it to the mesh
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
	# TODO: apply motion from cameraFrame

	cameraFrame = -1 # cameraFrame 0 is frame 1 in the skeleton
	
	pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)			
	bvhSkelFile = os.path.join(pathToMotion, 'motion_capture', 'skeleton.bvh')
	skelConnections_asNp = os.path.join(pathToMotion, 'motion_capture', 'skelConnections.npy')
	bvhSkelFile_asNp = bvhSkelFile[:bvhSkelFile.rfind('.')] + '.npy'

	# ----------------------------------- load the skeleton -----------------------------
	if os.path.isfile(bvhSkelFile_asNp):
		bvhJoints = np.load(bvhSkelFile_asNp)
		skelConnections = np.load(skelConnections_asNp)
	else:
		# bvhJoints, skelConnections, nodes = utils.ReadBvhFile(bvhSkelFile, False, 10)
		bvhJoints, skelConnections, nodes = utils.ReadBvhFile(bvhSkelFile, False)
		bvhJoints = utils.MovePointsOutOfMayaCoordSystem(bvhJoints, 1)
		np.save(bvhSkelFile_asNp, bvhJoints)
		np.save(skelConnections_asNp, skelConnections)

	# ----------------------------------- load the skeleton timecodes -----------------------------	
	# pathToSkelTimecodes = os.path.join(pathToMotion, 'motion_capture', 'timecodes.json')
	# with open(pathToSkelTimecodes, 'r') as f:
		# skelFrameTimecodes = json.load(f)

	bvhJoints = bvhJoints[:,:,cameraFrame+1]
	
	pathToSkinningWeights = os.path.join(pathToMainDataset, dog, 'meta', 'skinningWeights.mat')
	weights = utils.LoadMatFile(pathToSkinningWeights)
	# TEMP:
	# np.save(os.path.join(pathToMainDataset, dog, 'meta', 'skinningWeights.npy'), weights)
	
	pathToObj = os.path.join(pathToMainDataset, dog, 'meta', 'neutralMesh.obj')
	meshObj = utils.LoadObjFile(pathToObj, rotate=True)
	verts_neutral = meshObj['vertices']

	# plot the neutral skeleton and mesh without any skinning:
	fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')	
	ax, fig = utils.Plot3d(bvhJoints, connections=skelConnections, style='bo-', ax=ax, differentColoursForSides=False)
	ax.plot(verts_neutral[:,0], verts_neutral[:,1], verts_neutral[:,2], 'ro')
	plt.title('skeleton and mesh with no skinning applied')
	ax.set_xlabel('x'); ax.set_ylabel('y'); plt.show()
	
	kintree_table = np.array([[4294967295,0,1,2,3,4,5,6,7,2,9,10,11,12,13,2,15,16,17,18,19,19,21,19,23,0,25,26,27,28,0,30,31,32,33,0,35,36,37,38,39,40,41],
						 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]])
	numJoints = kintree_table.shape[1]
	id_to_col = { kintree_table[1, i]: i for i in range(kintree_table.shape[1])	}
	parent = {
		i: id_to_col[kintree_table[0, i]]
		for i in range(1, kintree_table.shape[1])
	}
	# parent is a dictionary, the entry for each index is the parent of index
		
	# J should be the joints in the neutral position for the dog
	
	J = bvhJoints
	rot = np.tile(np.eye(3), (numJoints,1,1)) # shape = 43x3x3
	rot[3,:,:] = np.array([[1,0,0],[0,0.7071,-0.7071],[0,0.7071,0.7071]]) # 45 degrees about X
	
	R_world = np.empty((numJoints, 3, 3))
	R_world[0] = rot[0]
	for i in range(1, numJoints):
		R_world[i] = R_world[parent[i]].dot(rot[i])
		
	# get joint locations in local space
	joint_local = J.copy()
	joint_local[1:] -= J[kintree_table[0,1:]]

	# joint_local is 43x3, R_world is 43x3x3
	# apply R_world to each corresponding joint_local	
	# TODO: see if this can be done in one line
	for i, j in enumerate(joint_local):
		joint_local[i,:] = np.dot(R_world[i,:,:], j) # note that the order matters
		
	# turn from local position to world position
	joint_world = joint_local
	for i in range(1, numJoints):
		parentPos = joint_world[parent[i],:]
		joint_world[i,:] += parentPos
	J = joint_world
	
	
	# G is the LOCAL transformation of each joint
	G = np.empty((numJoints, 4, 4))
	G[0] = utils.with_zeros(np.hstack((rot[0], J[0, :].reshape([3, 1]))))
	for i in range(1, numJoints):
		G[i] = G[parent[i]].dot(
			utils.with_zeros(
				np.hstack( [rot[i],((J[i, :]-J[parent[i],:]).reshape([3,1]))]	)
			)
		)
	
		
	
	# remove the transformation due to the rest pose. Note here that we're using self.J, which has the additional shoulder&ear offsets (if applicable)
	G = G - utils.pack(
		np.matmul(
			G,
			np.hstack([J, np.zeros([numJoints, 1])]).reshape([numJoints, 4, 1])
			)
		)
	# transformation of each vertex
	v_posed = verts_neutral
	T = np.tensordot(weights, G, axes=[[1], [0]])
	rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
	# T.shape (2426, 4, 4) rest_shape_h.shape (2426, 4)
	v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
	
	trans = np.zeros([3])
	verts = v + trans.reshape([1, 3]) # apply root transformation
	
	
	fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')	
	ax, fig = utils.Plot3d(J, connections=skelConnections, style='bo-', ax=ax, differentColoursForSides=False)
	ax.plot(verts[:,0], verts[:,1], verts[:,2], 'go')
	plt.title('skeleton and mesh with skinning applied')
	ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('auto')
	plt.show()
		
	
	
	
if __name__ == '__main__':
	pathToMainDataset, dog, motion, cameraFrame = SetValues()
	main(pathToMainDataset, dog, motion, cameraFrame)
	
	
	
	
	
	
	
	
	
	