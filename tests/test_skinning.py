'''
Note that this test is for the skinning weights of the dog
It does not yet take the animation from the bvh file and apply it to the mesh
'''

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import utils
from scipy.spatial.transform import Rotation as Rsci

def SetValues():
	pathToMainDataset = 'D:/DOG'
	dog = 'dog1'
	motion = 'trot'
	cameraFrame = 0
	
	return pathToMainDataset, dog, motion, cameraFrame
	
	
def main(pathToMainDataset, dog, motion, cameraFrame):

	skelFrame = cameraFrame+1
	
	pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)			
	bvhSkelFile = os.path.join(pathToMotion, 'motion_capture', 'skeleton.bvh')
	skelConnections_asNp = os.path.join(pathToMotion, 'motion_capture', 'skelConnections.npy')
	skelNodes_asPck = os.path.join(pathToMotion, 'motion_capture', 'skelNodes.pkl')
	bvhSkelFile_asNp = bvhSkelFile[:bvhSkelFile.rfind('.')] + '.npy'

	# ----------------------------------- load the skeleton -----------------------------
	if os.path.isfile(bvhSkelFile_asNp):
		bvhJoints = np.load(bvhSkelFile_asNp)
		skelConnections = np.load(skelConnections_asNp)
		with open(skelNodes_asPck, 'rb') as f:
			nodes = pickle.load(f)
	else:
		# bvhJoints, skelConnections, nodes = utils.ReadBvhFile(bvhSkelFile, False, 10)
		bvhJoints, skelConnections, nodes = utils.ReadBvhFile(bvhSkelFile, False)
		bvhJoints = utils.MovePointsOutOfMayaCoordSystem(bvhJoints, 1)
		
		with open(skelNodes_asPck, 'wb') as f:
			pickle.dump(nodes, f)
		np.save(bvhSkelFile_asNp, bvhJoints)
		np.save(skelConnections_asNp, skelConnections)

	# load skinning weights and neutral mesh
	pathToSkinningWeights = os.path.join(pathToMainDataset, dog, 'meta', 'skinningWeights.mat')
	weights = utils.LoadMatFile(pathToSkinningWeights)

	pathToObj = os.path.join(pathToMainDataset, dog, 'meta', 'neutralMesh.obj')
	meshObj = utils.LoadObjFile(pathToObj, rotate=True)
	verts_neutral = meshObj['vertices']
	
	J_neutral = bvhJoints[:,:,0] # the first frame of the bvh file is the neutral skeleton
	numJoints = J_neutral.shape[0]
	
	# rotate about z, to match dynaDog_np.py
	verts_neutral = verts_neutral[:,np.array((1,0,2))]
	verts_neutral[:,0] *= -1
	J_neutral = J_neutral[:,np.array((1,0,2))]
	J_neutral[:,0] *= -1
	
	# the world positions of the joints in this frame
	bvhJoints = bvhJoints[:,:,skelFrame]
	bvhJoints = bvhJoints[:,np.array((1,0,2))]
	bvhJoints[:,0] *= -1
	
	root_trans = bvhJoints[0]

	world_rot = utils.GetWorldRotationsForFrame(nodes, skelFrame, 'rodrigues')
	
	G = np.zeros((numJoints,4,4))
	# to get the correct rotations, we have to rearrange the axis
	for jntIdx, rot in enumerate(world_rot): 
		G[jntIdx,0:3,0:3] = Rsci.from_rotvec(rot).as_dcm()
		
		# remove root translation
		G[jntIdx,0:3,3] = bvhJoints[jntIdx] - bvhJoints[0]
		G[jntIdx,3,3] = 1

	# remove the neutral pose from the transformations
	G = G - utils.pack(
			np.matmul(
			G,
			np.hstack([J_neutral, np.zeros([numJoints, 1])]).reshape([numJoints, 4, 1])
			)
			)

	# transform each vertex
	T = np.tensordot(weights, G, axes=[[1], [0]])
	rest_shape_h = np.hstack((verts_neutral, np.ones([verts_neutral.shape[0], 1])))
	# T.shape (2426, 4, 4) rest_shape_h.shape (2426, 4)
	v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
	
	# apply root translation
	verts = v + root_trans.reshape([1, 3]) 
	
	fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')	
	ax, fig = utils.Plot3d(bvhJoints, connections=skelConnections, style='bo-', ax=ax, differentColoursForSides=False)
	ax.plot(verts[:,0], verts[:,1], verts[:,2], 'go')
	plt.title('skeleton and mesh with skinning applied')
	ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('auto')
	plt.show()
		
	
	
	
if __name__ == '__main__':
	pathToMainDataset, dog, motion, cameraFrame = SetValues()
	main(pathToMainDataset, dog, motion, cameraFrame)
	
	
	
	
	
	
	
	
	
	