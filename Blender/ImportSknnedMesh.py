'''
How to install numpy and scipy:

Open Blender, open python console
>>> import sys
>>> sys.exec_prefix
'/path/to/blender/python


Open a windows console as admin:
cd /path/to/blender/python/bin
python -m ensurepip
python -m pip install numpy
python -m pip install scipy


How to run this script:
Open a window in Blender and set it to "Text Editor"
Use the navigator to load this script

'''


import bpy
import sys
import numpy as np
import mathutils
import os
from os.path import join, dirname

os.system("cls")

# Add the folder that contains this script to the path, so we can access utils.py
# https://blender.stackexchange.com/questions/14911/get-path-of-my-script-file
for area in bpy.context.screen.areas:
	if area.type == 'TEXT_EDITOR':
		for space in area.spaces:
			if space.type == 'TEXT_EDITOR':
				scriptFolder = dirname(space.text.filepath)
				sys.path.insert(0, scriptFolder)	

from utils import LoadMatFile


def main():
	# ---- change these ------
	pathToMainDataset = 'D:/DOG'
	dog = 'dog2'
	motion = 'testSeq'
	# ---- change these ------

	
	pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)
	bvhSkelFile = os.path.join(pathToMotion, 'motion_capture', 'skeleton.bvh')
	pathToSkinningWeights = os.path.join(pathToMainDataset, dog, 'meta', 'skinningWeights.mat')
	pathToObj = os.path.join(pathToMainDataset, dog, 'meta', 'neutralMesh.obj')
		
	# the mesh and skeleton are defined in millimetres. Import into Blender in metres.
	OBJ_SCALE = 0.01
	BVH_SCALE = 0.01

	
	
	
	
	# ---------------------------------------------------------------

	#import bvh
	imported_object = bpy.ops.import_anim.bvh(filepath=bvhSkelFile, filter_glob="*.bvh", target='ARMATURE', global_scale=BVH_SCALE, frame_start=1, use_fps_scale=False, use_cyclic=False, rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')
	bvh_object = bpy.context.selected_objects[0] ####<--Fix
	# bvh_object.location = mathutils.Vector((0.0, 0.0, 0.0))
	# bvh_object.location = mathutils.Vector((0.0, 0.29094, -0.252148))

	# import obj
	imported_object = bpy.ops.import_scene.obj(filepath=pathToObj, split_mode='OFF')
	mesh_object = bpy.context.selected_objects[0] ####<--Fix
	mesh_object.scale = ( OBJ_SCALE, OBJ_SCALE, OBJ_SCALE )
	# mesh_object.location = mathutils.Vector((0.0, 0.29094, -0.252148))
		
	mesh_object.parent = bvh_object
	mesh_object.modifiers.new(name = 'Skeleton', type = 'ARMATURE')
	mesh_object.modifiers['Skeleton'].object = bvh_object


	bns = GetNamesInArmature(bvh_object)
	# bns = ['Root', 'Spine01', 'Spine02', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftWrist', 'LeftHand', 'LeftFinger', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightWrist', 'RightHand', 'RightFinger', 'Neck01', 'Neck02', 'Neck03', 'Neck04', 'Head', 'Nose', 'LeftEar', 'LeftEarEnd', 'RightEar', 'RightEarEnd', 'LeftLeg', 'LeftLowerLeg', 'LeftAnkle', 'LeftFoot', 'LeftToe', 'RightLeg', 'RightLowerLeg', 'RightAnkle', 'RightFoot', 'RightToe', 'TailBase', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'TailEnd']

	numBones = len(bns) # 42
	# add vertex groups
	for i in range(0, numBones):
		mesh_object.vertex_groups.new(bns[i])



	skinningWeights = LoadMatFile(pathToSkinningWeights)
	numBones = skinningWeights.shape[1] # 43
	
	# Blender doesn't let end effectors (ie, LeftFinger, RightToe, etc) have weights
	# transfer these weights to the parent bone (ie, LeftHand, RightFoot)
	# then select the indices of only non-end effector bones
	endEffectorBones = [8,14,20,22,24,29,34,42]
	jointIsEndEffector = np.zeros((numBones,))
	jointIsEndEffector[endEffectorBones] = 1
	for bn in endEffectorBones:
		skinningWeights[:,bn-1] = skinningWeights[:,bn-1] + skinningWeights[:,bn]
		skinningWeights[:,bn] *= 0
	skinningWeights = skinningWeights[:,jointIsEndEffector==0]
	
	
	print('skinningWeights.shape', skinningWeights.shape, 'numBones', numBones)
	SetSkinningMatrix(mesh_object, skinningWeights, 0)


	# root position at zero: [0.0, -0.29094, 0.252148]

def GetCurrentName(basename):
	# name might be "myMesh.001", so basename here would be "myMesh"
	obs = bpy.data.objects
	name = ''
	for o in obs:
		if basename in o.name:
			name = o.name
			break
	return name
	
def GetVertexGroupsNamesForMeshObject(mesh_object):

	if type(mesh_object) is str:
		mesh_object = GetCurrentName(mesh_object)
		mesh_object = bpy.data.objects[mesh_object]
		
	vgNames = []
	for vg in  mesh_object.vertex_groups:
		vgNames.append(vg.name)
	return vgNames
	
	
def SetSkinningMatrix(mesh_object, skinningMat, verbose=0):
	if type(mesh_object) is str:
		mesh_object = GetCurrentName(mesh_object)
		mesh_object = bpy.data.objects[name]
		
	vertex_groups = mesh_object.vertex_groups
	vertex_groups_names = GetVertexGroupsNamesForMeshObject(mesh_object)

	numBones = len(vertex_groups_names)
	
	me = mesh_object.data
	verts = me.vertices
	numVerts = len(verts)
	
	if type(skinningMat) is np.ndarray:
		print('skinningMat.shape', skinningMat.shape)
		skinningMat = skinningMat.tolist()

	
	for v_idx, s in enumerate(skinningMat):
		# loop through vertices
		# s is list of 43 values
		
		influencedByBones = np.where(np.array(s)>0)[0]
		weights = []
		for boneId, weight in enumerate(s):
			mesh_object.vertex_groups[boneId].add([v_idx], weight, "REPLACE")
			if boneId in influencedByBones: # save to print later
				weights.append(weight)
		
		if verbose:
			print('vertex id', v_idx, ', bones:', influencedByBones, ', weights', weights)
			
	return True
	
def GetNamesInArmature(arm_object):

	if type(arm_object) is str:
		arm_object = GetCurrentName(arm_object)
		arm_object = bpy.data.objects[arm_object]
		
	# if ob.type == 'ARMATURE':
		# armature = ob.data
	armature = arm_object.data
	boneNames = []
	for bone in armature.bones:
		boneNames.append(bone.name)
	return boneNames	
	

	


if __name__== "__main__":
	main()