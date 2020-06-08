'''
This code is based on smpl_np.py found at https://github.com/benjiebob/SMPL
as written by Yuxiao Zhou, https://github.com/CalciferZh 

For ease of comparision, this model has been structured similarly to SMPL, SMAL, etc.

You will need to set the base dataset directory path in "datasetFolder"
'''

import numpy as np
from os.path import join
from scipy.spatial.transform import Rotation as Rsci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle				
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import utils

# from utils import utils.GetLocalRotationsForFrame, utils.ReadBvhFile, utils.Plot3d


class DynaDog():
	def __init__(self, model_path, numShapePcs=-1):
	
		"""
		DynaDog model.

		Parameter:
		---------
		model_path: Path to the DyanDog model pickle file
		numShapePcs: int, optional, the number of principle components for the shape model

		"""
		
		
		with open(model_path, 'rb') as f:
			params = pickle.load(f, encoding='latin1')

			
		self.J_regressor = params['J_regressor'] # shape: (43, 2426)
		# note that the joint regressor is included in order to compare with SMAL
		# but is not used in the update() function
		
		self.skinning_weights = params['weights'] # shape: (2426,3)
		self.posedirs = params['posedirs'] # shape: (2426, 3, 378). 
		# These are all 0s and included only to match with SMAL
		
		self.v_template = params['v_template'] # shape: (2469, 3), mesh and vertices
		shapedirs = params['shapedirs'] # shape components, shape: (2504, 3, 17)
		if numShapePcs == -1:
			numShapePcs = shapedirs.shape[2]
		numShapePcs = min(numShapePcs, shapedirs.shape[2])
		self.numShapePcs = numShapePcs
		
		self.shapedirs = shapedirs[:,:,0:numShapePcs]
		self.faces = params['f']
		self.kintree_table = params['kintree_table']
		
		self.includeJointPosInModel = params['includeJointPosInModel']
		self.includeJointRotInModel = params['includeJointRotInModel']
		self.rotNormOpt_ret = params['rotNormOpt_ret']
		self.actualNumberOfRotations = self.rotNormOpt_ret['origBias'].shape[1]
		assert self.includeJointPosInModel and self.includeJointRotInModel
		
		
		# -------------- check if the mesh was scaled to a cube during pre-process stage -------------------
		vertNormOt_min = None
		vertNormOt_max = None
		meshScaledToCube = False
		if 'vertNormOt_ret' in params:
			vertNormOt_ret = params['vertNormOt_ret']
			meshScaledToCube = vertNormOt_ret['method'] == 'Normalise3dPoints'
			vertNormOt_min = vertNormOt_ret['min']
			vertNormOt_max = vertNormOt_ret['max']
		self.vertNormOt_min = vertNormOt_min
		self.vertNormOt_max = vertNormOt_max
		self.meshScaledToCube = meshScaledToCube


		id_to_col = {
			self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
		}
		self.parent = {
			i: id_to_col[self.kintree_table[0, i]]
			for i in range(1, self.kintree_table.shape[1])
		}
		self.numJoints = 43
		self.numVerts = 2426
		
		self.pose_shape = [self.numJoints, 3]
		self.beta_shape = [numShapePcs]
		self.trans_shape = [3]
		self.shoulderEarTrans_shape = [4,3]
		self.shoulderEarTrans_indices = [3, 9, 21, 23]

		self.pose = np.zeros(self.pose_shape)
		self.beta = np.zeros(self.beta_shape)
		self.rootTrans = np.zeros(self.trans_shape)
		self.shoulderEarTrans = np.zeros(self.shoulderEarTrans_shape)

		self.verts = None
		self.verts_neutral = None
		self.J = None
		self.J_neutral = None
		self.R = None
		self.preRot = None
		self.preRot_type = 'eul' # assume eul for now
		assert self.preRot_type == 'eul', 'self.preRot_type must be \'eul\' but is instead: %s' %  self.preRot_type
		
		# store a matrix of the "pre-rotation" of the dog, ie, the rotations that create the "flat" pose
		defaultPreRot = np.zeros((self.numJoints, 3))
		defaultPreRot[3,0] = -90
		defaultPreRot[9,0] = -90
		defaultPreRot[21,0] = 90
		defaultPreRot[23,0] = 90
		defaultPreRot[25,0] = -90
		defaultPreRot[30,0] = -90
		defaultPreRot_inv = defaultPreRot.copy() * -1 # since each bone has just one axis of rotation
		

		# convert from euler to rotation matrices
		defaultPreRot = Rsci.from_euler('xyz', defaultPreRot , degrees=True).as_dcm()
		defaultPreRot_inv = Rsci.from_euler('xyz', defaultPreRot_inv , degrees=True).as_dcm()
			
		# create "world" rotation matrices
		defaultPreRot_inv_world = np.empty((self.numJoints, 3, 3))
		defaultPreRot_inv_world[0] = defaultPreRot_inv[0]
		defaultPreRot_world = np.empty((self.numJoints, 3, 3))
		defaultPreRot_world[0] = defaultPreRot[0]
		for i in range(1, self.numJoints):
			defaultPreRot_inv_world[i] = defaultPreRot_inv_world[self.parent[i]].dot(defaultPreRot_inv[i])
			defaultPreRot_world[i] = defaultPreRot_world[self.parent[i]].dot(defaultPreRot[i])

		self.defaultPreRot_inv = defaultPreRot_inv
		self.defaultPreRot_inv_world = defaultPreRot_inv_world
		self.defaultPreRot_world = defaultPreRot_world
		self.defaultPreRot = defaultPreRot
	
		# load the weights for each shape
		shape_weights = None
		weights_path = model_path[:model_path.rfind('.')] + '_shapeWeights.p'
		if os.path.isfile(weights_path):
			with open(weights_path, 'rb') as f:
				weightParams = pickle.load(f, encoding='latin1')
				shape_weights = weightParams['toys_betas']
		self.shape_weights = shape_weights

		# by default, assume that the animated pose of the dog is in the form of rodrigues vectors
		self.pose_rotation_type = 'rodrigues'
	
	def set_params(self, pose=None, beta=None, rootTrans=None, shoulderEarTrans=None, pose_rotation_type=None):
		"""
		Set pose, shape, and/or translation parameters of SMPL model. Verices of the
		model will be updated and returned.

		Prameters:
		---------
		pose: Also known as 'theta', a [43,3] matrix indicating child joint rotation
		relative to parent joint. For root joint it's global orientation.
		Represented in the format specified by pose_rotation_type.

		beta: Parameter for model shape. A vector of shape [18]. Coefficients for
		PCA component

		rootTrans: Global translation of shape [3].

		shoulderEarTrans: [4,3] the translation values for additional offsets applied to the shoulders and ears of the dog
		
		pose_rotation_type: 'rodrigues' or 'quat', the type of rotation used for the pose of the dog
		
		Return:
		------
		Updated vertices.

		"""
		
		if pose is not None:
			self.pose = pose
		if beta is not None:
			self.beta = beta
		if rootTrans is not None:
			self.rootTrans = rootTrans
		if shoulderEarTrans is not None:
			self.shoulderEarTrans = shoulderEarTrans
		if pose_rotation_type is not None:
			self.pose_rotation_type = pose_rotation_type
		
		self.update()
		return self.verts

	def update(self):
		"""
		Called automatically when parameters are updated.

		"""
		produced = self.shapedirs.dot(self.beta) # produced.shape: (2504, 3)
		v_shaped = produced[:self.v_template.shape[0]] + self.v_template
		# extract the joint locations from the v_shaped array
		self.J = v_shaped[self.numVerts:,:]		
		v_shaped = v_shaped[0:self.numVerts,:]
		
		# rotation information from pose
		if self.pose_rotation_type == 'rodrigues':
			pose_cube = self.pose.reshape((-1, 1, 3))
			self.R = self.rodrigues(pose_cube)
		elif self.pose_rotation_type == 'quat':
			self.R = np.zeros((43,3,3))
			for rotIdx, rot in enumerate(self.pose):
				self.R[rotIdx] = Rsci.from_quat(rot).as_dcm()
		# self.R.shape (43, 3, 3)

		# create a matrix where the shoulderEarTrans have "trickled down" the kinematic chain
		allOffsets = np.zeros((self.numJoints, 3))
		allOffsets[self.shoulderEarTrans_indices,:] = self.shoulderEarTrans
		for i in range(1, self.numJoints):
			allOffsets[i,:] += allOffsets[self.parent[i],:]
					
		# ------------------- check to see if the mesh has been scaled to a cube -------------
		if self.meshScaledToCube:	
			v_shaped = np.multiply(v_shaped+1, self.vertNormOt_max-self.vertNormOt_min)
			v_shaped = v_shaped/2 + self.vertNormOt_min
			
			self.J = np.multiply(self.J+1, self.vertNormOt_max-self.vertNormOt_min)
			self.J = self.J/2 + self.vertNormOt_min
		
		
		# ---------------------------- apply pre-rotation from model, ie, the standing neutral pose ----------------------------
		# the end result of this is the dog skeleton and mesh in the standing neutral pose 
		preRot = produced[self.numVerts+self.numJoints:,:] # 35x3
		preRot = preRot[:,(1,2,0)] # reorder to be like matlab, where the model was created
		
		# turn from Nx3 matrix into N*3 x 1 matrix
		preRot = np.reshape(preRot, (preRot.shape[0]*3))
		# we might have added 1 or 2 extra 0's to make sure the amount was divisible by 3. Remove these
		preRot = preRot[0:self.actualNumberOfRotations] # shape = (104,3)

		# unnormalise preRot
		preRot = NormaliseRotations(preRot, opts=self.rotNormOpt_ret, unnormalise=1) # shape = (43,4, 1)
		# manipulate the quaternions to get the result we except to see
		preRot = preRot[:,(2,0,1,3),0] # shape = (43,4)
		preRot[:,3] *= -1
		rot = Rsci.from_quat(preRot).as_dcm()
		
		# the rotation is applied by first removing the "default pre-rotation", finally applying the "default pre-rotation" back on
		rot = np.matmul(rot, self.defaultPreRot_inv_world)
		rot = np.matmul(self.defaultPreRot_world, rot)

		G = np.empty((self.numJoints, 4, 4))
		G[0] = self.with_zeros(np.hstack((rot[0], self.J[0, :].reshape([3, 1]))))
		for i in range(1, self.numJoints):
			G[i] = G[self.parent[i]].dot(
			self.with_zeros(
				np.hstack(
				[rot[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
				)
			)
		)
		# get the joint locations out of G
		jnts_new = np.empty((self.numJoints,3))
		for i in range(self.numJoints):
			jnts_new[i,:] = G[i,0:3,3]
		self.J_neutral = jnts_new
		
		# new skinning
		# remove the transformation due to the rest pose. Note here that we're using self.J, which has the additional shoulder&ear offsets (if applicable)
		G = G - self.pack(
			np.matmul(
				G,
				np.hstack([self.J, np.zeros([self.numJoints, 1])]).reshape([self.numJoints, 4, 1])
				)
			)
			
		v_posed = v_shaped
		self.verts_neutral = v_shaped.copy()
		T = np.tensordot(self.skinning_weights, G, axes=[[1], [0]])
		rest_shape_h = np.hstack((v_shaped, np.ones([v_shaped.shape[0], 1])))
		# apply transformations to mesh
		self.verts_neutral = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
		# ------------  the dog is now in the standing neutral pose, for both the skeleton and mesh ------------

		
		

		# apply the pose rotation and translation
		# get the world transformation of each joint		
		jnts_current = self.J_neutral + allOffsets
		G = np.empty((self.kintree_table.shape[1], 4, 4))
		G[0] = self.with_zeros(np.hstack((self.R[0], self.J_neutral[0, :].reshape([3, 1]))))
		
		for i in range(1, self.kintree_table.shape[1]):
			G[i] = G[self.parent[i]].dot(
			self.with_zeros(
				np.hstack(
				[self.R[i],((jnts_current[i, :]-jnts_current[self.parent[i],:]).reshape([3,1]))]
				)
			)
			)
		# get the joint locations out of G
		jnts_new = np.empty((self.numJoints,3))
		for i in range(self.numJoints):
			jnts_new[i,:] = G[i,0:3,3]
		jnts_new = jnts_new + self.rootTrans.reshape([1, 3])
		self.J = jnts_new
		
		# remove the transformation due to the rest pose
		G = G - self.pack(
			np.matmul(
			G,
			np.hstack([self.J_neutral, np.zeros([self.numJoints, 1])]).reshape([self.numJoints, 4, 1])
			)
			)
		# transformation of each vertex
		T = np.tensordot(self.skinning_weights, G, axes=[[1], [0]])
		rest_shape_h = np.hstack((self.verts_neutral, np.ones([self.verts_neutral.shape[0], 1])))
		v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
		# apply root transformation
		self.verts = v + self.rootTrans.reshape([1, 3])
		

	def rodrigues(self, r):
		"""
		Rodrigues' rotation formula that turns axis-angle vector into rotation
		matrix in a batch-ed manner.

		Parameter:
		----------
		r: Axis-angle rotation vector of shape [batch_size, 1, 3].

		Return:
		-------
		Rotation matrix of shape [batch_size, 3, 3].

		"""
		theta = np.linalg.norm(r, axis=(1, 2), keepdims=True) # angle is the magnitude of the vector
		# avoid zero divide
		theta = np.maximum(theta, np.finfo(np.float64).tiny)
		r_hat = r / theta
		cos = np.cos(theta)
		z_stick = np.zeros(theta.shape[0])
		m = np.dstack([
			z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
			r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
			-r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
		).reshape([-1, 3, 3])
		i_cube = np.broadcast_to(
			np.expand_dims(np.eye(3), axis=0),
			[theta.shape[0], 3, 3]
		)
		A = np.transpose(r_hat, axes=[0, 2, 1])
		B = r_hat
		dot = np.matmul(A, B)
		R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
		return R

	def with_zeros(self, x):
		"""
		Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

		Parameter:
		---------
		x: Matrix to be appended.

		Return:
		------
		Matrix after appending of shape [4,4]

		"""
		return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

	def pack(self, x):
		"""
		Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
		manner.

		Parameter:
		----------
		x: Matrices to be appended of shape [batch_size, 4, 1]

		Return:
		------
		Matrix of shape [batch_size, 4, 4] after appending.

		"""
		return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

	def save_to_obj(self, path, scale=1):
		"""
		Save the DynaDog model to an .obj file. Also saves joints to text file

		Parameter:
		---------
		path: Path to save.
		scale: optional scale for the object and joints
		"""
		if not self.verts is None:
			verts = self.verts.copy() * scale
			with open(path, 'w') as fp:
				for v in verts:
					fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
				for f in self.faces + 1:
					fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

		if not self.J is None:
			path3 = path[0:-4] + '_jnts.txt'
			verts = self.J.copy() * scale
			with open(path3, 'w') as fp:
				for j in verts:
					fp.write('%f %f %f\n' % (j[0], j[1], j[2]))
'''
Function: NormaliseRotations

Parameters:
	applyRotToMesh - list, 3x1. Default makes kaya face the same direction as the SMAL model
	pathToJReg - string, full path to the joint regressor	
	pathToSkinning - string, full path to the skinning folder. We'll read mesh and skinning weights from this folder
	saveTo - string, full path to file where this model will be saved
	
Returns:
	True
'''		
def NormaliseRotations(data, opts=None, unnormalise=0):
	from scipy.spatial.transform import Rotation as R
	ret = {}
	withDofOnly = 0
	rotType = 'quat'
	centerAndVarTo1 = 1

	if opts is None:
		opts['withDofOnly'] = withDofOnly
		opts['rotType'] = rotType
		opts['centerAndVarTo1'] = centerAndVarTo1
		
	if 'withDofOnly' in opts:
		withDofOnly = opts['withDofOnly']
	if 'rotType' in opts:
		rotType = opts['rotType']
	if 'centerAndVarTo1' in opts:
		centerAndVarTo1 = opts['centerAndVarTo1']
		
	assert unnormalise # only this supported at the moment
	
	if unnormalise:
		# data is probably 1x104
		origBias = []
		origScale = []
		dof = np.ones((data.shape[0],2))
        
		if 'origBias' in opts:
			origBias = opts['origBias']
		if 'origScale' in opts:
			origScale = opts['origScale']
		if 'dof' in opts:
			dof = opts['dof']

        
		if not origScale == []:
			data = np.divide(data, origScale) # divide each entry in data with the corresponding entry in origScale
		if not origBias == []:
			data = data + origBias
		ind = np.nonzero(dof)
		ind = ind[0]
		data_new = np.zeros((data.shape[0], dof.shape[0]))
		
		for r in range(data_new.shape[0]):
			data_new[r,ind] = data[r,:]
 
		if rotType == 'quat':
			rotDim = 4
		else:
			rotDim = 3

		# reshape into numRots x rotDim x numFrames
		numFrames = data_new.shape[0]
		numRots = int(data_new.shape[1]/rotDim)
		
		data_new = np.reshape(data_new, (numRots,rotDim,numFrames))
		if rotType == 'quat':
			for i in range(numFrames):
				tmp = R.from_quat(data_new[:,:,i])
				data_new[:,:,i] = np.array(tmp.as_quat())

	return data_new
	
	
if __name__ == '__main__':


	datasetFolder = 'D:/DOG'
	modelPath = os.path.join(datasetFolder, 'shapeModel', 'dynaDog_v1.0.p')

	print('loading model from', modelPath)
	dynaDog = DynaDog(modelPath)
	
	# shape parameters
	beta = dynaDog.shape_weights[0,:]

	# pose parameters
	pose = np.zeros(dynaDog.pose_shape)
	rootTrans = np.zeros(dynaDog.trans_shape)
	shldr = np.zeros(dynaDog.shoulderEarTrans_shape)
	
	'''
	# Other examples:
	# dynaDog.set_params()
	
	# or 
	
	# read animation from bvh file. NOTE the translation for the shoulder and ears is not currently included in this feature
	applyMotionFrom_dog = 'dog1'
	applyMotionFrom_motion = 'trot'
	applyMotionFrom_frame = 0
	pathToBvh = os.path.join(datasetFolder, applyMotionFrom_dog, 'motion_%s'%applyMotionFrom_motion, 'motion_capture', 'skeleton.bvh')
	bvhPoints, connections, nodes = utils.ReadBvhFile(pathToBvh)
	pose = utils.GetLocalRotationsForFrame(nodes, applyMotionFrom_frame, asType='rodrigues')
	'''
	
	dynaDog.set_params(beta=beta, pose=pose, rootTrans=rootTrans, shoulderEarTrans=shldr)
	
	# save to obj file
	# dynaDog.save_to_obj('full/path/to/file.obj')

	# plot the dog in the animated pose
	fig = plt.figure(); ax = fig.add_subplot(111, projection='3d') 
	ax.scatter(dynaDog.verts[:,0], dynaDog.verts[:,1], dynaDog.verts[:,2], c="limegreen")
	ax, fig = utils.Plot3d(dynaDog.J, style='bo-', ax=ax, differentColoursForSides=True)
	ax.set_xlabel('x');ax.set_ylabel('y');ax.set_aspect('auto');plt.show()
	
	# plot the dog in the neutral standing pose
	'''
	fig = plt.figure(); ax = fig.add_subplot(111, projection='3d') 
	ax.scatter(dynaDog.verts_neutral[:,0], dynaDog.verts_neutral[:,1], dynaDog.verts_neutral[:,2], c="limegreen")
	ax, fig = utils.Plot3d(dynaDog.J_neutral, style='bo-', ax=ax, differentColoursForSides=True)
	ax.set_xlabel('x');ax.set_ylabel('y');ax.set_aspect('auto');plt.show()
	'''
		