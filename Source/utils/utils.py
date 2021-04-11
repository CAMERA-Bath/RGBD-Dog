import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

import json
import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as Rsci	
from scipy.io import loadmat

# import reader as reader
from utils import reader as reader
	
'''
Function: Plot2d

plot in 2d using opencv. Uint16 images are converted to uint8. Greyscale is converted to RGB

Parameters:
	img - numpy image or full path to image (string)
	projPoints - matrix, numBones x 2
	connections - array where each connections[i] is the index of i's parent
	clr - tuple, optional, default is (255, 255, 255). Colour for bones
	lineThickness - integer, optional, default is 1. Thickness in pixels for connecting bones
	plotJoints - bool, optional, default True. Plot the joints themselves not just bones
	differentColoursForSides - bool, optional, default False. Plots left bones in magenta
	mirIm - bool, optional, default False. Mirror the image
	markerClr - bool, optional, default is <clr>. Colour for markers
	leftRightClrs - list of tuples, optional, default is [(255,0,255), (255,0,0)]. Colours for plotting left and right as different colours
	radius - integer, optional, default is 2*lineThickness. Marker radius
	
Returns:

	image with plot
'''
def Plot2d(img, projPoints, connections=[], clr = (255, 255, 255), lineThickness=1, plotJoints=True, differentColoursForSides=False, mirIm=False, markerClr=None, leftRightClrs=None, radius=-1, visibility=[], jointSpecificClrs=[]):

	if connections == []:
		connections = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 2, 15, 16, 17, 18, 19, 19, 21, 19, 23, 0, 25, 26, 27, 28, 0, 30, 31, 32, 33, 0, 35, 36, 37, 38, 39, 40, 41]
	
	if visibility == []:
		visibility = [1]*43
		
	idx2name = ['root', 'spine01', 'spine02', 'lShoulder', 'lArm', 'lForearm', 'lWrist', 'lHand', 'lFinger', 
							'rShoulder', 'rArm', 'rForearm', 'rWrist', 'rHand', 'rFinger', 
							'neck01', 'neck02', 'neck03', 'neck04', 'head', 'nose', 'lEarInner', 'lEarOuter', 'rEarInner', 'rEarOuter',
							'lLeg', 'lLowerLeg', 'lAnkle', 'lFoot', 'lToe', 
							'rLeg', 'rLowerLeg', 'rAnkle', 'rFoot', 'rToe', 
							'tailBase', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail6', 'tailEnd']
							
	spineIsYellow = 0
	
	assert type(projPoints) is np.ndarray, 'the joints must be a numpy array'
	if len(projPoints.shape) == 3 and projPoints.shape[1] == 1 and projPoints.shape[2] == 2:
		projPoints = np.reshape(projPoints, (projPoints.shape[0],2))
	
	if markerClr is None:
		markerClr = clr
		
	if radius == -1:
		radius = 2*lineThickness
		
	# print('lineThickness', lineThickness, 'radius', radius)
	
	if leftRightClrs is None:
		leftRightClrs = [(255,0,255), (255,0,0)]
	# print('img.shape', img.shape)	
	numBones = projPoints.shape[0]	
	if not numBones == len(idx2name):
		differentColoursForSides=False	
		
	if img is None:
		# use matplotlib to plot on axis
		
		fig = plt.figure()
		
		# plot the bones
		for i, p in enumerate(projPoints):
			# print(p)
			cx = p[0]
			cy = p[1]
			
			parentId = connections[i]
			# if cx > -1 and cy > -1:
			if visibility[i] and visibility[parentId]:
				if i > 0: #not the root
					parentPoint = projPoints[parentId,:]
					px = parentPoint[0]
					py = parentPoint[1]

					if differentColoursForSides:
						boneName = idx2name[i]
						parentName = idx2name[connections[i]]
						
						if spineIsYellow:
							if boneName[0] == 'l' and parentName != 'root':
								plt.plot([cx, px], [cy, py], color=(leftRightClrs[0][0]/255, leftRightClrs[0][1]/255, leftRightClrs[0][2]/255), linewidth=lineThickness)
							elif boneName[0] == 'r' and parentName != 'root':
								plt.plot([cx, px], [cy, py], color=(leftRightClrs[1][0]/255, leftRightClrs[1][1]/255, leftRightClrs[1][2]/255), linewidth=lineThickness)
							else:
								plt.plot([cx, px], [cy, py], color=(0,1,1), linewidth=lineThickness)
						else:
							if boneName[0] == 'l':
								plt.plot([cx, px], [cy, py], color=(leftRightClrs[0][0]/255, leftRightClrs[0][1]/255, leftRightClrs[0][2]/255), linewidth=lineThickness)
							else:
								plt.plot([cx, px], [cy, py], color=(leftRightClrs[1][0]/255, leftRightClrs[1][1]/255, leftRightClrs[1][2]/255), linewidth=lineThickness)
					elif len(jointSpecificClrs) > 0:
						plt.plot([cx, px], [cy, py], color=(jointSpecificClrs[i][0]/255, jointSpecificClrs[i][1]/255, jointSpecificClrs[i][2]/255), linewidth=lineThickness)
					else:
						plt.plot([cx, px], [cy, py], color=clr, linewidth=lineThickness)	
							
		return plt
		
	else:
		if type(img) is str:
			#load the image
			img = cv2.imread(img, -1)
		if img.dtype == 'uint16':
			img = img.astype('float')/np.amax(img) # 0-1
			img = img*255
			img = img.astype('uint8')
		if mirIm:
			img = np.fliplr(img)
					
		if len(img.shape) == 2:
			# is greyscale. Turn into 3 channels for colour annotations
			img = np.stack((img,)*3, axis=-1)
			
		# plot the bones
		for i, p in enumerate(projPoints):
			# print(p)
			cx = int(round(p[0]))
			cy = int(round(p[1]))
			
			parentId = connections[i]
			# if cx > -1 and cy > -1:
			if cx > -1 and cy > -1 and cx < img.shape[1] and cy < img.shape[0] and visibility[i] and visibility[parentId]:
				if i > 0: #not the root
					parentPoint = projPoints[parentId,:]
					px = int(round(parentPoint[0]))
					py = int(round(parentPoint[1]))
					if px > -1 and py > -1 and px < img.shape[1] and py < img.shape[0]:
						if differentColoursForSides:
							boneName = idx2name[i]
							parentName = idx2name[connections[i]]
							
							if spineIsYellow:
								if boneName[0] == 'l' and parentName != 'root':
									img =cv2.line(img, (cx, cy), (px, py), leftRightClrs[0] ,lineThickness) # yellow?
								elif boneName[0] == 'r' and parentName != 'root':
									img =cv2.line(img, (cx, cy), (px, py), leftRightClrs[1] ,lineThickness) # blue?
								else:
									img =cv2.line(img, (cx, cy), (px, py), (0,255,255) ,lineThickness) # yellow
							else:
								if boneName[0] == 'l':
									img =cv2.line(img, (cx, cy), (px, py), leftRightClrs[0] ,lineThickness) # yellow?
								else:
									img =cv2.line(img, (cx, cy), (px, py), leftRightClrs[1] ,lineThickness) # blue?
						elif len(jointSpecificClrs) > 0:
							img =cv2.line(img, (cx, cy), (px, py), jointSpecificClrs[i],lineThickness)
						else:
							img =cv2.line(img, (cx, cy), (px, py), clr ,lineThickness)
							
		# plot the joint OVER the bone
		if plotJoints:
			for i, p in enumerate(projPoints):
				# print(p)
				cx = int(round(p[0]))
				cy = int(round(p[1]))
				# if cx > -1 and cy > -1:
				if cx > -1 and cy > -1 and cx < img.shape[1] and cy < img.shape[0]  and visibility[i]:
					if len(jointSpecificClrs) > 0:
						img = cv2.circle(img, (cx, cy), radius, jointSpecificClrs[i], lineThickness, 1)
					else:
						img = cv2.circle(img, (cx, cy), radius, markerClr, lineThickness, 1)
		return img

	
'''
Function: Plot2d_markers

plot markers and sticks on image. Markers connections is got using

vars = GetMocapData.GetData('kaya', 'walk', 1, 1)

markerNames =  vars['markerNames']  

markerConn =  vars['markerConn']  

Parameters:

	img - numpy image or full path to image (string)
	projPoints - matrix, numMarkers x 2
	connections - array where each connections[i] is the index of i's parent
	lineThickness - integer, optional, default is 1. Thickness in pixels for connecting sticks
	mirIm - bool, optional, default False. Mirror the image
	markerClr - bool, optional, default is [(255,0,255), (0,255,255), (255,255,0)]. Colour for markers
	radius - integer, optional, default is int(round(lineThickness*1.5)). Marker radius
	markerNames - array, optional, default []. The name of each marker
	indivMarkerClr - tuple, optional, default (0,255,255). Separate colour for all markers
	
Returns:

	image
'''
def Plot2d_markers(img=None, projPoints=[], connections=[], lineThickness=1,  mirIm=False, markerClr=None, radius=-1, markerNames=[], indivMarkerClr=None):
# note that markers can be plotted with Plot2d() if I want only a single colour and no connections between them

	assert( type(projPoints) is np.ndarray)
	assert( connections != [])

	if len(projPoints.shape) == 3 and projPoints.shape[1] == 1 and projPoints.shape[2] == 2:
		projPoints = np.reshape(projPoints, (projPoints.shape[0],2))


	if radius == -1:
		# radius = 2*lineThickness
		radius = int(round(lineThickness*1.5))
	# print('marker lineThickness', lineThickness, 'radius', radius)
	
	if markerClr is None:
		# left, centre, right
		# markerClr = [(255,0,255), (255,0,0), (0,255,0)] # magenta, blue, green
		# markerClr = [(0,0,255), (0,255,0), (255,0,0)] # red, green, blue
		# markerClr = [(0,0,255), (0,255,255), (255,0,0)] # red, yellow, blue
		markerClr = [(255,0,255), (0,255,255), (255,255,0)] # magenta, yellow, cyan
	if indivMarkerClr is None:
		indivMarkerClr = (0,255,255)
		
	if not img is None:
		if type(img) is str:
			#load the image
			img = cv2.imread(img, -1)
			if img.dtype == 'uint16':
				img = img.astype('float')/np.amax(img) # 0-1
				img = img*255
				img = img.astype('uint8')
		if mirIm:
			img = np.fliplr(img)

		if len(img.shape) == 2:
			# is greyscale. Turn into 3 channels for colour annotations
			img = np.stack((img,)*3, axis=-1)

	# print('img.shape', img.shape)	
	numMarkers = projPoints.shape[0]
	# plot the bones
	for i, p in enumerate(connections):
        
		#location
		p0 = projPoints[p[0],:]
		cx = int(round(p0[0]))
		cy = int(round(p0[1]))
			
			
		p1 = projPoints[p[1],:]
		px = int(round(p1[0]))
		py = int(round(p1[1]))
		
		#colour
		if not markerNames == []:
			# name
			name0 = markerNames[p[0]]
			name1 = markerNames[p[1]]
			if str.lower(name0[0]) == 'l' or str.lower(name1[0]) == 'l':
				clr = markerClr[0]
			elif str.lower(name0[0]) == 'r' or str.lower(name1[0]) == 'r':
				clr = markerClr[2]
			else:
				clr = markerClr[1]
		else:
			clr = markerClr[1]
			
		if px > -1 and py > -1 and px < img.shape[1] and py < img.shape[0] and cx > -1 and cy > -1 and cx < img.shape[1] and cy < img.shape[0]:
			img =cv2.line(img, (cx, cy), (px, py), clr,lineThickness) # blue?
		
        #print(p, name0, name1, clr, p0, p1)
        
	#plot the markers
	for i, p in enumerate(projPoints):
		name0 = markerNames[i]
		#colour
		# if str.lower(name0[0]) == 'l':
			# clr = markerClr[0]
		# elif str.lower(name0[0]) == 'r':
			# clr = markerClr[2]
		# else:
			# clr = markerClr[1]
		# clr = (0,255,255)
		
		cx = int(round(p[0]))
		cy = int(round(p[1]))
		if cx > -1 and cy > -1 and cx < img.shape[1] and cy < img.shape[0]:
			img = cv2.circle(img, (cx, cy), radius, indivMarkerClr, lineThickness, 1)
	return img
	
	
'''
Function: Plot3d

plot in 3d using matplotlib

Parameters:

	points - matrix, numJoints x 3
	connections - array where each connections[i] is the index of i's parent
	style - string, optional, default is 'bo-'. The style used for plotting
	ax - matplotlib axis, optional, default is [] and creates new figure.
	differentColoursForSides - bool, optional, default False. Plots left bones in magenta
	
Returns:

	ax, fig for matplotlib
'''
def Plot3d(points, connections=[], style='bo-', ax=[], differentColoursForSides=False, sideColours=[], jointSpecificClrs=[], markerStyle='o', lineStyle='-', jointNames=[], markerSize=6):

	numBones = points.shape[0]	
	if connections == []:
		if numBones == 43:
			connections = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 2, 15, 16, 17, 18, 19, 19, 21, 19, 23, 0, 25, 26, 27, 28, 0, 30, 31, 32, 33, 0, 35, 36, 37, 38, 39, 40, 41]
		elif numBones == 33: # SMAL
			connections = [-1, 0,1,2,3,4,5,6,7,8,9,6,11,12,13,6,15,0,17,18,19,0,21,22,23,0,25,26,27,28,29,30,16]
		elif numBones == 37: # BADJA
			connections = [-1, 0,1,2,3,4,5,6,7,8,9,6,11,12,13,6,15,0,17,18,19,0,21,22,23,0,25,26,27,28,29,30,16,16,16,16,16]	
			
	isMarkers = type(connections[0]) is list	
		# eg: [[0, 7], [0, 28], [36, 0], ...]
		
	jointNames = []
	if not isMarkers and numBones == 43:
		jointNames = ['root', 'spine01', 'spine02', 'lShoulder', 'lArm', 'lForearm', 'lWrist', 'lHand', 'lFinger', 
					'rShoulder', 'rArm', 'rForearm', 'rWrist', 'rHand', 'rFinger', 
					'neck01', 'neck02', 'neck03', 'neck04', 'head', 'nose', 'lEarInner', 'lEarOuter', 'rEarInner', 'rEarOuter',
					'lLeg', 'lLowerLeg', 'lAnkle', 'lFoot', 'lToe', 
					'rLeg', 'rLowerLeg', 'rAnkle', 'rFoot', 'rToe', 
					'tailBase', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail6', 'tailEnd']
								
	assert( type(points) is np.ndarray)
	fig = []
	if ax == []:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')		
		
	
	if differentColoursForSides and sideColours == []:
		sideColours = ['magenta', 'blue']
		
	if jointSpecificClrs != []:
		jnts = np.array(jointSpecificClrs)
		if np.amax(jnts) > 1:
			jnts = jnts/255
			jointSpecificClrs = jnts.tolist()
		
	if isMarkers:
		for i, p in enumerate(connections):
			# p = [N,M]
			bone = points[p[0],:]
			pBone = points[p[1],:]
			
			#colour
			if not jointNames == [] and differentColoursForSides:
				# name
				boneName = jointNames[p[0]]
				parentName = jointNames[p[1]]
				if boneName[0] == 'l' or parentName[0] == 'l':
					# img =cv2.line(img, (cx, cy), (px, py), (255,0,255) ,scale) # yellow?
					ax.plot([bone[0], pBone[0]], [bone[1], pBone[1]], [bone[2], pBone[2]], color=sideColours[0], marker=markerStyle, linestyle=lineStyle)
				else:
					# img =cv2.line(img, (cx, cy), (px, py), (255,0,0) ,scale) # blue?
					ax.plot([bone[0], pBone[0]], [bone[1], pBone[1]], [bone[2], pBone[2]], color=sideColours[1], marker=markerStyle, linestyle=lineStyle)
			elif len(jointSpecificClrs) > 0:
				ax.plot([bone[0], pBone[0]], [bone[1], pBone[1]], [bone[2], pBone[2]], color=jointSpecificClrs[b], marker=markerStyle, linestyle=lineStyle)
			else:
				ax.plot([bone[0], pBone[0]], [bone[1], pBone[1]], [bone[2], pBone[2]], style)
	else:
		# is skeleton
		if not numBones == len(jointNames):
			differentColoursForSides=False
			# we use bone name to decide which side of the body the joint is located. Can't do this if we don't have bone names
			
		for b in range(1, numBones):
			parent = connections[b]
			bone = points[b,:]
			pBone = points[parent,:]
			
			if differentColoursForSides:
				boneName = jointNames[b]
				parentName = jointNames[connections[b]]
				if boneName[0] == 'l' or parentName[0] == 'l':
					# img =cv2.line(img, (cx, cy), (px, py), (255,0,255) ,scale) # yellow?
					ax.plot([bone[0], pBone[0]], [bone[1], pBone[1]], [bone[2], pBone[2]], color=sideColours[0], marker=markerStyle, linestyle=lineStyle, markerSize=markerSize)
				else:
					# img =cv2.line(img, (cx, cy), (px, py), (255,0,0) ,scale) # blue?
					ax.plot([bone[0], pBone[0]], [bone[1], pBone[1]], [bone[2], pBone[2]], color=sideColours[1], marker=markerStyle, linestyle=lineStyle, markerSize=markerSize)
			elif len(jointSpecificClrs) > 0:
				ax.plot([bone[0], pBone[0]], [bone[1], pBone[1]], [bone[2], pBone[2]], color=jointSpecificClrs[b], marker=markerStyle, linestyle=lineStyle, markerSize=markerSize)
			else:
				ax.plot([bone[0], pBone[0]], [bone[1], pBone[1]], [bone[2], pBone[2]], style, markerSize=markerSize)
		
		
		
	ax.set_aspect('auto')
	# ax.set_aspect('equal') # doesn't work...
	
	return ax, fig
	
	
'''
Function: GetDefaultColours

Get the default bone-specific colours for the skeleton

Parameters:

	numJoints - int, the number of joints in the skeleton. Default 43
	
Returns:

	jointColours - list of RGB colours (each colour is a tuple)
	
'''	
def GetDefaultColours(numJoints=43):
	clr = ColourNameToNum()
	jointColours = []
	if numJoints==43:
		jointColours = [clr['red'], clr['red'], clr['red'], clr['cyan'], clr['cyan'], clr['cyan'], clr['cyan'], clr['cyan'], clr['cyan'], clr['purple'], clr['purple'], clr['purple'], clr['purple'], clr['purple'], clr['purple'], clr['green'], clr['green'], clr['green'], clr['green'], clr['green'], clr['green_dark'], clr['green_dark'], clr['green_dark'], clr['green_dark'], clr['green_dark'], clr['blue'], clr['blue'], clr['blue'], clr['blue'], clr['blue'], clr['magenta'], clr['magenta'], clr['magenta'], clr['magenta'], clr['magenta'], clr['yellow_dark'], clr['yellow_dark'], clr['yellow_dark'], clr['yellow_dark'], clr['yellow_dark'], clr['yellow_dark'], clr['yellow_dark'], clr['yellow_dark']]
	return jointColours
	
def ColourNameToNum():
	d = {'red': (255,0,0), 'green': (0,255,0), 'blue':(0,0,255), 'yellow':(255,255,0), 'cyan':(0,255,255), 'magenta':(255,0,255), 'purple':(127,0,127), 'green_dark':(0,127,127), 'yellow_dark':(255, 201, 14)}
	return d	
	
'''
Function: ReadBvhFile

load a bvh file

Parameters:

	filname - full path to bvh file
	onTheSpot - bool, default False. If True, root is fixed at 0
	scale - integer, default 1. Scale of skeleton
	transposeAfterReading - transpose the joint positions and rotations after reading the bvh file, if True

Returns:

	bvhPoints - matrix size numJoints x 3 x numFrames
	connections - array where each connections[i] is the index of i's parent
	nodes - nodes from BvhReader
'''	
def ReadBvhFile(filname, onTheSpot=False, scale=1, transposeAfterReading=True):

	readInst = reader.MyReader(filname, onTheSpot, scale)
	dt, bvhPoints, limits, nodes, data = readInst.read()
	
	# data is a list of lists. len(data) = numFrames, len(data[0]) = numJoints
	skelData = {}
	for i,n in enumerate(nodes):
		joint = {}
		joint['name'] = n.name
		joint['children'] = n.childrenIdx
		joint['parent'] = n.parentIdx
		joint['order'] = n.order
		joint['offset'] = n.offset
		
		rotation_mat = []
		rotation_local_euler = []
		rotation_local = []
		position = []
		for d in data:
			jointData = d[i]
			position.append(jointData.position)
			rotation_local_euler.append(jointData.rotation_local_euler)
			rotation_local.append(jointData.rotation_local)
			rotation_mat.append(jointData.rotation)
		
	
	# print('data[0]', data[0])
	connections = []
	for n in nodes:
		#a line can be drawn from this node to the node at n.parentIdx
		connections.append(n.parentIdx)
	
	
	nodeNames = []
	for n in nodes:
		childrenIdx = []
		nodeNames.append(n.name)
		# print(n.name, ', children: ', n.children)
		# print(n.name, n.idx, n.parentIdx)
		
		currNodeIdx = n.idx
		parentIdx = n.parentIdx
		if parentIdx >= 0:
			nodes[parentIdx].childrenIdx.append(currNodeIdx)
		
	# for n in nodes:
		# print(n.idx, n.name, ', children: ', n.childrenIdx)
		# Eg:
		# (0, 'Root', ', children: ', [1, 6, 11, 14])
		# (1, 'LeftLeg', ', children: ', [2])
		# (2, 'LeftLowerLeg', ', children: ', [3])

	
	# print(connections)
	
	# call JsonToNumpyArr.MovePointsOutOfMayaCoordSystem(bvhPoints, 1) instead. See ProjectMarkersOnSingleVideo.py
	# rot90X = np.array(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
	# bvhPoints = np.matmul(rot90X, bvhPoints)
	
	
	
	if transposeAfterReading:
		bvhPoints = np.transpose(bvhPoints)
		
		numFrames = nodes[0].rotation_world.shape[0]
		# transpose rotations
		for node in nodes:
			# rotation_world
			r = node.rotation_world.copy()
			r = np.reshape(r, (numFrames,9))
			r = r[:,[0,3,6,1,4,7,2,5,8]] # transpose
			node.rotation_world = np.reshape(r, (numFrames,3,3))
			
			# rotation_local
			r = node.rotation_local.copy()
			r = np.reshape(r, (numFrames,9))
			r = r[:,[0,3,6,1,4,7,2,5,8]] # transpose
			node.rotation_local = np.reshape(r, (numFrames,3,3))

			for i in range(numFrames):
				r = Rsci.from_dcm(node.rotation_local[i,:,:])
				r = r.as_euler('zyx', degrees=True)
				#print('r')
				node.rotation_local_euler[i,:] = r

	# bvhPoints is now : numJoints x 3 x numFrames
	
	return bvhPoints, connections, nodes
	
'''
Using the nodes from ReadBvhFile, extract the local rotations as either quaternions or rodrigues vectors
by default, the order is altered, to work with DynaDog.py
if returning the 3x3 rotation matrices, the order of rotation is not reordered
'''
def GetLocalRotationsForFrame(nodes, frame, asType='quat', reorder=True):
	assert asType in ['quat', 'rodrigues', 'dcm'], 'unsupported type %s' % asType
    
	numJoints = len(nodes)
	if asType == 'quat':
		rot = np.empty((numJoints,4))
		order = np.array((0,1,2,3))
		if reorder:
			order = np.array((2,0,1,3))
			
		for node in nodes:
			rot_local = np.array(node.rotation_local)
			r = Rsci.from_dcm(rot_local[frame]).as_quat()
			rot[node.idx] = r[order]
	elif asType == 'rodrigues':
		rot = np.empty((numJoints,3))
		order = np.array((0,1,2))
		if reorder:
			order = np.array((2,0,1))
			
		for node in nodes:
			rot_local = np.array(node.rotation_local)
			r = Rsci.from_dcm(rot_local[frame]).as_rotvec()
			rot[node.idx] = r[order]
	else:
		rot = np.empty((numJoints,3,3))
		for node in nodes:
			rot_local = np.array(node.rotation_local)
			rot[node.idx] = rot_local[frame]
	return rot
	
'''
Using the nodes from ReadBvhFile, extract the world rotations as either quaternions or rodrigues vectors
by default, the order is altered, to work with DynaDog.py
if returning the 3x3 rotation matrices, the order of rotation is not reordered
'''
def GetWorldRotationsForFrame(nodes, frame, asType='quat', reorder=True):
	assert asType in ['quat', 'rodrigues', 'dcm'], 'unsupported type %s' % asType

	numJoints = len(nodes)
	if asType == 'quat':
		rot = np.empty((numJoints,4))
		order = np.array((0,1,2,3))
		if reorder:
			order = np.array((2,0,1,3))
			
		for node in nodes:
			rot_world = np.array(node.rotation_world)
			r = Rsci.from_dcm(rot_world[frame]).as_quat()
			rot[node.idx] = r[order]
	elif asType == 'rodrigues':
		rot = np.empty((numJoints,3))
		order = np.array((0,1,2))
		if reorder:
			order = np.array((2,0,1))
			
		for node in nodes:
			rot_world = np.array(node.rotation_world)
			r = Rsci.from_dcm(rot_world[frame]).as_rotvec()
			rot[node.idx] = r[order]
	else:
		rot = np.empty((numJoints,3,3))
		for node in nodes:
			rot_world = np.array(node.rotation_world)
			rot[node.idx] = rot_world[frame]
	return rot
	
	
'''
Function: GetFilesInFolder
get all files of type <fileType>. Returns list of [imageName.ext, ...]

Parameters:
	myPath - string, full path to file
	fileType - string, optional, default ''. Get only files with this extension
	contains - string, optional, default ''. Get only files that contain this in the name
Returns:
	files - list of strings. filename.EXT only
'''
def GetFilesInFolder(myPath, fileType='', contains=''):

	contains = contains.lower()
	
	if fileType == '' and contains == '':
		files = [f for f in os.listdir(myPath) if os.path.isfile(os.path.join(myPath, f))]
	elif not fileType == '' and contains == '':
		files = [f for f in os.listdir(myPath) if os.path.isfile(os.path.join(myPath, f)) and f[-len(fileType):].lower() == fileType.lower()]
	elif fileType == '' and not contains == '':
		files = [f for f in os.listdir(myPath) if os.path.isfile(os.path.join(myPath, f)) and contains in f.lower()]	
	else:
		files = [f for f in os.listdir(myPath) if os.path.isfile(os.path.join(myPath, f)) and contains in f.lower() and f[-len(fileType):] == fileType]	
	return files
	
	
	
#move along the video to frame <frameNumber>. A similar function to SetVideoToTimeCode()
def SetVideoToFrameNumber(vidFile, frameNumber):

	ret = False
	if isinstance(vidFile, type(cv2.VideoCapture(0))):
		ret = vidFile.set(cv2.CAP_PROP_POS_FRAMES,frameNumber)
	else:
		vidFile = cv2.VideoCapture(vidFile)
		ret = vidFile.set(cv2.CAP_PROP_POS_FRAMES,frameNumber)
		# vidFile.release()
	return [ret, vidFile]
	
	
	
'''
Function: GetCalibData

Load all calibration files found in pathToCalibs. Files have the name calibFile%s, where %s comes from cams variable 
Parameters:

	pathToCalibs - (1) string or (2) list of strings, each being the full folder to where the calib files are stored	
	cams - list of strings, each is camera name

Returns:
	calibData_allCalibs_allCams - 2d array, where the first index is the same length as pathToCalibs, second index of length cams
'''		
def GetCalibData(pathToCalib, cams, prnt=False):
# pathToCalibs is a list of string, each being the full folder to where the calib files are stored	
	calibData_allCalibs_allCams = []
	
	isList = type(cams) is list
	if not isList:
		cams = [cams]
		
	if not 'calibFile%s' in pathToCalib:
		pathToCalib = os.path.join(pathToCalib, 'calibFile%s')
	calibData_allCams = []
	for c in cams:
		pathToCalibFile = pathToCalib % c
		assert os.path.isfile(pathToCalibFile), 'no file for %s' % pathToCalibFile

		if prnt:
			print('getting calibraton from %s...' % pathToCalibFile)
		# calib = ReadCalibFile(pathToCalibFile, 10)
		calib = ReadCalibFile(pathToCalibFile)
		# calib = ReadCalibFile(pathToCalibFile, focalLengthScale=1)
		if prnt:
			print('..done')	

		L = np.asarray(calib['L'])
		rot = L[0:3,0:3]
		[rot3x1, rot9x3] = cv2.Rodrigues(rot)
		t = L[0:3,3]
		distCoeffs = np.asarray([calib['k1'], calib['k2'], calib['p1'], calib['p2'], calib['k3']])
		K = np.asarray(calib['K'])
		calibData = {}
		calibData['calib'] = calib
		calibData['K'] = K
		calibData['rot3x1'] = rot3x1
		calibData['t'] = t
		calibData['distCoeffs'] = distCoeffs
		calibData_allCams.append(calibData)
	
	if not isList:
		calibData_allCams = calibData_allCams[0]
		
	return calibData_allCams
	
'''
Function: ReadCalibFile

Load calibration file to dict

Parameters:
	fullPathToCalibFile - string, full path to calib file
	translationScale - integer, optional, default 1. Scale applied to camera position
	additionalRot - list, optional, default [0,0,0]. Additional rotation applied to camera rotation
	asKinectInt - bool, optional, default False. Set intrinsics to dummy Kinect values

Returns:
	finalObject - dict
	finalObject.imWidth - 
	finalObject.imHeight - 
	finalObject.L - after applying translationScale and additionalRot
	finalObject.L_orig - directly from file
	finalObject.K - 
	finalObject.k1 - 
	finalObject.k2 - 
	finalObject.k3 - 
	finalObject.p1 - 
	finalObject.p2 -
'''			
def ReadCalibFile(fullPathToCalibFile, translationScale = 1, additionalRot = [0,0,0], asKinectInt=False, focalLengthScale=1):
	#eg fullPathToCalibFile = 'C:/Users/Sinead/Documents/data/Cats_And_Dogs_Home_Shoot_01/Kaya/data_final/calib/calibFile00'

	f = open(fullPathToCalibFile, 'r')

	#line 1
	imWidth = f.readline()
	imWidth = imWidth[0:-1]
	#line 2
	imHeight = f.readline()
	imHeight = imHeight[0:-1]

	#line 3
	line = f.readline()
	lineSplit = line.split(' ')
	fx = float(lineSplit[0]) * focalLengthScale
	fy = fx
	cx = lineSplit[2]

	#line 4
	line = f.readline()
	lineSplit = line.split(' ')
	cy = lineSplit[2]

	line = f.readline()
	line = f.readline()

	#line 7
	line = f.readline()
	lineSplit = line.split(' ')
	L11 = lineSplit[0]
	L12 = lineSplit[1]
	L13 = lineSplit[2]
	L14 = lineSplit[3]


	#line 8
	line = f.readline()
	lineSplit = line.split(' ')
	L21 = lineSplit[0]
	L22 = lineSplit[1]
	L23 = lineSplit[2]
	L24 = lineSplit[3]

	#line 9
	line = f.readline()
	lineSplit = line.split(' ')
	L31 = lineSplit[0]
	L32 = lineSplit[1]
	L33 = lineSplit[2]
	L34 = lineSplit[3]

	#line 10
	line = f.readline()
	lineSplit = line.split(' ')
	L41 = lineSplit[0]
	L42 = lineSplit[1]
	L43 = lineSplit[2]
	L44 = lineSplit[3]

	line = f.readline()


	#line 11
	line = f.readline()
	lineSplit = line.split(' ')
	k1 = lineSplit[0]
	k2 = lineSplit[1]
	p1 = lineSplit[2]
	p2 = lineSplit[3]
	k3 = lineSplit[4]

	f.close()

	L = [[float(L11), float(L12), float(L13), float(L14)], 
		[float(L21), float(L22), float(L23), float(L24)],
		[float(L31), float(L32), float(L33), float(L34)],
		[float(L41), float(L42), float(L43), float(L44)]]
	L_orig = L
	
	if (not translationScale == 1) or (not additionalRot == [0,0,0]):
	#apply an additional rotation and/or translation to the camera
		Linv = np.linalg.inv(np.asarray(L))
		cameraRot = Linv[0:3,0:3]
		addRot = euler_matrix(additionalRot[0], additionalRot[1], additionalRot[2], 'sxyz')
		cameraRot = addRot.dot(cameraRot)
		cameraPos = addRot.dot(Linv[0:3,3]) * translationScale
		Linv[0:3,3] = cameraPos;
		Linv[0:3,0:3] = cameraRot;
		L = np.linalg.inv(np.asarray(Linv))
		L = L.tolist()
	
	if asKinectInt:
		imWidth = 640
		imHeight = 480
		K = [[600.0, 0.0, 320.0],
			[0.0, 600.0, 240.0],
			[0.0, 0.0, 1.0]]
	else:
		K = [[float(fx), 0.0, float(cx)],
			[0.0, float(fy), float(cy)],
			[0.0, 0.0, 1.0]]
		
	finalObject = {'imWidth': float(imWidth), 'imHeight' :float(imHeight), 'L':L, 'L_orig':L_orig, 'K':K,
		'k1':float(k1), 'k2':float(k2), 'k3':float(k3), 'p1':float(p1), 'p2':float(p2)}
		
	return finalObject

	
def pixel2world(x, y, z, cx, cy, fx, fy):
		w_x = (x - cx) * z / fx
		w_y = (y - cy) * z / fy
		w_z = z
		return w_x, w_y, w_z
			
def ReconstructDepthImage(image, K, invalidVal=None):
	
	if type(image) is str:
		image = cv2.imread(image,  -1)
		
	assert str(image.dtype) == 'uint16', 'image type must be np.uint16 but is type %s' % str(image.dtype)
	
	h, w = image.shape
	x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
	
	points = np.zeros((h, w, 3), dtype=np.float32)
	# the K for all batched images is the same
	points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, K[0,2], K[1,2], K[0,0], K[1,1])
	points = points.reshape((-1, 3))
	
	if invalidVal is not None:
		# remove all points where the z value == invalidVal
		points = points[points[:,2] != invalidVal]
	
	return points
	
	
def MovePointsOutOfMayaCoordSystem(points3D, scale=1):

	rot90X = np.array(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
	#the points where generated in Maya -> multiply by 10 and rotate about x-axis by 90 degrees

	points3D = points3D*scale
	if points3D.shape[0] != 3:
		points3D = np.matmul(rot90X, np.transpose(points3D))
		return np.transpose(points3D)
	else:
		points3D = np.matmul(rot90X, points3D)
		return points3D
		
		
# filename = 'C:/Users/Sinead/Desktop/temp.json'
#return names, points
def GetPointsFromJsonFbx(filename):
	print('reading %s' % filename)
	j = json.load(open(filename, 'r'))
	# j = json.dumps(j, indent=4, sort_keys=True) #make sure the keys are sorted, include indentation to make it easier to read when printing
	# print(j)

	numPoints = int(round(len(j)/3)) #each entry in j is "___.TranslateX", "__.TranslateY", "____.TranslateZ" -> divide by 3 to get number of points

	namesAll = []
	for entry in j:
		namesAll.append(str(entry))
	namesAll.sort() #sorts a-z
		
	# print(namesAll)
	names = [] #unique, will not contain "____.TranslateX", etc

	#https://stackoverflow.com/questions/2990121/how-do-i-loop-through-a-list-by-twos?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	if sys.version_info[0] == 2:
		for i in xrange(0,len(namesAll),3): #in steps of 3
			n = namesAll[i]
			# print(n[0:n.find('.')])
			names.append(n[0:n.find('.')])
	else:
		for i in range(0,len(namesAll),3): #in steps of 3
			n = namesAll[i]
			# print(n[0:n.find('.')])
			names.append(n[0:n.find('.')])
	# print(names)


	numFrames = len(j[entry])
	# print('numFrames', numFrames)
	points = np.zeros((numPoints,3,numFrames))
			
	pointIdx = 0

	idx = 0
	for entry in namesAll:
		p = j[entry]
		
		if entry.find('translateX') >= 0:
			points[idx,0,:] = p
		elif entry.find('translateY') >= 0:
			points[idx,1,:] = p
		elif entry.find('translateZ') >= 0:
			points[idx,2,:] = p
			idx = idx+1
		else:
			assert False, 'ERROR reading marker file, entry is not X, Y or Z, entry=%s' % entry
	
	return names, points
	
	
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
	'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
	'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
	'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
	'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
	'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
	'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
	'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
	'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def euler_matrix(ai, aj, ak, axes='sxyz'):
	"""Return homogeneous rotation matrix from Euler angles and axis sequence.

	ai, aj, ak : Euler's roll, pitch and yaw angles
	axes : One of 24 axis sequences as string or encoded tuple

	>>> R = euler_matrix(1, 2, 3, 'syxz')
	>>> np.allclose(np.sum(R[0]), -1.34786452)
	True
	>>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
	>>> np.allclose(np.sum(R[0]), -0.383436184)
	True
	>>> ai, aj, ak = (4*math.pi) * (np.random.random(3) - 0.5)
	>>> for axes in _AXES2TUPLE.keys():
	...	R = euler_matrix(ai, aj, ak, axes)
	>>> for axes in _TUPLE2AXES.keys():
	...	R = euler_matrix(ai, aj, ak, axes)

	"""
	
	ai = math.radians(ai)
	aj = math.radians(aj)
	ak = math.radians(ak)
	axes = axes.lower()

	try:
		firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
	except (AttributeError, KeyError):
		_TUPLE2AXES[axes]  # validation
		firstaxis, parity, repetition, frame = axes

	i = firstaxis
	j = _NEXT_AXIS[i+parity]
	k = _NEXT_AXIS[i-parity+1]

	if frame:
		ai, ak = ak, ai
	if parity:
		ai, aj, ak = -ai, -aj, -ak

	si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
	ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
	cc, cs = ci*ck, ci*sk
	sc, ss = si*ck, si*sk

	M = np.identity(4)
	if repetition:
		M[i, i] = cj
		M[i, j] = sj*si
		M[i, k] = sj*ci
		M[j, i] = sj*sk
		M[j, j] = -cj*ss+cc
		M[j, k] = -cj*cs-sc
		M[k, i] = -sj*ck
		M[k, j] = cj*sc+cs
		M[k, k] = cj*cc-ss
	else:
		M[i, i] = cj*ck
		M[i, j] = sj*sc-cs
		M[i, k] = sj*cc+ss
		M[j, i] = cj*sk
		M[j, j] = sj*ss+cc
		M[j, k] = sj*cs-sc
		M[k, i] = -sj
		M[k, j] = cj*si
		M[k, k] = cj*ci
	return M[0:3,0:3]
	
'''
Function: LoadMatFile
load a .mat file specified by fullPath

Parameters:
	fullPath - string, full path to file
Returns:
	data from file
'''
def LoadMatFile(fullPath):
	mat = loadmat(fullPath)
	ky_final = ''
	for ky in mat.keys():
		if not ky[0:2] == '__':
			ky_final = ky
	assert not ky_final == '', 'no valid key found'
	return mat[ky_final]	
	
	
'''
Function: EnsureObjFacesInCorrectFormat

objloader can't handle a face line in the format: "f 787/1/1 788/2/2 279/3/3 5/4/4"

this line is changed to two lines: "f 787/787/787 788/788/788 279/279/279" and "f 787/787/787 279/279/279 5/5/5"

The current file is given the suffix "_OLD" and new contenst are written to the current filename

Parameters:

	pathToObj - string, full path to .obj file
	
Returns:

	writeFile - boolean, True is file was altered
	
'''
def EnsureObjFacesInCorrectFormat(pathToObj):
	assert sys.version_info[0] >= 3, 'only available in Python3'
	from objloader import Obj
	ob = Obj.open(pathToObj)

	writeFile = True
	keepReading = True
	content = []
	with open(pathToObj, 'r') as fp:
		line = fp.readline()
		cnt = 1
		while line and keepReading:
			#print('line', line)
			if line[0] == 'f':
				splt = line.split('/')
				numSlashes = len(splt) -1
				if numSlashes == 8:
					# splt eg: ['f 1931', '87', '93 1091', '88', '94 1171', '89', '95 1971', '90', '96\n']
					#print('splt',  splt)
					num1 = int(splt[0][2:])
					num2_temp = int(splt[1])
					if num1 == num2_temp:
						# the line is in the format 1/1/1 2/2/2 3/3/3
						# this means we don't need to do anything to it
						keepReading = False
						writeFile = False
					else:
						a = splt[2]
						num2 = int(a[a.find(' ')+1:])
						a = splt[4]
						num3 = int(a[a.find(' ')+1:])
						a = splt[6]
						num4 = int(a[a.find(' ')+1:])
						s1 = 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (num1, num1, num1, num2, num2, num2, num3, num3, num3)
						s2 = 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (num1, num1, num1, num3, num3, num3, num4, num4, num4)
						content.append(s1)
						content.append(s2)
				elif numSlashes == 6:
					# splt eg: ['f 917', '512', '797 916', '513', '798 922', '514', '799\n']
					# print('splt',  splt)
					num1 = int(splt[0][2:])
					num2_temp = int(splt[1])
					if num1 == num2_temp:
						# the line is in the format 1/1/1 2/2/2 3/3/3
						# this means we don't need to do anything to it
						keepReading = False
						writeFile = False
					else:
						a = splt[2]
						num2 = int(a[a.find(' ')+1:])
						a = splt[4]
						num3 = int(a[a.find(' ')+1:])
						s1 = 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (num1, num1, num1, num2, num2, num2, num3, num3, num3)
						content.append(s1)
						content.append(s2)
				else:
					# print('numSlashes', numSlashes)
					# print('line', line)
					keepReading = False
					writeFile = False
					# the file is in a correct format. Don't need to keep reading
			else:
				content.append(line)
			line = fp.readline()
			cnt += 1
		
	newFilename = pathToObj
	if writeFile:
		oldFilename = pathToObj
		pathOnly = GetPathFromFullFilename(oldFilename)
		filenameOnly = GetFilenameFromPath(oldFilename)
		# extension is assumed to be '.obj'
		
		newFilename = join(pathOnly, filenameOnly + '_OLD.obj')
		rename(oldFilename, newFilename)
		
		with open(oldFilename, 'w') as fp:
			for c in content:
				fp.write(c)
		print('old file at ', newFilename)
		print('new file at ', oldFilename)
	return writeFile, newFilename
	
	
'''
Function: LoadObjFile

load the obj file. Returns an object similar to that in Matlab

Parameters:

	fullPath - string, full path to .obj file
	
Returns:

	obj - dict
	obj['vertices'] - numpy array, Nx3
	obj['faces'] - numpy array, Mx3
	obj['vertex_normals'] - numpy array, Nx3
	obj['texture'] - numpy array, Nx2
	
	obj - dict
	obj['vertices'] - numpy array, Nx3
	obj['vertices_normal'] - numpy array, Nx3
	obj['vertices_texture'] - numpy array, Nx2
	
	obj['objects']['data']['vertices'] - faces, numpy array, Px3
	objects['type'] - 'f'
'''
def LoadObjFile(fullPath, rotate=False):
	assert sys.version_info[0] >= 3, 'only available in Python3'

	from objloader import Obj
	
	
	newFileCreated, fullPath = EnsureObjFacesInCorrectFormat(fullPath)
	ob = Obj.open(fullPath)
	verts = np.array(ob.vert)
	if rotate:
		# use this to make sure the neutral mesh aligns with 'bvhJoints_neutral' in GetMocapData.GetData()
		verts = verts[:,(0,2,1)]
		verts[:,1] *= -1
		# bvhJoints_neutral = JsonToNumpyArr.MovePointsOutOfMayaCoordSystem(bvhJoints[:,:,0])
		
	norms = np.array(ob.norm)
	texture = np.array(ob.text)
	# texture is Nx3, make Nx2
	texture = texture[:,0:2]
	
	faces = ob.face
	if faces[0][0] == faces[0][1] and faces[0][0] == faces[0][2]  or faces[0][1] is None:
		# faces in the file are in the format 'f 736/736/736 40/40/40 46/46/46', giving 3 entries per face. 
		# Change it to be 1 entry of '736/40/46'
		faces = [face[0] for face in ob.face]
		numFaces = len(faces)
		# print('numFaces', numFaces)
		faces = np.array(faces)
		faces = np.reshape(faces, ((int(numFaces/3), 3)))
		# print('faces.shape', faces.shape)
	else:
		# print('else, faces[0]', faces[0])
		faces = np.array(faces)
		
	faces -= 1 # start at 0

	# create dict that has the same format as reading an obj, LoadObj, in Matlab
	obj = {}
	obj['vertices'] = verts
	obj['vertices_normal'] = norms
	obj['vertices_texture'] = texture
	
	objects = {}
	objects['type'] = 'f'
	data = {}
	data['vertices'] = faces
	objects['data'] = data
	obj['objects'] = objects
	return obj
	
	
	
def TestSkinning(dog='kaya', motion='trot'):

	assert sys.version_info[0] >= 3, 'object loading only supported in Python 3'
	
	useV2 = 1
	data_dog = GetMocapData.GetData(dog, motion, 1,1, useV2=useV2)
	pathStart = data_dog['pathToDataFinal_withMotion']
	pathToVid = data_dog['pathToVid']
	bvhSkelFile = data_dog['bvhSkelFile']
	skelIdxStart = data_dog['skelIdxStart']
	sonyIdxStart = data_dog['sonyIdxStart']
	pathToCalib = data_dog['pathToCalibs'][0]
	tcStart = data_dog['tcStart']
	bvhPoints = data_dog['bvhJoints']
	bvhJoints_neutral = data_dog['bvhJoints_neutral']
	skelConnections = data_dog['skelConnections']
	
	pathToSkinningWeights = data_dog['pathToSkinningWeights']
	pathToSuitMesh = data_dog['pathToSuitMesh']
	if pathToSuitMesh == '':
		pathToSuitMesh = data_dog['pathToFurMesh']
	assert not pathToSuitMesh == '', 'no obj file found for ' + dog +', '+ motion
	
	points = bvhJoints_neutral
	
	suitMesh = PathFolderFunctions.LoadObjFile(pathToSuitMesh, rotate=True)
	verts_neutral = suitMesh['vertices'] * 100
	
	'''
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')	
	
	# plot skeleton
	ax, fig = Plot3d(points, connections=skelConnections, style='bo-', ax=ax, differentColoursForSides=False)
	
	# for v in verts_neutral:
	ax.plot(verts_neutral[:,0], verts_neutral[:,1], verts_neutral[:,2], 'ro-')
		
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	
	plt.show()
	'''
	s = time()
	kintree_table = np.array([[4294967295,0,1,2,3,4,5,6,7,2,9,10,11,12,13,2,15,16,17,18,19,19,21,19,23,0,25,26,27,28,0,30,31,32,33,0,35,36,37,38,39,40,41],
							 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]])
	numJoints = kintree_table.shape[1]
	id_to_col = { kintree_table[1, i]: i for i in range(kintree_table.shape[1])	}
	parent = {
		i: id_to_col[kintree_table[0, i]]
		for i in range(1, kintree_table.shape[1])
	}
	# parent is a dictionary, the entry for each index is the parent of index
		
	weights = PathFolderFunctions.LoadMatFile(pathToSkinningWeights)
	# J should be the joints in the neutral position for the dog
	
	J = bvhJoints_neutral
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
	G[0] = with_zeros(np.hstack((rot[0], J[0, :].reshape([3, 1]))))
	for i in range(1, numJoints):
		G[i] = G[parent[i]].dot(
			with_zeros(
				np.hstack( [rot[i],((J[i, :]-J[parent[i],:]).reshape([3,1]))]	)
			)
		)
	
		
	
	# remove the transformation due to the rest pose. Note here that we're using self.J, which has the additional shoulder&ear offsets (if applicable)
	G = G - pack(
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
	print(time() - s)
	
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')	
	
	# plot skeleton
	ax, fig = Plot3d(J, connections=skelConnections, style='bo-', ax=ax, differentColoursForSides=False)
	# ax.plot(verts[:,0], verts[:,1], verts[:,2], 'go-')
	ax.scatter(verts[:,0], verts[:,1], verts[:,2], 'go-')
		
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_aspect('auto')

	plt.show()
	
	
def with_zeros(x):
	"""
	Append a [0, 0, 0, 1] vector to a [3, 4] matrix. 
	Taken from ProcessingMocapData\SMALViewer-master\SMPL\rgbdog_np.py

	Parameter:
	---------
	x: Matrix to be appended.

	Return:
	------
	Matrix after appending of shape [4,4]

	"""
	return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

def pack(x):
	"""
	Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
	manner. . Taken from ProcessingMocapData\SMALViewer-master\SMPL\rgbdog_np.py

	Parameter:
	----------
	x: Matrices to be appended of shape [batch_size, 4, 1]

	Return:
	------
	Matrix of shape [batch_size, 4, 4] after appending.

	"""
	return np.dstack((np.zeros((x.shape[0], 4, 3)), x))