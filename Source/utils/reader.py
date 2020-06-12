'''
This script is based on:
https://github.com/lawrennd/mocap/blob/master/python/reader.py
'''

from pylab import *
import time
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from utils import bvh as bvh
	
	
	
class MyReader(bvh.BVHReader) :

	# Constructor to set up all the global variables
	def __init__(self, filename,onTheSpot = False, scale=1):

		self.onTheSpot = onTheSpot
		self.screen = None
		self.filename = filename
		# A list of unprocessed tokens (strings)
		self.tokenlist = []
		# The current line number
		self.linenr = 0
		self.frame = 0
		# Root node
		self._root = None
		self._nodestack = []
		self.points = []
		self.data = []
		self.channels = []
		# Total number of channels
		self._numchannels = 0
		self.count = 0
		self.nodes = []
		self.frames = 0
		self.dt = 0
		self.currIdx = 0
		self.scale = scale

	# When motion begins, record the number of frames and the desired length
	# of each frame (dt)
	def onMotion(self,frames,dt):

		self.frames = frames
		self.dt = dt



	# For each frame, append the xyz points to the self.points() array
	# and if current frame = number of frames then call self.render
	def onFrame(self,channels):
		if sys.version_info[0] == 3:
			channels = list(map(float,channels))
		
		# channels is the entire information for the current frame, ie, rotation AND translation
		self.channels.append(channels)
		
		pnts, xyzStruct = self.bvh2xyz(channels)
		self.points.append(pnts)
		self.data.append(xyzStruct)
		
		self.frame += 1

		if self.frame == self.frames:
			self.channels = array(self.channels)
			for row in range(1,len(self.channels)):
				for col in range(3,len(self.channels[0])):

					diff = self.channels[row,col]-self.channels[row-1, col]
					if abs(diff+360)<abs(diff):
						self.channels[row:,col]=self.channels[row:,col]+360
					elif abs(diff-360)<abs(diff):
						self.channels[row:,col]=self.channels[row:,col]-360


	# Initiated by the onHierarchy() method, this method adds all the nodes to the
	# list of nodes from the hierarchy stored within the nodes themselves
	def recurse(self,node):
		if(node.children) :
			for n in node.children:
				self.nodes.append(n)
				self.count+=1
				self.setOrder(n)
				self.recurse(n)


	# Sets the order of the angle channels
	def setOrder(self,node):
		order = ""
		for channel in node.channels:
			if(channel == "Xposition" or channel== "Yposition" or channel== "Zposition"):
				order = ""
			else :
				order += channel[0]
		node.order = order


	# Appends the root node to the list of nodes, then recurses through all the nodes
	# adding their children to the list of nodes, then creates a parent-child list
	# using findNode() and orderNodes()
	def onHierarchy(self,root):
		self.count+=1
		self.nodes.append(root)
		self.setOrder(root)
		self.recurse(root)
		unsortedNodes = [[0,0]]

		for n in range(len(self.nodes)):
			#append each node and its parent as a pair to the list
			for x in self.nodes[n].children:
				child = self.findNode(x)
				unsortedNodes.append([child,n])
		self.nodeOrder = self.orderNodes(unsortedNodes)

	# Return the index in the list 'nodes' for a given node
	def findNode(self,node):
		for n in range(len(self.nodes)):
			if(self.nodes[n] == node):
				return n

	# Takes an unordered list of node indicies and orders them
	def orderNodes(self,unordered):
		nodelist = []

		for n in unordered :
			if len(nodelist)==0 :
				nodelist.append(n)
			else :
				for i in range(len(nodelist)) :
					if nodelist[i][0]> n[0]   :
						nodelist.insert(i,n)
						break
				else:
					nodelist.append(n)
		return nodelist

	# Based on Prof Neil Lawrence's MATLAB mocap toolbox.
	# Converts bvh channels to XYZ co-ordinates
	def bvh2xyz(self,channels):
		curChan = 0
		xyzStruct = []

		#if the sequence has finished reading in, and this method is used
		#Then it must be being used from outside
		external = (self.frame == self.frames)
		for i in range(len(self.nodes)):
			#read in the channels
			xpos = ypos = zpos = 0
			if (len(self.nodes[i].channels) == 6) :

				#If this sequences is not to be performed on the spot
				#Then read in the root location
				if not self.onTheSpot:
					xpos = channels[curChan] * self.scale
					ypos = channels[curChan+1] * self.scale
					zpos = channels[curChan+2] * self.scale
				else:
					channels[curChan] = 0
					channels[curChan+1] = 0
					channels[curChan+2] = 0
				curChan+=3

			if(len(self.nodes[i].channels) >2):
				# we come in here after 'if (len(self.nodes[i].channels) == 6) :'

				if(self.nodes[i].order == "ZXY"):
					zangle = math.radians(channels[curChan])
					xangle = math.radians(channels[curChan+1])
					yangle =math.radians(channels[curChan+2])
				elif(self.nodes[i].order == "ZYX"):
					zangle = math.radians(channels[curChan])
					yangle = math.radians(channels[curChan+1])
					xangle = math.radians(channels[curChan+2])
				elif(self.nodes[i].order == "XYZ"):
					xangle = math.radians(channels[curChan])
					yangle = math.radians(channels[curChan+1])
					zangle = math.radians(channels[curChan+2])
				else :
					zangle = math.radians(channels[curChan])
					xangle =math.radians(channels[curChan+1])
					yangle = math.radians(channels[curChan+2])
				curChan+=3
			else :
				xangle = 0.
				yangle = 0.
				zangle = 0.
				
			
			offsets = matrix('0.,0.,0.')
			rotation_euler = [xangle,yangle,zangle]
			
			thisRotation = self.rotationMatrix(xangle,yangle,zangle) #order='zxy'
			# thisRotation can be acquired using
			# from scipy.spatial.transform import Rotation as Rsci
			# np.matmul(np.matmul(Rsci.from_euler('y',-yangle).as_dcm(), Rsci.from_euler('x',-xangle).as_dcm()), Rsci.from_euler('z',-zangle).as_dcm())
			
			# self.rotationMatrix(xangle,yangle,0)
			# np.matmul(Rsci.from_euler('y',-yangle).as_dcm(), Rsci.from_euler('x',-xangle).as_dcm())
			
			# self.rotationMatrix(0,yangle,zangle)
			# np.matmul(Rsci.from_euler('y',-yangle).as_dcm(), Rsci.from_euler('z',-zangle).as_dcm())
			
			# self.rotationMatrix(xangle,0,zangle)
			# np.matmul(Rsci.from_euler('x',-xangle).as_dcm(), Rsci.from_euler('z',-zangle).as_dcm()) 
			
			# but I haven't yet figured out how to get euler back from thisRotation, using Rsci.from_dcm()
			
			thisPosition = [xpos, ypos, zpos]
			
			isEndEff = len(self.nodes[i].children) == 0
			if isEndEff: #is an end effectors
				thisPosition = offsets + thisPosition
			struct = Position()
			xyzStruct.append(struct)
			
			addThis = self.nodes[i].offset
			# end effectors don't have information in Channels. Therefore if we want to scale them, we need to apply the scale to the original offset value. 
			# We don't want to do this to any other type of bone, as this scale is handled by applying to offset
			if isEndEff:		
				addThis = (addThis[0] * self.scale, addThis[1] * self.scale, addThis[2] * self.scale)
				
			offsets += addThis
			
			#Calculate the position and rotation
			if i==0 :
				xyzStruct[i].position = offsets + thisPosition
				xyzStruct[i].rotation = thisRotation
				xyzStruct[i].rotation_local = thisRotation
				xyzStruct[i].rotation_local_euler = rotation_euler
			else :
				parent = self.nodeOrder[i][1]
				if isEndEff:
					xyzStruct[i].position = (offsets + thisPosition)*xyzStruct[parent].rotation + xyzStruct[parent].position
				else:
					xyzStruct[i].position = (thisPosition)*xyzStruct[parent].rotation + xyzStruct[parent].position
				xyzStruct[i].rotation = thisRotation*xyzStruct[parent].rotation
				xyzStruct[i].rotation_local = thisRotation
				xyzStruct[i].rotation_local_euler = rotation_euler
				
				
			self.nodes[i].position.append(xyzStruct[i].position)
			self.nodes[i].rotation_world.append(xyzStruct[i].rotation)
			self.nodes[i].rotation_local.append(xyzStruct[i].rotation_local)
			self.nodes[i].rotation_local_euler.append(xyzStruct[i].rotation_local_euler)
		

		#Return a list of points (xyzStruct contains a position and rotation)
		points = []
		for m in xyzStruct :
			points.append(m.position)
		return points, xyzStruct

	#Converts xyz angles into a rotation matrix
	def rotationMatrix(self,xangle,yangle,zangle,order='zxy'):
		c1 = math.cos(xangle)

		c2 = math.cos(yangle)
		c3 = math.cos(zangle)
		s1 = math.sin(xangle)
		s2 = math.sin(yangle)
		s3 = math.sin(zangle)
		rM = array([[c2*c3-s1*s2*s3, c2*s3+s1*s2*c3, -s2*c1],
					 [-c1*s3, c1*c3, s1],
					 [s2*c3+c2*s1*s3, s2*s3-c2*s1*c3, c2*c1]])
		return matrix(rM)

	#Calls parent read method, returns necessary information
	def read(self):
		bvh.BVHReader.read(self)
		
		# each node contains an attribute 'postion' etc that is a list of matrix objects.
		# convert this and others into Nx3 numpy arrays
		points_numpy = np.array(self.points)
		numFrames = points_numpy.shape[0]
		numJoints = points_numpy.shape[1]
		
		points_numpy = np.reshape(points_numpy, (numFrames, numJoints, 3))
		
		for i in range(len(self.nodes)):
			self.nodes[i].position = points_numpy[:,i,:]
			self.nodes[i].rotation_local_euler = np.array(self.nodes[i].rotation_local_euler)
			self.nodes[i].rotation_world = np.array(self.nodes[i].rotation_world)
			self.nodes[i].rotation_local = np.array(self.nodes[i].rotation_local)
		
		limits = self.split()
		return self.dt,self.points,limits,self.nodes, self.data

	#Split the storage of the data into an array which contains xs, ys and zs,
	#returning a list of the max values
	def split(self, points=[]):
		internal = not points
		if internal:
			points = self.points
		else:
			points = [points]
		length = len(points[0])
		size = len(points)
		newPoints = zeros([size,3,length],'float')
		zmax = xmax = ymax = 0
		zmin = xmin = ymin = 0

		for i in range(size):
			zs = zeros(length)
			xs = zeros(length)
			ys = zeros(length)
			for j in range(0,length):

				line = points[i][j].getA()
				zs[j] = line[0][0]
				xs[j] = line[0][1]
				ys[j] = line[0][2]

			if(i==0):
				zmax = max(zs)
				xmax = max(xs)
				ymax = max(ys)
			maxz = max(zs)
			maxy = max(ys)
			maxx = max(xs)
			if(maxz > zmax):
				zmax = maxz
			if(maxx > xmax):
				xmax = maxx
			if(maxy > ymax):
				ymax = maxy

			if(i==0):
				zmin = min(zs)
				xmin = min(xs)
				ymin = min(ys)
			minz = min(zs)
			miny = min(ys)
			minx = min(xs)
			if(minz < zmin):
				zmin = minz
			if(minx < xmin):
				xmin = minx
			if(miny < ymin):
				ymin = miny
			newPoints[i][0] = zs
			newPoints[i][1] = xs
			newPoints[i][2] = ys

		if not internal:
			return newPoints[0]

		self.points = newPoints
		return ([zmax,xmax,ymax,zmin,xmin,ymin])

#Used to create a structure that enables the conversion
#of angles to xyz position
class Position():
	def __int__(self,rotation,position, rotation_euler=[0,0,0]):
		self.rotation = rotation
		self.rotation_local = rotation
		self.rotation_local_euler = rotation_euler
		self.position = position

