'''

Please note that the code in this script was based on that of https://github.com/benjiebob/SMALViewer

'''
import numpy as np

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QWidget, QFrame, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QScrollArea, QGridLayout, QCheckBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from functools import partial

import os
from os.path import join
import collections

import scipy.misc
import datetime

import pickle as pkl
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import cv2
import math

import utils_mc

sys.path.insert(0, os.path.abspath('..'))
from utils import utils

DISPLAY_SIZE = 700
pathToMainDataset = 'D:/DOG'
play_speed = 10 # 1000 ms = 1 s

class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		QMainWindow.__init__(self, parent)

		

		self.is_playing = False
		self.animTimer = QTimer()
		self.play_speed_userDef = play_speed
		self.play_speed = play_speed # speed will be slowed down x30 for Kinect, since the images are furth apart in time
		self.animTimer.setInterval(self.play_speed)
		self.animTimer.timeout.connect(self.read_next_frame) # connect timeout signal to signal handler
		self.animTimer.stop()
		
		self.viewDataType = ''
		
		self.setup_data()
		self.ChangeDog(self.dog)
		self.UpdateDataAfterMotionChange(self.availableMotions[0])
		
		self.setup_ui()
		
		print('self.camIdx', self.camIdx)
		self.ChangeCamera(self.camIdx)
		self.update_render()
		
	def setup_data(self):	
		
		self.vidFile = ''
		self.frameRaw = None
		self.ax3d = None

		clrs = {'red': (255,0,0), 'blue': (0,0,255), 'green': (0,255,0), 'yellow': (255,255,0), 'cyan': (0,255,255), 'magenta': (255,0,255), 'white':(255,255,255), 'darkerBlue':  (39,127,255)}

		self.skelJntClr = clrs['red']
		self.skelLeftRightClrs = [clrs['cyan'], clrs['magenta']] # [left, right]

		self.markerClr = [clrs['cyan'], clrs['darkerBlue'], clrs['darkerBlue']] # [left, center, right]
		self.indivMarkerClr = clrs['red']

		self.camColour = clrs['cyan']
		
		self.plotCameras = True
		self.camIdx = 0
		self.showMeshMask = False
		self.plotCalibs = True
		self.plotMarkers = False
		self.plotSkel = True

		self.availableDogs = utils_mc.GetDogs(pathToMainDataset)
		self.dog = self.availableDogs[0]
		
	
	def get_layout_region(self, control_set):
		layout_region = QVBoxLayout()

		scrollArea = QScrollArea()
		scrollArea.setWidgetResizable(True)
		scrollAreaWidgetContents = QWidget(scrollArea)
		scrollArea.setWidget(scrollAreaWidgetContents)
		scrollArea.setMinimumWidth(750)

		grid_layout = QGridLayout()

		for idx, (label, com_slider) in enumerate(control_set):
			grid_layout.addWidget(label, idx, 0)
			grid_layout.addWidget(com_slider, idx, 1)
		
		scrollAreaWidgetContents.setLayout(grid_layout)

		layout_region.addWidget(scrollArea)
		return layout_region

	def setup_ui(self):
		self.video_controls = []
		self.camera_controls = []
		self.displayData_controls = []
		
		def ctrl_layout_add_separator():
			line = QFrame()
			line.setFrameShape(QFrame.HLine)
			line.setFrameShadow(QFrame.Sunken)
			ctrl_layout.addWidget(line)
			
		def view_layout_add_separator():
			line = QFrame()
			line.setFrameShape(QFrame.HLine)
			line.setFrameShadow(QFrame.Sunken)
			view_layout.addWidget(line)
		
		# ---------------- rgb video --------------
		view_layout = QVBoxLayout()
		self.render_img_label = QLabel() # this is the RGB image
		view_layout.addWidget(self.render_img_label)
		# ---------------- rgb video --------------
		
		
		fontSize = 10
		fontWeight = QtGui.QFont.Bold
		fontType = "Times"
		
		# --------------- video controls -----------
		# reset
		self.is_playing = False
		self.markerFrame = 0
		self.markerFrame = 0
		self.animTimer.stop()
		
		self.anim_frame = QFrame()
		self.anim_layout = QGridLayout()
		self.play_video_pb = QPushButton('Play')
		self.play_video_pb.clicked.connect(self.playPauseVideo)
		self.anim_layout.addWidget(self.play_video_pb, 0, 1)
		
		nextFrame_pb = QPushButton('Next Frame')
		nextFrame_pb.clicked.connect(partial(self.read_next_frame, True))
		self.anim_layout.addWidget(nextFrame_pb, 0, 2)
		
		prevFrame_pb = QPushButton('Prev Frame')
		prevFrame_pb.clicked.connect(partial(self.read_next_frame, False))
		self.anim_layout.addWidget(prevFrame_pb, 0, 0)
		
		#  ----visibility controls ----
		labelA = QLabel('Toggle Visible:')
		labelA.setFont(QtGui.QFont(fontType, fontSize, fontWeight))
		self.anim_layout.addWidget(labelA, 0,3)
		
		markerVis_pb = QPushButton('Markers')
		markerVis_pb.clicked.connect(partial(self.ToggleVisibility, 'markers'))
		self.anim_layout.addWidget(markerVis_pb, 0,4)
		skelVis_pb = QPushButton('Skeleton')
		skelVis_pb.clicked.connect(partial(self.ToggleVisibility, 'skeleton'))
		self.anim_layout.addWidget(skelVis_pb, 0,5)
		camVis_pb = QPushButton('Cameras')
		camVis_pb.clicked.connect(partial(self.ToggleVisibility, 'cameras'))
		self.anim_layout.addWidget(camVis_pb, 0,6)
		maskVis_pb = QPushButton('Mask')
		maskVis_pb.clicked.connect(partial(self.ToggleVisibility, 'masks'))
		self.anim_layout.addWidget(maskVis_pb, 0,7)
		# ---- visibility controls ----
		
		
		self.anim_frame.setLayout(self.anim_layout)
		view_layout.addWidget(self.anim_frame)
		# ---- camera controls -------
		
		
		
		self.cam_frame = self.UpdateCameraButtons()
		
		
		labelA = QLabel('Cameras:')
		labelA.setFont(QtGui.QFont(fontType, fontSize, fontWeight))
		view_layout.addWidget(labelA)
		view_layout.addWidget(self.cam_frame)
		self.view_layout = view_layout
		# ---- camera controls -------
		

		# ------- available dogs-------
		self.ctrl_layout = QVBoxLayout()
		dog_layout = QGridLayout()
		for idx, dog in enumerate(self.availableDogs):
			dog_pb = QPushButton(dog)
			dog_pb.clicked.connect(partial(self.ChangeDog_buttonClick, dog))
			dog_layout.addWidget(dog_pb, idx, 0)
			
		self.dog_frame = QFrame()
		self.dog_frame.setLayout(dog_layout)	
		labelDogs = QLabel('Dogs:')
		labelDogs.setFont(QtGui.QFont(fontType, fontSize, fontWeight))
		self.ctrl_layout.addWidget(labelDogs)
		self.ctrl_layout.addWidget(self.dog_frame)
		# ------- available motions-------
		
		
		# ------- available motions-------
		self.motion_frame = self.UpdateMotionButtons()
		labelMotions = QLabel('Motions:')
		labelMotions.setFont(QtGui.QFont(fontType, fontSize, fontWeight))
		self.ctrl_layout.addWidget(labelMotions)
		self.ctrl_layout.addWidget(self.motion_frame)
		# ------- available motions-------
		
		# ------- available modalities --------
		self.modal_frame = self.UpdateModalitiesButtons()
		self.ctrl_layout.addWidget(self.modal_frame)
		# ------- available modalities --------
		
		
		self.main_layout = QHBoxLayout()
		self.main_layout.addLayout(view_layout)
		self.main_layout.addLayout(self.ctrl_layout)
		

		self.main_widget = QWidget()
		self.main_widget.setLayout(self.main_layout)
		self.setCentralWidget(self.main_widget)
		
		# WINDOW
		self.window_title_stem = 'Data Viewer'
		self.setWindowTitle(self.window_title_stem)
		
		self.statusBar().showMessage('Ready...')
		
		
		self.showMaximized()

	
	def UpdateCameraButtons(self):
		cam_numRows = 2
		cam_numCols = math.ceil(len(self.cams)/cam_numRows)
		cam_layout = QGridLayout()
		
		for idx, cam in enumerate(self.cams):
			cam_pb = QPushButton(cam)
			cam_pb.clicked.connect(partial(self.ChangeCamera, idx, True))
			cam_layout.addWidget(cam_pb, (idx//cam_numCols), (idx%cam_numCols))
			
		cam_frame = QFrame()
		cam_frame.setLayout(cam_layout)	
		return cam_frame
		
	def UpdateModalitiesButtons(self):
		mod_layout = QGridLayout()
		for idx, modal in enumerate(self.availableModalities):
			mod_pb = QPushButton(modal)
			mod_pb.clicked.connect(partial(self.ChangeModality_buttonClick, modal))
			mod_layout.addWidget(mod_pb, idx, 0)
		mod_frame = QFrame()
		mod_frame.setLayout(mod_layout)
		return mod_frame
		
	def UpdateMotionButtons(self):
		motion_layout = QGridLayout()
		for idx, motion in enumerate(self.availableMotions):
			motion_pb = QPushButton(motion)
			motion_pb.clicked.connect(partial(self.ChangeMotion, idx))
			motion_layout.addWidget(motion_pb, idx, 0)
			
		motion_frame = QFrame()
		motion_frame.setLayout(motion_layout)
		return motion_frame
		
	def update_render(self, useRawFrame=False):
		
		if not useRawFrame:
			# update self.frameRaw
			
			if self.viewDataType == 'sony':
				[ret, self.vidFile] = utils_mc.SetVideoToFrameNumber(self.vidFile, self.imageFrame)
				
				ret, frameRaw = self.vidFile.read()
				self.frameRaw = frameRaw[:,:,(2,1,0)] # reorder
			elif self.viewDataType == 'kinect_rgb':
				frameRaw = cv2.imread(join(self.pathToImageFolder_current, self.imageNames[self.imageFrame]))
				self.frameRaw = frameRaw[:,:,(2,1,0)] # reorder
			else: #if self.viewDataType == 'kinect_depth'
				frameRaw = cv2.imread(join(self.pathToImageFolder_current, self.imageNames[self.imageFrame]), -1)
				frameRaw = frameRaw.astype('float')/np.amax(frameRaw) # scaled 0-1
				frameRaw = frameRaw*255
				frameRaw = frameRaw.astype('uint8')
				self.frameRaw = np.stack((frameRaw,)*3, axis=-1)
			
			s = 'Dog: %s, Motion: %s, Camera: %s, Frame: %d' % (self.dog, self.motion, self.cams[self.camIdx], self.markerFrame)
			
			self.window_title_stem = s
			self.setWindowTitle(self.window_title_stem)
			
		if self.frameRaw is not None:
			self.frame = self.UpdatePlotData(self.frameRaw.copy())
			frame_origSize = self.frame.copy()
			self.render_img_label.setPixmap(self.image_to_pixmap(self.frame, DISPLAY_SIZE))
			self.render_img_label.update()

	def toggle_control(self, layout):
		sender = self.sender()
		layout.setHidden(not sender.isChecked())
		
	def playPauseVideo(self):
		self.is_playing = not self.is_playing
		print('self.is_playing', self.is_playing)
		
		if self.is_playing:
			self.play_video_pb.setText('Pause')
			self.animTimer.start()
		else:
			self.play_video_pb.setText('Play')
			self.animTimer.stop()
			
		
	def nextFrame(self):
		self.read_next_frame(True)
		
	def prevFrame(self):
		self.read_next_frame(False)
		
	def updateFrame(self, fr, nextFrameGoingForward=True):
		# for the sony videos, we can just increment frame
		# for kinect videos, we increment the image frame normally, but the marker frame needs to find the next valid timecode
		if nextFrameGoingForward:
			fr += 1
			if fr >= self.numFrames:
				fr = 0
		else:
			fr -= 1
			if fr < 0:
				fr = self.numFrames-1

		return self.allowableMarkerFrames[fr], fr
				
	def read_next_frame(self, nextFrameGoingForward=True):
		self.markerFrame, self.imageFrame = self.updateFrame(self.imageFrame, nextFrameGoingForward)
		self.update_render()

	def image_to_pixmap(self, img, img_size):
		qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).copy()
		pixmap = QPixmap(qim)
		# return pixmap.scaled(img_size, img_size, QtCore.Qt.KeepAspectRatio)
		return pixmap.scaledToHeight(img_size)
			
			
	# this is called every time we go back/forward a frame, change camera, etc
	def UpdatePlotData(self, frame):

		if self.showMeshMask:
			mask = cv2.imread(join(self.pathToMasks % self.cams[self.camIdx], self.imageNames[self.imageFrame]), -1)
			mask = cv2.resize(mask, (0,0), fx=self.postProjectionScale, fy=self.postProjectionScale, interpolation=cv2.INTER_NEAREST ) 
			mask = mask/np.amax(mask)
			mask = np.stack((mask.astype(np.uint8),)*3, axis=-1)
			frame = frame * mask

		if self.plotMarkers:
			points2d = self.GetProjectedPoints(self.markers[:,:,self.markerFrame])
			points2d *= self.postProjectionScale
			

			frame = utils.Plot2d_markers(frame, points2d, connections=self.markerConn, lineThickness=self.markerPlotSize, radius=-1, markerNames=self.markerNames, markerClr=self.markerClr, indivMarkerClr=self.indivMarkerClr)

		if self.plotSkel:
			points2d = self.GetProjectedPoints(self.bvhJoints[:,:,self.markerFrame+1])	
			points2d *= self.postProjectionScale
			frame = utils.Plot2d(frame, points2d, self.skelConnections, radius=self.skeletonRadiusSize, lineThickness=self.skeletonPlotSize, plotJoints=True, differentColoursForSides=True, markerClr=self.skelJntClr, leftRightClrs=self.skelLeftRightClrs)

		if self.plotCameras:
			cameraLoc_2d = self.GetProjectedPoints(self.cam_world_translation)
			cameraLoc_2d *= self.postProjectionScale
			
			cameraLoc_2d = np.round(cameraLoc_2d).astype(np.int)
			font = cv2.FONT_HERSHEY_SIMPLEX
			for camIdx, loc in enumerate(cameraLoc_2d):
				if loc[0] > -1 and loc[1] > -1 and loc[0] < frame.shape[1] and loc[1]  < frame.shape[0] and camIdx != self.camIdx:
					frame = cv2.circle(frame, (loc[0], loc[1]), self.cameraPlotRadius, self.camColour, self.cameraPlotThickness, 1)
					frame = cv2.putText(frame, self.cams[camIdx], (loc[0], loc[1]), font, self.cameraFontScale, self.camColour, self.cameraFontThickness)

		return frame
		
	def ToggleVisibility(self, dataType):
		if dataType == 'markers':
			self.plotMarkers = not self.plotMarkers
		elif dataType == 'skeleton':
			self.plotSkel = not self.plotSkel	
		elif dataType == 'cameras':
			self.plotCameras = not self.plotCameras
		elif dataType == 'masks':
			self.showMeshMask = not self.showMeshMask
		self.update_render(True) # don't go onto the next frame
		
	def GetProjectedPoints(self, points3d):
		#calibration data_dog
		rot3x1 = self.calibData['rot3x1']
		t = self.calibData['t']
		K = self.calibData['K']
		distCoeffs = self.calibData['distCoeffs']
		[points2d, jac] = cv2.projectPoints(points3d, rot3x1, t, K, distCoeffs)
		
		points2d = np.reshape(points2d, [points2d.shape[0], 2])
		return points2d	
		
	def ChangeDog_buttonClick(self, dog):
		self.ChangeDog(dog)
		
		# update the camera buttons
		self.view_layout.removeWidget(self.cam_frame)
		self.cam_frame.deleteLater()
		self.cam_frame = self.UpdateCameraButtons()
		self.view_layout.addWidget(self.cam_frame)
		
		# update the motion buttons
		self.ctrl_layout.removeWidget(self.motion_frame)
		self.motion_frame.deleteLater()
		# self.motion_frame = None

		self.motion_frame = self.UpdateMotionButtons()
		self.ctrl_layout.addWidget(self.motion_frame)
		self.main_layout.update()
		self.main_widget.update()
		self.ChangeMotion(0)
		
	def ChangeDog(self, dog):
		print(dog)
		
		self.dog = dog
		self.pathToDog = join(pathToMainDataset, dog)
		self.availableMotions = utils_mc.GetMotionsForDog(self.pathToDog)
		self.ChangeModality() # updates self.cams
		
		
	def ChangeModality_buttonClick(self, modal):
		print(modal)
		prev_model = self.viewDataType
		self.viewDataType = modal
		
		self.ChangeModality()
		self.UpdateDataAfterModalityChange()
		if not ('kinect' in prev_model and 'kinect' in modal):
			self.camIdx = 0
		# else: keep the camera ID the same if flicking between kinect cameras
		
		markerFrame_prev = self.markerFrame
		imageFrame_prev = self.imageFrame
		self.ChangeCamera(self.camIdx, updateRender=False)
		if 'kinect' in prev_model and 'kinect' in modal:
			self.markerFrame = markerFrame_prev
			self.imageFrame = imageFrame_prev
		# else keep the image frame the same
		
		# update the camera buttons
		self.view_layout.removeWidget(self.cam_frame)
		self.cam_frame.deleteLater()
		self.cam_frame = self.UpdateCameraButtons()
		self.view_layout.addWidget(self.cam_frame)
		self.main_layout.update()
		self.main_widget.update()
		
		self.update_render()
		
	def ChangeModality(self):
		if self.viewDataType == '':
			# this is true when we first run the script
			availableModalities = utils_mc.GetModalityForMotion(join(pathToMainDataset, self.dog, 'motion_%s'%self.availableMotions[0]))
			self.viewDataType = availableModalities[-1] # start with sony, if possible
			
		self.pathToCalibs = join(self.pathToDog, 'calibration', self.viewDataType)
		self.cams = utils_mc.GetCamerasForModality(self.pathToCalibs)
		
		assert len(self.pathToCalibs) > 0, 'no calibration path found'
		self.calibData_allCalibs_allCams = utils.GetCalibData(self.pathToCalibs, self.cams)
		assert len(self.calibData_allCalibs_allCams) > 0, 'no calibration data found'
		
		self.cam_world_rotations = np.zeros((len(self.calibData_allCalibs_allCams), 3, 3))
		self.cam_world_translation = np.zeros((len(self.calibData_allCalibs_allCams), 3))
		for idx, calib in enumerate(self.calibData_allCalibs_allCams):
			L = calib['calib']['L']
			Linv = np.linalg.inv(L) 
			self.cam_world_rotations[idx] = Linv[0:3,0:3]
			self.cam_world_translation[idx] = Linv[0:3,3]

		
		
	def UpdateDataAfterModalityChange(self):
		self.postProjectionScale = 1
		self.pathToVid = '' # for sony
		self.pathToImageFolder_current = '' # for kinect
		self.imageNames = []
		
		self.markerPlotSize = 3
		self.skeletonPlotSize = 4
		self.skeletonRadiusSize = 2
		self.cameraPlotRadius = 5
		self.cameraPlotThickness = 2
		self.cameraFontScale = 2
		self.cameraFontThickness = 3
		
		
		if self.viewDataType == 'sony':
			self.pathToVid = join(self.pathToMotion, 'sony', 'camera%s', 'camera%s_2K.mp4')
			self.postProjectionScale = 0.5 # 4K calibration values but 2K video
			self.play_speed = self.play_speed_userDef
			self.imageFrame = 0
			self.markerFrame = 0
		else:
			self.pathToImageFolder = join(self.pathToMotion, self.viewDataType, 'camera%s', 'images')
			self.play_speed = self.play_speed_userDef * 30
			
			if 'depth' in self.viewDataType:
				self.markerPlotSize = 2
				self.skeletonPlotSize = 2
				self.skeletonRadiusSize = 1
				self.cameraPlotRadius = 3
				self.cameraPlotThickness = 1
				self.cameraFontScale = 0.75
				self.cameraFontThickness = 1
				
		self.animTimer.setInterval(self.play_speed)
		
		
		self.pathToMasks = join(self.pathToMotion, self.viewDataType, 'camera%s', 'masks')
		self.vidFiles = []
		self.vidFilesPaths = []
		self.numCams = len(self.cams)
		if self.viewDataType == 'sony':
			for c in self.cams:
				pathToCurrVid = self.pathToVid %(c,c)
				self.vidFilesPaths.append(pathToCurrVid)
				
		
			
	def ChangeMotion(self, motionIdx):
		self.motionIdx = motionIdx
		print('change motion to %s' % self.availableMotions[motionIdx])
		self.UpdateDataAfterMotionChange(self.availableMotions[motionIdx])
		self.ChangeCamera(self.camIdx)	


		# update the modality buttons
		self.ctrl_layout.removeWidget(self.modal_frame)
		self.modal_frame.deleteLater()
		self.modal_frame = self.UpdateModalitiesButtons()
		self.ctrl_layout.addWidget(self.modal_frame)
		self.main_layout.update()
		self.main_widget.update()

	
	def UpdateDataAfterMotionChange(self, motion):
		self.statusBar().showMessage('Loading %s...' % motion)
		
		self.motion = motion
		dog = self.dog
		
		self.availableModalities = utils_mc.GetModalityForMotion(join(pathToMainDataset, dog, 'motion_%s'%motion))
		if self.viewDataType is None or self.viewDataType not in self.availableModalities:
			self.viewDataType = self.availableModalities[0]
			
		fullPathToMotionCaptureFolder = join(pathToMainDataset, dog, 'motion_%s'%motion, 'motion_capture')
		self.timecodes = utils_mc.GetTimecodesForSequence(fullPathToMotionCaptureFolder)
		self.numFrames = len(self.timecodes) - 1
		self.numFrames_skeletonTotal = self.numFrames
		
		self.videoIdxStart = 0
		self.videoIdxEnd = self.numFrames
		
		self.pathToMotion = os.path.join(pathToMainDataset, dog, 'motion_%s'%motion)
		self.UpdateDataAfterModalityChange()
		
		# ----------------------------------- load the skeleton -----------------------------
		bvhSkelFile = os.path.join(self.pathToMotion, 'motion_capture', 'skeleton.bvh')
		skelConnections_asNp = os.path.join(self.pathToMotion, 'motion_capture', 'skelConnections.npy')
		bvhSkelFile_asNp = bvhSkelFile[:bvhSkelFile.rfind('.')] + '.npy'
		print('loading skeleton...')
		if os.path.isfile(bvhSkelFile_asNp):
			bvhJoints = np.load(bvhSkelFile_asNp)
			skelConnections = np.load(skelConnections_asNp)
		else:
			bvhJoints, skelConnections, nodes = utils.ReadBvhFile(bvhSkelFile, False)
			bvhJoints = utils.MovePointsOutOfMayaCoordSystem(bvhJoints, 1)
			np.save(bvhSkelFile_asNp, bvhJoints)
			np.save(skelConnections_asNp, skelConnections)
		self.bvhJoints = bvhJoints
		self.skelConnections = skelConnections
			
		# ----------------------------------- load the markers -----------------------------	
		markerFile = os.path.join(self.pathToMotion, 'motion_capture', 'markers.json')
		markerFile_asNp = markerFile[:markerFile.rfind('.')] + '.npy'
		markerFile_namesText = markerFile[:markerFile.rfind('.')] + '_names.txt'
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
		self.markers = markers
		self.markerNames = markerNames
		# ----------------------------------- load the stick file that defines the relationship between the markers -----------------------------
		# use the vskSticks.txt file to get the connection between markers
		vskSticks = os.path.join(pathToMainDataset, dog, 'meta', 'vskSticks.txt')
		
		markerConn = []
		with open(vskSticks, 'r') as myFile:
			for f in myFile:
				res = [x.strip() for x in f.split(' ')]
				#convert from name to index
				markerConn.append([markerNames.index(res[0]), markerNames.index(res[1])])
		self.markerConn = markerConn
		self.imageFrame = 0 # this is the frame for the images. For sony, this is [0,1,2,...] for Kinect, this might be [2,5,10,13,...]
		self.markerFrame = 0 # markerFrame is the frame into the array of markers
		self.statusBar().showMessage('Loaded %s...' % motion)
		
		
	def ChangeCamera(self, camIdx, updateRender=True):
		self.camIdx = camIdx

		if self.viewDataType == 'sony':
			self.vidFile = cv2.VideoCapture(self.vidFilesPaths[self.camIdx])
			[ret, self.vidFile] = utils_mc.SetVideoToFrameNumber(self.vidFile, self.markerFrame)
			self.allowableMarkerFrames = np.arange(self.numFrames_skeletonTotal)
			self.numFrames = self.numFrames_skeletonTotal
			self.imageNames = utils_mc.GetFilesInFolder(self.pathToMasks  % self.cams[self.camIdx], 'png')
			# keep marker and image frame the same, since all cameras have the same footage
		else:
			self.pathToImageFolder_current = self.pathToImageFolder % self.cams[self.camIdx]
			self.imageNames = utils_mc.GetFilesInFolder(self.pathToImageFolder_current, 'png')
			self.allowableMarkerFrames = utils_mc.GetMarkerFramesForKinectCam(self.pathToImageFolder_current)
			# we might not have the same frame for this camera view, so go back to the start of the sequence
			self.markerFrame = self.allowableMarkerFrames[0]
			self.imageFrame = 0
			self.numFrames = len(self.imageNames)

		
		print('setting calib data to camera', self.camIdx)
		self.calibData = self.calibData_allCalibs_allCams[self.camIdx]
		print('updateRender', updateRender)
		if updateRender:
			self.update_render()