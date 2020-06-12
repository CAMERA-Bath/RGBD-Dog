from os import listdir, makedirs, walk, rename
from os.path import isfile, join, isdir, getctime, getmtime, getsize, exists, dirname
import json
import numpy as np
import cv2

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
		files = [f for f in listdir(myPath) if isfile(join(myPath, f))]
	elif not fileType == '' and contains == '':
		# files = [f for f in listdir(myPath) if isfile(join(myPath, f)) and f[-len(fileType):] == fileType]
		files = [f for f in listdir(myPath) if isfile(join(myPath, f)) and f[-len(fileType):].lower() == fileType.lower()]
	elif fileType == '' and not contains == '':
		files = [f for f in listdir(myPath) if isfile(join(myPath, f)) and contains in f.lower()]	
	else:
		files = [f for f in listdir(myPath) if isfile(join(myPath, f)) and contains in f.lower() and f[-len(fileType):] == fileType]	
	return files
	
'''
Function: GetMostRecentFileInFolder
get the names of the folders in myPath

Parameters:
	myPath - string, full path to file
	contains - string, optional, default ''. Get only folders that contain this in the name
Returns:
	latest_file - list of strings, folder names
'''
def GetFoldersInFolder(myPath, contains=''):
	if contains=='':
		fldrs = [f for f in listdir(myPath) if isdir(join(myPath, f))]
	else:
		fldrs = [f for f in listdir(myPath) if isdir(join(myPath, f)) and contains in f.lower()]
	# fldrs is just the folder name, not the full path to the folders
	return fldrs
	
	
#move along the video to frame <frameNumber>. A similar function to SetVideoToTimeCode()
def SetVideoToFrameNumber(vidFile, frameNumber):

	ret = False
	if type(vidFile) is cv2.VideoCapture:
		ret = vidFile.set(cv2.CAP_PROP_POS_FRAMES,frameNumber)
	else:
		vidFile = cv2.VideoCapture(vidFile)
		ret = vidFile.set(cv2.CAP_PROP_POS_FRAMES,frameNumber)
		# vidFile.release()
	return [ret, vidFile]
	
	
def GetCamerasForModality(fullPathToCalibration):
	cams = GetFilesInFolder(fullPathToCalibration, contains='calibFile')
	cams = [cam[-2:] for cam in cams]
	return cams
	
def GetTimecodesForSequence(fullPathToMotionCaptureFolder):
	pathToTimecodes = join(fullPathToMotionCaptureFolder, 'timecodes.json')
	tc = {}
	with open(pathToTimecodes, 'r') as f:
		tc = json.load(f)
	return tc
	
def GetDogs(fullPathToDataset):
	dogs = GetFoldersInFolder(fullPathToDataset, contains='dog')
	return dogs
	
def GetModalityForMotion(fullPathToMotionFolder):
	d = GetFoldersInFolder(fullPathToMotionFolder)
	d.remove('motion_capture')
	return d
	
	
def GetMotionsForDog(fullPathToDogFolder):
	motions = GetFoldersInFolder(fullPathToDogFolder, contains='motion_')
	motions = [motion[7:] for motion in motions]
	return motions
	
def GetMarkerFramesForKinectCam(fullPathToImagesOrMasks):
	# D:\DOG\dog1\motion_walk\kinect_depth\camera00\images
	ims = GetFilesInFolder(fullPathToImagesOrMasks, 'png')
	frameIdx = [int(im[3:11]) for im in ims]
	return np.array(frameIdx)
	
if __name__ == '__main__':
	cams = GetCamerasForModality('D:/DOG/dog5/calibration/sony')
	tc = GetTimecodesForSequence('D:/DOG/dog5/motion_walk1/motion_capture')
	frameIdx = GetMarkerFramesForKinectCam('D:/DOG/dog1/motion_walk/kinect_depth/camera00/images')
	motions = GetMotionsForDog('D:/DOG/dog5')
	dogs = GetDogs('D:/DOG')
	modal = GetModalityForMotion('D:/DOG/dog5/motion_walk1')
	print(modal)
	
	
	
	
	
	
	
	