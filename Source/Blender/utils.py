# import sys, os
# sys.path.append(os.path.dirname(sys.path[0]))

from scipy.io import loadmat

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