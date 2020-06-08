# Instructions

Use the script ImportSknnedMesh.py to import the animated dog skeleton and mesh into Blender.

## Installing Numpy and Scipy in Blender
The code requries that Blender's python has numpy as scipy installed. To do this, use the following Windows instructions:
1. Open Blender
2. Open a python console window
3. Find the path to Blender's installed python using
```console
>>> import sys
>>> sys.exec_prefix
```
4. This will print out a path, /path/to/blender/python
5. Open a windows console as admin:
```console
cd /path/to/blender/python/bin
python -m ensurepip
python -m pip install numpy
python -m pip install scipy
```

## How to run this script:
1. Open a window in Blender and set it to "Text Editor"
2. Use the navigator to load this script
3. Edit the values in pathToMainDataset, dog or motion
4. Click the "run script" button
