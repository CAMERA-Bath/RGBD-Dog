	
Each of these tests provide a basic feature of the data. To run:
```
cd path/to/RGBD-Dog/tests
python test_TESTNAME.py
```

* test_3dReconstruction: reconstruct the points in the Kinect depth image
* test_3dSkeletonAndMarkers: plot the skeleton and markers in 3D space
* test_mask: apply the mask to the image
* test_projection: project either the skeleton or marker positions onto the image
* test_skinning: apply the animation from a bvh file to the dog

Each file contains variables to edit before running. Please see the file itself.

