# Computer-Graphic
This repo contains all source code of Computer Graphic course

# How to run
1. pip install -r requirements.txt
2. export PYTHONPATH=./ (for MacOS)
3. python3 shape/viewer.py

# Step to visualize shapes in UI
1. Select omshape from dropdown
2. The shape will be in static view and cannot interact to rotate them
3. To rotate them, select "Using Trackball" in Option
4. To move around the object, select "Move camera" in Option, then click W,S,A,D, to go forward, backward, left, right
5. To visualize a single optimizer:
    - Select Mesh in Shape, and select the function
    - Select Sphere/SubdivideSphere in Shape
    - Click Confirm 
    - Select "Using Trackball" and "Optimizer" in Options
    - Select a type of optimizer
    - Increase Learning rate by clicking button "+" and see the ball moving
6. To visualize 2 optimizers:
    - Select Mesh in Shape, and select the function
    - Select "Using Trackball" and "Visualize 2 Optimizers" in Options
    - Click Confirm 
    - Increase Learning rate by clicking button "+" and see the balls moving 
7. To visualize multi-camera
    - Select an object in Shape
    - Select "Multi Camera" in Options
    - Click Confirm
    - By default, the whole multi-camera system can be rotated by trackball, which will not change in each camera view on the right viewport
    - To see the change in each camera view, select "Rotation" in Options and click Confirm