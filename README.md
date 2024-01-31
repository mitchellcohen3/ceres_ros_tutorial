# ROS Ceres Tutorials
A few simple tutorial examples of nonlinear least squares problems using Ceres,
with 3D visualization using ROS.


## Installation
This repo is a ROS package and should be built using ROS. It has been tested on
Ubuntu 20.04 with ROS Noetic and Ceres 2.0.0. To build the ROS package, create a
Catkin workspace with a source directory, clone this repo into the `src`
directory, and build the workspace by running the following in a terminal.
```bash
$ mkdir -p ~/catkin_ws_ceres_tutorials/src/
$ cd ~/catkin_ws_ceres_tutorials/src/
$ git clone https://github.com/mitchellcohen3/ceres_ros_tutorial.git
$ cd ..
$ catkin build
```

## Running the bundle adjustment example
For now, the only example implemented is a simple bundle adjustment example
where we wish to estimate robot poses and landmark positions, given relative
position measurements between the poses and the landmarks. To run the bundle adjustment example, first launch the visualization by running
the follwoing from the root of the workspace.
```bash
source devel/setup.bash
roslaunch ceres_ros_tutorial bundle_adjustment_rviz.launch
```

Then, in a separate terminal, launch the executable by navigating to the
directory `devel/lib/ceres_ros_tutorial/` and run
```bash
$ ./simple_ba_example
```
The visualizer should show the initial guess and final result from Ceres.
