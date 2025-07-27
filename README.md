# Autonoumous_Parking_Ackermann
![image-20250727133516758](C:\Users\zhuyu\AppData\Roaming\Typora\typora-user-images\image-20250727133516758.png)

## Project Pipeline

![image-20250727133522883](C:\Users\zhuyu\AppData\Roaming\Typora\typora-user-images\image-20250727133522883.png)

## Setup

1. Install Cartographer

On Ubuntu Focal with ROS Noetic use these commands to install the above tools:

```
sudo apt-get update
sudo apt-get install -y python3-wstool python3-rosdep ninja-build stow
```

After installed ninja (a installation tools)

```
mkdir catkin_ws
cd catkin_ws
wstool init src
wstool merge -t src https://raw.githubusercontent.com/cartographer-project/cartographer_ros/master/cartographer_ros.rosinstall
wstool update -t src

# make sure no warning in previous steps
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y

# must have, if previously installed please igonre
src/cartographer/scripts/install_abseil.sh
# if version conflict right try:
sudo apt-get remove ros-${ROS_DISTRO}-abseil-cpp

# must use isolated install
catkin_make_isolated --install --use-ninja
```

2. Calibration camera and robot extrinsic parameters

Oringial urdf is attached in s

2. 
