# DroneTest

For now there are two branches in this repo.
- Main -- the origin agile flight repo (unchanged)
- vitfly -- the vitfly runable code

## How to use it
the same with agile flight and in the main branch.
### set up workspace

```
cd     # or wherever you'd like to install this code
export ROS_VERSION=noetic
export CATKIN_WS=./icra22_competition_ws
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
catkin init
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color

cd src
git clone git@github.com:LuNa-shi/DroneSimTest.git agile_flight
cd agile_flight
```

### setup ros
in agile_flight folder
```
./setup_ros.bash
catkin build
```

### setup pytorch env
1. install anaconda
2. (optional) download cuda
3. create env `conda create -n vitfly`
4. `conda activate vitfly`
5. use pip to install requirement packages:matplotlib, numpy, pandas, pytorch, uniplot rospkg pyyaml scipy opencv-python
6. maybe you can use this command to to install everything (EXCEPT pytorch)
```
pip install matplotlib numpy pandas uniplot rospkg pyyaml scipy opencv-python
``` 




## How to run vitfly
### checkout
```
git checkout vitfly
```

### what did I changed?
- /envtest/ros/evaluation_node,run_competition,user_code
- /models (with pretrained models)
- launch_evaluation.bash (most important for automatic)

### how to run

- open one terminal, enter icra workspace and  `source devel/setup.bash`, recommand write in .bashrc
- run `roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True`
- this code may fail because unity, try it again when every thing is killed
- when you see two windows(rviz and unity), you succeed
--- 
- open another terminal
- `conda activate vitfly`
- in the agile_flight folder, run `bash launch_evaluation.bash (N)`
- N is optional for running loops
- you can check /envtest/ros/train_set for data gathered in the running process.

### Note
If you meet some problem, check out [notebook for simExperiment](https://notes.sjtu.edu.cn/JD3lo_HQSTu7VmxGpNqMuQ?both), or ask me.

## How to experiment on your own algorithm (recommended)
take astar for an example.
```
    git checkout -b astar
```
revise user_code, run_competition ...
copy the launch_evaluation.bash in your branch.

