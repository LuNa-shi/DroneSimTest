#!/bin/bash

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  N=10
fi

# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi

# Launch the simulator, unless it is already running
if [ -z $(pgrep visionsim_node) ]
then
  roslaunch envsim visionenv_sim.launch render:=True &
  ROS_PID="$!"
  echo $ROS_PID
  sleep 30
else
  ROS_PID=""
fi

SUMMARY_FILE="evaluation.yaml"
"" > $SUMMARY_FILE

# Perform N evaluation runs
for i in $(eval echo {1..$N})
do

  start_time=$(date +%s)
  # Publish simulator reset
  rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
  rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"

  cd ./envtest/ros/
  python3 evaluation_node.py &
  PY_PID="$!"

  python3 run_competition.py --vision_based --num_lstm_layers 5.0 --model_type "ViTLSTM" --model_path ../../models/ViTLSTM_model.pth &
  COMP_PID="$!"

  cd -

  sleep 2.0

  # Wait until the evaluation script has finished
  # Wait until the evaluation script has finished
  while ps -p $PY_PID > /dev/null
  do
    echo
    echo [LAUNCH_EVALUATION] Sending start navigation command
    echo
    rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
    sleep 2

    # if the current iteration has surpassed the time limit, something went wrong (possibly: [Pipeline]     Bridge failed!). Kill the simulator.
    if ((($(date +%s) - start_time) >= 300))
    then
      echo
      echo
      echo
      echo
      echo "Time limit exceeded. Exiting evaluation script loop."
      echo
      echo
      echo
      echo
      kill -SIGINT $PY_PID
      break
    fi

  done


  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
  mv "tmp.yaml" "$SUMMARY_FILE"

  kill -SIGINT "$COMP_PID"
done


if [ $ROS_PID ]
then
  kill -SIGINT "$ROS_PID"
fi
