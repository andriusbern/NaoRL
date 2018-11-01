#!/bin/bash
CHORE_PATH="`sed -n 2p settings.txt`"
VREP_PATH="`sed -n 1p settings.txt`"
cd "$CHORE_PATH"
./naoqi-bin -p 5995 &
sleep 2
cd "$VREP_PATH"
source vrep.sh &

