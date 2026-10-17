# g1_files/env.sh : source this in every robot terminal
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"                 # folder that contains g1_files and nav_source

source /opt/ros/foxy/setup.bash
source ~/ws_livox/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export PYTHONPATH="$ROOT/pylibs:$ROOT/nav_source:$HERE:$PYTHONPATH"
cd "$HERE"