#!/bin/bash
function cleanup() {
  xhost -local:root
  containers=$(docker container ls -f "name=demo" -aq)
  docker container stop "$containers"
  exit 1
}

demos="demos."
function main() {
  xhost +local:root
  docker run --rm -it --gpus all --net host --privileged --env NVIDIA_DISABLE_REQUIRE=1 --name "demo" --shm-size 64g\
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  \
  -v /home/"${USER}"/ivy_vision:/ivy_vision \
  \
  -v /home/"${USER}"/ivy/ivy:/ivy/ivy \
  -v /home/"${USER}"/demo_utils/ivy_demo_utils:/demo-utils/ivy_demo_utils \
  -v /home/"${USER}"/builder/ivy_builder:/builder/ivy_builder \
  -v /home/"${USER}"/mech/ivy_mech:/mech/ivy_mech \
  \
  -v /home/"${USER}"/PyRep/pyrep:/PyRep/pyrep \
  \
   unifyai/vision:latest python3 -m $demos"$1" "${@:2}"
}

main "$@" || cleanup
