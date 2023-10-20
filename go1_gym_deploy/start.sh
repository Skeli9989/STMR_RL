docker run -d\
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=${XAUTH}" \
    --volume="${XAUTH}:${XAUTH}" \
    --volume="/home/unitree/go1_gym:/home/isaac/go1_gym" \
    --privileged \
    --runtime=nvidia \
    --net=host \
    --workdir="/home/isaac/go1_gym" \
    --name="foxy_controller" \
    jetson-model-deployment tail -f /dev/null
    docker start foxy_controller
docker exec -it foxy_controller /bin/bash