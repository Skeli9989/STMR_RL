sudo rm -rf ~/go1_gym
docker stop foxy_controller
docker rm foxy_controller
sudo kill $(ps aux |grep lcm_position | awk '{print $2}')
