#!/bin/bash
sudo ifconfig enp108s0 down
sudo ifconfig enp108s0 192.168.123.162/24
sudo ifconfig enp108s0 up
ping 192.168.123.161
