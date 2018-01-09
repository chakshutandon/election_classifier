#!/usr/bin/env bash

sudo ln -s /vagrant/* /home/ubuntu/

sudo apt-get update
sudo apt-get -y install python3
sudo apt-get -y install python3-pip

sudo pip3 install --upgrade pip
sudo pip3 install -r /home/ubuntu/requirements.txt

# Use token/password for production environment
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --notebook-dir='/home/ubuntu/Notebooks' --NotebookApp.token='' &


