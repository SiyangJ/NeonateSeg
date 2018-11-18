#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --time=2-0:00:00
#SBATCH --job-name=tensorboard
#SBATCH --output=/nas/longleaf/home/siyangj/TB_logs/tensorboard-%J.log
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:0

unset OMP_NUM_THREADS

# Set SIMG path
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/1.9.0/simg

# Set SIMG name
SIMG_NAME=tensorflow1.9.0-py3-cuda9.0-ubuntu16.04.simg

DATA_PATH=/nas/longleaf/home/siyangj/

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=longleaf

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.unc.edu
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}.unc.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# DON'T USE ADDRESS BELOW. 
# DO USE TOKEN BELOW

singularity exec --nv -B /nas -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "export LC_ALL=C;cd $DATA_PATH;tensorboard --logdir=$1 --port=${port} --ip=${node}"
