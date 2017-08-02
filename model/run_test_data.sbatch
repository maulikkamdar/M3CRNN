#!/bin/bash
#
#SBATCH --job-name=CRNN
#SBATCH --output=run_test_data_%j.txt
#
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres gpu:1

# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lhan2@stanford.edu

module load tensorflow
export OPENCV_OPENCL_RUNTIME=

python run_test_data.py
