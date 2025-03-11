#!/bin/bash 
#SBATCH --mail-user=anhcao@tnt.uni-hannover.de          # Only <UserName>@tnt.uni-hannover.de is allowed
#SBATCH --mail-type=ALL                                 # Email on job start/end
#SBATCH --job-name=MyTestjob                            # Job name in the job history
#SBATCH --output=Logs/Slurm/slurm-%j-out.txt            # Output log file (job ID replaces %j)
#SBATCH --time=139:00:00                                # Maximum run time (HH:MM:SS)
#SBATCH --partition=gpu_long_stud                       # Partition for the job
#SBATCH --nodes=1                                       # Reserve 1 compute node
#SBATCH --tasks-per-node=6                              # Reserve 6 tasks per node
#SBATCH --cpus-per-task=4                               # Reserve 4 CPUs per task
#SBATCH --mem=64G                                       # Reserve 64GB RAM
#SBATCH --gres=gpu:1                                    # Reserve 1 GPU

# Change to working directory
cd ~/tmp/GitHub/DLProject_1/ModelTraining/

# Activate conda environment
source ~/tmp/miniconda3/etc/profile.d/conda.sh
conda activate MaProject

# Export or check PYTHONPATH
export PYTHONPATH="~/tmp/miniconda3/envs/MaProject/bin/python":$PYTHONPATH
echo "PYTHONPATH=$PYTHONPATH"

# Run training script with arguments
python3 Wafer-map/main_train.py --mode continual --config Wafer-map/config.yaml

# python3 Drive-data/Drive_data_dataset_training.py --config Drive-data/config.yaml

# python3 ~/tmp/GitHub/DLProject_1/Slurm-script-runner/test2.py

