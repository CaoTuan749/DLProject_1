#!/bin/bash 
#SBATCH --mail-user=anhcao@tnt.uni-hannover.de # only <UserName>@tnt.uni-hannover.de is allowed as mail address
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=MyTestjob        # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=slurm-%j-out.txt   # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)

#SBATCH --time=01:30:00             # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --partition=cpu_normal_stud     # Partition auf der gerechnet werden soll. Ohne Angabe des Parameters wird auf der
                                    #   Default-Partition gerechnet. Es können mehrere angegeben werden, mit Komma getrennt.
#SBATCH --tasks-per-node=4          # Reservierung von 4 CPUs pro Rechenknoten
#SBATCH --mem=10G                   # Reservierung von 10GB RAM

cd

cd tmp/GitHub/DLProject_1

#source ~/GitHub/DLProject_1/.venv/bin/activate

source ~/tmp/miniconda3/etc/profile.d/conda.sh

export PYTHONPATH="~/tmp/miniconda3/envs/MaProject/bin/python":$PYTHONPATH

echo PYTHONPATH=$PYTHONPATH

conda activate MaProject

#python Test_5/test.py

#export PYTHONPATH='~/GitHub/DLProject_1/.venv/lib/python3.11':$PYTHONPATH



# python3 Test_5/test.py

# python3 ModelTraining/Drive-data-dataset-remake.py

python3 ModelTraining/Wafer-data-dataset-pad.py

#deactivate
