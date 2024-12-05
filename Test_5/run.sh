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

cd GitHub/DLProject_1

source .venv/bin/activate

python3 Test_5/test.py

deactivate