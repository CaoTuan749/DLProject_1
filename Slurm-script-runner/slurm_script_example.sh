#!/bin/bash

#SBATCH --mail-user=lexuan@tnt.uni-hannover.de # only <UserName>@tnt.uni-hannover.de is allowed as mail address
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=CV-KF-5:hpo_FD00*:*_array_norm_SSDA_SSReDist_ReDist_2dFeat_mmd_hsigmoid_c-mm        #  Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=logs/slurm/slurm-%A_%a-out.log   # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurm job-ID ersetzt)

#SBATCH --time=192:00:00            # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --partition=gpu_long     	# Partition auf der gerechnet werden soll. Ohne Angabe des Parameters wird auf der
                                    #   Default-Partition gerechnet. Es können mehrere angegeben werden, mit Komma getrennt.
#SBATCH --nodes=1                  	# Reservierung von 1 Rechenknoten
                                    #   alle nachfolgend reservierten CPUs müssen sich auf den reservierten Knoten befinden
#SBATCH --tasks-per-node=6          # Reservierung von 16 CPUs pro Rechenknoten
#SBATCH --cpus-per-task=4

#SBATCH --mem=64G                   # Reservierung von 32GB RAM
#SBATCH --gres=gpu:1		   		# Reservierung von einer GPU. Es kann ein bestimmter Typ angefordert werden:

#SBATCH --array=0-11

trap "kill 0" SIGINT

home=/data/lexuan
working_dir=$home/workspace/UIDA; cd $working_dir
env_name=CMAPSS
a_python=/home/lexuan/anaconda3/tmp/envs/$env_name/bin/python

export PYTHONPATH=$home/workspace/UIDA:$PYTHONPATH
echo PYTHONPATH=$PYTHONPATH

export PYTHONFAULTHANDLER=1

file_to_run='library/experiments/experiment_runner.py'


#########################
source_data=('FD001' 'FD001' 'FD001' 'FD002' 'FD002' 'FD002' 'FD003' 'FD003' 'FD003' 'FD004' 'FD004' 'FD004')
target_data=('FD002' 'FD003' 'FD004' 'FD001' 'FD003' 'FD004' 'FD001' 'FD002' 'FD004' 'FD001' 'FD002' 'FD003')

save_dir_prefix=tmp/2023-03-14/hpo/cmapss/CV-KF/${source_data[$SLURM_ARRAY_TASK_ID]}

epoch=(100 100 100 100 100 100 100 100 100 100 100 100)

hparams_0='learn_rate,weight_decay,grl_lambda'

config_file='DANN-IDC_CMAPSS_b32_RC2_s-1-inf_h-norm_enc-lin_v1.2.ini'

input_transform=cluster-minmax

$a_python $file_to_run --model DACNN --data 'CMAPSS-'${source_data[$SLURM_ARRAY_TASK_ID]}:'CMAPSS-'${target_data[$SLURM_ARRAY_TASK_ID]} --base_config_file $config_file --save_dir_prefix $save_dir_prefix  --hparams_opt $hparams_0 --hpo_decision_metric v_task_loss,v_domain_mmd --hpo_decision_direction minimize,minimize --n_trials 100 --criterion_task SSReDistLoss --criterion_task_target ReDistLoss --criterion_domain MMDLoss --max_epoch ${epoch[$SLURM_ARRAY_TASK_ID]} --seed_num 0 --output_transform normalize --max_rul 125 --grl_mode r-auto --effective_batch_size 128 --input_transform $input_transform --n_splits 5 --n_folds 5 --cross_validation Kfold

########################
wait
