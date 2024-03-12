#!/bin/bash
#SBATCH --job-name=dsa:clustering                        # job name
#SBATCH --partition=GPUA100                              # select partion GPU
#SBATCH --nodes=1                                        # number of nodes requested by user
#SBATCH --gres=gpu:1                                     # use generic resource GPU, format: --gres=gpu:[n], n is the number of GPU card
#SBATCH --time=1-00:00:00                                # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=cuda.%j.out                             # redirect both standard output and erro output to the same file

source activate py39-cu117
export PYTHONPATH=/home2/s224636/Documents/Spatial_Biology_Project/src:/home2/s224636/Documents/dinov2

python run_dsa_clustering.py --src "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_CODEX/output/P2T1.csv" \
                             --out "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_CODEX/output/P2T1" \
                             --method "Leiden" --method_arg 0.2
