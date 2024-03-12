#!/bin/bash
#SBATCH --job-name=dinov2:metadata                       # job name
#SBATCH --partition=GPUA100                              # select partion GPU
#SBATCH --nodes=1                                        # number of nodes requested by user
#SBATCH --gres=gpu:1                                     # use generic resource GPU, format: --gres=gpu:[n], n is the number of GPU card
#SBATCH --time=1-00:00:00                                # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=cuda.%j.out                             # redirect both standard output and erro output to the same file

source activate py39-cu117

export PYTHONPATH=/work/bioinformatics/s217053/DINO/DINOv2/Spatial_Biology_Project-kate/src:/work/bioinformatics/s217053/DINO/DINOv2/dinov2-kate:/work/bioinformatics/s217053/DINO/DINOv2/Spatial_Biology_Project-kate/examples

python example_10_gen_metadata.py --root "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_root" \
                                  --seg "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/train" \
                                  --names "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/channelNames" \
                                  --picks "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/channelPicks" \
                                  --size 64 \
                                  --out "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_out"
