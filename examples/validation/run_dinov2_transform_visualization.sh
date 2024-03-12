#!/bin/bash
#SBATCH --job-name=dinov2:trans_visualization            # job name
#SBATCH --partition=GPUA100                              # select partion GPU
#SBATCH --nodes=1                                        # number of nodes requested by user
#SBATCH --gres=gpu:1                                     # use generic resource GPU, format: --gres=gpu:[n], n is the number of GPU card
#SBATCH --time=1-00:00:00                                # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=cuda.%j.out                             # redirect both standard output and erro output to the same file

source activate py39-cu117
export PYTHONPATH=/home2/s224636/Documents/Spatial_Biology_Project/src:/home2/s224636/Documents/dinov2

python example_11_run_dinov2_transform_visualization.py --src "backbone" \
                                          --model "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_dinov2/models/dinov2_vitb14_pretrain.pth" \
                                          --data "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_dinov2/test/test_0.tif" \
                                          --seg "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_CODEX/test/test_0.csv" \
                                          --ws 64 \
                                          --chans 37 \
                                          --names "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_CODEX/test/channelNames.txt" \
                                          --picks "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_CODEX/test/channelPicks.txt" \
                                          --add "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_dinov2/additional/test/test_0.tif" \
                                          --out "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/augmentations"
