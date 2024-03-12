#!/bin/bash
#SBATCH --job-name=dinov2:inference                      # job name
#SBATCH --partition=GPUv100s                              # select partion GPU
#SBATCH --nodes=1                                        # number of nodes requested by user
#SBATCH --gres=gpu:1                                     # use generic resource GPU, format: --gres=gpu:[n], n is the number of GPU card
#SBATCH --time=1-00:00:00                                # run time, format: D-H:M:S (max wallclock time)
#SBATCH --output=cuda.%j.out                             # redirect both standard output and erro output to the same file

source activate py39-cu117
export PYTHONPATH=/work/bioinformatics/s217053/DINO/DINOv2/Spatial_Biology_Project-kate/src:/work/bioinformatics/s217053/DINO/DINOv2/dinov2-kate:/work/bioinformatics/s217053/DINO/DINOv2/Spatial_Biology_Project-kate/examples

python example_08_run_dinov2_inference_tiff_only_batches.py --src "checkpoint" \
                                                    --model "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_root/outputs/eval/training_24999/teacher_checkpoint.pth" \
                                                    --data "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_root/test" \
                                                    --seg "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/test" \
                                                    --ws 256 \
                                                    --chans 4 \
                                                    --names "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/channelNames" \
                                  		            --picks "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/channelPicks" \
                                                    --outdir "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_out/feature-sep-21-epoch20" \
						                            --labels '{"sample_0":"A02_T", "sample_1":"A03_T", "sample_2":"A01_N", "sample_3":"E03_T", "sample_4":"A01_T", "sample_5":"E01_T", "sample_6":"E01_N", "sample_7":"E02_T", "sample_8":"E02_N", "sample_9":"E03_N", "sample_10":"A03_N"}' 
