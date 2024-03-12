source activate py39-cu117

export PYTHONPATH=/work/bioinformatics/s217053/DINO/DINOv2/Spatial_Biology_Project-kate/src:/work/bioinformatics/s217053/DINO/DINOv2/dinov2-kate:/work/bioinformatics/s217053/DINO/DINOv2/Spatial_Biology_Project-kate/examples

python /work/bioinformatics/s217053/DINO/DINOv2/dinov2-kate/dinov2/run/train/train.py \
       --nodes 1 --partition=GPU4v100 --gpus-per-node 4 \
       --config-file /work/bioinformatics/s217053/DINO/DINOv2/Spatial_Biology_Project-kate/src/myconfig.yaml \
       --output_dir /archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_root/outputs \
       train.dataset_path=TiffDataset:split=TRAIN:root=/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_root:extra=/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_root/metadata:seg=/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/train:names=/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/channelNames:picks=/archive/bioinformatics/Jamieson_lab/shared/spatial-core/K/analysis/huang/EO_AO/TMA_0622/DinoV2_in/channelPicks:size=64
