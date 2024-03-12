# Spatial_Biology_Project
This repo is for single cell analysis, start with DINO, keep update and will be migrated to UTSW GitLab in the future

## Run the Code
### Set PYTHONPATH
set /src in this repo and dinov2 repo as the PYTHONPATH
```
export PYTHONPATH=/home2/s224636/Documents/Spatial_Biology_Project/src:/home2/s224636/Documents/dinov2
```
### Quick examples
```
cd examples/playground
python example_*.py
```
### DINOv2
change configurations
```
cd src/
vim myconfig.yaml
```
create metadata
```
cd examples/train
sbatch run_dinov2_metadata.sh
```
run training script
```
cd examples/train
./run_dinov2_training.sh
```
run inference script
```
cd examples/inference
sbatch run_dinov2_inference.sh
```
### Downstream analysis and validation
run clustering/tSNE
```
cd examples/down_stream_analysis
sbatch run_dsa_clustering.sh
sbatch run_dsa_tSNE.sh
```
check data augmentation/PCA
```
cd examples/validation
sbatch run_dinov2_transform_visualization.sh
sbatch run_dinov2_PCA_visualization.sh
```

## Set Environment
### Fedora Linux
```
module load firefox/latest
module load git/v2.5.3
module load python/3.8.x-anaconda
module load ImageJ/latest
module load cuda112/toolkit/11.2.0
```
### Venv's
find the path of conda configuration file .condarc, then create one for specifying path for /envs and /pkgs
```
conda info
```
add these lines to .condarc if you want to keep these huge files somewhere else
```
pkgs_dirs:
  - /work/to/your/path/.conda/pkgs
envs_dirs:
  - /work/to/your/path/.conda/envs
```

## Activate Venv
### Add these to `requirements.txt`
```
--extra-index-url https://download.pytorch.org/whl/cu117
torch==2.0.0
torchvision==0.15.0
omegaconf
torchmetrics==0.10.3
fvcore
iopath
xformers==0.0.18
submitit
--extra-index-url https://pypi.nvidia.com
cuml-cu11
```
### Start from very begining using `conda.yaml`
```
conda env create -f conda.yaml
conda activate py39-cu117
```

## File Structure
#### Project/Dinov2
```
-- work_path
   |-- Spatial_Biology_Project
   |   |-- examples
   |   |-- src
   |   `-- README.md
   `-- dinov2
       |-- dinov2
       |   |-- data
       |   | ...
       |   |-- train
       |   |-- utils
       |   `--__init__.py
       |  ...
       `-- README.md
```

#### Dinov2 Training
```
-- training_supplies_data_path
   |-- output
   |   |-- test_0.csv
   |   `-- ...
   |-- channelNames
   |   |-- sample_0.txt
   |   `-- ...
   |-- channelPicks
   |   |-- sample_0.txt
   |   `-- ...
   |-- train
   |   |-- sample_0.csv
   |   `-- ...
   |-- val
   |   |-- sample_0.csv
   |   `-- ...
   `-- test
       |-- test_0.csv
       |-- channelNames.txt
       `-- channelPicks.txt

-- training_data_path
   |-- train
   |   |-- sample_0.tif
   |   `-- ...
   |-- val
   |   |-- sample_0.tif
   |   `-- ...
   |-- test
   |   `-- test_0.tif
   |-- models
   |   |-- vitb14_pretrain.pth
   |   `-- ...
   |-- metadata
   |   |-- class_ids_TRAIN.npy
   |   `-- ...
   |-- outputs
   |   |-- logs
   |   |-- evals
   |   `-- checkpoints.pth
   |-- additional
   |   |-- train
   |   |   |-- sample_0.tif
   |   |   `-- ...
   |   |-- val
   |   |   |-- sample_0.tif
   |   |   `-- ...
   |   `-- test
   |       `-- test_0.tif
   `-- label.txt
```

#### label.txt: `class_id, class_name`
```
sample_0,single cell 0
sample_1,single cell 1
...
```
