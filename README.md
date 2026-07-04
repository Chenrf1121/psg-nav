# PSG-Nav

This repository contains the codebase for **PSG-Nav**, an object-goal navigation system built on top of the original SG-Nav codebase and extended for our PSG-Nav paper accepted to **ICML 2026**.

We are currently cleaning and reorganizing the released code. The first release target is to make the code easy to run on the **HSSD ObjectNav benchmark**.

## Status

This repository is under active release preparation.

Current focus:

- HSSD benchmark adaptation and evaluation scripts.
- PSG-Nav naming cleanup after migration from SG-Nav.
- Local LLM/VLM server integration with Qwen2.5 models.
- Removal of old debug-only and ablation-only code paths.
- Minimal release documentation for reproducible HSSD evaluation.

Supported dataset entry points are registered in [configs/dataset_registry.py](configs/dataset_registry.py):

| Dataset key | Config | Default split | Notes |
| --- | --- | --- | --- |
| `hssd` | `configs/challenge_objectnav_hssd.local.rgbd.yaml` | `val` | Primary benchmark target for the current cleanup |

## Installation

The environment is based on the Habitat navigation stack. We recommend naming the navigation environment `psg_nav`.

### 1. Create Environment

```bash
conda create -n psg_nav python=3.9
conda activate psg_nav
```

Install Habitat:

```bash
conda install habitat-sim==0.2.4 -c conda-forge -c aihabitat
pip install -e habitat-lab
```

Replace the Habitat-Sim agent wrapper:

```bash
HABITAT_SIM_PATH=$(pip show habitat_sim | grep 'Location:' | awk '{print $2}')
cp tools/agent.py ${HABITAT_SIM_PATH}/habitat_sim/agent/
```

Install core packages:

```bash
conda install -c pytorch faiss-gpu=1.8.0
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Install Grounded-SAM dependencies:

```bash
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
mkdir -p data/models
wget -O data/models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O data/models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Install GLIP:

```bash
cd GLIP
python setup.py install
mkdir -p MODEL
cd MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
cd ../../
```

### 2. LLM/VLM Server Environment

PSG-Nav uses a local Flask server for text and vision-language model calls. The current code path uses:

- LLM: `/data/models/Qwen2.5-7B-Instruct`
- VLM: `/data/models/Qwen2.5-VL-7B-Instruct`

The server is launched from [server.py](server.py). In our local scripts the server environment is named `mllm`; make sure that environment contains a recent enough `transformers` version for Qwen2.5-VL and a PyTorch build compatible with your CUDA driver.

Run the server manually:

```bash
conda activate mllm
python server.py --port 5001
```

Then run PSG-Nav in another terminal:

```bash
conda activate psg_nav
python PSG_Nav.py --dataset hssd --server_port 5001
```

## HSSD Benchmark Setup

HSSD is the first benchmark we are adapting and validating for the cleaned PSG-Nav release.

Expected structure:

```text
data/HSSD/
├── datasets/
│   └── objectnav/
│       └── hssd/
│           └── v1/
│               ├── train/
│               │   └── train.json.gz
│               └── val/
│                   └── val.json.gz
└── scene_datasets/
    └── hssd/
        └── *.glb
```

Recommended download path:

```bash
bash tools/download_hssd_objectnav.sh
```

After downloading, verify:

```bash
ls -lh data/HSSD/datasets/objectnav/hssd/v1/val/val.json.gz
ls data/HSSD/scene_datasets/hssd | head
```

The HSSD config is:

```text
configs/challenge_objectnav_hssd.local.rgbd.yaml
```

Update `SCENES_DIR` and `DATA_PATH` in that config if your dataset is stored outside the default `data/HSSD/` layout.

## Running Evaluation

### Single Process

Run HSSD validation:

```bash
python PSG_Nav.py --dataset hssd --split_l 0 --split_r 40 --server_port 5001
```

Run a smaller range while debugging:

```bash
python PSG_Nav.py --dataset hssd --split_l 0 --split_r 5 --server_port 5001 --visualize
```

### Parallel Runner

Use [run_parallel_with_server.py](run_parallel_with_server.py) to start one server and one PSG-Nav process per GPU:

```bash
python run_parallel_with_server.py --gpus 5 --dataset hssd
```

Multiple GPUs:

```bash
python run_parallel_with_server.py --gpus 4,5,6,7 --dataset hssd
```

Manual scene allocation:

```bash
python run_parallel_with_server.py --gpus 4,5 --dataset hssd --scenes-per-gpu 20,20
```

Resume from a later scene:

```bash
python run_parallel_with_server.py --gpus 5 --dataset hssd --start-scene 20
```

Logs are written to:

```text
logs/parallel_run_<timestamp>/
```

Results and visualizations are written to timestamped folders under:

```text
data/results_<dataset>_<timestamp>/
data/visualization_<dataset>_<timestamp>/
```

## Repository Layout

```text
PSG_Nav.py                    Main navigation entry point
server.py                     Local Flask LLM/VLM server
run_parallel_with_server.py   Multi-GPU evaluation launcher
configs/                      Dataset registry and Habitat configs
graph/                        Scene graph and semantic reasoning modules
utils/                        Mapping, detection, planning, landmark selection
visualizations/               Visualization utilities
```

## Notes for Current Release Cleanup

- The codebase originated from SG-Nav and is being renamed to PSG-Nav.
- The recommended conda environment name for this release is `psg_nav`.
- Old ablation/debug options have been removed from the main runtime path.
- The local server no longer uses Ollama/llama3.2-vision in the current PSG-Nav path; it uses Qwen2.5 LLM/VLM checkpoints.
- HSSD support is the first benchmark target for this release pass.

## Acknowledgements

This repository builds on the original [SG-Nav](https://github.com/bagh2178/SG-Nav) codebase and the Habitat ecosystem. We thank the authors and maintainers of SG-Nav, Habitat-Sim, Habitat-Lab, GroundingDINO, Segment Anything, GLIP, and related open-source projects.

## Citation

The PSG-Nav paper has been accepted to ICML 2026. Citation information will be added here after the final camera-ready metadata is available.
