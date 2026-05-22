# Workflow: CHPC and Colab

## Local development

```bash
micromamba env create -f environment.yml -n csdv
micromamba activate csdv
pip install -e ".[dev]"
pytest
```

## CHPC (SLURM)

```bash
# Login node:
git clone <repo> ~/csdv && cd ~/csdv
micromamba env create -f environment.yml -p ./.micromamba/csdv
micromamba activate ./.micromamba/csdv
pip install -e ".[lidr]"

export CSDV_DATA_ROOT=/scratch/.../csdv/data
export CSDV_RESULTS_ROOT=/scratch/.../csdv/results

# Submit a batch job:
sbatch scripts/slurm/run_chm_inference.sbatch --site SCBI --year 2022
```

CHPC nodes have R and `lidR` available; use the `lidr` optional extra.

## Google Colab

```python
!git clone <repo> /content/csdv
%cd /content/csdv
!pip install -e .   # no [lidr] extra; rpy2 not supported

import os
os.environ["CSDV_DATA_ROOT"] = "/content/drive/MyDrive/csdv/data"
os.environ["CSDV_RESULTS_ROOT"] = "/content/drive/MyDrive/csdv/results"

from csdv_core.chm_inference import infer
```

On Colab, crown segmentation falls back to `segmentation.chm_watershed` (Python) since `lidr_bridge` requires rpy2 + R.

## Notebooks

Exploration notebooks live in `notebooks/` and import from `csdv_core`. Anything that grows beyond a screen of inline logic gets promoted to a module under `src/csdv_core/`.
