# CSDV: Eastern CONUS Forest Disturbance Mapping

Stand-level structural metric extraction and rule-based classification of forest disturbance and post-disturbance recovery from multi-temporal NAIP imagery across the eastern United States.

> Preliminary research code. Lead contributor: Eric L. Bullock, University of Utah.

## What this is

The `csdv_core` package extracts gap fraction, crown width statistics, GLCM texture, cover fractions, and spatial pattern metrics from NAIP (0.6 m RGBN) and NAIP-derived canopy height models, then classifies stands into Oliver-Larson developmental stages and tracks multi-date trajectories across 21 classes.

## Quickstart

```bash
micromamba env create -f environment.yml -n csdv
micromamba activate csdv
pip install -e ".[dev]"
pytest
csdv --help
```

See `docs/workflow_chpc.md` for CHPC and Colab setup.

## Layout

```
src/csdv_core/    # the library
scripts/          # CHPC/Colab batch entry points
notebooks/        # exploration
tests/            # pytest
docs/             # architecture, conventions, data layout, workflow
legacy/           # frozen pre-restructure scratch work (not tracked)
```

Read `docs/architecture.md` for the package map and pipeline diagram.

## Status

The codebase was restructured from a scratch `ProofOfConcept/` workspace. Module skeletons are in place with TODO markers pointing at the legacy code under `legacy/proof_of_concept/`. Implementation proceeds in phases (see `docs/architecture.md`).
