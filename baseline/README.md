# Baselines

This folder contains code to run an implementation of our baselines [Auto-Pipeline](https://www.vldb.org/pvldb/vol14/p2563-he.pdf) and [Ver](https://arxiv.org/pdf/2106.01543.pdf). 


## Baseline: Auto-Pipeline*
To run this baseline (in folder `auto-pipeline*/`):
```
python runFoofah.py
```
This calls `to_json.py` to convert every table to the format needed by a modification of Explain-Da-V's implementation of Auto-pipeline (https://github.com/shraga89/ExplainDaV/tree/main). The set of tables given is the same set of candidate tables.


## Baseline: Ver
We ran the code provided by Ver's authors (https://github.com/TheDataStation/ver). 
To run this baseline (in folder `ver-main/`):
```
python run_ver.py
```
This calls add_csv_tables.py that performs preprocessing (e.g., indexing each source table's integration set, found in `tptr_small_integSet.json`).

Ver produces list of views that are recorded in `materialized_views_fullDL.json`. To evaluate the results, you can run `evaluate_projected_outputs.py`. The output tables from Ver are in folder `ver_outputs/tptr_small`, and the results reported in the paper are found in `ver_outputs/Ver_proj_result_stats.json`.