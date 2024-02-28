import json
import add_csv_tables
import ver_quick_start
import glob
from tqdm import tqdm
import time

SOURCES_TABLE_FILEPATH = "../../Datasets/tptr_small/queries/"
previous_output_filepath = "materialized_views_fullDL.json"
with open(previous_output_filepath) as f: source_views = json.load(f)

all_sources = glob.glob(SOURCES_TABLE_FILEPATH+"*.csv")
for source_table in tqdm(all_sources[1:]):
    source_name = source_table.split("/")[-1]
    if source_name in source_views: continue
    # Create Source Config File
    # Add CSV files to sources list
    # Build Data Profiles
    # Build discovery index on top of data profiles
    start_time = time.time()
    add_csv_tables.main(source_name)
    print("Finished offline processing in %.3f seconds" % (time.time()-start_time))
    
    # Query with example table, and return a list of materialized views (post-pruning)
    start_time = time.time()
    ver_quick_start.main(source_table)
    print("Finished querying in %.3f seconds" % (time.time()-start_time))
    
    # Next Step: Evaluation
    # python ver_outputs/evaluate_projected_outputs.py