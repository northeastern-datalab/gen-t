import os
import json
import glob
import argparse
from tqdm import tqdm
from ver_cli import VerCLI

'''
Run for every source table
'''
def offline_professing(ver, source_name, runOnIntegSet, integration_set_dict, path_to_csv_files, profile_path):
    print("======= Processing Source %s" % (source_name))
    # Remove any source config file that may be there
    sources_file_name = "tptr_"+ source_name
    if os.path.exists(sources_file_name):
        os.remove(sources_file_name)
    ## Run the following three VerCLI scripts one time, then build index for each source you want to run for
    # Create Source Config File
    ver.create_sources_file(sources_file_name)
    print("Finished Creating Source Config File")
    # Add csv files to the sources list 
    if runOnIntegSet:
        # get integration set for each source table
        csv_files = integration_set_dict[source_name]
    else:
        # get all data lake tables for each source table
        csv_files = [file.split("/")[-1] for file in glob.glob(path_to_csv_files+"/*.csv")]
    for csv_name in tqdm(csv_files):
        ver.add_csv(sources_file_name, csv_name, path_to_csv_files)
    print("Finished Adding CSV files")
    # Build Data Profiles
    ver.profile(sources_file_name, profile_path)
    print("Finished building data profiles")
    
def main(source_table, runOnIntegSet=1):
    
    path_to_csv_files = "../../Datasets/tptr_small/datalake"
    profile_path = "../../Datasets/tptr_small/data_profiles"
    ver = VerCLI()
    
    if runOnIntegSet:
        path_to_integ_set_json = "../tptr_small_integSet.json"
        with open(path_to_integ_set_json) as f: 
            integration_set_dict = json.load(f) # source table: [expected integration set, including err and null tables]
            profile_path += "_%s" % (source_table)
            offline_professing(ver, source_table, runOnIntegSet, integration_set_dict, path_to_csv_files, profile_path)
            
    else:
        profile_path += "_%s" % ("all")
        offline_professing(ver, "all", runOnIntegSet, {}, path_to_csv_files, profile_path)
    
    
    # Run for each run you want to do 
    # Build discovery indices on top of data profiles
    ver.build_dindex(profile_path, force=True)
    
    