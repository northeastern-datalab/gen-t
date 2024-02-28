'''
Iterate through (1) findCandidates and (2) discoveryGraph until we find a path with KL-divergence of 0
'''
import time
import os
import glob
import pandas as pd
import argparse
import sys
import json
sys.path.append('../findCandidates/')
import set_similarity
sys.path.append('../')
import recover_matrix_ternary
from tqdm import tqdm

def get_lake(benchmark):
    '''
    Get data lake tables found from Starmie
    Args: 
        benchmark(str): benchmark name for filepath
        sourceTableName (str): name of source table
        discludeTables (list): tables to disclude from data lake tables (source)
        includeStarmie (Boolean): get candidate tables from Starmie results or not
    Return: 
        lakeDfs(dict of dicts): filename: {col: list of values as strings}
        rawLakeDfs (dict): filename: raw DataFrame
    '''
    DATALAKE_PATH = '/home/gfan/Datasets/%s/' % (benchmark)
    lakeTables = []
    # Import all data lake tables
    totalLakeTables = glob.glob(DATALAKE_PATH+'datalake/*.csv')

    rawLakeDfs = {}
    allLakeTableCols = {}
    for filename in totalLakeTables:
        table = filename.split("/")[-1]
        df = pd.read_csv(filename, lineterminator="\n")
        rawLakeDfs[table] = df
        
        for index, col in enumerate(df.columns):    
            if table not in allLakeTableCols: allLakeTableCols[table] = {}
            # Convert every data value to strings, so there are no mismatches from data types
            allLakeTableCols[table][col] = [str(val).rstrip() for val in df[col] if not pd.isna(val)]
    return rawLakeDfs, allLakeTableCols

def get_starmie_candidates(benchmark):
    '''
    Get data lake tables found from Starmie
    Args: 
        benchmark(str): benchmark name for filepath
    Return: 
        starmieCandidatesForSources(dict): source table: list of candidate tables returned from Starmie
    '''
    # ==== Import the tables returned from Starmie and use that as reduced data lake
    with open("../../Starmie_candidate_results/%s/starmie_candidates.json" % (benchmark)) as json_file: starmieCandidatesForSources = json.load(json_file)
    return starmieCandidatesForSources

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="tptr", choices=['tptr', 'santos_large_tptr', 'tptr_groundtruth', 'tptr_small', 'tptr_large',
                                                                          't2d_gold', 'TUS_t2d_gold', 'wdc_t2d_gold'])
    parser.add_argument("--threshold", type=float, default=0.2)
    hp = parser.parse_args()
    
    runStarmie = 0
    if hp.benchmark == 'santos_large_tptr':
        runStarmie = 1
    
    FILEPATH = '/home/gfan/Datasets/%s/queries/' % (hp.benchmark)  
    datasets = glob.glob(FILEPATH+'*.csv')
    
    print("\t\t\t=========== GETTING DATA LAKE TABLES ===========")
    lakeDfs, allLakeTableCols = get_lake(hp.benchmark)
    print("%d lakeDfs" % (len(lakeDfs)))
    if runStarmie: starmie_candidates = get_starmie_candidates(hp.benchmark)
    CANDIDATE_OUTPUT_DIR = "../results_candidate_tables/%s/"%(hp.benchmark)
    if not os.path.exists(CANDIDATE_OUTPUT_DIR):
      os.makedirs(CANDIDATE_OUTPUT_DIR)
    allCandidatesForSources = {}
    for indx in tqdm(range(0, len(datasets))):
        source_table = datasets[indx].split(FILEPATH)[-1] 
        print("\t=========== %d) Source Table: %s =========== " % (indx, source_table))
        if runStarmie: source_candidates = starmie_candidates[source_table.replace('.csv', '')]
        else: source_candidates = []
        candidateTablesFound, noCandidates = set_similarity.main(hp.benchmark, source_table, hp.threshold,lakeDfs, allLakeTableCols, source_candidates)
        if noCandidates: 
            print("There were no candidates found (no candidate for key)")
        allCandidatesForSources[source_table] = candidateTablesFound
    CANDIDATE_TABLE_PATH = CANDIDATE_OUTPUT_DIR + "candidateTables.json"
    try: os.remove(CANDIDATE_TABLE_PATH)
    except FileNotFoundError: pass
    # Save JSON with Source Table: {Candidate Table: {Candidate's column name: mapped Source's column name}}
    with open(CANDIDATE_TABLE_PATH, "w+") as f: json.dump(allCandidatesForSources, f, indent=4)