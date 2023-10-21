'''
Iterate through (1) findCandidates and (2) discoveryGraph until we find a path with KL-divergence of 0
'''
import time
import os
import glob
import pandas as pd
import argparse
import json
import sys
sys.path.append('../findCandidates/')
import set_similarity
sys.path.append('../')
import recover_matrix_ternary
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="tpch", choices=['tpch', 'santos_large_tpch', 'tpch_groundtruth', 'tpch_small', 'tpch_large',
                                                                          't2d_gold', 'TUS_t2d_gold', 'wdc_t2d_gold'])
    parser.add_argument("--saveMT", type=int, default=1)
    hp = parser.parse_args()
    runStarmie = 0
    if hp.benchmark == 'santos_large_tpch':
        runStarmie = 1
    
    FILEPATH = '/home/gfan/Datasets/%s/queries/' % (hp.benchmark)  
    datasets = glob.glob(FILEPATH+'*.csv')
    candidateTablePath = "../results_candidate_tables/%s/candidateTables.json" % (hp.benchmark)
    with open(candidateTablePath) as json_file: allCandidateTableDict = json.load(json_file)
    
    avgRuntimes = {k: [] for k in ['matrix_initialization', 'matrix_traversal','all']}
    eachRuntimes = {}
    numSources = 0
    avgSizeOutput = []
    algStartTime = time.time()
    
    allOriginsForSources = {}
    for indx in tqdm(range(0, len(datasets))):
        source_table = datasets[indx].split(FILEPATH)[-1] 
        print("\t=========== %d) Source Table: %s =========== " % (indx, source_table))
        currStartTime = time.time()
        originating_tables, ternMatrixRuntimes = recover_matrix_ternary.main(hp.benchmark, source_table)
        if originating_tables: allOriginsForSources[source_table] = {t: allCandidateTableDict[source_table][t] for t in originating_tables}
        if not ternMatrixRuntimes:
            print("There were no candidates found in TERNARY MATRIX TRAVERSAL")
            continue
        
        avgRuntimes['matrix_initialization'] += ternMatrixRuntimes['matrix_initialization']
        avgRuntimes['matrix_traversal'] += ternMatrixRuntimes['matrix_traversal']
        avgRuntimes['all'].append(time.time() - currStartTime)
        
        eachRuntimes[source_table] = {'matrix_initialization': ternMatrixRuntimes['matrix_initialization'], 'matrix_traversal': ternMatrixRuntimes['matrix_traversal'],
                                                'all': time.time() - currStartTime}
        
        numSources += 1
    
    if hp.saveMT:
        ORIGIN_TABLE_PATH = "../results_candidate_tables/%s/originatingTables.json"%(hp.benchmark)
        try: os.remove(ORIGIN_TABLE_PATH)
        except FileNotFoundError: pass
        with open(ORIGIN_TABLE_PATH, "w+") as f: json.dump(allOriginsForSources, f, indent=4)
    print("\t\t\t=================================")
    
    TIMESTATS_OUTPUT_DIR = "../experiment_logs/%s/"%(hp.benchmark)
    if not os.path.exists(TIMESTATS_OUTPUT_DIR):
      os.makedirs(TIMESTATS_OUTPUT_DIR)
    sourceStats = {'num_sources': numSources}
    for k, v_list in avgRuntimes.items():
        sourceStats['avg_'+k] = sum(v_list)/len(v_list)
    sourceStats.update(eachRuntimes)
    with open(TIMESTATS_OUTPUT_DIR+"runtimes_genT.json", "w+") as f: json.dump(sourceStats, f, indent=4)
    print("FINISHED ALL %d SOURCES IN %.3f seconds (%.3f minutes, %.3f hrs)" % (numSources, time.time() - algStartTime, (time.time() - algStartTime)/60, ((time.time() - algStartTime)/60)/60))