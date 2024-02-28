'''
Iterate through (1) findCandidates and (2) discoveryGraph until we find a path with KL-divergence of 0
'''
import time
import os
import glob
import pandas as pd
import sys
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

if __name__ == '__main__':
    benchmark = 'tptr'
    # benchmark = 'santos_large_tptr'
    # benchmark = 'tptr_groundtruth'
    # benchmark = 'tptr_0_groundtruth'
    # benchmark = 't2d_gold'
    # benchmark = 'tptr_small'
    # benchmark = 'tptr_large'
    # benchmark = 'TUS_t2d_gold'
    # benchmark = 'wdc_t2d_gold'
    runStarmie = 0
    if benchmark == 'santos_large_tptr':
        runStarmie = 1
    saveMT = 1
    
    FILEPATH = '/home/gfan/Datasets/%s/queries/' % (benchmark)  
    datasets = glob.glob(FILEPATH+'*.csv')
    allTernTDR_recall, allTernTDR_prec = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    allTernInstanceSim, allTernDkl = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    allTernRuntimes, ternRuntimes = {k: [] for k in ['matrix_initialization', 'matrix_traversal','all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    
    numSources = 0
    avgSizeOutput = []
    algStartTime = time.time()
    
    print("\t\t\t=========== GETTING DATA LAKE TABLES ===========")
    lakeDfs, allLakeTableCols = get_lake(benchmark)
    print("%d lakeDfs" % (len(lakeDfs)))
    for indx in tqdm(range(0, len(datasets))):
        source_table = datasets[indx].split(FILEPATH)[-1] 
        threshold = 0.2
        print("\t=========== %d) Source Table: %s =========== " % (indx, source_table))

        print("\t\t\t=========== BEGIN SET SIMILARITY ===========")
        setSimTime = time.time()
        noCandidates = set_similarity.main(benchmark, source_table, threshold,lakeDfs, allLakeTableCols,includeStarmie=runStarmie)
        if noCandidates: 
            print("There were no candidates found (no candidate for key)")
            continue
        print("\t\t\t=========== END SET SIMILARITY FOR %s IN %.3f sec ===========\n" % (source_table, time.time() - setSimTime))
        
        print("\t\t\t=========== BEGIN TERNARY MATRIX TRAVERSAL ===========")
        currStartTime = time.time()
        ternMatrixTDR_recall, ternMatrixTDR_prec, ternInstanceSim, ternTableDkl, numOutputVals, ternMatrixRuntimes = recover_matrix_ternary.main(benchmark, source_table, saveTraversedTables=saveMT)
        if not ternMatrixRuntimes:
            print("There were no candidates found in TERNARY MATRIX TRAVERSAL")
            # os.remove("../results_candidate_tables/%s/%s_candidateTables.pkl" % (benchmark, source_table))
            continue
        allTernRuntimes['matrix_initialization'] += ternMatrixRuntimes['matrix_initialization']
        allTernRuntimes['matrix_traversal'] += ternMatrixRuntimes['matrix_traversal']
        allTernRuntimes['all'].append(time.time() - currStartTime)
        
        print("\t\t\t=========== END TERNARY MATRIX TRAVERSAL in %.3f seconds (%.3f minutes) =================================" % ((time.time() - currStartTime), (time.time() - currStartTime)/60))
        numSources += 1
    print("\t\t\t=================================")
    print("FINISHED ALL %d SOURCES IN %.3f seconds (%.3f minutes, %.3f hrs)" % (numSources, time.time() - algStartTime, (time.time() - algStartTime)/60, ((time.time() - algStartTime)/60)/60))
    print("\t\tAverage Runtimes: %.3f sec (%.3f min)" % (sum(allTernRuntimes['all'])/len(allTernRuntimes['all']), (sum(allTernRuntimes['all'])/len(allTernRuntimes['all']))/60))
    print("\t\tAverage matrix_initialization Runtimes: %.3f sec (%.3f min)" % (sum(allTernRuntimes['matrix_initialization'])/len(allTernRuntimes['matrix_initialization']), (sum(allTernRuntimes['matrix_initialization'])/len(allTernRuntimes['matrix_initialization']))/60))
    print("\t\tAverage matrix_traversal Runtimes: %.3f sec (%.3f min)" % (sum(allTernRuntimes['matrix_traversal'])/len(allTernRuntimes['matrix_traversal']), (sum(allTernRuntimes['matrix_traversal'])/len(allTernRuntimes['matrix_traversal']))/60))
    
    