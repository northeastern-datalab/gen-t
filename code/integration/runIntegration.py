import time
import glob
import pandas as pd
import math
import table_integration, alite_fd_original
from tqdm import tqdm
import sys
sys.path.append('../discovery/')
from evaluatePaths import setTDR, bestMatchingTuples, instanceSimilarity
from calcDivergence import table_Dkl,getQueryConditionalVals

if __name__ == '__main__':
    # benchmark = 't2d_gold'
    # benchmark = 't2d_gold_mtTables'
    
    # benchmark = 'TUS_t2d_gold'
    # benchmark = 'TUS_t2d_gold_mtTables'
    
    # benchmark = 'wdc_t2d_gold'
    # benchmark = 'wdc_t2d_gold_mtTables'
    # timeout = 100 # 100 seconds for t2d
    # ========= TPCH Variations
    # benchmark = 'santos_large_tpch'
    # benchmark = 'santos_large_tpch_groundtruth'
    # benchmark = 'tpch'
    # benchmark = 'tpch_groundtruth'
    # benchmark = 'tpch_0_groundtruth'
    
    # benchmark = 'santos_large_tpch_mtTables'
    # benchmark = 'tpch_mtTables'
    # benchmark = 'tpch_groundtruth_mtTables'
    # benchmark = 'tpch_small_mtTables_groundtruth'
    # benchmark = 'santos_large_tpch_mtTables_groundtruth'
    # benchmark = 'tpch_mtTables_groundtruth'
    # benchmark = 'tpch_large_mtTables_groundtruth'
    
    
    # benchmark = 'tpch_small'
    # benchmark = 'tpch_small_mtTables'
    # benchmark = 'tpch_small_groundtruth'
    
    benchmark = 'tpch_large'
    # benchmark = 'tpch_large_mtTables'
    # benchmark = 'tpch_large_groundtruth'
    timeout = 25263 # 7 hrs for tpch
    runAfterMt = 0
    if '_mtTables' in benchmark: runAfterMt = 1
    performProjSel = 1
    outputDir = "output_tables/"
    if runAfterMt:
        outputDir = "mtAlite_output_tables/"
    elif performProjSel:
        outputDir = "output_tables_projSel/"
        
    benchmarkTitle = benchmark
    # benchmarkTitle = 'santos_large_tpch'
    print("========= Benchmark: ", benchmark)
    FILEPATH = '/home/gfan/Datasets/%s/queries/' % (benchmarkTitle)  
    if '_mtTables' in benchmark and '_groundtruth' not in benchmark: FILEPATH = '/home/gfan/Datasets/%s/queries/' % (benchmark.split('_mtTables')[0])  
    elif '_mtTables_groundtruth' in benchmark: FILEPATH = '/home/gfan/Datasets/%s/queries/' % (benchmark.split('_mtTables_groundtruth')[0])  
    print(FILEPATH)
    datasets = glob.glob(FILEPATH+'*.csv')
    allTDR_recall, allTDR_prec, allInstanceSim, allDkl = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    allRuntimes = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    numSources = 0
    timedOutSources = []
    avgSizeOutput = []
    avgSizeRatio = []
    algStartTime = time.time()
    if '_mtTables' in benchmark: 
        print("Gen-T, Given the Candidate Tables from Set Similarity")
    else:         
        print("ALITE Baseline, Given the Candidate Tables from Set Similarity")
    for indx in tqdm(range(0, len(datasets))):
        source_table = datasets[indx].split(FILEPATH)[-1]
        sourceDf = pd.read_csv(FILEPATH+source_table)
        print("\t=========== %d) Source Table: %s =========== " % (indx, source_table))
        print("Source has %d cols, %d rows --> %d total values" % (sourceDf.shape[1], sourceDf.shape[0], sourceDf.shape[0]*sourceDf.shape[1]))
        primaryKey = sourceDf.columns.tolist()[0]
        foreignKeys = [colName for colName in sourceDf.columns.tolist() if 'key' in colName and colName != primaryKey] # for tpch
        if 't2d_gold' in benchmark:
            # ==== T2D_GOLD Datalake
            # Get another primary key if the first column only has NaN's
            if len([val for val in sourceDf[primaryKey].values if not pd.isna(val)]) == 0:
                for colIndx in range(1, sourceDf.shape[1]):
                    currCol = sourceDf.columns.tolist()[colIndx]
                    if len([val for val in sourceDf[currCol].values if not pd.isna(val)]) > 1:
                        primaryKey = currCol
                        break
            foreignKeys = []
        queryValPairs = getQueryConditionalVals(sourceDf, primaryKey)
        startTime = time.time()
        # RUN WITH MT
        if runAfterMt:
            timed_out, noCandidates, numOutputVals = table_integration.main(benchmark, source_table, timeout)
        # RUN BASELINE
        else:
            timed_out, noCandidates, numOutputVals = alite_fd_original.main(benchmark, source_table, performProjSel, timeout)
        
        if timed_out: 
            print("\t\t\tAlite Timed out for Source Table %s after %.3f seconds =================================" % (source_table, (time.time() - startTime)))
            timedOutSources.append(source_table)
            continue
        if noCandidates: 
            print("\t\t\tAlite Has No Candidates for Source Table %s after %.3f seconds =================================" % (source_table, (time.time() - startTime)))
            continue
        print("\t\t\tAlite Finished for Source Table %s in %.3f seconds (%.3f minutes), \n Output Table of Size %d =================================" % (source_table, (time.time() - startTime), (time.time() - startTime)/60, numOutputVals))
        avgSizeOutput.append(numOutputVals)
        avgSizeRatio.append(numOutputVals/(sourceDf.shape[0]*sourceDf.shape[1]))
        fd_result = outputDir + benchmark+source_table
        fd_result = pd.read_csv(fd_result)
        # Evaluation
        print("BEGIN Evaluation")
        TDR_recall, TDR_precision = setTDR(sourceDf, fd_result)
        bestMatchingDf = bestMatchingTuples(sourceDf, fd_result, primaryKey)
        if bestMatchingDf is None: continue
        instanceSim = instanceSimilarity(sourceDf, bestMatchingDf, primaryKey)
        bestMatchingDf = bestMatchingDf[sourceDf.columns]
        tableDkl, _, colDkls = table_Dkl(sourceDf, bestMatchingDf, primaryKey, queryValPairs, math.inf, log=1)
        if tableDkl == 0.0 or TDR_recall == 1.0 or TDR_precision == 1.0 or instanceSim == 1.0:
            print(" !!!! FOUND COMPLETE PATH !!!")

        print("=== FINISHED Source %s with \n TDR Recall = %.3f, TDR Precision = %.3f, instanceSim = %.3f, KL-DIVERGENCE = %.3f  === " % (source_table, TDR_recall, TDR_precision, instanceSim, tableDkl))
        print("\t\t\t=========== FINISHED in %.3f seconds (%.3f minutes) =================================" % ((time.time() - startTime), (time.time() - startTime)/60))
        if 'tpch' in benchmark and not timed_out and not noCandidates:
            if 'psql_many' in source_table:
                allTDR_recall['manyJoins'].append(TDR_recall)
                allTDR_prec['manyJoins'].append(TDR_precision)
                allInstanceSim['manyJoins'].append(instanceSim)
                allDkl['manyJoins'].append(tableDkl)
                allRuntimes['manyJoins'].append(time.time() - startTime)
            elif 'psql_edge' in source_table:
                allTDR_recall['simple'].append(TDR_recall)
                allTDR_prec['simple'].append(TDR_precision)
                allInstanceSim['simple'].append(instanceSim)
                allDkl['simple'].append(tableDkl)
                allRuntimes['simple'].append(time.time() - startTime)
            else:
                allTDR_recall['oneJoin'].append(TDR_recall)
                allTDR_prec['oneJoin'].append(TDR_precision)
                allInstanceSim['oneJoin'].append(instanceSim)
                allDkl['oneJoin'].append(tableDkl)
                allRuntimes['oneJoin'].append(time.time() - startTime)
                
        allTDR_recall['all'].append(TDR_recall)
        allTDR_prec['all'].append(TDR_precision)
        allInstanceSim['all'].append(instanceSim)
        allDkl['all'].append(tableDkl)
        allRuntimes['all'].append(time.time() - startTime)
        numSources += 1
        
    print("\t\t\t=================================")
    print("FINISHED ALL %d SOURCES IN %.3f seconds (%.3f minutes, %.3f hrs)" % (numSources, time.time() - algStartTime, (time.time() - algStartTime)/60, ((time.time() - algStartTime)/60)/60))
    print("%d Sources Timed Out: " % (len(timedOutSources)), timedOutSources)
    print("\t\t\t=========== TOTAL RESULTS ===========")
    print("Average TDR Recall: %.3f, Average TDR Precision: %.3f" % (sum(allTDR_recall['all'])/len(allTDR_recall['all']), sum(allTDR_prec['all']) / len(allTDR_prec['all'])))
    print("Average Instance Sim: %.3f, Average Dkl: %.3f" % (sum(allInstanceSim['all'])/len(allInstanceSim['all']), sum(allDkl['all'])/len(allDkl['all'])))
    print("\t\tAverage Runtimes: %.3f sec (%.3f min)" % (sum(allRuntimes['all'])/len(allRuntimes['all']), (sum(allRuntimes['all'])/len(allRuntimes['all']))/60))
    print("\t\t Average Size of Output: %d" % (sum(avgSizeOutput)/len(avgSizeOutput)))
    print("\t\t Average Ratio of Output Size: %.3f" % (sum(avgSizeRatio)/len(avgSizeRatio)))
    
    if 'tpch' in benchmark:
        print("\t\t\t=========== BREAK-DOWN OF RESULTS ===========")
        print("PROJ/SEL Queries: TDR Recall: %.3f, TDR Precision: %.3f, Instance Sim: %.3f, Dkl: %.3f" % (sum(allTDR_recall['simple'])/len(allTDR_recall['simple']), sum(allTDR_prec['simple']) / len(allTDR_prec['simple']), sum(allInstanceSim['simple']) / len(allInstanceSim['simple']), sum(allDkl['simple']) / len(allDkl['simple'])))
        print("\tRuntime: %.3f sec" % (sum(allRuntimes['simple']) / len(allRuntimes['simple'])))
        
        print("ONE-JOIN Queries: TDR Recall: %.3f, TDR Precision: %.3f, Instance Sim: %.3f, Dkl: %.3f" % (sum(allTDR_recall['oneJoin'])/len(allTDR_recall['oneJoin']), sum(allTDR_prec['oneJoin']) / len(allTDR_prec['oneJoin']), sum(allInstanceSim['oneJoin']) / len(allInstanceSim['oneJoin']), sum(allDkl['oneJoin']) / len(allDkl['oneJoin'])))
        print("\tRuntime: %.3f sec" % (sum(allRuntimes['oneJoin']) / len(allRuntimes['oneJoin'])))
        
        print("MANY-JOINS Queries: TDR Recall: %.3f, TDR Precision: %.3f, Instance Sim: %.3f, Dkl: %.3f" % (sum(allTDR_recall['manyJoins'])/len(allTDR_recall['manyJoins']), sum(allTDR_prec['manyJoins']) / len(allTDR_prec['manyJoins']), sum(allInstanceSim['manyJoins']) / len(allInstanceSim['manyJoins']), sum(allDkl['manyJoins']) / len(allDkl['manyJoins'])))
        print("\tRuntime: %.3f sec" % (sum(allRuntimes['manyJoins']) / len(allRuntimes['manyJoins'])))
    