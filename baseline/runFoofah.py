import pandas as pd
import glob
import os
import math
from Foofah import foofah
from Foofah.foofah_libs.operators import *
from Foofah.foofah_libs import operators as Op
from config import *
import time
from tqdm import tqdm
from evaluatePaths import setTDR, bestMatchingTuples, instanceSimilarity
from calcDivergence import table_Dkl,getQueryConditionalVals
import to_json


if __name__ == '__main__':
    # benchmark = 't2d_gold'
    # benchmark = 'TUS_t2d_gold'
    # benchmark = 'wdc_t2d_gold'
    # timeout = 100 # 100 seconds for t2d
    # ========= TPCH Variations
    # benchmark = 'santos_large_tpch'
    benchmark = 'tpch'
    # benchmark = 'tpch_groundtruth'
    # benchmark = 'tpch_0_groundtruth'
    # benchmark = 'tpch_mtTables'
    timeout = 1800
    projSel = 1
    
    outputDir = "output_tables/"
    if projSel: outputDir = "output_tables_projSel/"
    # FILEPATH = '../../Datasets/%s/queries/' % (benchmark)  
    FILEPATH = '../../Datasets/%s/queries/' % ('tpch')  
    # FILEPATH = '../../Datasets/%s/queries/' % ('t2d_gold')  
    sources = glob.glob(FILEPATH+'*.csv')
    allTDR_recall, allTDR_prec, allInstanceSim, allDkl = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}, {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    allRuntimes = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    numSources = 0
    timedOutSources = []
    avgSizeOutput = []
    avgSizeRatio = []
    algStartTime = time.time()
    print("Auto-Pipeline Baseline, Given the Candidate Tables from Set Similarity (Applied Project / Select)")
    for indx in tqdm(range(0, len(sources))):
        source_table = sources[indx].split(FILEPATH)[-1]
        sourceDf = pd.read_csv(FILEPATH+source_table)
        sourceDf.columns = [col.replace("'", '') for col in sourceDf.columns]
        
        candidateFileName = benchmark
        if 'tpch' in benchmark:
            b_list = benchmark.split('_')
            b_list.insert(1,'small')
            candidateFileName = '_'.join(b_list)
        if not os.path.isfile("../results_candidate_tables/%s/%s_candidateTables.pkl" % (candidateFileName, source_table)): 
            print("Source table %s has no candidates" % (source_table))
            continue
        print("\t=========== %d) Source Table: %s =========== " % (indx, source_table))
        print("Source has %d total values" % (sourceDf.shape[0]*sourceDf.shape[1]))
        to_json.main(benchmark, source_table, projSel)
        print("\t\t\t=========== BEGIN Auto-Pipeline ===========")
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
        input_folder_path = 'Inputs/%s/%s_for_autopipeline.txt' % (benchmark, source_table)
        if projSel:
            input_folder_path = 'Inputs_projSel/%s/%s_for_autopipeline.txt' % (benchmark, source_table)
        
        startTime = time.time()
        
        _, timedOut, numOutputVals = foofah.main(benchmark, source_table, input_folder_path, ext_time_limit=timeout, what_to_explain='tables', projSel=projSel)
        if timedOut: 
            print("\t\t\tAuto-Pipeline Timed out for Source Table %s after %.3f seconds =================================" % (source_table, (time.time() - startTime)))
            timedOutSources.append(source_table)
            continue
        if numOutputVals == 0:
            print("\t\t\tAuto-Pipeline Produced no output for Source Table %s after %.3f seconds =================================" % (source_table, (time.time() - startTime)))
            continue
        avgSizeOutput.append(numOutputVals)
        avgSizeRatio.append(numOutputVals/(sourceDf.shape[0]*sourceDf.shape[1]))

        transformation = ''
        print("Transformation")
        with open('foo.txt', 'r') as f:
            for op in f.readlines():
                transformation += op
            print(transformation)

        print("\t\t\t Auto-Pipeline Finished for Source Table %s in %.3f seconds (%.3f minutes), \n Output Table of Size %d =================================" % (source_table, (time.time() - startTime), (time.time() - startTime)/60, numOutputVals))
        ap_result = outputDir + benchmark+"_3/"+source_table
        ap_result = pd.read_csv(ap_result)
        overlapCols = [col for col in sourceDf.columns if col in ap_result.columns]
        ap_result = ap_result[overlapCols]
        
        # Evaluation
        TDR_recall, TDR_precision = setTDR(sourceDf, ap_result)
        bestMatchingDf = bestMatchingTuples(sourceDf, ap_result, primaryKey)
        if bestMatchingDf is None: 
            print("\t\t\tAuto-Pipeline Produced no output for Source Table %s (no bestMatchingDf) after %.3f seconds =================================" % (source_table, (time.time() - startTime)))
            continue
        bestMatchingDf = bestMatchingDf[overlapCols]
        instanceSim = instanceSimilarity(sourceDf, bestMatchingDf, primaryKey)
        tableDkl, _, colDkls = table_Dkl(sourceDf, bestMatchingDf, primaryKey, queryValPairs, math.inf, log=1)
        if tableDkl == 0.0 or TDR_recall == 1.0 or TDR_precision == 1.0 or instanceSim == 1.0:
            print(" !!!! FOUND COMPLETE PATH !!!")

        print("=== FINISHED Source %s with \n TDR Recall = %.3f, TDR Precision = %.3f, instanceSim = %.3f, KL-DIVERGENCE = %.3f  === " % (source_table, TDR_recall, TDR_precision, instanceSim, tableDkl))
        print("\t\t\t=========== FINISHED in %.3f seconds (%.3f minutes) =================================" % ((time.time() - startTime), (time.time() - startTime)/60))
        if 'tpch' in benchmark:
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
    print("Average TDR Recall of Auto-Pipeline: %.3f, Average TDR Precision of Auto-Pipeline: %.3f" % (sum(allTDR_recall['all'])/len(allTDR_recall['all']), sum(allTDR_prec['all']) / len(allTDR_prec['all'])))
    print("\tLength TDR Recall: %d, Length TDR Precision: %d" % (len(allTDR_recall['all']), len(allTDR_prec['all'])))
    
    print("Average Instance Sim of Auto-Pipeline: %.3f, Average Dkl of Auto-Pipeline: %.3f" % (sum(allInstanceSim['all'])/len(allInstanceSim['all']), sum(allDkl['all'])/len(allDkl['all'])))
    print("\tLength Instance Sim: %d, Length Dkl: %d" % (len(allInstanceSim['all']), len(allDkl['all'])))

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
    


