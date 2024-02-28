import time
import glob
import pandas as pd
import math
import argparse
import table_integration, alite_fd_original
from tqdm import tqdm
import os
import json
import sys
sys.path.append('../discovery/')
from evaluatePaths import setTDR, bestMatchingTuples, instanceSimilarity
from calcDivergence import table_Dkl,getQueryConditionalVals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="tptr", choices=['tptr', 'santos_large_tptr', 'tptr_groundtruth', 'tptr_small', 'tptr_large',
                                                                          't2d_gold', 'TUS_t2d_gold', 'wdc_t2d_gold'])
    parser.add_argument("--timeout", type=int, default=25263) # 7 hrs for tptr
    parser.add_argument("--genT", type=int, default=1) # 1 if candidate tables were pruned to originating tables in Gen-T, else 0
    parser.add_argument("--doPS", type=int, default=1) # 1 if perform projection and selection, else 0
    
    hp = parser.parse_args()
    
    benchmark = hp.benchmark
    # benchmark = 'santos_large_tptr'
    
    OUTPUT_DIR = "output_tables/%s/"%(benchmark)
    if hp.genT:
        OUTPUT_DIR = "genT_output_tables/%s/"%(benchmark)
    elif hp.doPS:
        OUTPUT_DIR = "output_tables_projSel/%s/"%(benchmark)
    if not os.path.exists(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)
    print("========= Benchmark: ", benchmark)
    FILEPATH = '/home/gfan/Datasets/%s/queries/' % (benchmark)  
    if '_groundtruth' in benchmark: FILEPATH = '/home/gfan/Datasets/%s/queries/' % (benchmark.split('_groundtruth')[0])  

    datasets = glob.glob(FILEPATH+'*.csv')
    allTDR_recall, allTDR_prec, allInstanceSim, allDkl = {}, {}, {}, {}
    allRuntimes = {}
    numSources = 0
    saved_sources = []
    timedOutSources = []
    avgSizeOutput = []
    avgSizeRatio = []
    individual_prec_recall = {}
    runtimes = {}
    
    
    
    originatingTablePath = "../results_candidate_tables/%s/originatingTables.json" % (benchmark)
    if not os.path.isfile(originatingTablePath):
        if hp.genT: print("Need to Generate Originating Tables before finishing Gen-T")
        else: allOriginatingTableDict = None
    else: 
        with open(originatingTablePath) as json_file: allOriginatingTableDict = json.load(json_file)
            
    algStartTime = time.time()
    if hp.genT: 
        print("Gen-T, Given the Candidate Tables from Set Similarity")
    else:         
        print("ALITE Baseline, Given the Candidate Tables from Set Similarity")
    for indx in tqdm(range(0, len(datasets))):
        source_table = datasets[indx].split(FILEPATH)[-1]
        if allOriginatingTableDict and source_table not in allOriginatingTableDict: continue
        sourceDf = pd.read_csv(FILEPATH+source_table)
        print("\t=========== %d) Source Table: %s =========== " % (indx, source_table))
        print("Source has %d cols, %d rows --> %d total values" % (sourceDf.shape[1], sourceDf.shape[0], sourceDf.shape[0]*sourceDf.shape[1]))
        primaryKey = sourceDf.columns.tolist()[0]
        foreignKeys = [colName for colName in sourceDf.columns.tolist() if 'key' in colName and colName != primaryKey] # for tptr
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
        if hp.genT:
            timed_out, noCandidates, numOutputVals = table_integration.main(benchmark, source_table, allOriginatingTableDict[source_table], hp.timeout)
        # RUN BASELINE
        else:
            candidateTablePath = "../results_candidate_tables/%s/candidateTables.json" % (benchmark)
            with open(candidateTablePath) as json_file: allCandidateTableDict = json.load(json_file)
            timed_out, noCandidates, numOutputVals = alite_fd_original.main(benchmark, source_table, allCandidateTableDict[source_table], hp.doPS, hp.timeout)
        
        if timed_out: 
            print("\t\t\tAlite Timed out for Source Table %s after %.3f seconds =================================" % (source_table, (time.time() - startTime)))
            timedOutSources.append(source_table)
            continue
        if noCandidates: 
            print("\t\t\tAlite Has No Candidates for Source Table %s after %.3f seconds =================================" % (source_table, (time.time() - startTime)))
            continue
        print("\t\t\tAlite Finished for Source Table %s in %.3f seconds (%.3f minutes), \n Output Table of Size %d =================================" % (source_table, (time.time() - startTime), (time.time() - startTime)/60, numOutputVals))
        runtimes[source_table] = time.time() - startTime
        avgSizeOutput.append(numOutputVals)
        avgSizeRatio.append(numOutputVals/(sourceDf.shape[0]*sourceDf.shape[1]))
        fd_result = OUTPUT_DIR+source_table
        fd_result = pd.read_csv(fd_result)
        
        timed_out = False
        noCandidates = False
        # Evaluation
        print("BEGIN Evaluation")
        TDR_recall, TDR_precision = setTDR(sourceDf, fd_result)
        bestMatchingDf = bestMatchingTuples(sourceDf, fd_result, primaryKey)
        if bestMatchingDf is None: continue
        instanceSim = instanceSimilarity(sourceDf, bestMatchingDf, primaryKey)
        bestMatchingDf = bestMatchingDf[sourceDf.columns]
        tableDkl, _, colDkls = table_Dkl(sourceDf, bestMatchingDf, primaryKey, queryValPairs, math.inf, log=1)

        print("=== FINISHED Source %s with \n TDR Recall = %.3f, TDR Precision = %.3f, instanceSim = %.3f, KL-DIVERGENCE = %.3f  === " % (source_table, TDR_recall, TDR_precision, instanceSim, tableDkl))
        print("\t\t\t=========== FINISHED in %.3f seconds (%.3f minutes) =================================" % ((time.time() - startTime), (time.time() - startTime)/60))
        
        try: f1_score = 2 * (TDR_precision * TDR_recall) / (TDR_precision + TDR_recall)
        except: f1_score = 0
        individual_prec_recall[source_table] = {'Precision': TDR_precision, 'Recall': TDR_recall,
                                                'F1_Score': f1_score}
        
        curr_metrics = [TDR_recall, TDR_precision, instanceSim, tableDkl, time.time() - startTime]
        if 'tptr' in benchmark and not timed_out and not noCandidates:
            if 'psql_many' in source_table:
                for metricIdx, metric_dict in enumerate([allTDR_recall, allTDR_prec, allInstanceSim, allDkl, allRuntimes]):
                    if 'manyJoins' not in metric_dict:
                        metric_dict['manyJoins'] = []
                    metric_dict['manyJoins'].append(curr_metrics[metricIdx])
            elif 'psql_edge' in source_table:
                for metricIdx, metric_dict in enumerate([allTDR_recall, allTDR_prec, allInstanceSim, allDkl, allRuntimes]):
                    if 'simple' not in metric_dict:
                        metric_dict['simple'] = []
                    metric_dict['simple'].append(curr_metrics[metricIdx])
            else:
                for metricIdx, metric_dict in enumerate([allTDR_recall, allTDR_prec, allInstanceSim, allDkl, allRuntimes]):
                    if 'oneJoin' not in metric_dict:
                        metric_dict['oneJoin'] = []
                    metric_dict['oneJoin'].append(curr_metrics[metricIdx])
        
        for metricIdx, metric_dict in enumerate([allTDR_recall, allTDR_prec, allInstanceSim, allDkl, allRuntimes]):
            if 'all' not in metric_dict:
                metric_dict['all'] = []
            metric_dict['all'].append(curr_metrics[metricIdx])
        numSources += 1
        saved_sources.append(source_table)
        
    print("\t\t\t=================================")
    TIMESTATS_OUTPUT_DIR = "../experiment_logs/%s/"%(benchmark)
    
    EXPSTATS_OUTPUT_DIR = "../experiment_logs/%s/"%(benchmark)
    if not os.path.exists(EXPSTATS_OUTPUT_DIR):
      os.makedirs(EXPSTATS_OUTPUT_DIR)
    sourceStats = {'num_sources': [numSources, saved_sources], 'timed_out_sources': [len(timedOutSources), timedOutSources]}
    metricDicts = {'TDR_Recall': allTDR_recall, 'TDR_Precision': allTDR_prec, 'Instance_Similarity': allInstanceSim, 'Instance_Divergence': allInstanceSim, 'KL_Divergence': allDkl, 'Runtimes': allRuntimes}
    for metricName, mDict in metricDicts.items():
        sourceStats[metricName] = {}
        for k, v_list in mDict.items():
            if metricName == 'Instance_Divergence':
                v_list = [1.0-val for val in v_list]
            sourceStats[metricName][k] = round(sum(v_list)/len(v_list), 3)
    print("Final KL_Divergence: ", sourceStats['KL_Divergence']['all'])
    sizeLists = {'ouptut_size': avgSizeOutput, 'size_ratio': avgSizeRatio}
    for sName, sList in sizeLists.items():
        sourceStats[sName] = round(sum(sList)/len(sList), 3)

    if hp.genT:
        with open(EXPSTATS_OUTPUT_DIR+"final_genT_results.json", "w+") as f: json.dump(sourceStats, f, indent=4)
        with open(EXPSTATS_OUTPUT_DIR+"each_source_result_genT.json", "w+") as f: json.dump(individual_prec_recall, f, indent=4)
        with open(TIMESTATS_OUTPUT_DIR+"runtimes_genT.json") as timesJson:
            matrix_times = json.load(timesJson)
        for sTable, timesDict in matrix_times.items():
            if '.csv' not in sTable: continue
            if sTable in runtimes: matrix_times[sTable]['table_integration'] = runtimes[sTable]
        with open(TIMESTATS_OUTPUT_DIR+"runtimes_genT.json", mode='w') as f: json.dump(matrix_times, f, indent=4)
        
    elif hp.doPS:
        with open(EXPSTATS_OUTPUT_DIR+"final_alitePS_results.json", "w+") as f: json.dump(sourceStats, f, indent=4)
        with open(EXPSTATS_OUTPUT_DIR+"each_source_result_alitePS.json", "w+") as f: json.dump(individual_prec_recall, f, indent=4)
        with open(EXPSTATS_OUTPUT_DIR+"runtimes_alitePS.json", "w+") as f: json.dump(runtimes, f, indent=4)
        
    else:
        with open(EXPSTATS_OUTPUT_DIR+"final_alite_results.json", "w+") as f: json.dump(sourceStats, f, indent=4)
        with open(EXPSTATS_OUTPUT_DIR+"each_source_result_alite.json", "w+") as f: json.dump(individual_prec_recall, f, indent=4)
        with open(EXPSTATS_OUTPUT_DIR+"runtimes_alite.json", "w+") as f: json.dump(runtimes, f, indent=4)
    
    print("FINISHED ALL %d SOURCES IN %.3f seconds (%.3f minutes, %.3f hrs)" % (numSources, time.time() - algStartTime, (time.time() - algStartTime)/60, ((time.time() - algStartTime)/60)/60))
   