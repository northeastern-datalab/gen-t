import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="tptr", choices=['tptr', 'santos_large_tptr', 'tptr_groundtruth', 'tptr_small', 'tptr_large',
                                                                          't2d_gold', 'TUS_t2d_gold', 'wdc_t2d_gold'])
    hp = parser.parse_args()

    with open("%s/runtimes_genT.json"%(hp.benchmark)) as timesJson:
        all_runtimes = json.load(timesJson)
        
    total_runtime = []
    for sTable, timesDict in all_runtimes.items():
        if '.csv' not in sTable: continue
        if "all" not in timesDict or "table_integration" not in timesDict: continue
        total_runtime.append(timesDict["all"]+timesDict["table_integration"])
    print("For %d Source Tables, the Average Runtimes (sec) is %.3f" % (len(total_runtime), sum(total_runtime)/len(total_runtime)))