import glob
import pandas as pd
import json
from tabulate import tabulate
from io import StringIO
import math
from statistics import mean
from evaluatePaths import setTDR, bestMatchingTuples, instanceSimilarity
from calcDivergence import table_Dkl,getQueryConditionalVals


SOURCES_TABLE_FILEPATH = "/Users/gracefan/Documents/Datasets/tpch_small/queries/"
EXPSTATS_OUTPUT_DIR = "./llm_outputs/"


def clean_table_string(table_string):
    lines = table_string.split('\n')
    for i in range(len(lines)):
        cells = lines[i].split('|')
        cleaned_cells = [cell.strip() for cell in cells]
        lines[i] = '|'.join(cleaned_cells)
    cleaned_table_string = '\n'.join(lines)
    return cleaned_table_string

def get_df_from_string(llm_output):
    num_rows = 0
    with open(llm_output, 'r') as file:
        llm_output_string = ""
        for line in file.readlines():
            if line.startswith("|"):
                num_rows += 1
                if num_rows == 2:
                    continue
                
                llm_output_string += line
    # Remove leading and trailing whitespaces from each line
    llm_output_string = clean_table_string(llm_output_string)
    llm_output_string = '\n'.join(line[1:-1].strip() for line in llm_output_string.split('\n'))
    # Create a pandas DataFrame
    df = pd.read_csv(StringIO(llm_output_string), sep="|")
    return df

def check_schemas(source_df, llm_output_df):
    aligned_schemas = True
    if set(source_df.columns) != set(llm_output_df.columns):
        print("\t\t\tFound different schemas for Source Table %s" % (source_table_name))
        print(tabulate(source_df.columns, headers="keys", tablefmt='psql'))
        print(tabulate(llm_output_df.columns, headers="keys", tablefmt='psql'))
        aligned_schemas = False
    return aligned_schemas

# Function to convert textual columns to lowercase
def convert_to_lowercase(column):
    if column.dtypes == 'O':  # 'O' stands for Object (textual)
        return column.str.lower()
    else:
        return column
    
    
if __name__ == '__main__':
    sourceStats = {}
    all_recall, all_prec, all_instanceSim, all_instanceDiv, all_klDiv = [], [], [], [], []
    
    all_llm_outputs = glob.glob(EXPSTATS_OUTPUT_DIR+"*.txt")
    for llm_output in all_llm_outputs:
        source_table_name = llm_output.split("/")[-1].replace(".txt", ".csv")
        source_df = pd.read_csv(SOURCES_TABLE_FILEPATH+source_table_name)
        source_df = source_df.apply(convert_to_lowercase)
        llm_output_df = get_df_from_string(llm_output)
        llm_output_df = llm_output_df.apply(convert_to_lowercase)
        
        # extract columns from LLM outputted table that are also in the source table
        common_cols = [col for col in source_df.columns if col in llm_output_df.columns]
        llm_output_df = llm_output_df[common_cols]
        print(check_schemas(source_df, llm_output_df))
            
        ## Start Evaluation Evaluation
        primaryKey = source_df.columns.tolist()[0]
         
        # Get TDR Recall and TDR Precision
        TDR_recall, TDR_precision = setTDR(source_df, llm_output_df)
        bestMatchingDf = bestMatchingTuples(source_df, llm_output_df, primaryKey)
        
        if bestMatchingDf is None: 
            print("\t\t\t Source Table %s: TDR Recall = %.3f, TDR Precision = %.3f =================================" % (source_table_name, TDR_recall, TDR_precision))  
            print("\t\t\tGPT Produced no output for Source Table %s =================================" % (source_table_name))
            continue
        # Get Instance Similarity
        bestMatchingDf = bestMatchingDf[common_cols]
        instanceSim = instanceSimilarity(source_df, bestMatchingDf, primaryKey)
        # Get Conditional KL-divergence
        queryValPairs = getQueryConditionalVals(source_df, primaryKey)
        tableDkl, _, colDkls = table_Dkl(source_df, bestMatchingDf, primaryKey, queryValPairs, math.inf, log=0)
        print("\t\t\t Source Table %s: TDR Recall = %.3f, TDR Precision = %.3f, Instance Similarity = %.3f, KL-Divergence = %.3f =================================" % (source_table_name, TDR_recall, TDR_precision, instanceSim, tableDkl))
        
        # Store stats for this source table
        sourceStats[source_table_name] = {
            "TDR_Recall": TDR_recall,
            "TDR_Precision": TDR_precision,
            "Instance_Similarity": instanceSim,
            "Instance_Divergence": 1.0-instanceSim, 
            "KL_Divergence": tableDkl
        }
        all_recall.append(TDR_recall)
        all_prec.append(TDR_precision)
        all_instanceSim.append(instanceSim)
        all_instanceDiv.append(1.0-instanceSim)
        all_klDiv.append(tableDkl)
    print("Found results for %d Source Tables"%(len(sourceStats)))
        
    sourceStats["Total"] = {
        "Number of Sources": len(sourceStats),
        "TDR_Recall": mean(all_recall),
        "TDR_Precision": mean(all_prec),
        "Instance_Similarity": mean(all_instanceSim),
        "Instance_Divergence": mean(all_instanceDiv), 
        "KL_Divergence": mean(all_klDiv)
    }
    # Save results to output folder
    with open(EXPSTATS_OUTPUT_DIR+"LLM_results.json", "w+") as f: json.dump(sourceStats, f, indent=4)