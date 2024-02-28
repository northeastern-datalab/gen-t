import json
import math
import numpy as np
from tabulate import tabulate
import pandas as pd
from tqdm import tqdm
import glob
from statistics import mean
from functools import reduce
from evaluatePaths import setTDR, bestMatchingTuples, instanceSimilarity
from calcDivergence import table_Dkl,getQueryConditionalVals


all_views_filepath = "../tptr_small_outputs/"
save_results_folderpath = "tptr_small/"
source_tables_filepath = "../../Datasets/tptr_small/queries/"
def num_similar_values(df1, df2):
    '''
    Find size of value overlap between two columns from two DFs with the highest overlap
    '''
    overlap_values = {}
    # Iterate through each column in the first dataframe
    for col1 in df1.columns:
        overlap_values[col1] = 0
        # Iterate through each column in the second dataframe
        for col2 in df2.columns:
            # Check if the values in the two columns overlap
            common_values = set(df1[col1]) & set(df2[col2])
            if common_values:
                if overlap_values[col1] < len(common_values):
                    overlap_values[col1] = len(common_values)
    return sum(overlap_values.values())


def get_most_similar_view(projected_source_df, view_list):
    '''
    From the list of views returned from Ver, find the most similar view to the projected source table
    '''
    most_similar_view, most_similar_view_df = None, pd.DataFrame()
    num_most_similar = 0
    
    for mat_view in view_list:
        if "_union_" in mat_view:
            unioned_view_list = mat_view.split("_union_")
            view1_df = pd.read_csv("%s%s/%s/%s.csv"%(all_views_filepath,source,schema,unioned_view_list[0]))
            view2_df = pd.read_csv("%s%s/%s/%s.csv"%(all_views_filepath,source,schema,unioned_view_list[0]))
            curr_view_df = pd.concat([view1_df, view2_df], ignore_index=True)
        else:
            curr_view_df = pd.read_csv("%s%s/%s/%s.csv"%(all_views_filepath,source,schema,mat_view))
        
        num_curr_overlap_vals = num_similar_values(projected_df, curr_view_df)
        if num_most_similar < num_curr_overlap_vals or (num_curr_overlap_vals == num_most_similar and num_curr_overlap_vals == 0):
            num_most_similar = num_curr_overlap_vals
            most_similar_view = mat_view
            most_similar_view_df = curr_view_df

    return most_similar_view_df, most_similar_view
    
def convert_to_lowercase(column): 
    '''
    Convert textual columns to lowercase
    '''
    if column.dtypes == 'O':  # 'O' stands for Object (textual)
        return column.str.lower()
    else:
        return column
    
def inherit_source_key(source_key_col, og_source_df, df): 
    '''
    For the output table, add in key column with key values associated with value in output table from source table
    '''
    og_source_df = og_source_df.apply(convert_to_lowercase)
    if source_key_col not in df.columns:
        # if key is not in result, use source table's column
        common_cols = [col for col in og_source_df if col in df]
        common_cols_with_key = [col for col in og_source_df if col in df or col == source_key_col]
        df_with_key = pd.merge(og_source_df[common_cols_with_key], df, on=common_cols, how='right')[common_cols_with_key] 
        
        # add the rest of the key values from the source table
        source_key = og_source_df[source_key_col]
        source_key.loc[len(source_key)] = np.nan
        output_df = pd.merge(df_with_key, source_key, on=source_key_col, how='outer')
    return output_df

def join(df1, df2, common_column):
    return pd.merge(df1, df2, on=common_column, how='outer')


def evaluate_output(source_df, output_df):
    '''
    Evaluate TDR Recall, Precision, Instance Similarity, and KL-divergence of output
    '''
    primaryKey = source_df.columns.tolist()[0]
        
    # Get TDR Recall and TDR Precision
    TDR_recall, TDR_precision = setTDR(source_df, output_df)
    bestMatchingDf = bestMatchingTuples(source_df, output_df, primaryKey)
    if bestMatchingDf is None: 
        bestMatchingDf = output_df
    # Get Instance Similarity
    bestMatchingDf = bestMatchingDf[list(output_df.columns)]
    instanceSim = instanceSimilarity(source_df, bestMatchingDf, primaryKey)
    # Get Conditional KL-divergence
    queryValPairs = getQueryConditionalVals(source_df, primaryKey)
    tableDkl, _, _ = table_Dkl(source_df, bestMatchingDf, primaryKey, queryValPairs, math.inf, log=0)
    
    return TDR_recall, TDR_precision, instanceSim, tableDkl

if __name__ == '__main__':
    with open("../materialized_views_fullDL.json") as f: source_views = json.load(f)
    
    allStats = {}
    for source, schema_view in tqdm(source_views.items()):
        recall, prec, instSim, instDiv = [], [], [], []
        source_df = pd.read_csv(source_tables_filepath+source)
        source_key_col = list(source_df.columns)[0]
        source_cols = list(source_df.columns)
        all_result_dfs = []
        # For each pair of columns, find the view that is most similar to projected source df
        for schema, views in schema_view.items():
            column_list = schema.split(",")
            projected_df = source_df[column_list[-1]].to_frame()
            # convert textual columns to lowercase
            projected_df = projected_df.apply(convert_to_lowercase)
            view_list = json.loads(views)
            if not view_list: # if no view was found, record 0 scores
                TDR_recall, TDR_precision, instanceSim = 0.0, 0.0, 0.0
                continue
            else:
                view_df, most_similar_view = get_most_similar_view(projected_df, view_list)
                # Clean the resulting DF: drop nan columns, duplicate rows, and nan rows
                common_cols = [col for col in projected_df.columns if col in view_df.columns]
                view_df = view_df[common_cols].dropna(axis=1, how='all') # project on common columns and drop nan columns
                view_df = view_df.drop_duplicates().dropna(how='all')
                if view_df.empty: # if the view found is empty, record 0 scores
                    TDR_recall, TDR_precision = 0.0, 0.0
                    continue
                else:
                    # Convert resulting DF to lowercase and inherit key values from source table
                    view_df = view_df.apply(convert_to_lowercase)
                    view_df = inherit_source_key(source_key_col, source_df, view_df)
                    view_df = view_df.drop_duplicates().dropna(how='all')
                    
                    df_with_source_vals = view_df.dropna(subset=[source_key_col]) # remove rows in output where key value is nan
                    if not df_with_source_vals.empty:all_result_dfs.append(df_with_source_vals)
                    # Get projected columns from source table that we used to query Ver
                    proj_key_df = source_df[column_list].apply(convert_to_lowercase)
                    proj_key_df[source_key_col] = proj_key_df[source_key_col].astype(float)
                    # Evaluation for each result
                    TDR_recall, TDR_precision, instanceSim, _ = evaluate_output(proj_key_df, view_df)
                    print(f"\t\tResulting View {most_similar_view} has Schema {list(view_df.columns)}, {view_df.shape[0]} rows, Exp Results: {TDR_recall, TDR_precision}")
                    print(view_df.head(3))
            # Save Recall and Precision of results for each projected Source Table
            recall.append(TDR_recall)
            prec.append(TDR_precision)
            instSim.append(instanceSim)
            instDiv.append(1-instanceSim)
        
        # Get Instance Divergence and KL divergence of integration of all results for current Source Table
        try: 
            # Evaluate divergence measures on outer join result of all DFs, taking tuples whose key value is not nan
            result_df = reduce(lambda df1, df2: join(df1, df2, source_key_col), all_result_dfs)
            source_df[source_key_col] = source_df[source_key_col].astype(float)
            _, _, _, tableDkl = evaluate_output(source_df.apply(convert_to_lowercase), result_df)
        except: continue
        
        # Save output table for entire source table
        result_df.to_csv(save_results_folderpath+source, index=False)
        
        # Record current output table's statistics
        allStats[source] = {
            "TDR_Recall": mean(recall),
            "TDR_Precision": mean(prec),
            "Instance_Similarity": mean(instSim),
            "Instance_Divergence":mean(instDiv), 
            "KL_Divergence": tableDkl
        }
    # Record total statistics (average over all source tables)
    allStats["Total"] = {
        "Number of Sources": len(allStats),
        "TDR_Recall": mean([stats["TDR_Recall"] for stats in allStats.values()]),
        "TDR_Precision": mean([stats["TDR_Precision"] for stats in allStats.values()]),
        "Instance_Similarity": mean([stats["Instance_Similarity"] for stats in allStats.values()]),
        "Instance_Divergence": mean([stats["Instance_Divergence"] for stats in allStats.values()]), 
        "KL_Divergence": mean([stats["KL_Divergence"] for stats in allStats.values()])
    }
    # Save results to output folder
    with open("Ver_proj_result_stats.json", "w+") as f: json.dump(allStats, f, indent=4)