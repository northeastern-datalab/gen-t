import config
from dindex_store.discovery_index import load_dindex
from aurum_api.algebra import AurumAPI
from qbe_module.query_by_example import ExampleColumn, QueryByExample
from qbe_module.materializer import Materializer
from tqdm import tqdm
import os
import json
import pandas as pd
import glob
from tabulate import tabulate
from view_distillation import vd
from view_distillation.vd import ViewDistillation


def main(qt):
    cnf = {setting: getattr(config, setting) for setting in dir(config)
        if setting.islower() and len(setting) > 2 and setting[:2] != "__"}

    dindex = load_dindex(cnf)
    print("Loading DIndex...OK")

    api = AurumAPI(dindex)
    print("created aurum api")

    # QBE interface
    qbe = QueryByExample(api)

    """
    Specify an example query
    """
    print("Query Table %s..." % (qt.split("/")[-1]))
    qt_df = pd.read_csv(qt)
    all_atts = list(qt_df.columns)
    
    # first add key column
    key_att = list(qt_df.columns)[0]
    
    for att in all_atts[1:]:
        qt_df = pd.read_csv(qt)
        print("Starting new round: ", key_att, att)
        print(qt_df[att].head(5))
        attr_examples = [{
            "attr": key_att,
            "examples": [str(val) for val in qt_df[key_att].values]},
            {
            "attr": att,
            "examples": [str(val) for val in qt_df[att].values]}] # must be a list of string values
        included_atts = [att_example["attr"] for att_example in attr_examples] #for now, include all rows
        qt_df = qt_df[included_atts]

        source_columns = []
        for curr_column_example_dict in attr_examples:
            source_columns.append(ExampleColumn(**curr_column_example_dict))
            

        # example_columns = [
        #     ExampleColumn(attr='school_name', examples=["Ogden International High School", "University of Chicago - Woodlawn"]),
        #     ExampleColumn(attr='school type', examples=["Charter", "Neighborhood"]),
        #     ExampleColumn(attr='level', examples=["Level 1", "Level 2+"])
        # ]


        """
        Find candidate columns
        """
        candidate_list = qbe.find_candidate_columns(source_columns, cluster_prune=True)

        """
        Display candidate columns (for debugging purpose)
        """
        for i, candidate in enumerate(candidate_list):
            print('column {}: found {} candidate columns'.format(format(i), len(candidate)))
            # for col in candidate:
            #     print(col.to_str())


        cand_groups, tbl_cols = qbe.find_candidate_groups(candidate_list)

        """
        Find join graphs
        """
        print("Finding join graphs...")
        join_graphs = qbe.find_join_graphs_for_cand_groups(cand_groups)
        print(f"number of join graphs: {len(join_graphs)}")
        """
        Display join graphs (for debugging purpose)
        """
        for i, join_graph in enumerate(join_graphs):
            print(f"----join graph {i}----")
            join_graph.display()

        """
        Materialize join graphs
        """
        ## PARAMETERS: EDIT
        data_path = '../../Datasets/tptr_small/datalake/'  # path where the raw data is stored
        output_path = './tptr_small_outputs/%s/%s/'%(qt.split("/")[-1],",".join(list(qt_df.columns)))  # path to store the output views
        max_num_views = 100  # how many views you want to materialize
        sep = ','  # csv separator

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        materializer = Materializer(data_path, tbl_cols, 200, sep)

        result_dfs = []

        j = 0
        print("Query Table schema", list(qt_df.columns))
        for join_graph in tqdm(join_graphs):
            """
            a join graph can produce multiple views because different columns are projected
            """
            df_list = materializer.materialize_join_graph(join_graph)
            for df in df_list:
                if len(df) != 0:
                    metadata = {}
                    metadata["join_graph"] = join_graph.to_str()
                    metadata["columns_proj"] = list(df.columns)
                    with open(f"./{output_path}/view{j}.json", "w") as outfile:
                        json.dump(metadata, outfile)
                    j += 1
                    new_cols = []
                    k = 1
                    for col in df.columns:
                        new_col = col.split(".")[-1]
                        if new_col in new_cols:
                            new_col += str(k)
                            k += 1
                        new_cols.append(new_col)
                    df.columns = new_cols
                    # # Grace added 2/18
                    # # perform "selection": select values for example column that it has highest overlap with
                    # col_rowIndx = {}
                    # for df_col in df.columns:
                    #     col_vals = set([str(val) for val in df[df_col].values])
                    #     num_highest_overlap = 0
                    #     largest_overlap_indx = None
                    #     for qt_col in qt_df.columns:
                    #         qt_vals = set([str(val) for val in qt_df[qt_col].values])
                    #         value_overlap = col_vals.intersection(qt_vals)
                    #         if len(value_overlap) > num_highest_overlap:
                    #             num_highest_overlap = len(value_overlap)
                    #             largest_overlap_indx = df[df[df_col].isin(value_overlap)].index
                    #     if not largest_overlap_indx:
                    #         continue
                    #     col_rowIndx[df_col] = list(largest_overlap_indx)
                    # df = df[[*col_rowIndx]]
                    # print("col_rowIndx", col_rowIndx, qt_df.columns, df.columns)
                    # if df.empty:
                    #     continue
                    # common_row_indices = set.intersection(*(set(indices) for indices in col_rowIndx.values()))
                    # selected_rows = df.loc[common_row_indices]
                    # df = selected_rows
                    
                    df.to_csv(f"./{output_path}/view{j}.csv", index=False)

                    result_dfs.append(df)

            if j >= max_num_views:
                break

        """
        View Distillation
        """

        vd = ViewDistillation(dfs=result_dfs)

        # Generates a networkx graph representing 4C relationships among views (nodes)
        vd.generate_graph()
        print("Finished Generating Graph. Onto pruning...")
        # Prune the graph with the given pruning strategies, returning the updated graph
        graph = vd.prune_graph(remove_identical_views=True,
                            remove_contained_views=True,
                            union_complementary_views=True)
        # Grace added: write to JSON output
        graph_nodes = list(graph.nodes.data())
        qt_name = qt.split("/")[-1]
        qt_schema = ",".join(list(qt_df.columns))
        found_views = [tpl[0] for tpl in graph_nodes]
        found_views = json.dumps(found_views)
        json_output_path = "/materialized_views_fullDL.json"
        #read existing file and append new data
        loaded_json = {}
        if os.path.exists(json_output_path):
            with open(json_output_path,"r") as f:
                loaded_json = json.load(f)
        if qt_name in loaded_json:
            loaded_json[qt_name][qt_schema] = found_views
        else:
            loaded_json[qt_name] = {qt_schema: found_views}
        #overwrite/create file
        with open(json_output_path,"w") as f:
            json.dump(loaded_json,f,indent=4)