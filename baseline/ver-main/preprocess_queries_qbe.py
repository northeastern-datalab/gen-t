## Format we want

# example_columns = [
#     ExampleColumn(attr='school_name', examples=["Ogden International High School", "University of Chicago - Woodlawn"]),
#     ExampleColumn(attr='school type', examples=["Charter", "Neighborhood"]),
#     ExampleColumn(attr='level', examples=["Level 1", "Level 2+"])
# ]

import glob
import pandas as pd

example_columns = []
col_vals = {}
query_tables = glob.glob("/Users/gracefan/Documents/Datasets/tpch_small/queries/*.csv")
for qt in query_tables:
    if qt.split("/")[-1] != "psql_12_n_ij_s.csv": continue
    # print(qt.split("/")[-1])
    qt_df = pd.read_csv(qt)
    atts = qt_df.columns
    for attInd, att in enumerate(atts):
        col_vals[att] = [str(val) for indx, val in enumerate(qt_df[att].values) ]
        # if attInd > 2: break
        example_columns.append(
            "ExampleColumn(attr='%s', examples=%s),"%(att, str([str(val) for indx, val in enumerate(qt_df[att].values) ])) # if indx < 2
        )
    break
for col in example_columns: 
    print(col)