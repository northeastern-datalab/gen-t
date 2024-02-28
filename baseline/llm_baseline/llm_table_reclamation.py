import glob
import json
import pandas as pd
from openai import OpenAI
import tiktoken
from tqdm import tqdm
import time

client = OpenAI(api_key='')

# Filepath setup
SOURCES_TABLE_FILEPATH = "../../Datasets/tptr_small/queries/"
DATA_LAKE_FILEPATH = "../../Datasets/tptr_small/datalake/"
INTEG_SET_FILEPATH = "../tptr_small_integSet.json"
all_sources = glob.glob(SOURCES_TABLE_FILEPATH+"*.csv")

def clean_table_string(table_string):
    lines = table_string.split('\n')
    for i in range(len(lines)):
        cells = lines[i].split('|')
        cleaned_cells = [cell.strip() for cell in cells]
        lines[i] = '|'.join(cleaned_cells)
    cleaned_table_string = '\n'.join(lines)
    return cleaned_table_string

def get_tables(source_table, integration_set_dict, frac_table_size, fullDataLake=False):
    '''
    Get Source Table, and a data lake (either the full data lake or integration set)
    Return formatted string to use in GPT prompt
    '''
    source_table_name = source_table.split("/")[-1]
    source_df = pd.read_csv(source_table)
    
    # # Select the first num_cols and first num_rows
    # source_df = source_df.iloc[:num_rows, :num_cols]
    # Convert the Source DataFrame to a formatted string
    source_table_string = source_df.to_markdown(index=False, tablefmt='pipe')
    source_table_string = clean_table_string(source_table_string)
    # Get the schema of the source table
    source_schema = ",".join(source_df.columns.tolist())
    
    data_lake_string = ""
    if fullDataLake: dl_tables = glob.glob(DATA_LAKE_FILEPATH+"*.csv")
    else: dl_tables = [DATA_LAKE_FILEPATH+tbl for tbl in integration_set_dict[source_table_name]]
    for dl_tbl in dl_tables:
        dl_tbl_name = dl_tbl.split("/")[-1]
        dl_table_df = pd.read_csv(dl_tbl)
        common_columns = [col for col in dl_table_df.columns if col in source_df.columns]
        dl_table_df = dl_table_df[common_columns]
        # Select first num_cols and first num_rows
        num_rows = dl_table_df.shape[0] // frac_table_size
        dl_table_df = dl_table_df.iloc[:num_rows, :]
        data_lake_string += "%s: \n %s \n" % (dl_tbl_name, dl_table_df.to_markdown(index=False, tablefmt='pipe'))
    data_lake_string = clean_table_string(data_lake_string)
    return source_table_name, source_table_string, source_schema, data_lake_string

def generate_sample_input_output():
    '''
    Generate sample inputs and outputs that we can instruct GPT with
    '''
    # Sample dataframe for illustration purposes
    data_X = {'A': [1, 2, 3],
            'B': ['b1', 'b2', 'b3']}

    data_Y = {'C': ['b2', 'b3', 'b4'],
            'D': ['dd2', 'dd3', 'dd4']}

    data_Z = {'E': [4, 5, 6],
            'F': ['b4', 'b5', 'b6']}
    df_X = pd.DataFrame(data_X)
    df_Y = pd.DataFrame(data_Y)
    df_Z = pd.DataFrame(data_Z)

    # perform integration to produce "source table"
    df_Z.columns = df_X.columns
    unioned_df = pd.concat([df_X, df_Z], ignore_index=True)
    result_df = pd.merge(unioned_df, df_Y, left_on='B', right_on='C', how='inner')
    # Convert the "sample source" DataFrame to a formatted string
    source_table_string = result_df.to_markdown(index=False, tablefmt='pipe')
    # Get the schema of the source table
    source_schema = ",".join(result_df.columns.tolist())
    data_lake_string = ""
    for tableIndx, dl_table in enumerate([df_X, df_Y, df_Z]):
        data_lake_string += "%s: \n %s \n" % ("table%d"%(tableIndx), dl_table.to_markdown(index=False, tablefmt='pipe'))
    
    sample_prompt = get_prompt_message("Source Table", source_table_string, source_schema, data_lake_string)
    sample_output = source_table_string
    return sample_prompt, sample_output
    
    
def get_prompt_message(source_table_name, source_table_string, source_schema, data_lake_string):
    # Define table reclamation
    table_reclamation_explanation = "Given an example table and a data lake, take a subset of tables from the data lake and integrate it using some Select-Project-Join-Union queries to reproduce all rows in the example table. Show me the tabular result of the integration ONLY."
    # Create GPT Prompt message
    message = table_reclamation_explanation
    message += "\n Here is an example table %s: " % (source_table_name)
    message += source_table_string
    message += "\n Here is the data lake, from which you find tables that can be used to reproduce the example table: "
    message += data_lake_string
    message+='\n Show me ONLY the resulting table from integrating a set of data lake tables such that the resulting table is as close as possible to the example tuple. The table you show me should have the same schema as the example table: %s.   Desired format: <table>. Answer with just the table output.' % (source_schema)
    return message
    
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



if __name__ == '__main__':
    model_name = "gpt-3.5-turbo-1106"

    # load in integration set for each source table
    with open(INTEG_SET_FILEPATH) as f: 
        integration_set_dict = json.load(f) # source table: [expected integration set, including err and null tables]

    all_source_messages = {}
    for source_table in tqdm(all_sources):
        # GPT3.5
        max_tokens = 9000
        num_tokens_message = max_tokens
        num_attempts = 0
        while num_tokens_message >= max_tokens:
            # Reduce  size of data lake tables if the current token length exceeds maximum number of tokens
            num_attempts += 1
            source_table_name, source_table_string, source_schema, data_lake_string = get_tables(source_table, integration_set_dict, num_attempts, fullDataLake=False)
            message = get_prompt_message(source_table_name, source_table_string, source_schema, data_lake_string)
            num_tokens_message = num_tokens_from_string(message, model_name)
            
        
        # Define system message, and prompt GPT with problem definition, source table and data lake
        system_prompt = "You are a data scientist who reproduces an example table using data lake tables. You always return the reproduced example table."
        messages = [ 
            {
                "role": "system", 
                "content": system_prompt,
                "role": "user", 
                "content": message
            }
        ] 
        
        print("%s: User's message has %s tokens" % (source_table_name, num_tokens_message))
        all_source_messages[source_table_name] = {
            "source table": source_table_name,
            "message": message 
        }
        # Run GPT
        # response = client.chat.completions.create(model=model_name,
        # messages=messages)
        
        # gpt_response = response.choices[0].message.content
        # print(f"ChatGPT: {gpt_response}") 
        
        # print("Sleeping 20 seconds to ensure API call rate limit not surpassed")
        # time.sleep(20)
    with open("gpt_prompt.json", "w+") as f: json.dump(all_source_messages, f, indent=4)
        