# Given a set of candidate tables, convert every table to text file (similar to json)
import glob
import pandas as pd
import numpy as np
import pickle
import pickle5 as p

def loadDictionaryFromPickleFile(dictPath):
    ''' Load the pickle file as a dictionary
    Args:
        dictPath: path to the pickle file
    Return: dictionary from the pickle file
    '''
    filePointer=open(dictPath, 'rb')
    dictionary = p.load(filePointer)
    filePointer.close()
    return dictionary


def projectAtts(tableDfs, queryTable):
    ''' For each table, project out the attributes that are in the query table.
        If the table has no shared attributes with the query table, remove
    '''
    projectedDfs = {}
    queryCols = queryTable.columns
    for table, df in tableDfs.items():
        tableCols = df.columns
        projectedTable = df.drop(columns=[c for c in tableCols if c not in queryCols])
        if not projectedTable.empty:
            projectedDfs[table] = projectedTable
    return projectedDfs

def selectKeys(tableDfs, queryTable, primaryKey, foreignKeys):
    ''' For each table, select tuples that contain key value from queryTable
        If the table has no shared keys with the queryTable, remove
    '''    
    selectedDfs = {}
    queryKeyVals = {}
    # for key in [primaryKey]+foreignKeys:
    for key in queryTable.columns:
        queryKeyVals[key] = queryTable[key].tolist()

    commonKey = primaryKey
    for table, df in tableDfs.items():
        dfCols = df.columns.tolist()
        commonKeys = [k for k in [primaryKey]+foreignKeys if k in df.columns]
        if not commonKeys: continue
        
        if len(commonKeys) == 1:
            commonKey = commonKeys[0]
            numUniqueVals, tableFK = 0, None
            for col in df.columns:
                if col == commonKey: continue
                if df[col].count() > numUniqueVals:
                    numUniqueVals = df[col].count()
                    tableFK = col
            commonKeys = [commonKey, tableFK]
            print("%s only has 1 key, now commonKeys = " % (table), commonKeys)
        if len(commonKeys) > 1: 
            conditions = [df[commonKeys[0]].isin(queryKeyVals[commonKeys[0]]).values]
            for commonKey in commonKeys[1:]:
                conditions.append([df[commonKey].isin(queryKeyVals[commonKey]).values])                
            selectedTuplesDf = df.loc[np.bitwise_and.reduce(conditions)[0]].drop_duplicates().reset_index(drop=True)
            
        if not selectedTuplesDf.empty:
            selectedDfs[table] = selectedTuplesDf
                
    return selectedDfs



def restructureDf(originalDf):
    colNames = [i for i in range(originalDf.shape[1])]
    newData = []
    newData.append(originalDf.columns.to_list())
    newData += originalDf.values.tolist()
    newDf = pd.DataFrame(newData, columns=colNames)
    return newDf

def main(benchmark, sourceTableName, projSel=0):
    FILEPATH = '../../Datasets/%s/' % (benchmark)  
    if '_groundtruth' in benchmark or '_mtTables' in benchmark: FILEPATH = '../../Datasets/%s/' % ('tpch')  
    sourceTable = pd.read_csv(FILEPATH+"queries/"+sourceTableName)#, header=None)
    sourceTable.columns = [col.replace("'", '') for col in sourceTable.columns]
    # remove apostrophe
    for col in sourceTable.columns:
        try:
            sourceTable[col] = sourceTable[col].str.replace("'", '')
        except: continue
    DATA_PATH = FILEPATH+'datalake/'
    
    candidateFileName = benchmark
    if 'tpch' in benchmark:
        # candidateFileName = 'tpch_small'
        b_list = benchmark.split('_')
        b_list.insert(1,'small')
        candidateFileName = '_'.join(b_list)
    print("candidateFileName", candidateFileName)
    candidateTableDict = loadDictionaryFromPickleFile("/Users/gracefan/Documents/Meetings with Renee/auto-pipeline/results_candidate_tables/%s/%s_candidateTables.pkl" % (candidateFileName, sourceTableName))
    print("Reformatting %d Candidate Tables" % (len(candidateTableDict)))
    
    tableDfs = {}
    for tableName in candidateTableDict:
        table = DATA_PATH+tableName
        df = pd.read_csv(table)#, header=None)#, encoding='latin1',warn_bad_lines=True, error_bad_lines=False)
        if len(candidateTableDict[tableName]) > 0: print("RENAME COLUMNS: ", tableName, candidateTableDict[tableName])
        df = df.rename(columns=candidateTableDict[tableName])
        df.columns = [col.replace("'", '') for col in df.columns]
        for col in df.columns:
            try:
                df[col] = df[col].str.replace("'", '')
            except: continue
        tableDfs[table] = df
    finalTableDfs = tableDfs
    if projSel:
        # ==== PROJECT / SELECT Source Table's Columns / Keys
        sourceCols = sourceTable.columns.tolist()
        primaryKey = sourceCols[0]
        foreignKeys = [colName for colName in sourceCols if 'key' in colName and colName != primaryKey]
        
        if 't2d_gold' in benchmark:
            # ==== T2D_GOLD Datalake
            # Get another primary key if the first column only has NaN's
            if len([val for val in sourceTable[primaryKey].values if not pd.isna(val)]) == 0:
                for colIndx in range(1, sourceTable.shape[1]):
                    currCol = sourceCols[colIndx]
                    if len([val for val in sourceTable[currCol].values if not pd.isna(val)]) > 1:
                        primaryKey = currCol
                        break
            foreignKeys = []
        projectedTableDfs = projectAtts(tableDfs, sourceTable)
        finalTableDfs = selectKeys(projectedTableDfs, sourceTable, primaryKey, foreignKeys)
        for table, df in finalTableDfs.items():
            df.reset_index(drop=True, inplace=True)
        
    # Move source table's column headers to first row
    sourceTable = restructureDf(sourceTable)
    foofah_input = '{"InputTable": ' 
    # foofah_input_tables = ' '
    foofah_input_tables = ' ['
    numInputTables = 0
    for table in finalTableDfs:
        df = restructureDf(finalTableDfs[table])
        if numInputTables != 0: foofah_input_tables += ', '
        foofah_input_tables += str(df.to_json(orient="values"))
        numInputTables += 1
        
        
    foofah_input_tables += ']'
    foofah_input += foofah_input_tables
    foofah_input += ', "NumSamples": %d' % (numInputTables)
        
    foofah_input += ', "TestName": "1"'
    
    foofah_input += ', "TestingTable": []'
    # foofah_input += foofah_input_tables
    
    foofah_input += ', "OutputTable": '
    foofah_input += str(sourceTable.to_json(orient="values"))
    
    foofah_input += ', "TestAnswer": []'
    # foofah_input += str(sourceTable.to_json(orient="values"))
    foofah_input += '}'
    
    input_folder_path = 'Inputs/%s/%s_for_foofah.txt' % (benchmark, sourceTableName)
    if projSel:
        input_folder_path = 'Inputs_projSel/%s/%s_for_foofah.txt' % (benchmark, sourceTableName)
        
    with open(input_folder_path, 'w') as f:
        f.write(foofah_input)