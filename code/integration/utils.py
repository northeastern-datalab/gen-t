import pickle
import pickle5 as p
import numpy as np
import pandas as pd

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

def saveDictionaryAsPickleFile(dictToSave, dictPath):
    ''' Save dictionary as a pickle file
    Args:
        dictToSave to be saved
        dictPath: filepath to which the dictionary will be saved
    '''
    filePointer=open(dictPath, 'wb')
    pickle.dump(dictToSave,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
    filePointer.close()
    
def projectAtts(tableDfs, queryTable):
    ''' For each table, project out the attributes that are in the query table.
        If the table has no shared attributes with the query table, remove
    '''
    projectedDfs = {}
    queryCols = queryTable.columns
    for table, df in tableDfs.items():
        projectedDfs[table] = df
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
    for col in queryTable.columns:
        queryKeyVals[col] = queryTable[col].tolist()
        
    commonKey = primaryKey
    for table, df in tableDfs.items():
        selectedDfs[table] = df.drop_duplicates().reset_index(drop=True)
        
        dfCols = df.columns
        commonKeys = [k for k in [primaryKey]+foreignKeys if k in dfCols]
        allColNumVals = {}
        commonKey = None
        if commonKeys: 
            commonKey = commonKeys[0]
            numCommonKeyUniqueVals = len(set([val for val in df[commonKey].values.tolist() if not pd.isna(val)]))
        
        for col in dfCols:
            uniqueVals = set([val for val in df[col].values.tolist() if not pd.isna(val)])            
            if col in commonKeys and col != commonKey:
                if len(uniqueVals) > numCommonKeyUniqueVals:
                    numCommonKeyUniqueVals = len(uniqueVals)                    
                    commonKey = col
            elif col not in commonKeys: allColNumVals[col] = len(uniqueVals)
        allColNumVals = {k: v for k, v in sorted(allColNumVals.items(), key=lambda item: item[1], reverse=True)}
        tableFK = list(allColNumVals.keys())
        if commonKey:
            commonKeys = [commonKey]
            if tableFK: commonKeys.append(tableFK[0])
            conditions = [df[commonKeys[0]].isin(queryKeyVals[commonKeys[0]]).values]
            if len(commonKeys) > 1:
                for commonKey in commonKeys[1:]:
                    conditions.append([df[commonKey].isin(queryKeyVals[commonKey]).values])    
            conditions.append(np.full((1,len(conditions[0])), True, dtype=bool))              
            selectedTuples = df.loc[np.bitwise_and.reduce(conditions)[0]]
        else: 
            commonKeys = tableFK[:2]
            print("%s commonKeys: " % (table), commonKeys)
            conditions = [df[commonKeys[0]].isin(queryKeyVals[commonKeys[0]]).values]
            if len(commonKeys) > 1:
                for commonKey in commonKeys[1:]:
                    conditions.append([df[commonKey].isin(queryKeyVals[commonKey]).values])                
            else: conditions.append(np.full((1,len(conditions[0])), False, dtype=bool))  
            selectedTuples = df.loc[np.bitwise_or.reduce(conditions)[0]]
             
        if not selectedTuples.empty:
            selectedDfs[table] = selectedTuples 
    return selectedDfs
