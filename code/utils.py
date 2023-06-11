import pickle
import pickle5 as p
import numpy as np

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
    
def loadListFromTxtFile(listPath): 
    ''' Save list as a text file
    Args:
        listPath: filepath with the stored list
    '''
    savedList = []
    with open(listPath) as f:
        listItems = f.read().splitlines()
        for i in listItems:
            savedList.append(i)
    return savedList
    
def saveListAsTxtFile(listToSave, listPath): 
    ''' Save list as a text file
    Args:
        listToSave to be saved
        listPath: filepath to which the list will be saved
    '''
    with open(listPath, "w") as output:
        for item in listToSave:
            output.writelines(str(item)+'\n')
            
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