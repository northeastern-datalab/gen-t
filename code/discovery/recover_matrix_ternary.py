import numpy as np
import pandas as pd
import glob
import time
from tqdm import tqdm
from preprocess import getTableDfs, preprocessForeignKeys
import sys
sys.path.append('../')
import utils
from utils import projectAtts, selectKeys


def initializeTableMatrices(sourceTable, tableDfs, primaryKey, foreignKeys):
    ''' Create matrixes for each data lake table, 1 = value matches source table, 0 = doesn't match, -1 = value is non-null and doesn't match
    Input:
        sourceTable: DataFrame of source table
        tableDfs (dict): tableName and table Df
        primaryKey (string): primary key of source table
        foreignKeys (list): list of foreign keys of source table
    Return dictionary of each candidate table with its matrix representation
    '''
    tableMatrices = {}
    tableDklDict = {}
    for table, df in tqdm(tableDfs.items()):
        dfMatrix = {k: np.zeros((1, sourceTable.shape[1])).tolist() for k in sourceTable[primaryKey].values} # dictionary: source keys: matrix
        for indx, qKey in enumerate(sourceTable[primaryKey].values):
            if pd.isna(qKey): continue # skip if the source table's primary key value is Null
            i = sourceTable.index[sourceTable[primaryKey]==qKey].tolist()[0]
            commonKey = primaryKey
            commonKeys = [k for k in [primaryKey]+foreignKeys if k in df.columns]
            if len(commonKeys) > 0: commonKey = commonKeys[0]
            qKeyVal = sourceTable.iloc[i, sourceTable.columns.tolist().index(commonKey)]
            numAddedCols = 0
            if qKeyVal in df[commonKey].values:
                if df.loc[df[commonKey] == qKeyVal].shape[0] > 1: 
                    for _ in range(df.loc[df[commonKey] == qKeyVal].shape[0]-1):
                        dfMatrix[qKey] += np.zeros((1, sourceTable.shape[1])).tolist()
                        numAddedCols += 1
                        
            for qCol in sourceTable.columns:
                sourceVal = sourceTable.loc[sourceTable[primaryKey] == qKey, qCol].tolist()[0]
                # get i, j index of matrix
                if qCol not in df.columns: continue
                j = list(sourceTable.columns).index(qCol)
                
                if qKeyVal in df[commonKey].values:
                    if df.loc[df[commonKey] == qKeyVal].shape[0] > 1: 
                        # there are multiple rows in DataFrame with same key
                        dfVals = df.loc[df[commonKey] == qKeyVal, qCol].values.tolist()
                    elif type(df.loc[df[commonKey] == qKeyVal, qCol]) == pd.DataFrame: dfVals = df.loc[df[commonKey] == qKeyVal, qCol].values.tolist()[0]
                    else: dfVals = df.loc[df[commonKey] == qKeyVal, qCol].tolist()
                    # check NaNs
                    for valIndx, val in enumerate(dfVals):
                        if pd.isna(val) and pd.isna(sourceVal): # both are NULL
                            dfMatrix[qKey][valIndx][j] = 1.0
                        if val == sourceVal:
                            dfMatrix[qKey][valIndx][j] = 1.0
                        if val != sourceVal and not pd.isna(val):
                            dfMatrix[qKey][valIndx][j] = -1.0                        
                            
                elif primaryKey not in df.columns or (primaryKey in df.columns and qKey not in df[primaryKey].values):
                    if sourceVal in set(df[qCol].values): 
                        # df does not contain primary key but overlaps property values
                        dfMatrix[qKey][0][j] = 1.0
                if pd.isna(sourceVal):
                    dfMatrix[qKey][0][j] = 1.0
        tableMatrices[table] = dfMatrix
    return tableMatrices
         
def combineTernaryMatrices(aTable, bTable):
    '''
    combine ternary matrices
    '''
    combinedMatrix = {}
    for key, aRows in aTable.items():
        combinedMatrix[key] =  []
        bRows = bTable[key] # aRows, bRows = list of lists
        toCombineRows = {} # list of Booleans, if exists False then don't combine rows
        for colIndx in range(len(aRows[0])):
            aVals, bVals = [row[colIndx] for row in aRows], [row[colIndx] for row in bRows]
            for aIndx, aVal in enumerate(aVals):
                for bIndx, bVal in enumerate(bVals):
                    if (aIndx, bIndx) not in toCombineRows: toCombineRows[(aIndx, bIndx)] = []
                    if aVal != bVal and aVal != 0 and bVal != 0:
                        # if they are -1 / 1
                        toCombineRows[(aIndx, bIndx)].append(False)
                    else: toCombineRows[(aIndx, bIndx)].append(True)
        
        combinedAIndexes, combinedBIndexes = set(), set()
        for combIndx, toCombine in toCombineRows.items():
            # combine if all equal, max() if there exists a 0
            # DO NOT combine if 1 and -1 are both there
            if all(toCombine):
                combRow = np.maximum(aRows[combIndx[0]], bRows[combIndx[1]]).astype(float).tolist()
                combinedMatrix[key].append(combRow)
                combinedAIndexes.add(combIndx[0])
                combinedBIndexes.add(combIndx[1])
            else:
                if combIndx[0] not in combinedAIndexes: combinedMatrix[key].append(aRows[combIndx[0]])
                if combIndx[1] not in combinedBIndexes: combinedMatrix[key].append(bRows[combIndx[1]])
        
    return combinedMatrix
    

def traverseGraph(tableMatrices, startTable, sourceTable):
    '''
    Traverse space of matrices to and combine pairs of matrices,
    end when the resulting matrix is all 1's or all matrices have been combined
    Return:
        list of tables whose matrix representations were combined, and percentage of 1s in resulting matrix
    '''
    startTable, startMatrix = startTable, tableMatrices[startTable]
    traversedTables, nextTable = [startTable], None
    # Using Normalized VSS as evaluateSimilarity()
    prevCorrect = mostCorrect = findPercentageCorrect_norm(startMatrix)
    
    testCount = 0
    exitEarly = 0
    while len(traversedTables) < len(tableMatrices) and mostCorrect < 1.0 and not exitEarly:
        startTime = time.time()
        prevCorrect = mostCorrect
        testCount += 1
        for table, matrix in tableMatrices.items():
            if table not in traversedTables:
                intermediateMatrix = startMatrix
                if len(traversedTables) > 1: 
                    for tTable in traversedTables:
                        intermediateMatrix = combineTernaryMatrices(intermediateMatrix, tableMatrices[tTable])                
                combinedMatrix = combineTernaryMatrices(intermediateMatrix, matrix)
                # Using Normalized VSS metric as evaluateSimilarity()
                percentCorrectVals = findPercentageCorrect_norm(combinedMatrix)
    
                
                if percentCorrectVals > mostCorrect:
                    mostCorrect = percentCorrectVals
                    nextTable = table
        print(nextTable, mostCorrect)
        if mostCorrect == prevCorrect: exitEarly = 1 # iterated through all tables, and no improvement
        if not exitEarly:
            traversedTables.append(nextTable)
    print(traversedTables)
    print("Traverse %d tables with %.2f correct values" % (len(traversedTables), mostCorrect))
    return traversedTables, mostCorrect


def findPercentageCorrect_norm(matrixDict):
    '''
    evaluate Similarity: find the percentage of 1's or correct values in the current matrix
    '''
    checkTuples = []
    for key, tuples in matrixDict.items():
        if len(tuples) == 1: 
            checkTuples += tuples
        else:
            mostCorrectTuple, mostCorrectPercent = [], -0.1
            for t in tuples:
                correctPercent = len([val for val in t if val>0]) / len(t)
                if correctPercent > mostCorrectPercent:
                    mostCorrectPercent = correctPercent
                    mostCorrectTuple = t
            checkTuples.append(mostCorrectTuple)
    checkMatrix = [item for sublist in checkTuples for item in sublist]
    
    percentCorrectVals = len([val for val in checkMatrix if val>0]) / len(checkMatrix)
    percentErrVals = len([val for val in checkMatrix if val<0]) / len(checkMatrix)
    
    if percentCorrectVals == 0.0 or percentErrVals > percentCorrectVals: return None
    return 0.5*(1 + percentCorrectVals - percentErrVals)

def getDLMatrices(sourceTable, tableDfs, primaryKey,foreignKeys, startNode=None):
    '''
    initialize candidate tables as matrices to align their values with the Source Table,
    then combine matrices
    '''
    startTime = time.time()
    
    tableMatrices = initializeTableMatrices(sourceTable, tableDfs, primaryKey, foreignKeys) 
    matrixInitTime = time.time() - startTime
    print("=========== MATRIX INITIALIZATION of %d matrices took %.2f seconds =================================" % (len(tableMatrices), (matrixInitTime)))
    # pick start node
    startTime = time.time()
    startTable, mostCorrect = {}, 0
    tableCorrectVals = {}
    removeTables = []
    for table, matrix in tableMatrices.items():
        # Using Normalized VSS metric as evaluateSimilarity()
        percentCorrectVals = findPercentageCorrect_norm(matrix)
        
        if not percentCorrectVals: 
            print("REMOVING", table, percentCorrectVals)
            removeTables.append(table)
            continue
        tableCorrectVals[table] = percentCorrectVals
        print(table, percentCorrectVals)
        if percentCorrectVals > mostCorrect: 
            startTable = table
            mostCorrect = percentCorrectVals
    for table in removeTables:
        print("Removing Table ", table)
        tableDfs.pop(table)
        tableMatrices.pop(table)
    if mostCorrect == 0.0: return None, None, None, None, None, None
    
    tableMatrices = {k: v for k, v in sorted(tableMatrices.items(),key = lambda item : tableCorrectVals[item[0]], reverse=True)}
    print("Start Table: ", startTable, mostCorrect)
    print("=========== Took %.2f seconds to pick Start Table =================================" % (time.time() - startTime))
    startTime = time.time()
    traversedTables, correctVals = traverseGraph(tableMatrices, startTable, sourceTable)
    matTraverseTime = time.time() - startTime
    print("=========== MATRIX TRAVERSAL took %.2f seconds =================================" % (matTraverseTime))
    return tableDfs, tableMatrices, traversedTables, correctVals, matrixInitTime, matTraverseTime
 

def getPreprocessedTables(benchmark, datasets, sourceTableName, sourceTable):
    # Get preprocessed tables
    tableDfs = getTableDfs(benchmark, datasets, sourceTableName)
    if tableDfs is None: return tableDfs
    
    for table, df in tableDfs.items():
        # check types
        for col in df.columns:
            if col in sourceTable:
                try: df[col] = df[col].astype(sourceTable[col].dtypes.name)
                except: 
                    df = df.dropna()
                    try: df[col] = df[col].astype(sourceTable[col].dtypes.name)
                    except: df = df.drop(col, axis=1) # if cannot convert to same type, delete   
    print("There are %d preprocessed tables" % (len(tableDfs)))
    return tableDfs


def main(benchmark, sourceTableName='source'):
    FILEPATH = '/home/gfan/Datasets/%s/' % (benchmark) 
    timesStats = {}
    startTime = time.time()
    # ===== REAL Data Lake ====
    datasets = glob.glob(FILEPATH+'datalake/*.csv')
    foundPaths = []
    print("\t=========== Source Table: %s =========== " % (sourceTableName))
    sourceTable = pd.read_csv(FILEPATH+"queries/"+sourceTableName)
    # ==== TPTR Datalake
    # Primary Key is first column in dataframe
    primaryKey = sourceTable.columns.tolist()[0]
    foreignKeys = [colName for colName in sourceTable.columns.tolist() if 'key' in colName and colName != primaryKey]
    # ==== T2D_GOLD Datalake
    if 't2d_gold' in benchmark:
        primaryKey = sourceTable.columns.tolist()[0]
        # Get another primary key if the first column only has NaN's
        if len([val for val in sourceTable[primaryKey].values if not pd.isna(val)]) == 0:
            for colIndx in range(1, sourceTable.shape[1]):
                currCol = sourceTable.columns.tolist()[colIndx]
                if len([val for val in sourceTable[currCol].values if not pd.isna(val)]) > 1:
                    primaryKey = currCol
                    break
        foreignKeys = []

    print("Source Table Columns: ", sourceTable.columns.tolist())
    tableDfs = getPreprocessedTables(benchmark, datasets, sourceTableName, sourceTable)
    if not tableDfs: return None, None
    
    projectedTableDfs = projectAtts(tableDfs, sourceTable)
    finalTableDfs = selectKeys(projectedTableDfs, sourceTable, primaryKey, foreignKeys)
    print("There are %d tables after projection and selection" % (len(finalTableDfs)))
    finalTableDfs = preprocessForeignKeys(finalTableDfs, primaryKey, foreignKeys, sourceTable)
    print("There are %d tables after foreign key join" % (len(finalTableDfs)))
    for table, df in finalTableDfs.items():
        print(table, df.shape, df.columns.tolist())
    if not finalTableDfs: return None, None
    print("Source table %s has %d rows and %d columns, columns: " % (sourceTableName, sourceTable.shape[0], sourceTable.shape[1]), sourceTable.columns.tolist())
    print("Keys: ", primaryKey, foreignKeys)
    
    finalTableDfs, tableMatrices, traversedTables, correctVals, matrixInitTime, matTraverseTime = getDLMatrices(sourceTable, finalTableDfs, primaryKey, foreignKeys)
    timesStats['matrix_initialization'] = [matrixInitTime]
    timesStats['matrix_traversal'] = [matTraverseTime]
    if tableMatrices == None and traversedTables == None and correctVals == None: timesStats = None
    
    # return list of traversed tables for Source as originating tables
    return traversedTables, timesStats
    