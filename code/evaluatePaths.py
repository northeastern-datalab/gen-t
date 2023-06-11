import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def convertDfToLists(df):
    tupleList = []
    for row in df.itertuples(index=False):
        rowTuple = ['NAN' if pd.isna(val) else val for val in list(row)]
        tupleList.append(tuple(rowTuple))
    return tupleList
    
def alignSchemas(queryDf, integratedDf):
    qSchema = sorted(queryDf.columns.tolist())
    resultSchema = sorted(integratedDf.columns.tolist())
    for col in qSchema:
        if col not in resultSchema:
            integratedDf[col] = np.nan
    resultSchema = sorted(integratedDf.columns.tolist())
    queryDf = queryDf[qSchema]
    integratedDf = integratedDf[resultSchema]
    return queryDf, integratedDf

def alignTypes(queryDf, integratedDf):
    # align types
    for col in queryDf.columns.tolist():
        if col in integratedDf.columns:
            if is_numeric_dtype(queryDf[col]) and is_numeric_dtype(integratedDf[col]) and queryDf[col].dtype != integratedDf[col]:
                qColType = queryDf[col].dtypes.name
                integratedColType = integratedDf[col].dtypes.name
                try: integratedDf[col] = integratedDf[col].astype(qColType) 
                except: queryDf[col] = queryDf[col].astype(integratedColType)
    return queryDf, integratedDf
            
def setTDR(queryDf, integratedDf):
    '''
    From ALITE
    convert DFs to sets of tuples, arranged lexicographically, then find intersection
    '''
    if queryDf.columns.tolist() != integratedDf.columns.tolist():
        queryDf, integratedDf = alignSchemas(queryDf, integratedDf)
        
    queryDf, integratedDf = alignTypes(queryDf, integratedDf)
    qTuples = convertDfToLists(queryDf) # list of tuples
    resultTuples = convertDfToLists(integratedDf) # list of tuples
    commonTuples = list(set(qTuples).intersection(set(resultTuples)))
    TDR_recall = len(commonTuples) / queryDf.shape[0]
    TDR_precision = len(commonTuples) / integratedDf.shape[0]
    return TDR_recall, TDR_precision


def bestMatchingTuples(queryDf, integratedDf, primaryKey):
    '''
    for row in queryDf:
        get primary key value
        find rows in integratedDf with primary key value
        pick best row and add to resultingDf 
    '''
    bestMatchingTuples = []
    for rowIndx in range(queryDf.shape[0]):
        queryRow = queryDf.loc[rowIndx,:]
        keyVal = getattr(queryRow, primaryKey)
        qRowList = list(queryRow)
        correspondingRows = integratedDf.loc[integratedDf[primaryKey] == keyVal]
        mostCorrectTuple, mostCorrectPercent = [], -0.1
        for row in correspondingRows.itertuples(index=False):
            numCorrectVals = 0
            for indx, qVal in enumerate(qRowList):
                if qVal == list(row)[indx]: numCorrectVals += 1
                elif pd.isna(qVal) and pd.isna(list(row)[indx]): numCorrectVals += 1
            if numCorrectVals/len(qRowList) > mostCorrectPercent:
                mostCorrectPercent = numCorrectVals/len(qRowList)
                mostCorrectTuple = list(row)
        if mostCorrectTuple: bestMatchingTuples.append(mostCorrectTuple)
    bestMatchingDf = pd.DataFrame(bestMatchingTuples, columns=integratedDf.columns)
    if not bestMatchingTuples: return None
    queryDf, bestMatchingDf = alignTypes(queryDf, bestMatchingDf)
    return bestMatchingDf


def instanceSimilarity(queryDf, integratedDf, primaryKey):
    # From MapMerge
    instanceSims = []
    for keyVal in queryDf[primaryKey]:
        queryTuples = queryDf.loc[queryDf[primaryKey] == keyVal].values.tolist()
        if len(queryTuples) == 0: continue # if keyVal is nan
        queryTuple = queryTuples[0]
        resultTuples = integratedDf.loc[integratedDf[primaryKey] == keyVal].values.tolist()
        if len(resultTuples) == 0: continue
        resultTuple = resultTuples[0]
        numCommonVals = 0
        for colIndx in range(queryDf.shape[1]):
            if queryTuple[colIndx] == resultTuple[colIndx]: numCommonVals += 1
            elif pd.isna(queryTuple[colIndx]) and pd.isna(resultTuple[colIndx]): numCommonVals += 1
        
        tupleSim = numCommonVals / queryDf.shape[1]
        instanceSims.append(tupleSim)
    return sum(instanceSims) / queryDf.shape[0]
    