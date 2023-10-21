import numpy as np
from math import e, inf
import pandas as pd

def getColProbs(col, candidateTable, queryColVals, queryTable, primaryKey, currMinDklSums):
    # queryTable, candidateTable, primaryKey, currMinDkl, queryTime=1, log=0
    '''
    Pass in a query table dataframe and a candidate table dataframe
    Iterate through each column in query table and calculate the probability that the candidate table has correct and incorrect value
    '''    
    currColDkl = {}
    keyOccur = {keyVal: 0 for keyVal in queryTable[primaryKey]}
    correctVals = {keyVal: [] for keyVal in queryTable[primaryKey]}
    incorrectVals = {keyVal: [] for keyVal in queryTable[primaryKey]}
    for _, row in candidateTable.iterrows():
        keyValAtRow = row[primaryKey]
        if keyValAtRow not in queryColVals: continue
        valAtRow = row[col]
        if type(valAtRow)  == pd.Series: valAtRow = next(iter(set(valAtRow.values)))
        if pd.isna(valAtRow) and not pd.isna(queryColVals[keyValAtRow]): continue
        keyOccur[keyValAtRow] += 1
        
        if not type(queryColVals[keyValAtRow]) == type(valAtRow):
            try:
                queryValChangedType = type(valAtRow)(queryColVals[keyValAtRow])
                valAtRowChangedType = type(queryColVals[keyValAtRow])(valAtRow)
            
                if queryValChangedType == valAtRow:
                    queryColVals[keyValAtRow] = queryValChangedType
                elif valAtRowChangedType == queryColVals[keyValAtRow]:
                    valAtRow = valAtRowChangedType
            except:
                pass
        if valAtRow == queryColVals[keyValAtRow]: # correct value found
            correctVals[keyValAtRow].append(valAtRow)
        elif pd.isna(valAtRow) and pd.isna(queryColVals[keyValAtRow]): # both found and expected are NULLs
            correctVals[keyValAtRow].append(valAtRow)
        else:
            if not pd.isna(valAtRow): # found erroneous value
                incorrectVals[keyValAtRow].append(valAtRow)
    for keyVal in correctVals:
        if keyOccur[keyVal] == 0:
            # doesn't appear, add KL divergence of 0.5
            if col not in currColDkl: currColDkl[col] = 0.5
            else: currColDkl[col] += 0.5
            continue
        valProb = len(correctVals[keyVal]) / keyOccur[keyVal]
        errValProb = len(incorrectVals[keyVal]) / keyOccur[keyVal]
        valDKL = getVal_Dkl(valProb, errValProb)
        
        if col not in currColDkl: currColDkl[col] = valDKL
        else: currColDkl[col] += valDKL
    
    return currColDkl
    
def getVal_Dkl(prob_x, prob_notx):
    '''
    Arguments:
        prob_x (float): probability that the values are correct given the keys are correct
        prob_notx (float)
    Finds KL-divergence of output for each correct value given each key
    '''
    col_dkl = 0
    # Assume P(x | key) is always 1
    if (prob_x * (1 - prob_notx)) == 0:
        # causes error in log
        # Occurs when (1) prob_x = 0 or (2) prob_notx = 1
        col_dkl += -1.0
    else:
        ind_dkl = np.log(prob_x * (1 - prob_notx))
        col_dkl = ind_dkl
    return -col_dkl

def getQueryConditionalVals(queryTable, primaryKey):
    # get query table value pairs
    queryPairs = {}
    for col in queryTable.columns:
        if col == primaryKey:
            continue
        valPairs = {}
        for i, val in enumerate(queryTable[col]):
            # unique keys: one correct value for each query key value
            valPairs[list(queryTable[primaryKey])[i]] = val
        queryPairs[col] = valPairs
    return queryPairs

def table_Dkl(queryTable, candidateTable, primaryKey, queryValPairs, currMinDkl, queryTime=1, log=0):
    breakEarly = 0
    colDKL, tableDkl, finalTableDkl = {}, 0, inf
    currMinDklSums = currMinDkl*(len(queryTable.columns)-1)
    # Calculate probability that key is correct
    if primaryKey in list(candidateTable.columns):
        # unique keys
        overlapKeys = set(candidateTable[primaryKey]).intersection(set(queryTable[primaryKey]))
        keyProb = len(overlapKeys) / len(queryTable[primaryKey])
        colDKL[primaryKey] = keyProb
        if keyProb == 0: breakEarly = 1
    else: 
        print("PrimaryKey not in columns -> inf DKL")
        breakEarly = 1
    if not breakEarly:
        for col in queryTable.columns:
            if col not in candidateTable.columns:
                # column not found in candidate table
                colDKL[col] = 1.0*len(queryValPairs[col])
                tableDkl += colDKL[col]  / colDKL[primaryKey]
                if tableDkl > currMinDklSums: breakEarly = 1
                if queryTime and breakEarly: break
        
        for col in candidateTable.columns:
            if col == primaryKey or col not in queryTable.columns: continue
            queryColVals = queryValPairs[col] # dictionary of key value: expected value
            currColDkl = getColProbs(col, candidateTable, queryColVals, queryTable, primaryKey, currMinDklSums)
            colDKL.update(currColDkl)
            # Check if we need to break if it is already > currMinDkl
            tableDkl += colDKL[col]  / colDKL[primaryKey]
            if tableDkl > currMinDklSums: breakEarly = 1
            if queryTime and breakEarly: break
        finalTableDkl = tableDkl / (len(colDKL)-1)
    if log:
        print("colDKL: ", colDKL)
        print("\tTABLE has Divergence", finalTableDkl)
    return finalTableDkl, breakEarly, colDKL