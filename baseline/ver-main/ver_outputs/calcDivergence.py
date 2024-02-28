import numpy as np
from math import e, inf
import random
import pandas as pd
import itertools
from functools import reduce
from sklearn.metrics import mutual_info_score
# from scipy.stats import entropy

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
        
        

def getAddVal_DKl(prob_x):
    col_dkl = 0
    # Assume P(x | key) is always 1
    if prob_x == 0: col_dkl += -1.0 # causes error in log
    else: col_dkl = np.log(prob_x)
    return -col_dkl
        
def getAddColProbs(sourceCol, addCol):    
    sumValueDkls = 0
    for val in sourceCol:
        # TODO: do we count nulls?
        numOcc = addCol.tolist().count(val)
        if pd.isna(val): numOcc = sum(pd.isna(x) for x in addCol.tolist())
        valProb = numOcc / len(addCol)
        sumValueDkls += getAddVal_DKl(valProb)
    return sumValueDkls
        
    
    
def getAddCol_Dkl(queryTable, candidateTable, primaryKey):
    ''' For each column that is NOT in the queryTable, find the Q Col that renders the minimum KL-divergence
    Don't consider additional tuples
    Key assumption here!
    '''
    additionalColDf = candidateTable.copy()
    # additionalColDf = additionalColDf[additionalColDf[primaryKey].isin(queryTable[primaryKey])]
    additionalColDf = additionalColDf.drop(columns=[c for c in queryTable.columns])
    # iterate through each additional column, and calculate its KL-divergence with every column from query table
    # Get column from query table that it has the lowest KL-divergence with
    colMinDkl = {}
    for addColName in additionalColDf.columns:
        colMinDkl[addColName] = inf
        addCol = additionalColDf[addColName]
        for sourceColName in queryTable.columns:
            sourceCol = queryTable[sourceColName]
            # sum over all values in sourceCol of log(probability that value is in addCol)
            colDkl = getAddColProbs(sourceCol, addCol)
            if colDkl < colMinDkl[addColName]: colMinDkl[addColName] = colDkl
    return sum(colMinDkl.values())


def getAddRowProbs(sourceRow, addRow):    
    sumValueDkls = 0
    for val in sourceRow:
        # TODO: do we count nulls?
        numOcc = addRow.count(val)
        if pd.isna(val): numOcc = sum(pd.isna(x) for x in addRow)
        valProb = numOcc / len(addRow)
        sumValueDkls += getAddVal_DKl(valProb)
    return sumValueDkls
        
    
    
def getAddRow_Dkl(queryTable, candidateTable, primaryKey):
    ''' For each column that is NOT in the queryTable, find the Q Col that renders the minimum KL-divergence
    Don't consider additional tuples
    Key assumption here!
    '''
    additionalRowDf = candidateTable.copy()
    additionalRowDf = additionalRowDf[~additionalRowDf[primaryKey].isin(queryTable[primaryKey])]
    # iterate through each additional row, and calculate its KL-divergence with every row from query table
    # Get row from query table that it has the lowest KL-divergence with
    tableMinDkls = 0
    additionalRows = [list(x) for x in additionalRowDf.values.tolist()]
    sourceRows = [list(x) for x in queryTable.values.tolist()]
    for addRow in additionalRows:
        rowMinDkl = inf
        for sourceRow in sourceRows:
            rowDkl = getAddRowProbs(sourceRow, addRow)
            if rowDkl < rowMinDkl: 
                rowMinDkl = rowDkl
        tableMinDkls += rowMinDkl
    return tableMinDkls

def getMutualInformation(addCol, sourceCol):
    def entropy(*X):
        return np.sum(-p * np.log2(p) if p > 0 else 0 for p in
            (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
                for classes in itertools.product(*[set(x) for x in X])))

    addEntropy = entropy(addCol)
    sourceEntropy = entropy(sourceCol)
    addSourceEntropy = entropy(addCol,sourceCol)
    print(addEntropy,sourceEntropy,addSourceEntropy)
    # mutual information = H(X) + H(Y) - H(X,Y)
    return addEntropy, addEntropy+sourceEntropy-addSourceEntropy

def getConditionalEntropy(addCol, sourceCol):
    jointVals = []
    for x, y in zip(addCol,sourceCol):
        jointVals.append((x,y))
    condEntropy = []
    for indx, addVal in enumerate(addCol.tolist()):
        sourceVal = sourceCol.tolist()[indx]
        jointProb = jointVals.count((addVal, sourceVal))
        jointProb /= len(jointVals)
        sourceProb = sourceCol.tolist().count(sourceVal)
        sourceProb /= len(sourceCol.tolist())
        condEntropy.append(jointProb*np.log(jointProb/sourceProb))
    return -sum(condEntropy)
        

def getAddColMutualInfo(queryTable, candidateTable,primaryKey):
    ''' Minimize the maximal Mutual Information with a query column
    '''
    additionalColDf = candidateTable.copy()
    # additionalColDf = additionalColDf[additionalColDf[primaryKey].isin(queryTable[primaryKey])]
    additionalColDf = additionalColDf.drop(columns=[c for c in queryTable.columns])
    overlapColDf = candidateTable.copy().drop(columns=[c for c in candidateTable.columns if c not in queryTable.columns])
    # iterate through each additional column, and calculate its KL-divergence with every column from query table
    # Get column from query table that it has the lowest KL-divergence with
    colMaxGain = {}
    for addColName in additionalColDf.columns:
        colMaxGain[addColName] = -inf
        addCol = additionalColDf[addColName].fillna(additionalColDf[addColName].mode(dropna=True)[0])
        for sourceColName in overlapColDf.columns:
            if sourceColName == primaryKey: continue
            sourceCol = overlapColDf[sourceColName].fillna(overlapColDf[sourceColName].mode(dropna=True)[0])
            condEntropy = getConditionalEntropy(addCol, sourceCol)
            print(addColName,sourceColName, condEntropy)
            if condEntropy > colMaxGain[addColName]: colMaxGain[addColName] = condEntropy
            # mutualInfo = mutual_info_score(addCol, sourceCol)
    #         addEntropy, mutualInfo = getMutualInformation(addCol, sourceCol)
    #         # GOAL: maximize addEntropy and minimize (mutualInfo / addEntropy)
    #         # alt goal: maximize conditional entropy: H(X|Y)
    #         infoGain = addEntropy
    #         if addEntropy != 0: infoGain -= (mutualInfo/addEntropy)
                
    #         print(addColName,sourceColName,mutualInfo, addEntropy, infoGain)
    #         if mutualInfo > colMaxGain[addColName]: colMaxGain[addColName] = infoGain
    print(colMaxGain)
    return sum(colMaxGain.values())