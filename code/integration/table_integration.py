import sys
import pandas as pd
import numpy as np
import glob
import os
import time
import utils
from utils import projectAtts, selectKeys
sys.path.append('../')
from evaluatePaths import setTDR

replaceNull = '*NAN*'
def FindCurrentNullPattern(tuple1):
    current_pattern = ""
    current_nulls = 0
    for t in tuple1:
        if pd.isna(t):
            current_pattern += "0"
            current_nulls += 1
        else:
            current_pattern += "1"
    return current_pattern, current_nulls

#used to check what are the ancestor buckets of the child bucket
def CheckAncestor(child_bucket, parent_bucket):
    for i in range(len(child_bucket)):
        if int(child_bucket[i]) == 1 and int(parent_bucket[i])==0:
            return 0
    return 1

def CheckNonNullPositions(tuple1, total_non_nulls):
    non_null_positions = set()
    for i in range(0, len(tuple1)):
        if int(tuple1[i]) == 1:
            non_null_positions.add(i)
            if len(non_null_positions) == total_non_nulls:
                return non_null_positions
    return (non_null_positions)

def GetProjectedTuple(tuple1, non_null_positions, m):
    projected_tuple = tuple()
    for j in range(0,m):
        if j in non_null_positions:
            projected_tuple += (tuple1[j],)
    return projected_tuple

#preprocess input tables
def preprocess(table):
    table.columns = map(str.lower, table.columns)
    table = table.applymap(str) 
    table = table.apply(lambda x: x.str.lower()) #convert to lower case
    table = table.apply(lambda x: x.str.strip()) #strip leading and trailing spaces, if any
    return table

def labelNullsDf(queryDf, df, keyCols):
    try: 
        queryDf = queryDf[df.columns]
    except: return None
    labeledNullsDf = df.copy()
    originalCols = labeledNullsDf.columns
    dfKeyCols = [col for col in keyCols if col in originalCols]
    if not dfKeyCols: return None
    keyCol = dfKeyCols[0]
    for keyVal in queryDf[keyCol]:
        if pd.isna(keyVal) and not [val for val in labeledNullsDf[keyCol].values if pd.isna(val) or val == replaceNull]:
            nullTuple = [[np.nan]*len(df.columns)]
            labeledNullsDfVals = labeledNullsDf.values.tolist()
            labeledNullsDf = pd.DataFrame(labeledNullsDfVals+nullTuple, columns=originalCols)
            df = labeledNullsDf
        qTuples = queryDf.loc[queryDf[keyCol]==keyVal].values.tolist()
        candTuples = df.loc[df[keyCol]==keyVal].values.tolist()
        candIndxes = df.index[df[keyCol]==keyVal].tolist()
        if pd.isna(keyVal):
            qTuples = queryDf[queryDf[keyCol].isnull()].drop_duplicates().values.tolist()
            candTuples = df[df[keyCol].isnull()].drop_duplicates().values.tolist()
            candIndxes = df.index[df[keyCol].isnull()].drop_duplicates().values.tolist()
        qTuple = qTuples[0]
        qTupleNullIndxs = [i for i, val in enumerate(qTuple) if pd.isna(val)]
        for cInd, cT in enumerate(candTuples):
            qTupleNullIndxs = [i for i, val in enumerate(qTuple) if pd.isna(val)]
            candTupleNullIndxs = [i for i, cval in enumerate(cT) if pd.isna(cval)]
            commonNullIndxs = list(set(qTupleNullIndxs).intersection(set(candTupleNullIndxs)))
            for colInd in commonNullIndxs:
                labeledNullsDf.loc[candIndxes[cInd], originalCols[colInd]] = replaceNull
    labeledNullsDf.columns = originalCols
    return labeledNullsDf

def ReplaceNulls(table, null_count):
    null_set = set()
    for colname in table:
        for i in range (0, table.shape[0]):
            try:
                if pd.isna(table[colname][i]):
                    table[colname][i] = "null"+ str(null_count)
                    null_set.add("null"+ str(null_count))
                    null_count += 1
            except:
                print(colname)
                print(table.shape[0])
                print(i)
                sys.exit()
    return table, null_count, null_set

def AddNullsBack(table, nulls):
    columns = list(table.columns)
    input_rows = list(tuple(x) for x in table.values)
    output_rows = []
    for t in input_rows:
        new_t = tuple()
        for i in range(0, len(t)):
            if str(t[i]) in nulls:
                new_t += (np.nan,)
            else:
                new_t += (t[i],)
        output_rows.append(new_t)
    final_table = pd.DataFrame(output_rows, columns =columns)
    return final_table


def CountProducedNulls(list_of_tuples):
    labeled_nulls = 0
    for row in list_of_tuples:
        for value in row:
            if value == "nan":
                labeled_nulls += 1
    return labeled_nulls


# =============================================================================
# Efficient complementation using partitioning starts here
# =============================================================================
def complementTuples(tuple1, tuple2):
    keys = 0 #find if we have common keys
    alternate1= 0 #find if we have alternate null position with non-null value in the first tuple
    alternate2 = 0 #find if we have alternate null position with non-null value in the second tuple
    newTuple = list()
    
    for i in range(0,len(tuple1)):
        first = tuple1[i]
        if pd.isna(first): first = "nan"
        second = tuple2[i]
        if pd.isna(second): second = "nan"
        if first != "nan" and second!="nan" and first != second:
            return (tuple1,False)
        elif first == "nan" and second =="nan":
            # newTuple.append(first)
            newTuple.append(np.nan)
            
        elif first != "nan" and second!="nan" and first == second: #both values are equal
            keys+=1
            newTuple.append(first)
        #second has value and first is null
        elif first == "nan" and second != "nan":
            alternate1+=1
            newTuple.append(second)
        #first has value and second is null
        elif (second =="nan" and first != "nan"):
            alternate2+=1
            newTuple.append(first)
    count = 0
    for item in newTuple:
        if(pd.isna(item)):    
            count+=1     
    if (keys >0 and alternate1 > 0 and alternate2>0 and count != len(newTuple)):
        return (tuple(newTuple),True)
    else:
        return (tuple(tuple1),False)
    

        
def PartitionTuples(table, partitioning_index):
    partitioned_tuple_dict = dict()
    all_tuples = [tuple(x) for x in table.values]
    for t in all_tuples:
        if t[partitioning_index] in partitioned_tuple_dict:
            partitioned_tuple_dict[t[partitioning_index]].append(t)
        else:
            partitioned_tuple_dict[t[partitioning_index]] = [t]
    return partitioned_tuple_dict

def GetPartitionsFromList(all_tuples, partitioning_index):
    partitioned_tuple_dict = dict()
    for t in all_tuples:
        if t[partitioning_index] in partitioned_tuple_dict:
            partitioned_tuple_dict[t[partitioning_index]].add(t)
        else:
            partitioned_tuple_dict[t[partitioning_index]] = {t}
    null_partition = partitioned_tuple_dict.pop(np.nan, None)
    if null_partition is None:
        for each in partitioned_tuple_dict:
            partitioned_tuple_dict[each] = list(partitioned_tuple_dict[each])
        return partitioned_tuple_dict
    else:
        if len(partitioned_tuple_dict) == 0:
            partitioned_tuple_dict[np.nan] = list(null_partition)
            return partitioned_tuple_dict
        for each in partitioned_tuple_dict:
            temp_list = partitioned_tuple_dict[each]
            temp_list = temp_list.union(null_partition)
            partitioned_tuple_dict[each] = list(temp_list)            
    return partitioned_tuple_dict

def SelectPartitioningOrder(table):
    statistics = dict()
    stat_unique = {}
    stat_nulls = {}
    total_rows = table.shape[0]
    unique_weight = 0
    null_weight = 1 - unique_weight #only based on null weight
    i = 0
    for col in table:
        unique_count = len(set(table[col]))
        null_count = total_rows - table[col].isna().sum()
        score = (unique_count * unique_weight) + null_count * null_weight
        statistics[i] = score
        stat_unique[i] = unique_count
        stat_nulls[i] = total_rows - null_count
        i += 1
    stat_nulls = sorted(stat_nulls, key = stat_nulls.get, reverse = True)
    stat_unique = sorted(stat_unique, key = stat_unique.get, reverse = True)
    final_list = [stat_nulls[0]]
    stat_unique.remove(stat_nulls[0])
    final_list += stat_unique
    #return final_list    
    return sorted(statistics, key = statistics.get, reverse = True)

def FineGrainPartitionTuples(table, timeout, fdaStartTime):  
    input_tuples = list({tuple(x) for x in table.values})
    partitioning_order = SelectPartitioningOrder(table)
    debug_dict = {}
    list_of_list = []
    assign_tuple_id = {}
    for tid, each_tuple in enumerate(input_tuples):
        assign_tuple_id[each_tuple] = tid 
        if (time.time() - fdaStartTime) > timeout: return None, None
    list_of_list.append(input_tuples)
    finalized_list = []
    for i in partitioning_order:
        new_tuples = []
        track_used_tuples = {}
        for all_tuples in list_of_list:
            if len(all_tuples) > 100:
                partitions = GetPartitionsFromList(all_tuples, i)
                for each in partitions:
                    current_partition = partitions[each]
                    create_tid = set()
                    for current_tuple in current_partition:
                        create_tid.add(assign_tuple_id[current_tuple])
                    create_tid = tuple(sorted(create_tid))
                    if create_tid not in track_used_tuples:
                        if len(current_partition) > 100:
                            new_tuples.append(current_partition)
                        else:
                            finalized_list.append(current_partition)
                        track_used_tuples[create_tid] = 1
            else:
                finalized_list.append(all_tuples)
            if (time.time() - fdaStartTime) > timeout: return None, None
        list_of_list = new_tuples
        debug_dict[i] = list_of_list
    if len(list_of_list) > 0:    
        finalized_list = list_of_list + finalized_list
    return finalized_list, debug_dict


def ComplementAlgorithm(tuple_list, timeout, fdaStartTime):
    receivedTuples = dict()
    for t in tuple_list:
        receivedTuples[t] = 1
    complementResults = dict()
    while (1):
        i = 1
        used_tuples = dict()
        for tuple1 in tuple_list:
            complementCount = 0
            for tuple2 in tuple_list[i:]:
                (t, flag) = complementTuples(tuple1, tuple2)
                if (flag == True):
                    complementCount += 1
                    complementResults[t] = 1
                    used_tuples[tuple2] = 1
            i += 1
            if complementCount == 0 and tuple1 not in used_tuples:
                complementResults[tuple1] = 1
            if (time.time() - fdaStartTime) > timeout: return None
        if receivedTuples.keys() == complementResults.keys():
            break
        else:
            receivedTuples = complementResults
            complementResults = dict()
            tuple_list = [tuple(x) for x in receivedTuples]
    return [tuple(x) for x in complementResults]

def MoreEfficientComplementation(table, timeout, fdaStartTime):
    partitioned_tuple_list, debug_dict = FineGrainPartitionTuples(table, timeout, fdaStartTime)
    if not partitioned_tuple_list: return None, None, None, None, None
    complemented_list = set()
    count = 0 
    max_partition_size = 0
    for current_partition_tuples in partitioned_tuple_list:
        current_size = len(current_partition_tuples)
        if current_size > max_partition_size:
            max_partition_size = current_size
        complemented_tuples = ComplementAlgorithm(current_partition_tuples, timeout, fdaStartTime)
        if not complemented_tuples: return None, None, None, None, None
        for item in complemented_tuples:
            complemented_list.add(item)
        count +=1
        if count % 100000 == 0:
            print("partitions processed: ", count)
            print("generated tuples until now: ",len(complemented_list))
            print("Total partitions :", len(partitioned_tuple_list))
        if (time.time() - fdaStartTime) > timeout: return None, None, None, None, None
        
    print("largest partition size:", max_partition_size)    
    return complemented_list, len(partitioned_tuple_list), max_partition_size, "full", debug_dict


# =============================================================================
# Efficient complementation using partitioning ends here
# =============================================================================


def EfficientSubsumption(tuple_list, timeout, fdaStartTime):
    #start_time = time.time_ns()
    subsumed_list = []
    m = len(tuple_list[0]) #number of columns
    bucket = dict()
    minimum_null_tuples = dict()
    bucketwise_null_count = dict()
    first_pattern, minimum_nulls = FindCurrentNullPattern(tuple_list[0])
    bucket[first_pattern] = [tuple_list[0]]
    bucketwise_null_count[minimum_nulls] = {first_pattern}
    minimum_null_tuples[minimum_nulls] = [tuple_list[0]]
    for key in tuple_list[1:]:
        current_pattern, current_nulls = FindCurrentNullPattern(key)
        if current_nulls not in bucketwise_null_count:
            bucketwise_null_count[current_nulls] = {current_pattern}
        else:
            bucketwise_null_count[current_nulls].add(current_pattern)
        if current_pattern not in bucket:
            bucket[current_pattern] = [key]
        else:
            bucket[current_pattern].append(key)
        if current_nulls < minimum_nulls:
            minimum_null_tuples[current_nulls] = [key]
            minimum_null_tuples.pop(minimum_nulls)
            minimum_nulls = current_nulls
        elif current_nulls == minimum_nulls:
            minimum_null_tuples[current_nulls].append(key)
        if (time.time() - fdaStartTime) > timeout: return None
        
    #output all tuples with k null values
    subsumed_list = minimum_null_tuples[minimum_nulls]
    for i in range(minimum_nulls+1, m):
        if i in bucketwise_null_count:
            related_buckets = bucketwise_null_count[i]
            parent_buckets = set()
            temp = [v for k,v in bucketwise_null_count.items()
                                    if int(k) < i]
            parent_buckets = set([item for sublist in temp for item in sublist])
            
            for each_bucket in related_buckets:
                #do something
                current_bucket_tuples = bucket[each_bucket]
                if len(current_bucket_tuples) == 0:
                    continue
                non_null_positions = CheckNonNullPositions(each_bucket, m-i)
                parent_bucket_tuples = set()
                for each_parent_bucket in parent_buckets:
                    if CheckAncestor(each_bucket, each_parent_bucket) == 1:
                        list_of_parent_tuples = bucket[each_parent_bucket]
                        for every_tuple in list_of_parent_tuples:
                            projected_parent_tuple = GetProjectedTuple(
                                every_tuple, non_null_positions, m)
                            parent_bucket_tuples.add(projected_parent_tuple)
                new_bucket_item = []     
                for each_tuple in current_bucket_tuples:
                    projected_child_tuple = set()
                    for j in range(0,m):
                        if j in non_null_positions:
                            projected_child_tuple.add(each_tuple[j])
                    projected_child_tuple = GetProjectedTuple(
                                each_tuple, non_null_positions, m)
                    
                    if projected_child_tuple not in parent_bucket_tuples:
                        new_bucket_item.append(each_tuple)
                        subsumed_list.append(each_tuple)
                bucket[each_bucket] = new_bucket_item
        if (time.time() - fdaStartTime) > timeout: return None
    return subsumed_list

def checkAccuracy(old_df, new_df, sourceTable):
    commonCols = [col for col in sourceTable.columns if col in old_df.columns]
    sourceTable = sourceTable[commonCols]
    oldDf = old_df[commonCols]
    newDf = new_df[commonCols]
    unary_operator_applied = True
    originalTDR_r, originalTDR_p = setTDR(sourceTable, oldDf)
    changeTDR_r, changeTDR_p = setTDR(sourceTable, newDf)
    if changeTDR_r < originalTDR_r or changeTDR_p < originalTDR_p:
        return old_df, False
    return new_df, unary_operator_applied

def FDAlgorithm(candidate_tableDfs, source_table, cluster, timeout):
    fdaStartTime = time.time()
    #stats
    print("-----x---------x--------x---")
    print("Processing cluster:", cluster)
    m = len(candidate_tableDfs)
    null_count = 0
    null_set = set()
    table1Name = list(candidate_tableDfs.keys())[0]
    table1 = list(candidate_tableDfs.values())[0]
    table1 = table1.reset_index(drop=True)
    if table1.isnull().sum().sum() > 0:
        table1, null_count, current_null_set = ReplaceNulls(table1, null_count)
        null_set = null_set.union(current_null_set)
    tableOpsApplied = str(table1Name)
    # == BEGIN Outer union
    for tableName, files in candidate_tableDfs.items():
        if tableName == table1Name: continue
        if table1.isnull().sum().sum() > 0:
            table1, null_count, current_null_set = ReplaceNulls(table1, null_count)
            null_set = null_set.union(current_null_set)
            
        table2 = files.reset_index(drop=True)
        if table2.isnull().sum().sum() > 0:
            table2, null_count, current_null_set = ReplaceNulls(table2, null_count)
            null_set = null_set.union(current_null_set)
        table1 = pd.concat([table1,table2])
        
        if (time.time() - fdaStartTime) > timeout: return None, None, None
        print("Outer union done!", tableName)
        tableOpsApplied += ' OUTER UNION '
        #measure time after preprocessing
        start_time = time.time_ns()
        #print(null_set)
        s = table1.shape[0]
        total_cols = table1.shape[1]
        schema = list(table1.columns)
        start_complement_time = time.time_ns()
        complementationResults, complement_partitions, largest_partition_size, partitioning_used, debug_dict = MoreEfficientComplementation(table1, timeout, fdaStartTime)
        if not complementationResults and not complement_partitions and not largest_partition_size and not partitioning_used and not debug_dict: return None, None, None
        end_complement_time = time.time_ns()
        complement_time = int(end_complement_time - start_complement_time)/ 10**9
        print("Finished Complementation in %.2f sec" % (complement_time))
        
        fd_table = pd.DataFrame(complementationResults, columns =schema)

        print("Adding nulls back...")
        if len(null_set) > 0:
            fd_table =  AddNullsBack(fd_table, null_set)
        print("Added nulls back...")
        fd_table, unary_operator_applied = checkAccuracy(table1, fd_table, source_table)
        if unary_operator_applied: tableOpsApplied += ' Complementation '
        
        old_fd_table = fd_table.copy()
        fd_data = {tuple(x) for x in fd_table.values}
        start_subsume_time = time.time_ns()
        subsumptionResults = EfficientSubsumption(list(fd_data), timeout, fdaStartTime)
        if not subsumptionResults: return None, None, None
        end_subsume_time = time.time_ns()
        subsume_time = int(end_subsume_time - start_subsume_time)/ 10**9
        print("Finished Subsumption in %.2f sec" % (subsume_time))
        subsumed_tuples = len(list(fd_data)) - len(subsumptionResults)
        fd_table = pd.DataFrame(subsumptionResults, columns =schema)
        
        fd_table, unary_operator_applied = checkAccuracy(old_fd_table, fd_table, source_table)
        if unary_operator_applied: tableOpsApplied += ', Subsumption '
        tableOpsApplied += str(tableName)
        
        table1 = fd_table
        fd_data = [tuple(x) for x in fd_table.values]
        print("Output tuples: ( total", len(fd_data),")")
        # for t in fd_data:
        #     print(t)
        end_time = time.time_ns()
        total_time = int(end_time - start_time)/10**9
        print("---------------------------------")
        print("Time taken FD algorithm: %.2f sec" % (total_time))
        print("Table Operators Applied: " + tableOpsApplied)  
    
    numValues = table1.shape[0]* table1.shape[1]
    return table1.drop_duplicates(), numValues

def innerUnion(tableDfs, primaryKey, foreignKeys):
    '''
    Directly union tables that have the same schemas
    '''
    innerUnionOp = ''
    print("FIRST perform INNER UNION")    
    unionableTables = {}
    for tableA in tableDfs:
        if tableA in sorted({x for v in unionableTables.values() for x in v}):
            continue
        ASchema = list(tableDfs[tableA].columns)
        for tableB in tableDfs:
            if tableA != tableB:
                BSchema = list(tableDfs[tableB].columns)
                commonCols = set(ASchema).intersection(set(BSchema))
                if commonCols == set(ASchema) and commonCols == set(BSchema):
                    if tableA not in unionableTables:
                        unionableTables[tableA] = [tableB]
                    else:
                        unionableTables[tableA].append(tableB)
    delTables = set()
    
    for tableA in unionableTables:
        null_count = 0
        null_set = set()
        delTables.add(tableA)
        tableADf = tableDfs[tableA]
        union_df, union_name  = pd.DataFrame(), ""
        canUnion = True
        for tableB in unionableTables[tableA]:
            tableBDf = tableDfs[tableB][tableADf.columns]
            delTables.add(tableB)
            # iterate through aligned schemas and union
            union_df = pd.concat([tableADf, tableBDf], ignore_index=True)  
            
            union_name = tableA.split(".csv")[0] + "," + tableB
            tableA, tableADf = union_name, union_df
        tableDfs[union_name] = union_df
    for table in delTables:
        del tableDfs[table]
        
    # maintain order of tables in tableDfs
    orderedTables = [list(tableDfs.keys())[0]]
    for tableIndx, (table, df) in enumerate(tableDfs.items()):
        if tableIndx == (len(tableDfs)-1): 
            if table not in orderedTables: orderedTables.append(table)
            break
        nextTable = list(tableDfs.keys())[tableIndx+1]
        nextDf = list(tableDfs.values())[tableIndx+1]
        commonCols = [col for col in df.columns if col in nextDf.columns]
        if len(commonCols) > 0: orderedTables.append(nextTable)
        else:
            for nextTableIndx, (nextTable, nextDf) in enumerate(tableDfs.items()):
                if nextTableIndx <= (tableIndx+1): continue
                commonCols = [col for col in df.columns if col in nextDf.columns]
                if len(commonCols) > 0: 
                    orderedTables.append(nextTable)
                    # swap positions in tableDfs
                    allTables = list(tableDfs.items())
                    allTables[tableIndx+1], allTables[nextTableIndx] = allTables[nextTableIndx], allTables[tableIndx+1]
                    tableDfs = dict(allTables)
                    break
    orderedTableDfs = {table: tableDfs[table] for table in orderedTables}
    return orderedTableDfs

def compSubsumInnerUnion(table1, source_table, timeout):
    # == Apply Comp and Subsump on FIRST TABLE
    fdaStartTime = time.time()
    tableOpsApplied = ''
    schema = list(table1.columns)
    complementationResults, complement_partitions, largest_partition_size, partitioning_used, debug_dict = MoreEfficientComplementation(table1, timeout, fdaStartTime)
    if not complementationResults and not complement_partitions and not largest_partition_size and not partitioning_used and not debug_dict: return None, None
    fd_table = pd.DataFrame(complementationResults, columns =schema)
            
    fd_table, unary_operator_applied = checkAccuracy(table1, fd_table, source_table)
    if unary_operator_applied: tableOpsApplied += ' Complementation'
 
    old_fd_table = fd_table.copy()
    fd_data = {tuple(x) for x in fd_table.values}
    subsumptionResults = EfficientSubsumption(list(fd_data), timeout, fdaStartTime)
    if not subsumptionResults: return None, None
    subsumed_tuples = len(list(fd_data)) - len(subsumptionResults)
    fd_table = pd.DataFrame(subsumptionResults, columns =schema)
    fd_table, unary_operator_applied = checkAccuracy(old_fd_table, fd_table, source_table)
    if unary_operator_applied: tableOpsApplied += ', Subsumption '
    return fd_table, tableOpsApplied
        
def labelNulls(candidate_tables, source_table, keyCols):
    primaryKey = keyCols[0]
    for table, df in candidate_tables.items():
        dfCols = df.columns
        # add empty rows to DF with repeat keys as dummy rows
        nullRows = []
        if primaryKey in df.columns:
            for keyVal in df[primaryKey]:
                nRow = [keyVal]+[np.nan]*(df.shape[1]-1)
                nullRows.append(nRow)
            commonForeignKeys = [col for col in keyCols[1:] if col in dfCols]
            if commonForeignKeys:
                for keyVal in df[primaryKey]:
                    nRows = df[df[primaryKey]==keyVal].values.tolist()
                    for rIndx, row in enumerate(nRows):
                        for foreignKey in commonForeignKeys:
                            fKeyIndx = list(dfCols).index(foreignKey)
                            row[fKeyIndx] = np.nan
                        nullRows.append(row)
        
        dfRows = df.values.tolist()
        df = pd.DataFrame(dfRows+nullRows, columns=dfCols).drop_duplicates()
        lNullsDf = labelNullsDf(source_table, df, keyCols)
        if lNullsDf is not None: candidate_tables[table] = lNullsDf
    source_table = source_table.replace(np.nan, replaceNull)
    return candidate_tables, source_table
    
def loadCandidateTables(benchmark, sourceTableName):
    # originalBenchmark ='tpch' when benchmark ='tpch_mtTables'
    originalBenchmark = '_'.join(benchmark.split("_")[:-1])
    if '_groundtruth' in benchmark: 
        originalBenchmark = 'tpch'
        if '_mtTables_groundtruth' in benchmark: originalBenchmark = benchmark
    print("originalBenchmark", originalBenchmark)
    FILEPATH = '/home/gfan/Datasets/%s/' % (originalBenchmark)
    if '_mtTables_groundtruth' in benchmark: FILEPATH = '/home/gfan/Datasets/%s/' % (benchmark.split('_mtTables_groundtruth')[0])
    print("FILEPATH", FILEPATH)
    sourceTable = pd.read_csv(FILEPATH+"queries/"+sourceTableName)
    datasets = glob.glob(FILEPATH+"datalake/*.csv")    
    dataLakePath = FILEPATH+"datalake/"
    ssCandidateTableDict = utils.loadDictionaryFromPickleFile("../../results_candidate_tables/%s/%s_candidateTables.pkl" % (originalBenchmark, sourceTableName))
    candidateTableDict = utils.loadDictionaryFromPickleFile("../../results_candidate_tables/%s/%s_candidateTables.pkl" % (benchmark, sourceTableName))
    tableDfs = {}
    
    # for table in datasets:
    for tableName in candidateTableDict:
        table = dataLakePath+tableName
        if tableName == sourceTableName: continue
        table_df = pd.read_csv(table)
        if len(ssCandidateTableDict[tableName]) > 0: print("RENAME COLUMNS: ", tableName, ssCandidateTableDict[tableName])
        table_df = table_df.rename(columns=ssCandidateTableDict[tableName])
        # check types
        for col in table_df.columns:
            if col in sourceTable:
                try: table_df[col] = table_df[col].astype(sourceTable[col].dtypes.name)
                except: 
                    table_df = table_df.dropna()
                    try: table_df[col] = table_df[col].astype(sourceTable[col].dtypes.name)
                    except: table_df = table_df.drop(col, axis=1) # if cannot convert to same type, delete    

        tableDfs[tableName] = table_df
    for table, df in tableDfs.items():
        print(table, sorted(df.columns.tolist()))
    print("Imported %d Candidate Tables: " % (len(list(tableDfs.keys()))), list(tableDfs.keys()))
    
    finalTableDfs = tableDfs
    # ==== PROJECT / SELECT Source Table's Columns / Keys
    primaryKey = sourceTable.columns.tolist()[0]
    foreignKeys = [colName for colName in sourceTable.columns.tolist() if 'key' in colName and colName != primaryKey]
    
    if 't2d_gold' in benchmark:
        # ==== T2D_GOLD Datalake
        # Get another primary key if the first column only has NaN's
        if len([val for val in sourceTable[primaryKey].values if not pd.isna(val)]) == 0:
            for colIndx in range(1, sourceTable.shape[1]):
                currCol = sourceTable.columns.tolist()[colIndx]
                if len([val for val in sourceTable[currCol].values if not pd.isna(val)]) > 1:
                    primaryKey = currCol
                    break
        foreignKeys = []
    projectedTableDfs = projectAtts(tableDfs, sourceTable)
    finalTableDfs = selectKeys(projectedTableDfs, sourceTable, primaryKey, foreignKeys)
    for table, df in finalTableDfs.items():
        df.reset_index(drop=True, inplace=True)
        print(table, df.shape)
    return finalTableDfs, sourceTable, primaryKey, foreignKeys

def main(benchmark, sourceTableName, timeout):
    candidate_tables, sourceTable, primaryKey, foreignKeys = loadCandidateTables(benchmark, sourceTableName)
    sourceCols = sourceTable.columns.tolist()
    print("Source table Cols: ", sourceCols)
    print("Primary Key: ", primaryKey, " Foreign Keys: ", foreignKeys)
    print("%d Candidate Tables: " % (len(list(candidate_tables.keys()))), list(candidate_tables.keys()))
    noCandidates = False
    timed_out = False
    numOutputVals = 0
    if not candidate_tables: 
        noCandidates = True
        return timed_out, noCandidates, numOutputVals
    output_path = r"gen-t_output_tables/"+ benchmark
    
# =============================================================================
    if not os.path.exists(output_path):
      # Create a new directory because it does not exist 
      os.makedirs(output_path)
      print("output directory is created!")
# =============================================================================
    candidate_tables = innerUnion(candidate_tables, primaryKey, foreignKeys)
    candidate_tables, sourceTable = labelNulls(candidate_tables, sourceTable, [primaryKey]+foreignKeys)
    
    print("%d Inner Unioned Candidate Tables: " % (len(list(candidate_tables.keys()))), list(candidate_tables.keys()))
    for table, df in candidate_tables.items():
        fd_table, tableOps = compSubsumInnerUnion(df, sourceTable, timeout)
        candidate_tables[table] = fd_table
        print("Applied Unary: ", table, tableOps)
        
    result_FD = list(candidate_tables.values())[0]
    if result_FD is None: 
        noCandidates = True
        return timed_out, noCandidates, numOutputVals
    numOutputVals = result_FD.shape[0]* result_FD.shape[1]
    if len(candidate_tables) > 1:
        cluster_name = sourceTableName
        result_FD, numOutputVals = FDAlgorithm(candidate_tables, sourceTable, cluster_name, timeout)
    
    if result_FD is None: 
        noCandidates = True
        timed_out = True
    #save result to hard drive
    commonCols = [col for col in sourceTable.columns if col in result_FD.columns]
    if not commonCols: 
        noCandidates = True
        return timed_out, noCandidates, numOutputVals
    result_FD = result_FD[commonCols]
    
    sourceTable = sourceTable.replace(replaceNull, np.nan) 
    result_FD = result_FD.replace(replaceNull, np.nan)  
    result_FD = result_FD.dropna(axis=0, subset=[primaryKey])
    if result_FD.empty: 
        noCandidates = True
        return timed_out, noCandidates, numOutputVals
    
    if not timed_out: result_FD.to_csv(output_path+sourceTableName,index = False)
    return timed_out, noCandidates, numOutputVals
