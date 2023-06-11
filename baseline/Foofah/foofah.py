from timeit import default_timer as timer
import json
import queue
import argparse
import os
import csv
from tabulate import tabulate
from Foofah.foofah_libs.foofah_node import FoofahNode
# from .foofah_libs.foofah_node import FoofahNode
from Foofah.foofah_libs.operators import *
# from . import foofah_libs.operators as Operations
import numpy as np
from Foofah.foofah_libs.generate_prog import create_python_prog
# from .foofah_libs.generate_prog import create_python_prog
import contextlib
import math
import ast

MAX_STEPS = float("inf")

ALGO_BFS = 0
ALGO_A_STAR = 1
ALGO_A_STAR_NAIVE = 2
ALGO_AWA = 3


def reconstruct_path(current):
    if current is None:
        total_path = []
    else:
        total_path = [current]
        while current.parent is not None:
            current = current.parent
            total_path.append(current)
    return total_path

def pick_root_node(raw_data, goal_node, batch, epsilon):
    print("Choosing Root from %d Possible Nodes" % (len(raw_data)))
    min_score = math.inf
    root_node = None
    goalCols = goal_node.contents[0]
    print("Goal Node has %d columns" % (len(goalCols)), goalCols)
    maxColOverlap = 0
    for i, table in enumerate(raw_data):
        table_op = ({'fxn': None, 'name': 'start', 'char': 'start', 'cost': 1.0}, 0)
        t = FoofahNode(table, table_op, None, {})
        t.h_score = t.get_h_score_modified()
        print(i, "FINDING Root Node %d has %f values overlap: " % (t.node_id, t.h_score))
        if t.h_score < min_score:
            min_score = t.h_score
            root_node = t
        # tableCols = ast.literal_eval(table[0])   
        # print(tableCols)
        # colOverlap = [col for col in tableCols if col in goalCols]
        # if len(colOverlap) >= maxColOverlap:
        #     maxColOverlap = len(colOverlap)
        #     table_op = ({'fxn': None, 'name': 'start', 'char': 'start', 'cost': 1.0}, 0)
        #     t = FoofahNode(table, table_op, None, {})
        #     t.h_score = t.get_h_score(batch=batch)
        #     print("FINDING Root Node %d has %d columns overlap: " % (t.node_id, len(colOverlap)), t.h_score)
        #     if t.h_score < min_score:
        #         min_score = t.h_score
        #         root_node = t
                
            
    print("Root Node: ", root_node.node_id, root_node.contents[0])
    return root_node

def a_star_search(raw_data, target, ops, debug=0, timeout=300, algo=ALGO_A_STAR, batch=True,
                  epsilon=1, bound=float("inf"), p1=True, p2=True, p3=True):
    start_time = timer()
    
    FoofahNode.target = target

    # root_op = ({'fxn': None, 'name': 'start', 'char': 'start', 'cost': 1.0}, 0)
    # root = FoofahNode(raw_data, root_op, None, {})
    goal_op = ({'fxn': None, 'name': 'end', 'char': 'end', 'cost': 0.0}, 0)
    goal_node = FoofahNode(target, goal_op, None, {})

    FoofahNode.goal_node = goal_node
    
    root = pick_root_node(raw_data, goal_node, batch, epsilon)
    print("ROOT H score", root.h_score)
    root.g_score = 0.0

    # if algo == ALGO_BFS:
    #     root.h_score = 0
    # elif algo == ALGO_A_STAR:
    #     root.h_score = root.get_h_score(batch=batch)
    # elif algo == ALGO_A_STAR_NAIVE:
    #     root.h_score = root.get_h_score_rule()

    root.f_score = root.g_score + epsilon * root.h_score

    # Switch to using priority queue because it is thread safe
    open_q = queue.PriorityQueue()
    open_q_cache = None
    cost_q = {}
    closed_nodes = set()
    final_node = None
    reached_success = 0
    num_edges = 0
    timedOut = False

    # open_q.put(root)
    for i, table in enumerate(raw_data):
        table_op = ({'fxn': None, 'name': str(i), 'char': str(i), 'cost': 1.0}, 0)
        table_node = FoofahNode(table, table_op, None, {})
        if table_node.contents == root.contents: continue
        print("PUTTING NODE %d IN QUEUE" % (table_node.node_id))
        open_q.put(table_node)
    print("Open q Size in the Beginning: ", open_q.qsize())
    if open_q.empty():
        final_node = root
        return final_node, open_q, closed_nodes, reached_success, timedOut
    node = root
    while not open_q.empty():
        print("=============== Edge %d ============== " % (num_edges))
        if num_edges > 0: node = open_q.get(block=True)
        print("NEW ITERATION: node id %d, open_q has size %d" % (node.node_id, open_q.qsize()), sorted([node.node_id for node in open_q.queue]))
        cur_time = timer()

        if cur_time - start_time > timeout:
            print(("*** Exceeded time limit of %d seconds" % timeout))
            timedOut = True
            return final_node, open_q, closed_nodes, reached_success, timedOut

        if debug >= 1:
            if node.parent:
                print(("f_score:", node.f_score, "h_score:", node.h_score, "g_score:", node.g_score, "id:", node.node_id, "p_id:", node.parent.node_id, "depth:", node.depth, node, node.contents))
                print()
            else:
                print(("f_score:", node.f_score, "h_score:", node.h_score, "g_score:", node.g_score, "id:", node.node_id, "p_id:", "None", "depth:", node.depth, node, node.contents))
                print()

        closed_nodes.add(node)

        if node == goal_node:
            reached_success = 1
            final_node = node
            break
        my_children, tablesOpedOn = node.make_children(open_q, ops, bound=bound, p1=p1, p2=p2, p3=p3)
        # remove tables joined / unioned with
        for table in tablesOpedOn:
            open_q.queue.remove(table)
        
        min_child_score = math.inf
        best_child = None
        print("NEW PARENT: ", node, "has %d children, now open_q has size %d" % (len(my_children), open_q.qsize()))
        for c in my_children:
            if c in closed_nodes:
                continue
            
            # elif algo == ALGO_A_STAR:
            # c.h_score = c.get_h_score(batch=batch)
            c.h_score = c.get_h_score_modified()
            
            c.g_score = node.g_score + node.operation[0]['cost']
            # Check if destination has been found, if it is, return.
            if c.h_score == 0:
                if c == goal_node:
                    reached_success = 1
                    final_node = c
                    print("PUTTING NODE %d IN QUEUE (in traversal - reached goal)" % (c.node_id))
                    open_q.put(c)
                    cost_q[c] = c.f_score
                    if debug >= 2:
                        if c.parent:
                            print(("***", "f_score:", c.f_score, "h_score:", c.h_score, "g_score:", c.g_score, "id:", c.node_id, "p_id:", c.parent.node_id, "depth:", c.depth, c, c.contents))
                        else:
                            print(("***", "f_score:", c.f_score, "h_score:", c.h_score, "g_score:", c.g_score, "id:", c.node_id, "p_id:", "None", "depth:", c.depth, c, c.contents))

                    return final_node, open_q, closed_nodes, reached_success, timedOut

            c.f_score = c.g_score + epsilon * c.h_score
            print("CHILD %d F Score: " % (c.node_id),c.f_score)
            if c.f_score < min_child_score:
                min_child_score = c.f_score
                best_child = c
            if (c not in cost_q or (c in cost_q and c.f_score < cost_q[c])) and c.f_score < float("inf"):
                print("PUTTING NODE %d IN QUEUE (in traversal)" % (c.node_id), c)
                open_q.put(c)
                cost_q[c] = c.f_score

                if debug >= 2:
                    if c.parent:
                        print(("***", "f_score:", c.f_score, "h_score:", c.h_score, "g_score:", c.g_score, "id:", c.node_id, "p_id:", c.parent.node_id, "depth:", c.depth, c, c.contents))
                    else:
                        print(("***", "f_score:", c.f_score, "h_score:", c.h_score, "g_score:", c.g_score, "id:", c.node_id, "p_id:", "None", "depth:", c.depth, c, c.contents))
                        
        num_edges += 1
        if len(closed_nodes) == len(my_children):
            final_node = node
            return final_node, open_q, closed_nodes, reached_success, timedOut
        if open_q.empty():
            print("EMPTY: ", best_child)
            final_node = best_child
            if not my_children:
                final_node = node
            

    if open_q_cache:

        while open_q.qsize() > 0:
            open_q_cache.put(open_q.get())
        return final_node, open_q_cache, closed_nodes, reached_success, timedOut
    else:
        return final_node, open_q, closed_nodes, reached_success, timedOut


def extract_table(raw_data):
    if len(raw_data) == 1 and len(raw_data[0]) == 1:
        input_str = raw_data[0][0]

        rows = input_str.splitlines()

        delimiter_list = ["\t", ",", " "]
        quotechar_list = ["'", '"']

        for delimiter in delimiter_list:
            for quote_char in quotechar_list:
                temp_table = list(csv.reader(rows, delimiter=delimiter, quotechar=quote_char))
                row_len = set()
                for row in temp_table:
                    row_len.add(len(row))

                if len(row_len) == 1:
                    return temp_table

        return raw_data

    else:
        return raw_data


def main(benchmark, sourceTableName, ext_input = None, ext_time_limit = 60, what_to_explain = 'tables', projSel=0, target_type = 'TEXTUAL',
         ext_if_validate = True):
    final_node = None
    open_nodes = None
    closed_nodes = None
    FoofahNode.if_awa = False
    timedOut = False
    numOutputVals = 0

    #
    # Command Line Arguments
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('--details', action='store_true', default=False,
                        help="Print the detailed synthesized programs and intermediate tables")

    parser.add_argument('--input', type=str, nargs='+',
                        help="List of input test data files separated by spaces")
    parser.add_argument('--debug_level', type=int, default=0,
                        help="Debug level. 0 = none, 1 = simple, etc.")
    parser.add_argument('--timeout', type=int, default=300,
                        help="Search will stop after this many seconds.")

    parser.add_argument('--auto_read', action='store_true', help="Automatically read csv file using csv reader")

    parser.add_argument('--validate', action='store_true', default=False,
                        help="Validating the correctness of synthesized program")

    parser.add_argument('--search_algo', type=int, default=1,
                        help="Searh algorithm: 0 = BFS, 1 (default) = A*, 2 = naive heuristic")

    parser.add_argument('--no_batch', action='store_true', default=False, help="Disable batch")

    parser.add_argument('--weight', type=float, default=1, help="Weighted A*")

    parser.add_argument('--bound', type=float, default=float("inf"), help="Depth bound")

    parser.add_argument('--p1off', action='store_true', default=False, help="turn off prune rule 1")
    parser.add_argument('--p2off', action='store_true', default=False, help="turn off prune rule 2")
    parser.add_argument('--p3off', action='store_true', default=False, help="turn off prune rule 3")

    parser.add_argument('--globalPruneOff', action='store_true', default=False, help="turn off global pruning rules")
    parser.add_argument('--opPruneOff', action='store_true', default=False, help="turn off operator pruning rules")

    parser.add_argument('--wrap1off', action='store_true', default=False, help="turn off 1st wrap operator")
    parser.add_argument('--wrap2off', action='store_true', default=False, help="turn off 2nd wrap operator")
    parser.add_argument('--wrap3off', action='store_true', default=False, help="turn off 3rd wrap operator")

    #
    # Read Command Line Arguments
    #

    args = parser.parse_args()

    if_detail = args.details
    # input_files = args.input
    if ext_input:
        input_files = [ext_input,]
    debug_level = args.debug_level
    timeout = args.timeout

    if_auto_read = False
    if args.auto_read:
        if_auto_read = True

    if_validate = args.validate
    if_validate = ext_if_validate

    search_algo = args.search_algo

    if_batch = not args.no_batch

    epsilon = args.weight

    bound = args.bound

    p1off = args.p1off
    p2off = args.p2off
    p3off = args.p3off

    op_prune_off = args.opPruneOff

    wrap1off = args.wrap1off
    wrap2off = args.wrap2off
    wrap3off = args.wrap3off

    if op_prune_off:
        Operations.PRUNE_1 = False

    if wrap1off:
        Operations.WRAP_1 = False
    
    if wrap2off:
        Operations.WRAP_2 = False
    
    if wrap3off:
        Operations.WRAP_3 = False


    global_prune_off = args.globalPruneOff

    if global_prune_off:
        p1off = True
        p2off = True
        p3off = True

    if input_files is None or len(input_files) == 0:
        print("*** No test input file specified. ***")
        exit()

    test_files = input_files
    for test_file in test_files:
        with open(test_file, 'rb') as f:
            test_data = json.load(f)

        # raw_datas = [list(map(str, xi)) for x in test_data['InputTable'] for xi in x]
        raw_data = []
        for table in test_data['InputTable']:
            raw_data.append(list(map(str, table)))
        target = [list(map(str, x)) for x in test_data['OutputTable']]
        # print(target)

        if if_auto_read:
            for table in raw_data:
                table = extract_table(table)
                print("Raw Data Table: ", table)
        if not raw_data:
            return None, timedOut, numOutputVals
        start = timer()
        # a = lambda x, p1, s: f_split_first(x, p1, s)
        # print(a(raw_data, 0, ' '))
        # exit()
        search_space = None
        # print(what_to_explain)
        if what_to_explain == 'columns':
            if target_type == 'TEXTUAL':
                search_space = add_ops_column()
            elif target_type == 'BINARY_CLASSIFICATION':
                search_space = add_ops_column_text_to_class()
            # elif target_type == 'MULTICLASS_CLASSIFICATION':
            #     search_space = add_ops_column_text_to_class()
            else:
                search_space = add_ops_column_text_to_numeric()
        elif what_to_explain == 'row':
            search_space = add_ops_row()
        elif what_to_explain == 'full':
            search_space = add_ops_full()
        elif what_to_explain == 'foofah':
            search_space = add_ops()
        elif what_to_explain == 'auto-pipeline':
            search_space = add_ops_auto_pipeline()
        elif what_to_explain == 'foofah_plus':
            search_space = add_ops_plus()
        elif what_to_explain == 'tables':
            search_space = add_ops_tables()
        else:
            # print('here')
            search_space = add_ops()
        final_node, open_nodes, closed_nodes, reached_success, timedOut = a_star_search(raw_data, target, search_space, debug_level,
                                                             timeout=ext_time_limit, batch=if_batch, epsilon=epsilon,
                                                             bound=bound, algo=search_algo, p1=not p1off,
                                                             p2=not p2off,
                                                             p3=not p3off)

        end = timer()
        # if_detail = ext_if_detail
        # if final_node:
        print("Final Node: ", final_node)
        print("Open nodes: ", open_nodes.qsize())
        print("Closed nodes: ", len(closed_nodes))
        if not final_node: return None, timedOut, numOutputVals
        # SAVING Final node 
        
        output_path = "output_tables/"+ benchmark+"_3/"
        if projSel: output_path = "output_tables_projSel/"+ benchmark+"_3/"
        print("Saving to ", output_path)
        rows_to_save = []
        for row in final_node.contents:
            rows_to_save.append(ast.literal_eval(row))
        df_to_save = pd.DataFrame(rows_to_save[1:], columns=rows_to_save[0])
        df_to_save.to_csv(output_path+sourceTableName, index=False)
        numOutputVals = df_to_save.shape[0]*df_to_save.shape[1]

        path = reconstruct_path(final_node)
        # Some statistics
        num_visited = len(closed_nodes)
        nodes_created = open_nodes.qsize() + len(closed_nodes)
        poly = np.ones(len(path) + 1)
        poly[len(path)] = -nodes_created
        branch_factor = max(np.real(np.roots(poly)))
        create_python_prog(path, raw_data, True)

        # if not if_detail:
        #     return
            # program = create_python_prog(path, raw_data, True)
            #
            # print("# A Program Has Been Successfully Synthesized")
            # print("#")
            # print(("# Input file:", test_file))
            # print(("# Total operations:", len(path) - 1))
            # print(("# Time elapsed:    %.3f s   Nodes visited:  %d   Nodes created: %d" % (
            #     (end - start), num_visited, nodes_created)))
            # print(("# Naive branching factor: %d   Effective branching factor: %.2f" % (
            # len(add_ops()), branch_factor)))
            # print(("# Make child time: %.2f s   Heuristic time: %.2f s" % (
            #     sum(final_node.times['children']), sum(final_node.times['scores']))))
            # print(("#", "-" * 50))
            # print()
            # print(program)

        # else:
        with open("logs/%s/foofah_log_%s" % (benchmark+"_3", sourceTableName), "w") as o:
            with contextlib.redirect_stdout(o):
                print(("-" * 50))
                train_data = []

                for i, n in enumerate(reversed(path)):
                    listContents = []
                    for row in n.contents:
                        listContents.append(ast.literal_eval(row))
                    n.contents = listContents
                    print("n.contents", n.contents)
                    # Operations including transpose, unfold and unfold_header do not have parameters
                    if len(n.operation) > 1:
                        if n.operation[1]:
                            print(("%2d. %-13s at %d: H-score: %.1f Actual: %d" % (
                                i + 1, n.operation[0]['name'], n.operation[1], n.h_score, len(path) - i - 1)))
                        else:
                            print(("%2d. %-13s      : H-score: %.1f Actual: %d" % (
                                i + 1, n.operation[0]['name'], n.h_score, len(path) - i - 1)))
                        print((tabulate(n.contents, tablefmt="grid")))
                    else:
                        print(("%2d. %-13s: H-score: %.1f Actual: %d" % (
                            i + 1, n.operation[0]['name'], n.h_score, len(path) - i - 1)))
                        print((tabulate(n.contents, tablefmt="grid")))
                    remaining_steps = len(path) - i - 1
                    if remaining_steps > 0:
                        temp = dict()
                        temp["raw_table"] = n.contents
                        temp["target_table"] = target
                        temp["steps"] = remaining_steps
                        train_data.append(temp)

                if final_node.contents != target:
                    print()
                    print("%2d. Only \"Moves\" are needed to create a extact same view as target (TO BE COMPLETED)." % (
                        len(path) + 1))
                    print()

                print("-" * 50)
                print("Input file:", test_file)
                print("Total operations:", len(path) - 1)
                print("Time elapsed:    %.3f s   Nodes visited:  %d   Nodes created: %d" % (
                    (end - start), num_visited, nodes_created))
                print("Naive branching factor: %d   Effective branching factor: %.2f" % (len(add_ops()), branch_factor))
                print("Make child time: %.2f s   Heuristic time: %.2f s" % (
                    sum(final_node.times['children']), sum(final_node.times['scores'])))

                if if_validate:
                    test_table = test_data['TestingTable']
                    try:
                        for i, node in enumerate(reversed(path)):

                            if i > 0:
                                op = node.operation[0]
                                if op['num_params'] == 1:
                                    test_table = op['fxn'](test_table)
                                else:
                                    test_table = op['fxn'](test_table, node.operation[1])
                    except:
                        test_table = None
                    if test_table:
                        test_data["TransformedTestTable"] = test_table
                        test_data["Success"] = True
                        print("-" * 50)
                        print("Experiment 1: Apply the synthetic program on other data")
                        print("-" * 30)
                        print("Testing Table")
                        print(tabulate(test_data['TestingTable'], tablefmt="grid"))

                        print("Transformed Table")
                        print(tabulate(test_data["TransformedTestTable"], tablefmt="grid"))
                        print("-" * 30)
                        if reached_success: print("Result: Success")
                        else: print("Result: Failure")
                        print("-" * 50)

                    else:
                        test_data["TransformedTestTable"] = test_table
                        test_data["Success"] = False
                        print("-" * 50)
                        print("Experiment 1: Apply the synthetic program on other data")
                        print("-" * 30)
                        print("Testing Table")
                        print(tabulate(test_data['TestingTable'], tablefmt="grid"))
                        print("-" * 30)
                        if reached_success: print("Result: Success")
                        else: print("Result: Failure")
                        print("-" * 50)

                    dirname = os.getcwd() + "/test_results/validate"
                    filename = dirname + "/exp0_results_" + str(test_data['TestName']) + "_" + str(
                        test_data['NumSamples']) + ".txt"
                    if not os.path.exists(dirname):
                        try:
                            os.makedirs(dirname)
                        except OSError:
                            raise

                    with open(filename, 'w') as outfile:
                        json.dump(test_data, outfile)

        # else:
        #     # print("*** Solution Not Found ***")
        #     return False
    return True, timedOut, numOutputVals


if __name__ == "__main__":
    main()
