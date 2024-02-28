from archived.modelstore import StoreHandler
from knowledgerepr import fieldnetwork
from knowledgerepr import networkbuilder
from knowledgerepr.fieldnetwork import FieldNetwork
# from inputoutput import inputoutput as io

import pickle
import sys
import time
import csv


def serialize_object(obj, path):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()


def deserialize_object(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    return obj


def main(output_path=None, table_path=None):
    start_all = time.time()
    network = FieldNetwork()
    store = StoreHandler()
    # log = open('log_opendata_large_05.txt', 'w', buffering=1)
    # network_file = open('network_05_opendata_large.csv', 'w', encoding='UTF8', buffering=1)
    log = open('log_chicago_05.txt', 'w', buffering=1)
    network_file = open('network_05_chicago.csv', 'w', encoding='UTF8', buffering=1)
    csv_writer = csv.writer(network_file)
    csv_writer.writerow(['tbl1', 'col1', 'tbl2', 'col2'])

    start_fields_gen = time.time()
    # Get all fields from store
    fields_gen = store.get_all_fields()
    end_fields_gen = time.time()
    log.write("time to gen fields: {}\n".format(end_fields_gen-start_fields_gen))

    # Network skeleton and hierarchical relations (table - field), etc
    start_schema = time.time()
    col_cnt = network.init_meta_schema(fields_gen)
    print("number of columns:", col_cnt)
    log.write("number of columns: {}\n".format(col_cnt))
    end_schema = time.time()
    print("Total skeleton: {0}".format(str(end_schema - start_schema)))
    print("!!1 " + str(end_schema - start_schema))
    log.write("time to get schema: " + str(end_schema - start_schema) + '\n')

    print("begin to extract minhash signature")
    start_text_sig_sim = time.time()
    mh_signatures = store.get_all_mh_text_signatures()
    end_text_sig_sim = time.time()
    print("!!3 " + str(end_text_sig_sim - start_text_sig_sim))
    log.write("time to extract minhash signatures: " + str(end_text_sig_sim - start_text_sig_sim) + '\n')
    
    print("Begin to build graph")
    lsh_threshold = 0.5
    start_build_graph = time.time()
    content_sim_index, edges_cnt = networkbuilder.build_content_sim_mh_text(network, mh_signatures, lsh_threshold, log, csv_writer)
    end_build_graph = time.time()
    print("Total text-sig-sim (minhash): {0}; edges count: {1};".format(str(end_text_sig_sim - start_text_sig_sim), str(edges_cnt)))
    print("!!4 " + str(end_build_graph - start_build_graph))
    log.write("time to build graph: " + str(end_build_graph - start_build_graph) + '\n')
    log.write("edge count: {}\n".format(edges_cnt))

    # Content_sim num relation
    # start_num_sig_sim = time.time()
    # id_sig = store.get_all_fields_num_signatures()
    # #networkbuilder.build_content_sim_relation_num(network, id_sig)
    # networkbuilder.build_content_sim_relation_num_overlap_distr(network, id_sig, table_path)
    # #networkbuilder.build_content_sim_relation_num_overlap_distr_indexed(network, id_sig)
    # end_num_sig_sim = time.time()
    # print("Total num-sig-sim: {0}".format(str(end_num_sig_sim - start_num_sig_sim)))
    # print("!!5 " + str(end_num_sig_sim - start_num_sig_sim))

    # Primary Key / Foreign key relation
    # start_pkfk = time.time()
    # networkbuilder.build_pkfk_relation(network)
    # end_pkfk = time.time()
    # print("Total PKFK: {0}".format(str(end_pkfk - start_pkfk)))
    # print("!!6 " + str(end_pkfk - start_pkfk))

    # end_all = time.time()
    # print("Total time: {0}".format(str(end_all - start_all)))
    # print("!!7 " + str(end_all - start_all))

    path = "test/datagov/"
    if output_path is not None:
        path = output_path
    fieldnetwork.serialize_network(network, path)

    # Serialize indexes
    # path_schsim = path + "/schema_sim_index.pkl"
    # io.serialize_object(schema_sim_index, path_schsim)
    path_cntsim = path + "/content_sim_index.pkl"
    serialize_object(content_sim_index, path_cntsim)

    print("DONE!")


def plot_num():
    network = FieldNetwork()
    store = StoreHandler()
    fields, num_signatures = store.get_all_fields_num_signatures()

    xaxis = []
    yaxis = []
    numpoints = 0
    for x, y in num_signatures:
        numpoints = numpoints + 1
        xaxis.append(x)
        yaxis.append(y)
    print("Num points: " + str(numpoints))
    import matplotlib.pyplot as plt
    plt.plot(xaxis, yaxis, 'ro')
    plt.axis([0, 600000, 0, 600000])
    #plt.axis([0, 10000, 0, 10000])
    #plt.axis([0, 500, 0, 500])
    plt.show()


def test_content_sim_num():

    '''
    SETUP
    '''

    start_all = time.time()
    network = FieldNetwork()
    store = StoreHandler()

    # Get all fields from store
    fields_gen = store.get_all_fields()

    # Network skeleton and hierarchical relations (table - field), etc
    start_schema = time.time()
    network.init_meta_schema(fields_gen)
    end_schema = time.time()
    print("Total skeleton: {0}".format(str(end_schema - start_schema)))

    '''
    ACTUAL TEST
    '''

    # Content_sim num relation
    start_num_sig_sim = time.time()
    id_sig = store.get_all_fields_num_signatures()
    # networkbuilder.build_content_sim_relation_num(network, id_sig)
    networkbuilder.build_content_sim_relation_num_overlap_distr(network, id_sig)
    end_num_sig_sim = time.time()
    print("Total num-sig-sim: {0}".format(str(end_num_sig_sim - start_num_sig_sim)))


if __name__ == "__main__":

    #test_content_sim_num()
    #exit()

    path = None
    if len(sys.argv) == 5:
        path = sys.argv[2]
        table_path = sys.argv[4]

    else:
        print("USAGE: ")
        print("python networkbuildercoordinator.py --opath <model_path> --tpath <table_path>")
        print("where opath must be writable by the process")
        exit()
    main(path, table_path)

    #test_read_store()

    #test()
    #plot_num()
    #test_cardinality_propagation()