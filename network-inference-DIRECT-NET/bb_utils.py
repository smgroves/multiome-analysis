import pandas as pd
import os
import resource
import numpy as np
from graph_tool import all as gt
import booleabayes as bb

def log_job(dir_prefix, brcd, random_state, network_path, data_path, data_t1_path, cellID_table, node_normalization,
            node_threshold, split_train_test, write_binarized_data,fit_rules,validation,validation_averages,
            find_average_states,find_attractors,tf_basin,filter_attractors,on_nodes,off_nodes, time = None,
            linux = False, memory = False):
    T = {}
    if memory:
        if linux:
            T['memory_Mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
        else:
            T['memory_Mb'] = np.nan
    T['barcode'] = brcd
    T['random_state'] = random_state
    T['dir_prefix'] = dir_prefix
    T['network_path'] = network_path
    T['data_path'] = data_path
    T['data_t1_path'] = data_t1_path
    T['cellID_table'] = cellID_table
    T['node_normalization'] = node_normalization
    T['node_threshold'] = node_threshold
    T['split_train_test'] = split_train_test
    T['write_binarized_data'] = write_binarized_data
    T['fit_rules'] = fit_rules
    T['validation'] = validation
    T['validation_averages'] = validation_averages
    T['find_average_states'] = find_average_states
    T['find_attractors'] = find_attractors
    T['tf_basin'] = tf_basin # if -1, use average distance between clusters. otherwise use the same size basin for all phenotypes
    T['filter_attractors'] = filter_attractors
    T['on_nodes'] = on_nodes
    T['off_nodes'] = off_nodes
    T['total_time'] = time

    T = pd.DataFrame([T])
    if not os.path.isfile(dir_prefix + 'Job_specs.csv'):
        T.to_csv(dir_prefix + 'Job_specs.csv')
    else:
        with open(dir_prefix + 'Job_specs.csv', 'a') as f:
            T.to_csv(f, header=False)


def print_graph_info(graph, nodes, fname, brcd = "", dir_prefix = "", plot = True):
    print("==================================")
    print("Graph properties")
    print("==================================")
    print(graph)
    # print("Edge and vertex properties: ", graph.list_properties())
    print("Number of nodes:", len(nodes))
    print('Nodes: ', nodes)
    sources = []
    sinks = []
    for i in range(len(nodes)):
        if graph.vp.source[i] == 1: sources.append(graph.vp.name[i])
        if graph.vp.sink[i] == 1: sinks.append(graph.vp.name[i])
    print("Sources: ", sources)
    print("Sinks: ", sinks)

    #treat network as if it is undirected to ensure largest component includes all nodes and edges
    u = gt.extract_largest_component(graph, directed=False)
    print("Network is a single connected component: ", gt.isomorphism(graph,u))
    if gt.isomorphism(graph,u) == False:
        print("\t Largest component of network: ")
        print("\t", u)
    print("Directed acyclic graph: ", gt.is_DAG(graph))
    print("==================================")

    if plot:
        pos = gt.sfdp_layout(graph, mu = 1,  max_iter=1000)
        gt.graph_draw(graph, pos=pos, output=f"{dir_prefix}/{brcd}/{fname}_simple_network.pdf", output_size=(1000, 1000))


def draw_grn(G, gene2vertex, rules, regulators_dict, fname, gene2group=None, gene2color=None, type = "", B_min = 5,
             save_edge_weights = True, edge_weights_fname = "edge_weights.csv"):
    vertex2gene = G.vertex_properties['name']

    vertex_group = None
    if gene2group is not None:
        vertex_group = G.new_vertex_property("int")
        for gene in gene2group.keys():
            vertex_group[gene2vertex[gene]] = gene2group[gene]

    vertex_colors = [0.4, 0.2, 0.4, 1]
    if gene2color is not None:
        vertex_colors = G.new_vertex_property("vector<float>")
        for gene in gene2color.keys():
            vertex_colors[gene2vertex[gene]] = gene2color[gene]

    edge_weight_df = pd.DataFrame(index=sorted(regulators_dict.keys()), columns=sorted(regulators_dict.keys()))
    edge_binary_df = pd.DataFrame(index=sorted(regulators_dict.keys()), columns=sorted(regulators_dict.keys()))

    edge_markers = G.new_edge_property("string")
    edge_weights = G.new_edge_property("float")
    edge_colors = G.new_edge_property("vector<float>")
    for edge in G.edges():
        edge_colors[edge] = [0., 0., 0., 0.3]
        edge_markers[edge] = "arrow"
        edge_weights[edge] = 0.2

    for edge in G.edges():
        vs, vt = edge.source(), edge.target()
        source = vertex2gene[vs]
        target = vertex2gene[vt]
        regulators = regulators_dict[target]
        if source in regulators:
            i = regulators.index(source)
            n = 2 ** len(regulators)

            rule = rules[target]
            off_leaves, on_leaves = bb.tl.get_leaves_of_regulator(n, i)
            if rule[off_leaves].mean() < rule[on_leaves].mean():  # The regulator is an activator
                edge_colors[edge] = [0., 0.3, 0., 0.8]
                edge_binary_df.loc[target,source] = 1
            else:
                edge_markers[edge] = "bar"
                edge_colors[edge] = [0.88, 0., 0., 0.5]
                edge_binary_df.loc[target,source] = -1

            edge_weights[edge] = rule[on_leaves].mean() - rule[off_leaves].mean() + 0.2
            edge_weight_df.loc[target, source] = rule[on_leaves].mean() - rule[off_leaves].mean()
    G.edge_properties["edge_weights"] = edge_weights
    if save_edge_weights:
        edge_weight_df.to_csv(edge_weights_fname)
    pos = gt.sfdp_layout(G, groups=vertex_group,mu = 1, eweight=edge_weights, max_iter=1000)
    # pos = gt.arf_layout(G, max_iter=100, dt=1e-4)
    eprops = {"color": edge_colors, "pen_width": 2, "marker_size": 15, "end_marker": edge_markers}
    vprops = {"text": vertex2gene, "shape": "circle", "size": 20, "pen_width": 1, 'fill_color': vertex_colors}
    if type == 'circle':
        state = gt.minimize_nested_blockmodel_dl(G, B_min = B_min)
        state.draw(vprops=vprops, eprops=eprops)  # mplfig=ax[0,1])
    else:
        gt.graph_draw(G, pos=pos, output=fname, vprops=vprops, eprops=eprops, output_size=(1000, 1000))
    return G, edge_weight_df, edge_binary_df

