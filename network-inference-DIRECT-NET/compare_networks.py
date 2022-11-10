
from bb_utils import *


remove_sinks=False
remove_selfloops=False
remove_sources=False
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'

def run_network(network_path, dir_prefix, old_nodes = None, fillcolor = 'lightcyan', layout = None, plot = True,special_nodes = [],
                add_edge_weights = True):
    network_df = pd.read_csv(f'{dir_prefix}/{network_path}', header = None, index_col=None)
    network_df.columns = ['source','target','score','evidence']

    graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=remove_sinks, remove_selfloops=remove_selfloops,
                                              remove_sources=remove_sources)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    fname = network_path.split('/')[-1].split('.')[0]
    print(network_path)
    if old_nodes is None:
        print_graph_info(graph, vertex_dict, nodes,  fname, brcd = 'networks/network_plots', dir_prefix = dir_prefix,plot = plot, gene2color=None,
                         fillcolor = "lightcyan", layout = layout, add_edge_weights=add_edge_weights, ew_df=network_df)
    else:
        gene2color = {}
        for node in nodes:
            if node not in old_nodes:
                gene2color[node] = "green"
            elif node in special_nodes:
                gene2color[node] = "orange"
            else:
                gene2color[node] = fillcolor
        print_graph_info(graph, vertex_dict, nodes,  fname, brcd = 'networks/network_plots', dir_prefix = dir_prefix,plot = plot,
                         gene2color=gene2color, layout = layout, add_edge_weights=add_edge_weights, ew_df=network_df
                         )
    return graph, nodes

########################################
# DIRECT NET Network: no pruning
########################################
network_path = 'networks/DIRECT-NET_network_threshold_0.csv'
old_graph, old_nodes = run_network(network_path, dir_prefix, plot = False)
# run_network(network_path, dir_prefix, layout = 'circle')


########################################
# DIRECT NET Network with RORB
########################################

## use gene2color dictionary {gene:color} to color differences in network nodes (added or removed)
#
network_path = 'networks/DIRECT-NET_network_threshold_0_withRORB.csv'
old_graph, old_nodes = run_network(network_path, dir_prefix, old_nodes = old_nodes, special_nodes=['RORB'], plot = False)

########################################
# DIRECT NET Network with FIGR and RORB
########################################

## Highlighting NEUROG2 because it was removed in the next step due to lack of expression data

network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0.csv'
old_graph, old_nodes = run_network(network_path, dir_prefix, old_nodes = old_nodes, special_nodes = ['NEUROG2'], plot = False)

########################################
# DIRECT NET Network with FIGR and RORB
# Top 8 regulators, with sink nodes highlighted (to be removed)
########################################

network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs.csv'
old_graph, old_nodes = run_network(network_path, dir_prefix, old_nodes = old_nodes, add_edge_weights=False,
                                   special_nodes = ['STAT4', 'BBX', 'BACH2', 'EPAS1', 'LCOR'], plot = False)

########################################
# DIRECT NET Network with FIGR and RORB
# Top 8 regulators, wo sink nodes
# Extra sinks highlighted (to be removed)
########################################

network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_wo_sinks.csv'
old_graph, old_nodes = run_network(network_path, dir_prefix, old_nodes = old_nodes, add_edge_weights=False,
                                   special_nodes=['HIF1A', 'GLIS3', 'SOX6', 'AHR', 'ARID5B', 'FOSB', 'TCF12', 'MECOM',
                                                  'KMT2A', 'NFE2L2', 'ZBTB20', 'TCF7L1', 'THRB', 'STAT2', 'NPAS2', 'SOX5', 'BACH1'])


########################################
# DIRECT NET Network with FIGR and RORB
# Top 8 regulators, NO sink nodes
########################################

network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks.csv'
old_graph, old_nodes = run_network(network_path, dir_prefix, old_nodes = old_nodes, add_edge_weights=False, plot = False)