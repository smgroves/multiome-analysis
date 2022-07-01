import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
from graph_tool import all as gt
from graph_tool.topology import label_components
from collections import Counter
from graph_fit import *
# from graph_utils import state2idx, state_bool2idx

def state2idx(state):
    return int(state, 2)


# Returns 0 if state is []
def state_bool2idx(state):
    n = len(state) - 1
    d = dict({True: 1, False: 0})
    idx = 0
    for s in state:
        idx += d[s] * 2 ** n
        n -= 1
    return idx


# Hamming distance between 2 states
def hamming(x, y):
    s = 0
    for i, j in zip(x, y):
        if i != j: s += 1
    return s


# Hamming distance between 2 states, where binary states are given by decimal code
def hamming_idx(x, y, n):
    return hamming(idx2binary(x, n), idx2binary(y, n))

def idx2binary(idx, n):
    binary = "{0:b}".format(idx)
    return "0" * (n - len(binary)) + binary

# Given deterministic rules! Simulates the deterministic graph/rules and
# returns a state transition graph, as well as output of strongly connected
# component analysis. This is particularly useful for made-up networks and
# testing purposes.
def get_deterministic_stg(G, rules, regulators_dict):
    stg = gt.Graph()
    
    n = G.num_vertices()
    norm_fact = 1.*n
    nodes = [int(i) for i in G.vertices()]
    
    node_indices = dict(zip(nodes,range(len(nodes))))
    
    for i_ in range(2**n):
        stg.add_vertex()
    
    progress=0
    for idx in range(2**n):

        prog = int(idx/(1.*2**n)*100)
        if (prog > progress):
            print(prog)
            progress = prog

        state = idx2binary(idx,n)
        state_bool = [{'0':False,'1':True}[i] for i in state]
        v = stg.vertex(idx)

        for node_i, node in enumerate(nodes):
            rule = rules[node]
            regulators = regulators_dict[node]
            regulator_indices = [node_indices[i] for i in regulators]
            regulator_state = [state_bool[i] for i in regulator_indices]
            rule_leaf = state_bool2idx(regulator_state)
            neighbor_state = [i for i in state_bool]
            neighbor_state[node_i] = not neighbor_state[node_i]
            neighbor_idx = state_bool2idx(neighbor_state)
            neighbor_vertex = stg.vertex(neighbor_idx)

            # If the rule would change the node
            if rule[rule_leaf] != state_bool[node_i]:
                # Add a transition from state idx to state neighbor_idx
                edge = stg.add_edge(idx, neighbor_idx)
                
    print("Getting strongly connected components")
    comps, hists, atts = label_components(stg, directed=True, attractors=True)

    return stg, comps, hists, atts

# Given probabilistic rules! Simulates the probabilistic graph/rules and
# returns a state transition graph and edge_weights
def get_probabilistic_stg(rules, nodes, regulators_dict):
    stg = gt.Graph()
    edge_weights = stg.new_edge_property('float')
    stg.edge_properties['weight'] = edge_weights

    node_indices = dict(zip(nodes,range(len(nodes))))
    n = len(nodes)
    norm_fact = 1.*n

    for i_ in range(2**n):
        stg.add_vertex()
        
    progress = 0
    for idx in range(2**n):

        prog = int(idx/(1.*2**n)*100)
        if (prog > progress):
            print(prog)
            progress = prog

        state = idx2binary(idx,n)
        state_bool = [{'0':False,'1':True}[i] for i in state]
        v = stg.vertex(idx)

        for node_i, node in enumerate(nodes):
            rule = rules[node]
            regulators = regulators_dict[node]
            regulator_indices = [node_indices[i] for i in regulators]
            regulator_state = [state_bool[i] for i in regulator_indices]
            rule_leaf = state_bool2idx(regulator_state)
            neighbor_state = [i for i in state_bool]
            neighbor_state[node_i] = not neighbor_state[node_i]
            neighbor_idx = state_bool2idx(neighbor_state)
            neighbor_vertex = stg.vertex(neighbor_idx)

            flip = rule[rule_leaf]
            if state_bool[node_i]: flip = 1-flip
            
            if (flip > 0):
                edge = stg.add_edge(idx, neighbor_idx)
                edge_weights[edge] = flip/norm_fact
                
    return stg, edge_weights

# Do a random walk on an stg with known edge weights, with given start index and walk length
def random_walk(stg, edge_weights, start_idx, steps):
    verts = []
    next_vert = stg.vertex(start_idx)
    verts.append(next_vert)
    for i_ in range(steps):
        r = np.random.rand()
        running_p = 0
        for w in next_vert.out_neighbors():
            running_p += edge_weights[next_vert,w]
            if running_p > r:
                next_vert = w
                break
        verts.append(next_vert)
    return verts
    
def random_walk_raw(start_state, rules, regulators_dict, nodes, steps, on_nodes=[], off_nodes=[]):
    walk = []
    n = len(nodes)
    node_indices = dict(zip(nodes,range(len(nodes))))
    unperturbed_nodes = [i for i in nodes if not (i in on_nodes + off_nodes)]
    nu = len(unperturbed_nodes)
    flipped_nodes = []
    
    start_bool = [{'0':False,'1':True}[i] for i in idx2binary(start_state, n)]
    for i, node in enumerate(nodes):
        if node in on_nodes: start_bool[i] = True
        elif node in off_nodes: start_bool[i] = False
    
#    next_step = state_bool2idx(start_bool)
    next_step = start_bool
    next_idx = state_bool2idx(start_bool)
    
    while (steps > 0):
        r = np.random.rand()
        for node_i, node in enumerate(nodes):
            if node in on_nodes+off_nodes: continue
            neighbor_idx, flip = update_node(rules, regulators_dict, node, node_i, nodes, node_indices, next_step)
            r = r - flip**2/(1.*nu)
            if r <= 0:
                next_step = [{'0':False,'1':True}[i] for i in idx2binary(neighbor_idx, n)]
                next_idx = neighbor_idx
                flipped_nodes.append(node)
                break
        if r > 0: flipped_nodes.append(None)
        walk.append(next_idx)
        steps -= 1
    return walk, Counter(walk), flipped_nodes
    
def random_walk_until_leave_basin(start_state, rules, regulators_dict, nodes, radius, max_steps = 10000, on_nodes=[], off_nodes=[]):
    walk = []
    n = len(nodes)
    node_indices = dict(zip(nodes,range(len(nodes))))
    unperturbed_nodes = [i for i in nodes if not (i in on_nodes + off_nodes)]
    nu = len(unperturbed_nodes)
    flipped_nodes = []
    
    start_bool = [{'0':False,'1':True}[i] for i in idx2binary(start_state, n)]
    for i, node in enumerate(nodes):
        if node in on_nodes: start_bool[i] = True
        elif node in off_nodes: start_bool[i] = False
    
#    next_step = state_bool2idx(start_bool)
    next_step = start_bool
    next_idx = state_bool2idx(start_bool)
    distance = 0
    distances = []
    step_i = 0
    while (distance <= radius and step_i < max_steps):
        r = np.random.rand()
        for node_i, node in enumerate(nodes):
            if node in on_nodes+off_nodes: continue
            neighbor_idx, flip = update_node(rules, regulators_dict, node, node_i, nodes, node_indices, next_step)
            r = r - flip**2/(1.*nu)
            if r <= 0:
                next_step = [{'0':False,'1':True}[i] for i in idx2binary(neighbor_idx, n)]
                next_idx = neighbor_idx
                flipped_nodes.append(node)
                distance = hamming(next_step, start_bool)
                break
        if r > 0:
            flipped_nodes.append(None) ###TODO EDITED
        distances.append(distance)
        walk.append(next_idx)
        step_i += 1
    return walk, Counter(walk), flipped_nodes, distances


def random_walk_until_reach_basin(start_state, rules, regulators_dict, nodes, radius = 2, max_steps=10000, on_nodes=[],
                                  off_nodes=[], basin = None):
    walk = []
    n = len(nodes)
    node_indices = dict(zip(nodes, range(len(nodes))))
    unperturbed_nodes = [i for i in nodes if not (i in on_nodes + off_nodes)]
    nu = len(unperturbed_nodes)
    flipped_nodes = []

    start_bool = [{'0': False, '1': True}[i] for i in idx2binary(start_state, n)]
    for i, node in enumerate(nodes):
        if node in on_nodes:
            start_bool[i] = True
        elif node in off_nodes:
            start_bool[i] = False

    #    next_step = state_bool2idx(start_bool)
    next_step = start_bool
    next_idx = state_bool2idx(start_bool)
    distance = 0
    if isinstance(basin, list):
        min_dist = 200  # random high number to be replaced by actual distances
        for i in basin:
            distance = hamming_idx(start_state, i, len(nodes))
            if distance < min_dist:
                min_dist = distance
        distance = min_dist
    elif isinstance(basin, int):
        distance = hamming_idx(start_state, basin, len(nodes))  # find the distance to a certain basin and stop when
        # within radius
    else:
        print("Only integer state or list of integer states accepted for basin argument.")

    distances = []
    step_i = 0
    while (distance >= radius and step_i < max_steps):
        r = np.random.rand()
        for node_i, node in enumerate(nodes):
            if node in on_nodes + off_nodes: continue
            neighbor_idx, flip = update_node(rules, regulators_dict, node, node_i, nodes, node_indices, next_step)
            r = r - flip ** 2 / (1. * nu)
            if r <= 0:
                next_step = [{'0': False, '1': True}[i] for i in idx2binary(neighbor_idx, n)]
                next_idx = neighbor_idx
                flipped_nodes.append(node)

                if isinstance(basin, list):
                    min_dist = 200 #random high number to be replaced by actual distances
                    for i in basin:
                        distance = hamming_idx(next_idx, i, len(nodes))
                        if distance < min_dist:
                            min_dist = distance
                    distance = min_dist
                elif isinstance(basin, int):
                    distance = hamming(next_step, basin) #find the distance to a certain basin and stop when within
                # radius
                else: print("Only integer state or list of integer states accepted for basin argument.")

                break
        if r > 0:
            flipped_nodes.append(None)  ###TODO EDITED
        distances.append(distance)
        walk.append(next_idx)
        step_i += 1
    return walk, Counter(walk), flipped_nodes, distances


# Given an stg with edge weights, each pair of neighboring states has a weighted edge
# from A->B and another from B->A. This prunes all edges with weight < threshold.
# WARNING: If threshold > 0.5, it is possible for both A->B and B->A to get pruned.
# If you are using a reprogrammed stg, make sure to use nu, not n
def prune_stg_edges(stg, edge_weights, n, threshold = 0.5):
    d_stg = gt.Graph()
    for edge in stg.edges():
        if edge_weights[edge]*n > threshold:
            d_stg.add_edge(edge.source(), edge.target())
    return d_stg
    print("Finding strongly connected components")
    

# Returns dict mapping attractor components to states (idx) within that component
# atts = list of True/False indicating whether component_i is an attractor or not
# c_vertex_dict = dict mapping vertex component to list of states in the component
# vert_idx is a vertex_property mapping vertex -> idx. Used if the internal index
# of vertices in the stg are not equivalent to their state index.
def get_attractors(atts, c_vertex_dict, vert_idx = None):
    attractors = dict()
    for i, is_attractor in enumerate(atts):
        if is_attractor:
            if vert_idx is None: attractors[i] = [int(state) for state in c_vertex_dict[i]]
            else: attractors[i] = [vert_idx[state] for state in c_vertex_dict[i]]
    return attractors
    
def strip_dead_vertices(stg, out_of_bounds_vertex = None):
    G = stg.copy()
    vertices_to_strip = set()
    if out_of_bounds_vertex is None: out_of_bounds_vertex=G.vertex(0)
    G.remove_vertex(out_of_bounds_vertex.in_neighbors())
    G.remove_vertex(out_of_bounds_vertex)
#    for v in out_of_bounds_vertex.in_neighbors(): vertices_to_strip.add(v)
#        G.remove_vertex(v)
#    vertices_to_strip.add(out_of_bounds_vertex)
#    G.remove_vertex(vertices_to_strip)
    return G
    
# TODO: if    X <- A -> B -> C -> D -> E, then we can just say X <- A -> E for drawing purposes, and condense B,C,D,and E together
def condense_straight_paths():
    return

    
# Get full STG with perturbations defined in on_nodes and off_nodes
def perturb_node(rules, nodes, on_nodes, off_nodes, regulators_dict):
    stg = gt.Graph()
    edge_weights = stg.new_edge_property('float')
    stg.edge_properties['weight'] = edge_weights
    
    state_offset = len(on_nodes + off_nodes)

    unperturbed_nodes = [i for i in nodes if not (i in on_nodes or i in off_nodes)]
    nodes = on_nodes + off_nodes + unperturbed_nodes
    
    node_indices = dict(zip(nodes,range(len(nodes))))
    n = len(unperturbed_nodes)
    n_full = len(nodes)
    norm_fact = 1.*n

    for i_ in range(2**n):
        stg.add_vertex()
        
    progress = 0
    for idx in range(2**n):

        prog = int(idx/(1.*2**n)*100)
        if (prog > progress):
            print(prog)
            progress = prog

        state = idx2binary(idx,n)
        state_bool = [True]*len(on_nodes) + [False]*len(off_nodes) + [{'0':False,'1':True}[i] for i in state]
        v = stg.vertex(idx)

        for node_i, node in enumerate(unperturbed_nodes):
            rule = rules[node]
            regulators = regulators_dict[node]
            regulator_indices = [node_indices[i] for i in regulators]
            regulator_state = [state_bool[i] for i in regulator_indices]
            rule_leaf = state_bool2idx(regulator_state)
            neighbor_state = [i for i in state_bool]
            neighbor_state[node_i+state_offset] = not neighbor_state[node_i+state_offset]
            neighbor_idx = state_bool2idx(neighbor_state[state_offset:])
            neighbor_vertex = stg.vertex(neighbor_idx)

            flip = rule[rule_leaf]
            if state_bool[node_i+state_offset]: flip = 1-flip
            
            if (flip > 0):
                edge = stg.add_edge(idx, neighbor_idx)
                edge_weights[edge] = flip/norm_fact
                
    return stg, edge_weights
    


def get_reprogramming_rules(rules, regulators_dict, on_nodes, off_nodes):
    rules = rules.copy()
    regulators_dict = regulators_dict.copy()
    for node in on_nodes:
        rules[node] = np.asarray([1.])
        regulators_dict[node] = []
    for node in off_nodes:
        rules[node] = np.asarray([0.])
        regulators_dict[node] = []
    return rules, regulators_dict

def update_node(rules, regulators_dict, node, node_i, nodes, node_indices, state_bool, return_state=False):
    rule = rules[node]
    regulators = regulators_dict[node]
    regulator_indices = [node_indices[i] for i in regulators]
    regulator_state = [state_bool[i] for i in regulator_indices]
    rule_leaf = state_bool2idx(regulator_state)
    flip = rule[rule_leaf]
    if state_bool[node_i]: flip = 1-flip
    
    neighbor_state = [i for i in state_bool]
    neighbor_state[node_i] = not neighbor_state[node_i]
    neighbor_idx = state_bool2idx(neighbor_state)
    
    if return_state: return neighbor_idx, neighbor_state, flip
    return neighbor_idx, flip




### Not implemented here to 460 ###
# Starts from start_states, and updates until it has only hit unlikely nodes.
# NOTE: THIS DOES NOT ACTUALLY WORK!
def reprogram_walk(rules, regulators_dict, nodes, on_nodes, off_nodes, start_states, p_threshold = 0.000001):
    if len(np.intersect1d(on_nodes, off_nodes)) > 0: raise ValueError("on_nodes and off_nodes cannot contain the same node: %s"%repr(np.intersect1d(on_nodes, off_nodes)))
    stg = gt.Graph()
    edge_weights = stg.new_edge_property('float')
    vert_probs = stg.new_vertex_property('float')
    stg.edge_properties['weight'] = edge_weights
    stg.vertex_properties['prob'] = vert_probs # This is the best probability of reaching this state from one of the start_states
    vert_idx = stg.new_vertex_property('int') # This is the vertex's real idx, which corresponds to it's real state
    stg.vertex_properties['idx'] = vert_idx
    
    idx_vert_dict = dict() # This maps idx -> vertex
    
    rules, regulators_dict = get_reprogramming_rules(rules, regulators_dict, on_nodes, off_nodes)
    
    node_indices = dict(zip(nodes,range(len(nodes))))
    n = len(nodes)
    norm_fact = 1.*(n-len(on_nodes)-len(off_nodes))

    pending_vertices = set()
    dead_end_vertices = set() # These are vertices whose probability of reaching is less than p_threshold
    added_indices = set() # Keeps track of what indices are in the graph
    done_indices = set() # Keeps track of which indices we have already explored from

    for idx in start_states:
        v = stg.add_vertex()
        vert_idx[v]=idx
        vert_probs[v]=1.
#        pending_vertices.add(v)
        idx_vert_dict[idx]=v
        added_indices.add(idx)
        
        
    # Update all the start_states that are actively reprogrammed
    for idx in start_states:
        done_indices.add(idx)
        v = idx_vert_dict[idx]
        state = idx2binary(idx,n)
        state_bool = [{'0':False,'1':True}[i] for i in state]
        neighbor_bool = [i for i in state_bool]
        for node in on_nodes:
            ni = node_indices[node]
            neighbor_bool[ni]=True
        for node in off_nodes:
            ni = node_indices[node]
            neighbor_bool[ni]=False
        neighbor_idx = state_bool2idx(neighbor_bool)
        
        if (neighbor_idx != idx):
            if neighbor_idx in added_indices: w = idx_vert_dict[idx] # BUG TODO BUG
            else:
                w = stg.add_vertex()
                vert_idx[w]=neighbor_idx
                vert_probs[w]=1.
                pending_vertices.add(w)
                added_indices.add(neighbor_idx)
                idx_vert_dict[neighbor_idx] = w
            edge = stg.add_edge(v, w)
            edge_weights[edge] = 1.
        
    while len(pending_vertices) > 0:
        print(len(pending_vertices))
        v = pending_vertices.pop()
        idx = vert_idx[v]
        done_indices.add(idx)
        state = idx2binary(idx,n)
        state_bool = [{'0':False,'1':True}[i] for i in state]

        # Update each possible gene
        for node_i, node in enumerate(nodes):
            if node in on_nodes + off_nodes: continue
            neighbor_idx, flip_prob = update_node(rules, regulators_dict, node, node_i, nodes, node_indices, state_bool)
            
            # If there is a non-zero chance of flipping the gene
            if (flip_prob > 0):
            
                ew = flip_prob/norm_fact # This is the new edge-weight
            
                # Have we seen this neighbor before?
                if neighbor_idx in added_indices:
                    w = idx_vert_dict[neighbor_idx]
                    vert_probs[w] = max(vert_probs[w], vert_probs[v]*ew) # TODO: THIS IS TOTALLY WRONG! In reality, I should regenerate p purely fro edge weights. Which means, in reality, that I can't do it with p_threshold.
                    if vert_probs[w] > p_threshold:
                        if w in dead_end_vertices:
                            dead_end_vertices.remove(w)
                            pending_vertices.add(w)
                else:
                    w = stg.add_vertex()
                    vert_idx[w] = neighbor_idx
                    vert_probs[w] = vert_probs[v]*ew
                    added_indices.add(neighbor_idx)
                    idx_vert_dict[neighbor_idx] = w
                    if vert_probs[w] > p_threshold: pending_vertices.add(w)
                    else: dead_end_vertices.add(w)
                    
                edge = stg.add_edge(v, w)
                edge_weights[edge] = ew
                print("Adding edge from %d to %d"%(vert_idx[v], vert_idx[w]))
                
    print("Dead nodes: %d"%len(dead_end_vertices))
    if len(dead_end_vertices) > 0:
        w = stg.add_vertex()
        vert_idx[w]=0
        for v in dead_end_vertices:
            edge = stg.add_edge(v,w)
            edge_weights[edge]=1.
    return stg, edge_weights, idx_vert_dict
    

# Not done yet. vertex_dict maps idx -> vertex. vidx maps vertex -> idx
# If you have STG computed and turn nodes on or off to reduce STG (not implemented)
def perturb_stg(stg, vertex_dict, vidx = None, on_nodes=[], off_nodes=[]):
    n = len(nodes)
    nu = n-len(on_nodes)-len(off_nodes)

    if vidx is None: vidx = stg.vertex_properties['idx']

    on_indices = []
    off_indices = []
    for i,node in enumerate(nodes):
        if node in on_nodes: on_indices.append(i)
        elif node in off_nodes: off_indices.append(i)

    for v in stg.vertices():
        idx = vidx[v]
        if idx < 0: continue
        state = idx2binary(state, n)
        state_bool = [{'0':False,'1':True}[i] for i in state]
        for i in on_indices: state_bool[i] = True
        for i in off_indices: state_bool[i] = False
        newidx = state_bool2idx(state_bool)
    
    for state in states:
        state = idx2binary(state, n)
        for i in on_indices: state[i] = 1
        for i in off_indices: state[i] = 1
        return_states.append(state2idx(state))
    


# Given probabilistic rules! Simulates the probabilistic graph/rules and
# returns a state transition graph and edge_weights
def get_partial_stg(start_states, rules, nodes, regulators_dict, radius, on_nodes = [], off_nodes = [], pthreshold=0.):
    stg = gt.Graph()
    edge_weights = stg.new_edge_property('float')
    stg.edge_properties['weight'] = edge_weights

    vert_idx = stg.new_vertex_property('long') # This is the vertex's real idx, which corresponds to it's real state
    stg.vertex_properties['idx'] = vert_idx

    node_indices = dict(zip(nodes,range(len(nodes))))
    n = len(nodes)
    
    unperturbed_nodes = [i for i in nodes if not i in on_nodes+off_nodes]
    nu = len(unperturbed_nodes)
    norm_fact = 1.*nu
    
    added_indices = set()
    pending_vertices = set()
    out_of_bounds_indices = set()
    # if it reaches the radius without finding an attractor, it is out of bounds
    out_of_bounds_vertex = stg.add_vertex()
    vert_idx[out_of_bounds_vertex]=-1
    
    # Add the start states to the stg
    idx_vert_dict = dict()
    for idx in start_states:
        v = stg.add_vertex()
        vert_idx[v]=idx
        idx_vert_dict[idx]=v
        added_indices.add(idx)
        
    # Add edges from the start states to 
    for idx in start_states:
        v = idx_vert_dict[idx]
        state = idx2binary(idx,n)
        state_bool = [{'0':False,'1':True}[i] for i in state]
        neighbor_bool = [i for i in state_bool]
        for node in on_nodes:
            ni = node_indices[node]
            neighbor_bool[ni]=True
        for node in off_nodes:
            ni = node_indices[node]
            neighbor_bool[ni]=False
        neighbor_idx = state_bool2idx(neighbor_bool)
        
        if (neighbor_idx != idx):
            if neighbor_idx in added_indices: w = idx_vert_dict[neighbor_idx]
            else:
                w = stg.add_vertex()
                vert_idx[w]=neighbor_idx
                added_indices.add(neighbor_idx)
                idx_vert_dict[neighbor_idx] = w
            edge = stg.add_edge(v, w)
            edge_weights[edge]=1.
            pending_vertices.add(w)
        else: pending_vertices.add(v)
    
    start_states = set([idx2binary(i,n) for i in start_states]) # This is remembered and used to calculate the hamming distance of every visited state from the start states
    start_bools = [[{'0':False,'1':True}[i] for i in state] for state in start_states]
        
    # Go through the full list of visited vertices
    while len(pending_vertices) > 0:

        # Get the state for the next vertex
        v = pending_vertices.pop()
        idx = vert_idx[v]
        state = idx2binary(idx,n)
        state_bool = [{'0':False,'1':True}[i] for i in state]

        # Go through all the nodes and update it
        for node_i, node in enumerate(nodes):
            if not node in unperturbed_nodes: continue
            neighbor_idx, neighbor_state, flip_prob = update_node(rules, regulators_dict, node, node_i, nodes, node_indices, state_bool, return_state=True)

            if (flip_prob > pthreshold): # Add an edge to this neighbor
            
                # Have we seen this neighbor before in out_of_bounds?
                if neighbor_idx in out_of_bounds_indices:
                    if out_of_bounds_vertex in v.out_neighbors():
                        edge = stg.edge(v,out_of_bounds_vertex)
                        edge_weights[edge] += flip_prob / norm_fact
                    else:
                        edge = stg.add_edge(v, out_of_bounds_vertex)
                        edge_weights[edge] = flip_prob / norm_fact
                    continue
                        
                # Otherwise check to see if it IS out of bounds   
                min_dist = radius + 1
                for start_bool in start_bools:
                    dist = hamming(start_bool, neighbor_state)
                    if dist < min_dist:
                        min_dist = dist
                        break
                if min_dist > radius: # If it is out of bounds, add an edge to out_of_bounds_vertex
                    out_of_bounds_indices.add(neighbor_idx) # Add it to out_of_bounds_indices
                    if out_of_bounds_vertex in v.out_neighbors():
                        edge = stg.edge(v,out_of_bounds_vertex)
                        edge_weights[edge] += flip_prob / norm_fact
                    else:
                        edge = stg.add_edge(v, out_of_bounds_vertex)
                        edge_weights[edge] = flip_prob / norm_fact
                    continue
                        
                else: # Otherwise, it is in bounds
                    if neighbor_idx in added_indices: w = idx_vert_dict[neighbor_idx]
                    else: # Create the neighbor, and add it to the pending_vertices set
                        w = stg.add_vertex()
                        vert_idx[w]=neighbor_idx
                        added_indices.add(neighbor_idx)
                        idx_vert_dict[neighbor_idx] = w
                        pending_vertices.add(w)
                
                    
                    # Either way, add an edge from the current state to this one
                    edge = stg.add_edge(v, w)
                    edge_weights[edge]=flip_prob / norm_fact
                
                
    return stg, edge_weights



def get_flip_probs(idx, rules, regulators_dict, nodes, node_indices = None):
    n = len(nodes)
    if node_indices is None: node_indices = dict(zip(nodes,range(len(nodes))))
    state_bool = [{'0':False,'1':True}[i] for i in idx2binary(idx,n)]
    flips = []
    for i,node in enumerate(nodes):
        rule = rules[node]
        regulators = regulators_dict[node]
        regulator_indices = [node_indices[j] for j in regulators]
        regulator_state = [state_bool[j] for j in regulator_indices]
        rule_leaf = state_bool2idx(regulator_state)


        flip = rule[rule_leaf]
        if state_bool[i]: flip = 1-flip
        flips.append(flip)
        
    return flips
