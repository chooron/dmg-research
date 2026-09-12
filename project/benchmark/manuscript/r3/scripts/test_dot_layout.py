#!/usr/bin/env python3
"""Test script for PyGraphviz layout generation and coordinate parsing for F6a."""
import pygraphviz as pgv
import numpy as np

def get_ihacres_graph():
    G = pgv.AGraph(directed=True, strict=False)
    G.graph_attr.update(rankdir='TB', nodesep='0.6', ranksep='0.7', margin='0.1')
    
    # Nodes
    G.add_node('P', label='Precipitation P', shape='circle')
    G.add_node('PET', label='PET', shape='circle')
    G.add_node('S1', label='S1: CMD Deficit', shape='box')
    G.add_node('Ea', label='Ea', shape='circle')
    G.add_node('Split', label='Split', shape='point')
    G.add_node('UH_q', label='UH Quick (tau_q)', shape='box')
    G.add_node('UH_s', label='UH Slow (tau_s)', shape='box')
    G.add_node('Q', label='Streamflow Q', shape='circle')
    
    # Ranks
    G.add_subgraph(['P', 'PET'], rank='same')
    G.add_subgraph(['Ea', 'Split'], rank='same')
    G.add_subgraph(['UH_q', 'UH_s'], rank='same')
    
    # Edges
    G.add_edge('P', 'S1')
    G.add_edge('S1', 'Ea')
    G.add_edge('S1', 'Split')
    G.add_edge('Split', 'UH_q')
    G.add_edge('Split', 'UH_s')
    G.add_edge('UH_q', 'Q')
    G.add_edge('UH_s', 'Q')
    
    G.layout(prog='dot')
    return G

G = get_ihacres_graph()
print("IHACRES bb:", G.graph_attr['bb'])
for n in G.nodes():
    print(f"  Node {n.name:10s} pos={n.attr['pos']}")
