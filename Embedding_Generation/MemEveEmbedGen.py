#Importing all Necessary Libraries for Embedding Generation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
from node2vec import Node2Vec

# Initialising all the Data Required
def data_init():
    meta_events = pd.read_csv('*/meta-events.csv')
    meta_members = pd.read_csv('*/meta-members.csv')
    rsvps = pd.read_csv('*/rsvps.csv')
    return meta_events, meta_members, rsvps

#Preliminary Data Changes
def dataset_changes(meta_events, meta_members, rsvps):
    meta_events['new_id'] = range(len(meta_events))
    meta_members['new_id'] = range(len(meta_members))
    rsvps.drop(['id', 'group_id'], axis = 1, inplace=True)
    return meta_events, meta_members, rsvps

#Setting dataset indexes according to Numpy Array Indexing
def data_index(meta_events, meta_members):
    event_id_dict = {}
    for i, row in meta_events.iterrows():
        id, event_id = row['new_id'], row['event_id']
        event_id_dict[event_id] = id

    member_id_dict = {}
    for i, row in meta_members.iterrows():
        id, member_id = row['new_id'], row['member_id']
        member_id_dict[member_id] = id

    event_id_dict_rev = {}
    for i, row in meta_events.iterrows():
        id, event_id = row['new_id'], row['event_id']
        event_id_dict_rev[id] = event_id

    member_id_dict_rev = {}
    for i, row in meta_members.iterrows():
        id, member_id = row['new_id'], row['member_id']
        member_id_dict_rev[id] = member_id
    
    return event_id_dict, member_id_dict, event_id_dict_rev, member_id_dict_rev

#Saving Created indexed files
def save_files(event_id_dict, member_id_dict, event_id_dict_rev, member_id_dict_rev):
    with open('event_id_dict.pkl', 'wb') as f:
        pickle.dump(event_id_dict, f)

    with open('member_id_dict.pkl', 'wb') as f:
        pickle.dump(member_id_dict, f)
    
    with open('event_id_dict_rev.pkl', 'wb') as f:
        pickle.dump(event_id_dict_rev, f)

    with open('member_id_dict_rev.pkl', 'wb') as f:
        pickle.dump(member_id_dict_rev, f)   


# Initialising graph for 
def graph_create(rsvps):
    # Create a graph
    G = nx.Graph()

    # Add edges between members and events
    for _, row in rsvps.iterrows():
        G.add_edge(f"member_{row['member_id']}", f"event_{row['event_id']}")
    
    return G


def rsvps_index_rel(rsvps, event_id_dict, member_id_dict):
    # Function to replace event_id with its corresponding numpy array index
    def replace_event_id(event_id):
        idx = event_id_dict[event_id]
        return idx

    # Function to replace member_id with its corresponding numpy array index
    def replace_member_id(member_id):
        idx = member_id_dict[member_id]
        return idx

    rsvps['event_id'] = rsvps['event_id'].apply(replace_event_id)
    rsvps['member_id'] = rsvps['member_id'].apply(replace_member_id)
    rsvps['weight'] = 1

    return rsvps


def graph_proc(G, member_embeddings_dict):
    
    # Get all connected components (subgraphs)
    connected_components = nx.connected_components(G)

    # Iterate over each subgraph
    for component in connected_components:
        subgraph = G.subgraph(component).copy()
        
        # Generate embeddings using Node2Vec for the subgraph
        node2vec = Node2Vec(subgraph, dimensions=1024, walk_length=32, num_walks=100, workers=8)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Get embeddings for member nodes in this subgraph
        member_nodes = [node for node in subgraph.nodes() if node.startswith('member_')]
        for node in member_nodes:
            # Convert member_id to int when adding to the dictionary
            member_id = int(node.replace('member_', ''))
            member_embeddings_dict[member_id] = model.wv[node]
    
    return member_embeddings_dict

def main():
    meta_events, meta_members, rsvps = data_init()
    meta_events, meta_members, rsvps = dataset_changes(meta_events, meta_members, rsvps)
    event_id_dict, member_id_dict, event_id_dict_rev, member_id_dict_rev = data_index(meta_events, meta_members)
    save_files(event_id_dict, member_id_dict, event_id_dict_rev, member_id_dict_rev)
    rsvps = rsvps_index_rel(rsvps, event_id_dict, member_id_dict)

    # Creating a Graph G from the rsvps dataset with member id's and event id's
    G = graph_create(rsvps)

    member_embeddings_dict = {}

    member_embeddings_dict = graph_proc(G, member_embeddings_dict)

    member_embeddings_df = pd.DataFrame.from_dict(member_embeddings_dict, orient='index')

    member_embeddings_df = member_embeddings_df.sort_index()
    member_embeddings_df.reset_index(inplace=True)
    member_embeddings_df.rename(columns={'index': 'member_id'}, inplace=True) 
    member_embeddings_df.to_csv('member_embeddings_node2vec_allg_1024_2.csv')


if __name__ == "__main__":
    main()