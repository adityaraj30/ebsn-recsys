#Importing all Necessary Libraries for Embedding of Event Names
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import pickle


#Importing all event and group data combined
def data_init():
    full_event_group_edges = pd.read_csv('full_event_group_edges.csv')
    return full_event_group_edges


#Creating a dictionary with a mapping of event id and event name
def event_map(full_event_group_edges):
    event_name_map = {}
    for i, row in full_event_group_edges.iterrows():
        eventid, eventn = row['event_id'], row['name']
        event_name_map[eventid] = eventn

    return event_name_map

#Dictionaries with reference to index number to map with NumPy Array after generation of embeddings
def ref_event_arr(event_name_map):
    idx_to_event = {}
    event_to_idx = {}
    idx = 0
    for i in event_name_map:
        idx_to_event[idx] = i
        idx=idx+1
    
    idx=0
    for i in event_name_map:
        event_to_idx[i] = idx
        idx=idx+1
    
    return idx_to_event, event_to_idx 

#Saving Created indexed files
def save_files(idx_to_event, event_to_idx):
    with open('idx_to_event.pkl', 'wb') as f:
        pickle.dump(idx_to_event, f)

    with open('event_to_idx.pkl', 'wb') as f:
        pickle.dump(event_to_idx, f)

#Loading files
def load_files():
    with open('idx_to_event.pkl', 'rb') as f:
        idx_to_event = pickle.load(f)
    
    with open('event_to_idx.pkl', 'rb') as f:
        event_to_idx = pickle.load(f)
    
    return idx_to_event, event_to_idx

# Preparing the data to be sent into the Sentence Transformer 
def event_transf(full_event_group_edges):
    event_name_map = {}
    for i, row in full_event_group_edges.iterrows():
            eventid, eventn = row['event_id'], row['name']
            event_name_map[eventid] = eventn

    event_name_list=[]
    for i in event_name_map:
        event_name_list.append(event_name_map[i])
    event_name_list

    return event_name_list

# Running the all-mpnet-base-v2 SentenceTransformer on the data
def sent_transf_mod(event_name_list):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    event_embeddings = model.encode(event_name_list)
    return event_embeddings


def main():
    full_event_group_edges = data_init()
    event_name_map = event_map(full_event_group_edges)
    idx_to_event, event_to_idx = ref_event_arr(event_name_map)

    save_files(idx_to_event, event_to_idx)

    idx_to_event, event_to_idx = load_files()

    event_name_list = event_transf(full_event_group_edges)
    event_embeddings = sent_transf_mod(event_name_list)
    
    embedding_data = pd.DataFrame(event_embeddings)
    
    # Saving embeddings of the event names
    embedding_data.to_csv('event_embeddings_st.csv')


if __name__=='__main__':
    main()
