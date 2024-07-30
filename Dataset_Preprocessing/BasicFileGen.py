# Importing Necessary Modules and Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
import torch
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

# Importing all the data for preprocessing

group_edges = pd.read_csv('*/group-edges.csv')
member_edges = pd.read_csv('*/member-edges.csv')
member_group_edges = pd.read_csv('*/member-to-group-edges.csv')
meta_events = pd.read_csv('*/meta-events.csv')
meta_group = pd.read_csv('*/meta-groups.csv')
meta_members = pd.read_csv('*/meta-members.csv')
rsvps = pd.read_csv('*/rsvps.csv')

# Creating a new table which consists of an inner merge between Meta Events and Meta Group on the column group_id
full_event_group_edges = pd.merge(meta_events, meta_group, how ='inner', on ='group_id') 
full_event_group_edges.to_csv('full_event_group_edges.csv')
