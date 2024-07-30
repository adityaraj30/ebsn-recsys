# Basic Libraries Imported
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle

#Libraries imported for TSNE Calculation and Graph Plotting
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Initialising all Required Data
def data_init():
    member_group_edges = pd.read_csv('/Users/adityaraj/Aditya/IIT KGP/MeetUp Dataset - Nashville/member-to-group-edges.csv')
    meta_events = pd.read_csv('/Users/adityaraj/Aditya/IIT KGP/MeetUp Dataset - Nashville/meta-events.csv')
    meta_members = pd.read_csv('/Users/adityaraj/Aditya/IIT KGP/MeetUp Dataset - Nashville/meta-members.csv')
    full_event_group_edges = pd.read_csv('/Users/adityaraj/Aditya/IIT KGP/full_event_group_edges.csv')
    n2v_embed = pd.read_csv('/Users/adityaraj/Aditya/IIT KGP/Local_Env/member_embeddings_node2vec_allg_1024.csv')

    return member_group_edges, meta_events, meta_members, full_event_group_edges, n2v_embed


def load_files():
    with open('member_id_dict.pkl', 'rb') as f:
        member_id_dict = pickle.load(f)
    
    with open('member_id_dict_rev.pkl', 'rb') as f:
        member_id_dict_rev = pickle.load(f)
    
    return member_id_dict, member_id_dict_rev


#Cleaning Data
def clean_up_transform(n2v_embed):
    n2v_embed.drop('Unnamed: 0', axis = 1, inplace = True)
    n2v_embed_np = n2v_embed.to_numpy(dtype = 'float32')
    return n2v_embed_np

#Scaling Data and Passing through TSNE for dimensionality Reduction
def scaling_tsne(n2v_embed_np, n=2):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(n2v_embed_np)
    tsne = TSNE(n_components=n)
    reduced_embeddings = tsne.fit_transform(standardized_data)
    return reduced_embeddings

# Processing data for plotting
def data_for_graph(member_group_edges):
    member_ids = member_group_edges['member_id'].to_numpy()
    group_ids = member_group_edges['group_id'].to_numpy()
    return member_ids, group_ids

# Creating a member id to embedding map
def member_embed_map(member_ids, n2v_embed_np_red, member_id_dict):
    member_embed_dict={}
    for i in member_ids:
        member_embed_dict[i] = n2v_embed_np_red[member_id_dict[i]]
    return member_embed_dict
    
# Plot for 2D Graph
def plot_2D(group_ids, member_ids, member_embed_dict):
    unique_group_ids = np.unique(group_ids)
    num_groups = len(unique_group_ids)
    colors = plt.cm.get_cmap('tab20', num_groups)  # Use a colormap for distinct colors

    group_colors = {group_id: colors(i / num_groups) for i, group_id in enumerate(unique_group_ids)}

    plt.figure(figsize=(10, 8))
    for i, member_id in enumerate(member_ids):
        mem = member_embed_dict[member_id]
        group_id = group_ids[i]
        color = group_colors[group_id]
        plt.scatter(mem[0], mem[1], color=color)

    plt.title('2D Embeddings of member_id')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)

    # Save the plot
    plt.savefig('2d_PCA_1024_15G_TRIAL.png', format='png', dpi=300)

    plt.show()

# Plot for 3D Graph
def plot_3D(group_ids, member_ids, member_embed_dict):
    unique_group_ids = np.unique(group_ids)
    num_groups = len(unique_group_ids)
    colors = plt.cm.get_cmap('tab20', num_groups)

    group_colors = {group_id: 'rgb' + str(colors(i / num_groups)[:3]) for i, group_id in enumerate(unique_group_ids)}

    fig = go.Figure()

    for i, member_id in enumerate(member_ids):
        mem = member_embed_dict[member_id]
        group_id = group_ids[i]
        color = group_colors[group_id]
        
        fig.add_trace(go.Scatter3d(
            x=[mem[0]], y=[mem[1]], z=[mem[2]],
            mode='markers',
            marker=dict(size=5, color=color),
            name=f'Group {group_id}'
        ))

    fig.update_layout(
        title='3D Embeddings of member_id',
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        showlegend=False  # Set to True if you want to display the legend
    )

    # Save the plot to an HTML file
    fig.write_html('3d_PCA_1024_15G_TRIAL.html')


def main():
    member_group_edges, meta_events, meta_members, full_event_group_edges, n2v_embed = data_init()
    member_id_dict, member_id_dict_rev = load_files()
    member_ids, group_ids = data_for_graph(member_group_edges)

    meta_events['new_id'] = range(len(meta_events))
    meta_members['new_id'] = range(len(meta_members))

    n2v_embed_np = clean_up_transform(n2v_embed)
    
    n2v_embed_np_red = scaling_tsne(n2v_embed_np, 2)
    member_embed_dict = member_embed_map(member_ids, n2v_embed_np_red, member_id_dict)
    plot_2D(group_ids, member_ids, member_embed_dict)

    n2v_embed_np_red = scaling_tsne(n2v_embed_np, 3)
    member_embed_dict = member_embed_map(member_ids, n2v_embed_np_red, member_id_dict)
    plot_3D(group_ids, member_ids, member_embed_dict)



if __name__ == "__main__":
    main()
