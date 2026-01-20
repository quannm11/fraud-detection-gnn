import pandas as pd
import networkx as nx
import torch
import numpy as np
import os
from torch_geometric.data import HeteroData
from tqdm import tqdm

PKL_PATH = 'data/train_merged.pkl'
OUTPUT_PATH = 'data/fraud_graph_full.pt'

EDGE_CONFIG = {
    'card1': 50,       
    'DeviceInfo': 300  
}

FEATURE_COLS = [
    'TransactionAmt', 'dist1', 
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
    'D1', 'D2', 
    'card1'
]

def load_and_preprocess(path, sample_size=None):
    print(f"Loading data from {path}...")
    df = pd.read_pickle(path)
    
    if 'TransactionID' not in df.columns:
        df = df.reset_index()
        
    return df

def build_networkx_graph(df, edge_config):
    G = nx.Graph()
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Adding Nodes"):
        tx_id = f"TX_{row['TransactionID']}"
        G.add_node(tx_id, type='transaction', y=int(row['isFraud']))

    for col, threshold in edge_config.items():
        print(f"  Processing Edge Type: {col}...")
        
        # supernodes
        val_counts = df[col].value_counts()
        valid_values = val_counts[val_counts <= threshold].index
        valid_df = df[df[col].isin(valid_values)]
        
        for val, group in tqdm(valid_df.groupby(col), desc=f"Linking {col}"):
            if len(group) < 2: continue 
            
            hub_id = f"{col}_{val}"
            G.add_node(hub_id, type='entity', entity_type=col)
            
            for tx_id in group['TransactionID']:
                G.add_edge(f"TX_{tx_id}", hub_id)
                
    print(f"Graph Stats: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
    return G

def convert_to_heterodata(G, df):
    print("Converting to PyTorch HeteroData...")
    
    tx_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'transaction']
    entity_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'entity']
    
    tx_map = {n: i for i, n in enumerate(tx_nodes)}
    entity_map = {n: i for i, n in enumerate(entity_nodes)}
    
    y = torch.tensor([G.nodes[n].get('y', 0) for n in tx_nodes], dtype=torch.long)
    
    src, dst = [], []
    for u, v in G.edges():
        if u in tx_map and v in entity_map:
            src.append(tx_map[u]); dst.append(entity_map[v])
        elif v in tx_map and u in entity_map:
            src.append(tx_map[v]); dst.append(entity_map[u])
            
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    temp_df = df.set_index('TransactionID')
    
    ordered_ids = [int(n.replace("TX_", "")) for n in tx_nodes]
    
    subset = temp_df.loc[ordered_ids, FEATURE_COLS].fillna(-1).values
    mean = np.mean(subset, axis=0)
    std = np.std(subset, axis=0) + 1e-6
    x_features = (subset - mean) / std
    
    data = HeteroData()
    data['transaction'].x = torch.tensor(x_features, dtype=torch.float)
    data['transaction'].y = y
    data['entity'].x = torch.randn(len(entity_nodes), len(FEATURE_COLS)) # Random embedding for entities
    
    data['transaction', 'connected_to', 'entity'].edge_index = edge_index
    data['entity', 'rev_connected_to', 'transaction'].edge_index = torch.flip(edge_index, [0])
    
    return data

if __name__ == "__main__":
    df = load_and_preprocess(PKL_PATH, sample_size=100000) 
    G = build_networkx_graph(df, EDGE_CONFIG)
    pyg_data = convert_to_heterodata(G, df)
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    torch.save(pyg_data, OUTPUT_PATH)
    print(f"Success! Saved graph to {OUTPUT_PATH}")
    print(pyg_data)