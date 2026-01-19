import pandas as pd
import numpy as np
import gc
import os

DATA_DIR = 'data'
TRANSACTION_PATH = os.path.join(DATA_DIR, 'train_transaction.csv')
IDENTITY_PATH = os.path.join(DATA_DIR, 'train_identity.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'train_merged.pkl')

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df

print("Loading Data")
train_transaction = pd.read_csv(TRANSACTION_PATH)
train_identity = pd.read_csv(IDENTITY_PATH)

print(f"   Transaction Shape: {train_transaction.shape}")
print(f"   Identity Shape:    {train_identity.shape}")

print("Merging Data")
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

del train_transaction, train_identity
gc.collect()

print(f"Merged Shape: {train.shape}")

print("Optimizing Memory")
train = reduce_mem_usage(train)

print("Saving to Pickle")
train.to_pickle(OUTPUT_PATH)
print(f"DONE. Data saved to {OUTPUT_PATH}")