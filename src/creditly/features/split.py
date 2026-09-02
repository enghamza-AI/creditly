import numpy as np
import pandas as pd

def train_test_split_manual(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(df.index)

    n_test = int(len(df) * test_size)
    test_idx = shuffled_idx[:n_test]
    train_idx = shuffled_idx[n_test:]

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    return train_df, test_df

