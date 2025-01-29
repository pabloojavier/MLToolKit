import pandas as pd
import numpy as np
np.random.seed(42)

def create_simulated_data(n_features=20,n_rows=10_000):
    df = pd.DataFrame(np.random.randn(n_rows, n_features), columns=[f'col_{i}' for i in range(n_features)])
    df['target'] = np.random.choice([0, 1], n_rows, p=[0.8, 0.2])
    df['event_time'] = np.random.choice(pd.date_range('2022-01-01', '2023-01-01', freq='MS'), n_rows)
    df['cliente_id'] = np.random.choice([f'cliente_{i}' for i in range(n_rows)], n_rows)

    df = df.drop_duplicates(subset=['cliente_id', 'event_time'])
    return df