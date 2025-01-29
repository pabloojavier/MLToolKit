import pandas as pd
import numpy as np

def _get_n_std_plus(df: pd.DataFrame, timestamp_col: str,n,operator = '>') -> pd.DataFrame:
    df_dev = pd.DataFrame()
    for col in df.columns:
        if col == timestamp_col:
            continue
        if operator == '>':
            count = df.groupby(timestamp_col)[col].apply(lambda x: (x > (x.mean() + n * x.std())).sum())
        elif operator == '<':
            count = df.groupby(timestamp_col)[col].apply(lambda x: (x < (x.mean() - n * x.std())).sum())
        df_dev[col] = count
    df_dev = df_dev.T
    df_dev['Total'] = df_dev.sum(axis=1)
    return df_dev

def _get_standard_deviation(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df_std = df.groupby(by=[timestamp_col]).std().reset_index().set_index(timestamp_col).T.fillna(np.NaN)
    df_std['Total'] = df.drop(columns=[timestamp_col]).std()
    return df_std

def _get_mean(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df_mean = df.groupby(by=[timestamp_col]).mean().reset_index().set_index(timestamp_col).T.fillna(np.NaN)
    df_mean['Total'] = df.drop(columns=[timestamp_col]).mean()
    return df_mean

def _get_cnt_nulls(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    nulls = df.isnull().groupby(df[timestamp_col]).sum()
    nulls.loc['Total'] = nulls.sum()
    return nulls.drop(columns=timestamp_col).T

def _get_percent_nulls(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    nulls = df.isnull().groupby(df[timestamp_col]).sum()
    total = df.groupby(timestamp_col).size()
    nulls.loc['Total'] = nulls.sum()
    total.loc['Total'] = total.sum()
    return (nulls.div(total, axis=0)).drop(columns=timestamp_col).T

def _get_number_of_quantiles(df: pd.DataFrame, col: str) -> int:
    unique_values = len(df[col].round(decimals=4).unique())
    return min(unique_values, 10)

def _get_quantile_distribution(df: pd.DataFrame, col: str, n_quantiles: int,timestamp_col: str) -> pd.DataFrame:
    df.loc[:, 'quantile'] = pd.cut(df[col], n_quantiles)
    df = (
        df.groupby(by=[timestamp_col, 'quantile'])
        .agg({col: 'count'})
        .reset_index()
        .rename(columns={col: 'count'})
    )
    df = df.pivot(index='quantile', columns=timestamp_col, values='count')
    df = df.div(df.sum(axis=0), axis=1)
    return df