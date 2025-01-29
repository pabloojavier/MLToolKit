import pandas as pd
import numpy as np
from .utils.stability import (
    _get_n_std_plus, 
    _get_standard_deviation, 
    _get_mean, 
    _get_cnt_nulls, 
    _get_percent_nulls, 
    _get_number_of_quantiles, 
    _get_quantile_distribution
)

def stability_stat(
            df: pd.DataFrame,
            timestamp_col: str,
            features : list,
        ):

    df = df[features+[timestamp_col]]

    df_mean = _get_mean(df, timestamp_col)
    df_nulls_pct = _get_percent_nulls(df, timestamp_col)
    df_nulls_cnt = _get_cnt_nulls(df, timestamp_col)
    df_std = _get_standard_deviation(df, timestamp_col)
    df_over_three_std = _get_n_std_plus(df, timestamp_col,3,operator='>')
    df_under_three_std = _get_n_std_plus(df, timestamp_col,3,operator='<')
    df_over_five_std = _get_n_std_plus(df, timestamp_col,5,operator='>')
    df_under_five_std = _get_n_std_plus(df, timestamp_col,5,operator='<')

    return {
        'mean':df_mean,
        'std' :df_std,
        'nulls_pct':df_nulls_pct,
        'nulls_cnt':df_nulls_cnt,
        'over_three_std':df_over_three_std,
        'under_three_std':df_under_three_std,
        'over_five_std':df_over_five_std,
        'under_five_std':df_under_five_std,
    }

def dynamic_csi_stat(
            df: pd.DataFrame,
            timestamp_col: str,
            features : list,
        ) -> pd.DataFrame:
    cols_to_analyze = features

    df_append = pd.DataFrame()

    for feature in cols_to_analyze:
        df_feat = df[[feature, timestamp_col]].dropna()
        n_quantiles = _get_number_of_quantiles(df_feat, feature)
        if n_quantiles == 0:
            continue
        df_feat = _get_quantile_distribution(df_feat, feature, n_quantiles,timestamp_col)
        csi = _calculate_csi(df_feat, feature, n_quantiles)
        df_append = pd.concat([df_append, csi], axis=0)

    df_append = _assign_stability(df_append)

    return df_append

def get_features_with_low_variability(
        df: pd.DataFrame,
        features: list,
        threshold: float = 0.95):
    """
    Returns a list of features with low variability (i.e. features with a single value in more than 95%
    of the observations)
    """
    df = df[features]
    features_low_variability = []
    for feature in features:
        max_percent_concentration = df[feature].value_counts(normalize=True).max()
        try:
            if max_percent_concentration > threshold:
                features_low_variability.append(feature)
        except :
            pass
    return features_low_variability

def _calculate_csi(df: pd.DataFrame, col: str, n_quantiles: int) -> float:
    df_out = df.copy()

    for i in range(df.shape[1] - 1):
        actual = df.iloc[:, i + 1] + 10e-20
        expected = df.iloc[:, i] + 10e-20
        df_out.iloc[:, i + 1] = (actual - expected) * np.log(actual / expected)

    df_out = df_out.drop(columns=df.columns[0])
    df_out = pd.DataFrame(df_out.sum()).rename(columns={0: col}).T
    df_out['quantile'] = n_quantiles

    return df_out

def _assign_stability(df: pd.DataFrame) -> pd.DataFrame:
    df['status'] = '🟢'
    df['status_2'] = 'low'

    bool_med_csi = (df.iloc[:, :-3] >= 0.1) & (df.iloc[:, :-3] < 0.2)
    bool_med_csi = bool_med_csi.any(axis=1)

    bool_high_csi = df.iloc[:, :-3] > 0.2
    bool_high_csi = bool_high_csi.any(axis=1)

    df.loc[bool_med_csi, 'status'] = '🟡'
    df.loc[bool_med_csi, 'status_2'] = 'medium'
    df.loc[bool_high_csi, 'status'] = '🔴'
    df.loc[bool_high_csi, 'status_2'] = 'high'

    return df
