"""Cross-model agreement metrics (percentage agreement and Cohen's Kappa) for the MediTriageAI analysis framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def compute_pairwise_agreement(predictions_dict: dict[str, pd.DataFrame], target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Cohen's Kappa and percentage agreement between all model combinations.
    
    Args:
        predictions_dict: Mapping from model name to their predictions DataFrame.
        target_col: Column name to evaluate agreement (e.g. 'pred_specialist' or 'pred_severity').
        
    Returns:
        A tuple of (kappa_df, agreement_pct_df).
    """
    model_names = sorted(predictions_dict.keys())
    n = len(model_names)
    
    kappa_matrix = np.zeros((n, n))
    pct_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                kappa_matrix[i, j] = 1.0
                pct_matrix[i, j] = 1.0
                continue
                
            model_a = model_names[i]
            model_b = model_names[j]
            
            # Align predictions by sample_id
            df_a = predictions_dict[model_a].set_index("sample_id")
            df_b = predictions_dict[model_b].set_index("sample_id")
            
            common_ids = df_a.index.intersection(df_b.index)
            y_a = df_a.loc[common_ids, target_col].values
            y_b = df_b.loc[common_ids, target_col].values
            
            if len(common_ids) > 0:
                kappa_matrix[i, j] = cohen_kappa_score(y_a, y_b)
                pct_matrix[i, j] = np.mean(y_a == y_b)
            else:
                kappa_matrix[i, j] = 0.0
                pct_matrix[i, j] = 0.0
                
    kappa_df = pd.DataFrame(kappa_matrix, index=model_names, columns=model_names)
    pct_df = pd.DataFrame(pct_matrix, index=model_names, columns=model_names)
    
    return kappa_df, pct_df
