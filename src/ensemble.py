from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import TARGET_COLS, get_target_routing_weights


def rank_normalize_predictions(
    df: pd.DataFrame,
    target_cols: Sequence[str] = TARGET_COLS,
) -> pd.DataFrame:
    """Transform prediction probabilities into fractional percentile ranks in [0, 1].

    Because Macro ROC-AUC evaluates relative concordance rather than calibrated probabilities,
    rank normalization neutralizes distribution differences across heterogeneous model families
    (e.g., Vision Transformers vs. Convolutional Nets).

    Args:
        df: DataFrame containing study predictions.
        target_cols: List of target columns to normalize.

    Returns:
        DataFrame with ranked targets in [0, 1].
    """
    out = df.copy()
    n = len(out)
    if n <= 1:
        return out

    for col in target_cols:
        if col in out.columns:
            out[col] = out[col].rank(method="average", pct=True)
    return out


def blend_routed_predictions(
    cnn_df: pd.DataFrame,
    transformer_df: pd.DataFrame,
    routing_weights: dict[str, float] | None = None,
    target_cols: Sequence[str] = TARGET_COLS,
    id_col: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """Blend CNN/CoAtNet and Transformer predictions using clinical target routing.

    Applies the proven anatomical routing matrix in rank space:
    - Lateral Meniscus: 100% CNN/CoAtNet (CoAtNet dramatically dominates)
    - Medial Meniscus: 80% CNN / 20% Transformer
    - ACL, Lateral OA, Fracture: 75% CNN / 25% Transformer
    - Default findings: 60% CNN / 40% Transformer

    Args:
        cnn_df: Predictions from CNN/CoAtNet family models.
        transformer_df: Predictions from Vision Transformer family models (e.g. DINOv2).
        routing_weights: Optional custom per-target CNN weights in [0, 1].
        target_cols: List of 12 target column names.
        id_col: Identifier column name.

    Returns:
        Blended, rank-normalized DataFrame matching competition schema.
    """
    if routing_weights is None:
        routing_weights = get_target_routing_weights()

    # Align on Study IDs
    c_df = cnn_df.set_index(id_col)
    t_df = transformer_df.set_index(id_col).reindex(c_df.index)

    # Rank-transform both model streams
    c_ranked = rank_normalize_predictions(c_df, target_cols)
    t_ranked = rank_normalize_predictions(t_df, target_cols)

    blended = pd.DataFrame(index=c_df.index)

    for col in target_cols:
        w_cnn = float(routing_weights.get(col, 0.60))
        w_trans = 1.0 - w_cnn

        c_vals = c_ranked[col].to_numpy(dtype=np.float64)
        t_vals = t_ranked[col].to_numpy(dtype=np.float64)

        blended[col] = w_cnn * c_vals + w_trans * t_vals

    # Final rank normalization over the blended values
    blended = rank_normalize_predictions(blended, target_cols)
    return blended.reset_index()


def multi_model_rank_average(
    predictions_list: list[pd.DataFrame],
    weights: list[float] | None = None,
    target_cols: Sequence[str] = TARGET_COLS,
    id_col: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """Ensemble multiple model predictions via weighted rank averaging.

    Args:
        predictions_list: List of prediction DataFrames.
        weights: Optional blending weights per model (normalized to sum to 1.0).
        target_cols: Target column names.
        id_col: Study ID column name.

    Returns:
        Ensemble prediction DataFrame.
    """
    if not predictions_list:
        raise ValueError("predictions_list cannot be empty")

    n_models = len(predictions_list)
    if weights is None:
        w = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)

    base = predictions_list[0].set_index(id_col)
    accum = np.zeros((len(base), len(target_cols)), dtype=np.float64)

    for i, df in enumerate(predictions_list):
        aligned = df.set_index(id_col).reindex(base.index)
        ranked = rank_normalize_predictions(aligned, target_cols)
        accum += w[i] * ranked[list(target_cols)].to_numpy(dtype=np.float64)

    out = pd.DataFrame(accum, index=base.index, columns=list(target_cols))
    out = rank_normalize_predictions(out, target_cols)
    return out.reset_index()


DEFAULT_3WAY_ROUTING_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "Fracture": (0.50, 0.20, 0.30),
    "MCL": (0.55, 0.25, 0.20),
    "Lateral Meniscus": (0.60, 0.20, 0.20),
    "Contusion": (0.55, 0.25, 0.20),
    "__default__": (0.65, 0.25, 0.10),
}


def three_way_rank_blend(
    foundation_df: pd.DataFrame,
    convnext_df: pd.DataFrame,
    specialist_df: pd.DataFrame,
    routing_weights: dict[str, tuple[float, float, float]] | None = None,
    target_cols: Sequence[str] = TARGET_COLS,
    id_col: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """Blend Foundation, ConvNeXt, and Specialist predictions in rank space with 3-way clinical routing.

    Args:
        foundation_df: 4-Pillar Foundation consensus predictions.
        convnext_df: 2.5D ConvNeXt reader predictions.
        specialist_df: ASL Rare-Finding Specialist predictions.
        routing_weights: Mapping from target -> (w_found, w_cnxt, w_spec).
        target_cols: Target column names.
        id_col: Study identifier column.

    Returns:
        Rank-normalized DataFrame with columns [id_col] + target_cols.
    """
    if routing_weights is None:
        routing_weights = DEFAULT_3WAY_ROUTING_WEIGHTS

    f_df = foundation_df.set_index(id_col)
    c_df = convnext_df.set_index(id_col).reindex(f_df.index)
    s_df = specialist_df.set_index(id_col).reindex(f_df.index)

    f_ranked = rank_normalize_predictions(f_df, target_cols)
    c_ranked = rank_normalize_predictions(c_df, target_cols)
    s_ranked = rank_normalize_predictions(s_df, target_cols)

    blended = pd.DataFrame(index=f_df.index)
    for col in target_cols:
        weights = routing_weights.get(col, routing_weights.get("__default__", (0.65, 0.25, 0.10)))
        w_f, w_c, w_s = weights
        tot = w_f + w_c + w_s
        w_f, w_c, w_s = w_f / tot, w_c / tot, w_s / tot

        f_vals = f_ranked[col].to_numpy(dtype=np.float64)
        c_vals = c_ranked[col].to_numpy(dtype=np.float64)
        s_vals = s_ranked[col].to_numpy(dtype=np.float64)

        blended[col] = w_f * f_vals + w_c * c_vals + w_s * s_vals

    blended = rank_normalize_predictions(blended, target_cols)
    return blended.reset_index()


def probit_quantile_fusion(
    models_dict: dict[str, pd.DataFrame],
    target_weights: dict[str, dict[str, float]],
    target_cols: Sequence[str] = TARGET_COLS,
    id_col: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """Fuse multiple model predictions in Latent Gaussian space via Probit Quantile Normal Transformation.

    Rather than linear rank averaging (which compresses probability tails near 0.0 and 1.0),
    this transforms fractional percentiles into standard normal quantiles:
        z = Φ⁻¹(p) = ndtri(p)
    combines them with target-specific clinical weights:
        z_fused = Σ w_i * z_i
    and maps back to continuous probabilities:
        p_fused = Φ(z_fused) = ndtr(z_fused)

    Args:
        models_dict: Mapping from model_name -> DataFrame with [id_col] + target_cols.
        target_weights: Mapping from target_col -> {model_name: weight}.
        target_cols: List of target column names.
        id_col: Study identifier column name.

    Returns:
        Rank-normalized DataFrame with columns [id_col] + target_cols.
    """
    from scipy.special import ndtr, ndtri

    if not models_dict:
        raise ValueError("models_dict cannot be empty")

    first_name = next(iter(models_dict))
    base_df = models_dict[first_name].set_index(id_col)
    n = len(base_df)

    if n <= 1:
        # Fallback for single study: return first model
        return models_dict[first_name].copy()

    # Convert each model's predictions to fractional percentiles and then to latent normal z-scores
    z_scores: dict[str, pd.DataFrame] = {}
    for name, df in models_dict.items():
        aligned = df.set_index(id_col).reindex(base_df.index)
        ranks = aligned[list(target_cols)].rank(method="average", ascending=True)
        p = ((ranks - 0.5) / n).clip(1e-6, 1.0 - 1e-6)
        z_scores[name] = p.apply(ndtri)

    fused_z = pd.DataFrame(index=base_df.index)
    for col in target_cols:
        col_weights = target_weights.get(col, {})
        total_w = sum(float(col_weights.get(name, 1.0)) for name in z_scores)
        if total_w <= 0.0:
            total_w = float(len(z_scores))
            col_weights = {name: 1.0 for name in z_scores}

        z_col = np.zeros(n, dtype=np.float64)
        for name, z_df in z_scores.items():
            w = float(col_weights.get(name, 1.0)) / total_w
            z_col += w * z_df[col].to_numpy(dtype=np.float64)
        fused_z[col] = z_col

    fused_p = fused_z.apply(ndtr)
    final_df = pd.DataFrame(index=base_df.index)
    for col in target_cols:
        final_df[col] = (
            pd.Series(fused_p[col].values).rank(method="average", pct=True).to_numpy()
        )

    return final_df.reset_index()


