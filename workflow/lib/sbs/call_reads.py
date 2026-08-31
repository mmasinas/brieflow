"""Read Calling Utilities!

This module provides a set of functions for mapping and analyzing sequencing reads.
It includes functions for:

- Read Calling: Algorithms for calling bases from raw intensity data.
- Read Formatting: Tools to format and normalize sequencing reads.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# constants for calling reads
from lib.sbs.constants import (
    WELL,
    TILE,
    CELL,
    READ,
    CHANNEL,
    CYCLE,
    BARCODE,
    INTENSITY,
)


# # PER-CHANNEL DECODING APPROACH (per-read, PER-CHANNEL relative threshold) —
# # kept for reference. A channel was 'ON' for a cycle if its intensity was
# # >= relative_thresh * (that channel's max intensity across ALL cycles for that
# # read), i.e. each channel thresholded against its own peak brightness. To
# # revert, comment out the pooled-across-both-channels decode_2color_to_bases
# # below and uncomment this version.
# def decode_2color_to_bases(df_bases, relative_thresh=0.2):
#     """Convert 2-color intensity data to 4-channel base calls.
#
#     Uses a per-read, PER-CHANNEL relative threshold: a channel is 'ON' for a
#     given cycle if its intensity >= relative_thresh * (that channel's max
#     intensity across ALL cycles for that read). Each channel is thresholded
#     against its own peak brightness, so a dim channel is not penalized by a
#     bright one (and vice versa). This still uses the read's own peak brightness
#     as the reference, making G detection relative rather than absolute and
#     preventing globally bright/dark cells from producing the AAAAAAAA/GGGGGGGG
#     artifact seen with per-cycle absolute mean thresholds.
#
#     Encoding (matches Illumina 2-channel SBS chemistry):
#         Cy3 ON  + Cy5 ON  -> A  (both channels high)
#         Cy3 ON  + Cy5 OFF -> T  (only Cy3 high)
#         Cy3 OFF + Cy5 ON  -> C  (only Cy5 high)
#         Cy3 OFF + Cy5 OFF -> G  (both channels dark)
#
#     Args:
#         df_bases: DataFrame with columns [read, cycle, channel, intensity, ...]
#                   where channel values are 'Cy3' and 'Cy5'.
#         relative_thresh: Fraction of per-read, per-channel maximum used as on/off
#                          threshold. Default 0.2 (20%), following NIS-Seq
#                          convention. Increase to reduce A calls from partially-lit
#                          cycles, decrease to recover weak signals.
#
#     Returns:
#         DataFrame in same format as df_bases but with channel values replaced by
#         ['G', 'T', 'A', 'C'] and pseudo-intensities assigned.
#     """
#
#     id_cols = [col for col in df_bases.columns if col not in ['channel', 'intensity']]
#
#     df_pivot = df_bases.pivot_table(
#         index=id_cols,
#         columns='channel',
#         values='intensity',
#         aggfunc='first'
#     ).reset_index()
#
#     # Pre-compute per-read maximum SEPARATELY for each channel (across all cycles
#     # for that read). Thresholding each channel against its own peak makes G
#     # detection relative within each channel, so a dim-but-real channel signal is
#     # not suppressed by the other channel being brighter.
#     read_max = (
#         df_pivot.groupby('read')[['Cy3', 'Cy5']]
#         .max()
#         .rename(columns={'Cy3': 'read_max_Cy3', 'Cy5': 'read_max_Cy5'})
#     )
#     df_pivot = df_pivot.join(read_max, on='read')
#
#     records = []
#     for idx, row in df_pivot.iterrows():
#         base_dict = {col: row[col] for col in id_cols}
#
#         cy3_threshold = relative_thresh * row['read_max_Cy3']
#         cy5_threshold = relative_thresh * row['read_max_Cy5']
#
#         m_on = row['Cy3'] >= cy3_threshold
#         c_on = row['Cy5'] >= cy5_threshold
#
#         if m_on and c_on:
#             intensities = {'A': max(row['Cy3'], row['Cy5']), 'T': 0, 'C': 0, 'G': 0}
#         elif m_on and not c_on:
#             intensities = {'A': 0, 'T': row['Cy3'], 'C': 0, 'G': 0}
#         elif not m_on and c_on:
#             intensities = {'A': 0, 'T': 0, 'C': row['Cy5'], 'G': 0}
#         else:
#             intensities = {'A': 0, 'T': 0, 'C': 0, 'G': max(row['Cy3'], row['Cy5'])}
#
#         for base in ['G', 'T', 'A', 'C']:
#             record = base_dict.copy()
#             record['channel'] = base
#             record['intensity'] = intensities[base]
#             records.append(record)
#
#     df_decoded = pd.DataFrame(records)
#     df_decoded = df_decoded[df_bases.columns]
#
#     channel_order = {'G': 0, 'T': 1, 'A': 2, 'C': 3}
#     df_decoded['_channel_sort'] = df_decoded['channel'].map(channel_order)
#     df_decoded = df_decoded.sort_values(['read', 'cycle', '_channel_sort']).drop('_channel_sort', axis=1)
#     df_decoded = df_decoded.reset_index(drop=True)
#
#     return df_decoded


# Define 2-color chemistry base mapping
# CURRENT DECODING APPROACH (per-read relative threshold pooled across BOTH
# channels). A channel is 'ON' for a given cycle if its intensity is
# >= relative_thresh * (max intensity across ALL cycles and BOTH channels for
# that read). To use the per-channel variant instead, comment out this function
# and uncomment the per-channel decode_2color_to_bases above.
def decode_2color_to_bases(df_bases, relative_thresh=0.2):
    """Convert 2-color intensity data to 4-channel base calls.

    Uses a per-read relative threshold: a channel is 'ON' for a given cycle if
    its intensity >= relative_thresh * (max intensity across ALL cycles and BOTH
    channels for that read).
    """

    id_cols = [col for col in df_bases.columns if col not in ['channel', 'intensity']]

    df_pivot = df_bases.pivot_table(
        index=id_cols,
        columns='channel',
        values='intensity',
        aggfunc='first'
    ).reset_index()

    # Pre-compute per-read maximum across ALL cycles and BOTH channels.
    read_max = (
        df_pivot.groupby('read')[['Cy3', 'Cy5']]
        .max()
        .max(axis=1)
        .rename('read_max')
    )
    df_pivot = df_pivot.join(read_max, on='read')

    records = []
    for idx, row in df_pivot.iterrows():
        base_dict = {col: row[col] for col in id_cols}

        threshold = relative_thresh * row['read_max']

        m_on = row['Cy3'] >= threshold
        c_on = row['Cy5'] >= threshold

        if m_on and c_on:
            intensities = {'A': max(row['Cy3'], row['Cy5']), 'T': 0, 'C': 0, 'G': 0}
        elif m_on and not c_on:
            intensities = {'A': 0, 'T': row['Cy3'], 'C': 0, 'G': 0}
        elif not m_on and c_on:
            intensities = {'A': 0, 'T': 0, 'C': row['Cy5'], 'G': 0}
        else:
            intensities = {'A': 0, 'T': 0, 'C': 0, 'G': max(row['Cy3'], row['Cy5'])}

        for base in ['G', 'T', 'A', 'C']:
            record = base_dict.copy()
            record['channel'] = base
            record['intensity'] = intensities[base]
            records.append(record)

    df_decoded = pd.DataFrame(records)
    df_decoded = df_decoded[df_bases.columns]

    channel_order = {'G': 0, 'T': 1, 'A': 2, 'C': 3}
    df_decoded['_channel_sort'] = df_decoded['channel'].map(channel_order)
    df_decoded = df_decoded.sort_values(['read', 'cycle', '_channel_sort']).drop('_channel_sort', axis=1)
    df_decoded = df_decoded.reset_index(drop=True)

    return df_decoded


def _pivot_2color(df_bases):
    """Pivot raw Cy3/Cy5 long data to one row per read/cycle.

    Returns (df_pivot, id_cols) where df_pivot has the id columns plus 'Cy3' and
    'Cy5' intensity columns.
    """
    id_cols = [col for col in df_bases.columns if col not in ['channel', 'intensity']]
    df_pivot = df_bases.pivot_table(
        index=id_cols,
        columns='channel',
        values='intensity',
        aggfunc='first',
    ).reset_index()
    return df_pivot, id_cols


def correct_2color_channels(df_bases, bleed_select_q=0.75):
    """Channel-balance and crosstalk-unmix raw Cy3/Cy5 BEFORE decoding.

    Runs on the raw 2-color data (channel column still 'Cy3'/'Cy5') and returns
    the same long format with corrected 'intensity' values, ready to hand to
    decode_2color_to_bases. Applied in order:

      1. Crosstalk unmixing — a 2x2 linear correction (observed = M . true) whose
         off-diagonal bleed terms are estimated from the data: alpha = Cy3->Cy5
         bleed from confident T spots (Cy3 high, Cy5 low), beta = Cy5->Cy3 bleed
         from confident C spots. Recovered signal = M^-1 . observed.
      2. Gain balancing — equalize the two channels' 'on' levels (median of the
         confident single-channel spots) to their geometric mean, so a pooled
         relative threshold treats Cy3 and Cy5 fairly while keeping intensities
         on the raw magnitude scale (so Q / q_min stay interpretable).

    Background/offset subtraction is intentionally NOT performed here (assumed
    handled upstream at the image preprocessing stage).

    Note: this operates on the continuous 2-channel signal, which is the only
    stage where crosstalk unmixing is meaningful. The downstream median
    correction (transform_medians) runs on already-decoded one-hot data, where it
    collapses to an identity transform, so there is no double-correction.
    """
    df_pivot, id_cols = _pivot_2color(df_bases)

    # --- 1. Crosstalk unmixing ---
    # Select confident single-channel spots to estimate bleed coefficients.
    hi3 = df_pivot['Cy3'] >= df_pivot['Cy3'].quantile(bleed_select_q)
    lo5 = df_pivot['Cy5'] <= df_pivot['Cy5'].quantile(1 - bleed_select_q)
    hi5 = df_pivot['Cy5'] >= df_pivot['Cy5'].quantile(bleed_select_q)
    lo3 = df_pivot['Cy3'] <= df_pivot['Cy3'].quantile(1 - bleed_select_q)
    conf_t = hi3 & lo5  # confident T: Cy3 on, Cy5 off  -> residual Cy5/Cy3 = alpha
    conf_c = hi5 & lo3  # confident C: Cy5 on, Cy3 off  -> residual Cy3/Cy5 = beta

    alpha = (
        (df_pivot.loc[conf_t, 'Cy5'] / df_pivot.loc[conf_t, 'Cy3'])
        .replace([np.inf, -np.inf], np.nan)
        .median()
    )
    beta = (
        (df_pivot.loc[conf_c, 'Cy3'] / df_pivot.loc[conf_c, 'Cy5'])
        .replace([np.inf, -np.inf], np.nan)
        .median()
    )
    alpha = float(np.nan_to_num(alpha))
    beta = float(np.nan_to_num(beta))

    M = np.array([[1.0, beta], [alpha, 1.0]])
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.eye(2)
    obs = df_pivot[['Cy3', 'Cy5']].to_numpy(dtype=float)
    corr = obs @ M_inv.T
    df_pivot['Cy3'] = np.clip(corr[:, 0], 0, None)
    df_pivot['Cy5'] = np.clip(corr[:, 1], 0, None)

    # --- 2. Gain balancing ---
    # 'On' level of each channel = median among that channel's confident spots.
    on3 = df_pivot.loc[conf_t, 'Cy3'].median()
    on5 = df_pivot.loc[conf_c, 'Cy5'].median()
    if on3 and on5 and on3 > 0 and on5 > 0:
        target = np.sqrt(on3 * on5)  # common raw-scale magnitude
        df_pivot['Cy3'] = df_pivot['Cy3'] * (target / on3)
        df_pivot['Cy5'] = df_pivot['Cy5'] * (target / on5)

    # Melt back to long format (raw 'Cy3'/'Cy5' channel labels retained).
    df_corrected = df_pivot.melt(
        id_vars=id_cols,
        value_vars=['Cy3', 'Cy5'],
        var_name='channel',
        value_name='intensity',
    )
    return df_corrected[df_bases.columns]


# ============================================================================
# V2 / V3 DECODING APPROACHES (cluster the raw Cy3/Cy5 plane) — ACTIVE for v2/v3.
# These replace the threshold-based decode_2color_to_bases with a 4-cluster fit
# in the (Cy3, Cy5) plane, absorbing gain, crosstalk, and background jointly as
# cluster geometry. To activate one, uncomment the shared helpers below plus the
# desired decode_2color_* function, then dispatch to it in call_reads (see the
# decode block there). They run WITHOUT correct_2color_channels, since the
# clustering already handles channel scaling.
# ============================================================================
def _map_components_to_bases(centers):
    """Map 4 cluster centers in (Cy3, Cy5) space to bases by geometry.

    G = both channels low (nearest origin), A = both high, T = Cy3-high/Cy5-low,
    C = Cy5-high/Cy3-low. Returns a dict {base: component_index}.
    """
    c3, c5 = centers[:, 0], centers[:, 1]
    order = np.argsort(c3 + c5)
    g, a = int(order[0]), int(order[-1])
    mid = [i for i in range(4) if i not in (g, a)]
    # Of the two intermediate clusters, the more Cy3-dominant one is T, other C.
    if (c3[mid[0]] - c5[mid[0]]) >= (c3[mid[1]] - c5[mid[1]]):
        t, c = mid[0], mid[1]
    else:
        t, c = mid[1], mid[0]
    return {'G': g, 'T': t, 'A': a, 'C': c}


def _assemble_decoded(df_pivot, id_cols, columns):
    """Melt a wide (id_cols + G/T/A/C pseudo-intensity) frame back to the long
    df_bases format, ordered G, T, A, C within each read/cycle."""
    df_decoded = df_pivot.melt(
        id_vars=id_cols,
        value_vars=['G', 'T', 'A', 'C'],
        var_name='channel',
        value_name='intensity',
    )
    df_decoded = df_decoded[columns]

    channel_order = {'G': 0, 'T': 1, 'A': 2, 'C': 3}
    df_decoded['_channel_sort'] = df_decoded['channel'].map(channel_order)
    df_decoded = df_decoded.sort_values(['read', 'cycle', '_channel_sort']).drop('_channel_sort', axis=1)
    return df_decoded.reset_index(drop=True)


def _corner_seeds(Xs):
    """Percentile 'corner' initial centroids (G, T, C, A) for a stable, mostly
    deterministic cluster fit in scaled (Cy3, Cy5) space."""
    lo = np.percentile(Xs, 10, axis=0)
    hi = np.percentile(Xs, 90, axis=0)
    return np.array([
        [lo[0], lo[1]],  # G  (both low)
        [hi[0], lo[1]],  # T  (Cy3 high, Cy5 low)
        [lo[0], hi[1]],  # C  (Cy3 low, Cy5 high)
        [hi[0], hi[1]],  # A  (both high)
    ])


# V2 DECODING APPROACH (Gaussian mixture in the 2D Cy3/Cy5 plane).
def decode_2color_gmm(df_bases, random_state=0):
    """Convert 2-color intensity data to 4-channel base calls with a GMM.

    Fits a 4-component Gaussian mixture in the (Cy3, Cy5) plane and maps clusters
    to bases by geometry (G both-low, A both-high, T Cy3-high, C Cy5-high).
    Pseudo-intensities are posterior responsibilities scaled to the tile's
    typical brightness, so the base call is argmax posterior, the runner-up base
    is nonzero, and the quality score reflects call confidence (the margin
    between the two most likely clusters) rather than raw brightness.
    """
    from sklearn.mixture import GaussianMixture

    df_pivot, id_cols = _pivot_2color(df_bases)
    X = df_pivot[['Cy3', 'Cy5']].to_numpy(dtype=float)

    # Scale each channel by its median so neither dominates the (Euclidean)
    # geometry; this folds gain balancing into the clustering.
    med = np.median(X, axis=0)
    scale = np.where(med > 0, med, 1.0)
    Xs = X / scale

    gmm = GaussianMixture(
        n_components=4,
        covariance_type='full',
        means_init=_corner_seeds(Xs),
        random_state=random_state,
    )
    gmm.fit(Xs)
    resp = gmm.predict_proba(Xs)
    comp_to_base = _map_components_to_bases(gmm.means_ * scale)

    # Constant per-tile brightness scale keeps pseudo-intensities on the same
    # order of magnitude as raw signal (so Q / q_min stay interpretable) without
    # letting dark 'G' reads collapse to zero.
    nz = X[X > 0]
    bright = np.median(nz) if nz.size else 1.0
    for base in ['G', 'T', 'A', 'C']:
        df_pivot[base] = resp[:, comp_to_base[base]] * bright

    return _assemble_decoded(df_pivot, id_cols, df_bases.columns)


# V3 DECODING APPROACH (k-means in the 2D Cy3/Cy5 plane).
def decode_2color_kmeans(df_bases, random_state=0):
    """Convert 2-color intensity data to 4-channel base calls with k-means.

    Fits k-means (k=4) in the (Cy3, Cy5) plane and maps clusters to bases by
    geometry (same convention as decode_2color_gmm). The base call is the nearest
    centroid (hard assignment). Pseudo-intensities are distance-based soft
    weights (a Gaussian kernel over the distance to each centroid) scaled to
    typical brightness, giving a nonzero runner-up so the quality score is not
    degenerate. The nearest centroid gets the largest weight, so argmax of the
    pseudo-intensities equals the k-means label.
    """
    from sklearn.cluster import KMeans

    df_pivot, id_cols = _pivot_2color(df_bases)
    X = df_pivot[['Cy3', 'Cy5']].to_numpy(dtype=float)

    med = np.median(X, axis=0)
    scale = np.where(med > 0, med, 1.0)
    Xs = X / scale

    km = KMeans(
        n_clusters=4,
        init=_corner_seeds(Xs),
        n_init=1,
        random_state=random_state,
    )
    km.fit(Xs)
    comp_to_base = _map_components_to_bases(km.cluster_centers_ * scale)

    # Soft weights from distances to each centroid; bandwidth = median nonzero
    # distance. Nearest centroid -> largest weight -> matches the k-means label.
    dist = km.transform(Xs)
    bw = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    w = np.exp(-(dist ** 2) / (2.0 * bw ** 2))
    w = w / w.sum(axis=1, keepdims=True)

    nz = X[X > 0]
    bright = np.median(nz) if nz.size else 1.0
    for base in ['G', 'T', 'A', 'C']:
        df_pivot[base] = w[:, comp_to_base[base]] * bright

    return _assemble_decoded(df_pivot, id_cols, df_bases.columns)


# # OLD DECODING APPROACH (per-cycle mean thresholds) — kept for reference.
# # To revert, comment out the relative-threshold decode_2color_to_bases above,
# # uncomment this function, and restore the matching call_reads block below.
# def decode_2color_to_bases(df_bases, cycle_thresholds, baseline):
#     """Convert 2-color intensity data to 4-channel base calls.
#
#     cycle_thresholds: dict of {cycle: {'Cy3': threshold, 'Cy5': threshold}}
#     """
#
#     id_cols = [col for col in df_bases.columns if col not in ['channel', 'intensity']]
#
#     df_pivot = df_bases.pivot_table(
#         index=id_cols,
#         columns='channel',
#         values='intensity',
#         aggfunc='first'
#     ).reset_index()
#
#     records = []
#     for idx, row in df_pivot.iterrows():
#         base_dict = {col: row[col] for col in id_cols}
#
#         # Look up per-cycle thresholds
#         cy3_threshold = cycle_thresholds[row['cycle']]['Cy3']
#         cy5_threshold = cycle_thresholds[row['cycle']]['Cy5']
#
#         m_on = row['Cy3'] >= cy3_threshold
#         c_on = row['Cy5'] >= cy5_threshold
#
#         if m_on and c_on:
#             intensities = {'A': max(row['Cy3'], row['Cy5']), 'T': baseline, 'C': baseline, 'G': 0}
#         elif m_on and not c_on:
#             intensities = {'A': baseline, 'T': row['Cy3'], 'C': 0, 'G': 0}
#         elif not m_on and c_on:
#             intensities = {'A': baseline, 'T': 0, 'C': row['Cy5'], 'G': 0}
#         else:
#             intensities = {'A': 0, 'T': 0, 'C': 0, 'G': max(row['Cy3'], row['Cy5'])}
#
#         for base in ['G', 'T', 'A', 'C']:
#             record = base_dict.copy()
#             record['channel'] = base
#             record['intensity'] = intensities[base]
#             records.append(record)
#
#     df_decoded = pd.DataFrame(records)
#     df_decoded = df_decoded[df_bases.columns]
#
#     channel_order = {'G': 0, 'T': 1, 'A': 2, 'C': 3}
#     df_decoded['_channel_sort'] = df_decoded['channel'].map(channel_order)
#     df_decoded = df_decoded.sort_values(['read', 'cycle', '_channel_sort']).drop('_channel_sort', axis=1)
#     df_decoded = df_decoded.reset_index(drop=True)
#
#     return df_decoded


def call_reads(
    bases_data,
    peaks_data=None,
    correction_only_in_cells=True,
    normalize_bases_first=False,
    method="median",
    relative_thresh=0.35,
):
    """Call reads for in situ sequencing data.

    Call reads by compensating for channel cross-talk and calling the base
    with the highest corrected intensity for each cycle.

    Args:
    bases_data : pandas DataFrame
        Table of base intensity for all candidate reads, output of extract_bases.

    peaks_data : None or numpy array, default None
        Peaks/local maxima score for each pixel (output of find_peaks) to be included
        in the df_reads table for downstream QC or other analysis. If None, does not include
        peaks scores in returned df_reads table.

    correction_only_in_cells : boolean, default True
        If True, restricts correction/compensation step to account only for reads that
        are within a cell, as defined by the cell segmentation mask passed into
        extract_bases. Often identified spots outside of cells are not true sequencing
        reads.

    normalize_bases_first : boolean, default True
        If True, normalizes the base intensities before performing median correction.
        Only applies when method="median".

    method : str, default "median"
        Method to use for correction. Options are "median" or "percentile".
        - "median": Uses median-based correction, performed independently for each tile.
        - "percentile": Uses percentile-based correction, performed independently for each tile.

    relative_thresh : float, default 0.2
        Per-read, per-channel relative on/off threshold used when decoding raw
        2-color (Cy3/Cy5) intensities into 4-channel (G/T/A/C) base calls. A
        channel is "on" for a cycle if its intensity >= relative_thresh * that
        channel's max intensity across all cycles for the read. Only applied when
        bases_data still contains raw Cy3/Cy5 channels (i.e. has not already been
        decoded upstream).

    Returns:
 df_reads : pandas DataFrame
        Table of all reads with base calls resulting from SBS compensation and related metadata.
    """

    # Decode raw 2-color (Cy3/Cy5) intensities into 4-channel (G/T/A/C) base calls
    # using a per-read, per-channel relative threshold (see decode_2color_to_bases). This only
    # runs when the input still contains raw Cy3/Cy5 channels — e.g. when called
    # from 4.run_sbs_phenotype.sh. When called from the 2.configure_sbs_params
    # notebook the data is already decoded, so this step is skipped automatically.
    if bases_data["channel"].isin(["Cy3", "Cy5"]).any():
        # ACTIVE = v1 (gain/crosstalk correction + pooled relative threshold) — BEST config.
        bases_data = correct_2color_channels(bases_data)
        print("\n\n---DECODER: v1 (corrected) relative threshold---", relative_thresh, '\n\n')
        bases_data = decode_2color_to_bases(bases_data, relative_thresh=relative_thresh)

        # # To switch decoders, comment out the two v1 lines above and uncomment ONE block:
        # # v0 (pooled relative threshold only, no correction):
        # bases_data = decode_2color_to_bases(bases_data, relative_thresh=relative_thresh)
        # # v2 (Gaussian mixture clustering):
        # bases_data = decode_2color_gmm(bases_data)
        # # v3 (k-means clustering):
        # bases_data = decode_2color_kmeans(bases_data)

    # # OLD DECODING APPROACH (per-cycle mean thresholds) — kept for reference.
    # # To revert: comment out the relative-threshold decode block above, restore
    # # the per-cycle-mean decode_2color_to_bases signature near the top of this
    # # file, and uncomment the block below.
    # # IMPORTANT NOTE: COMMENT OUT this section when running 2.configure_sbs_params
    # # notebook; UNCOMMENT when running 4.run_sbs_phenotype.sh script
    # # # Per-cycle median/mean for each channel
    # # cycle_thresholds = (
    # #     bases_data[bases_data['channel'].isin(['Cy3', 'Cy5'])]
    # #     .groupby(['cycle', 'channel'])['intensity']
    # #     .mean()
    # #     .unstack('channel')  # columns become Cy3, Cy5
    # #     .to_dict('index')  # {cycle: {'Cy3': val, 'Cy5': val}}
    # # )
    # # # Baseline: 10th percentile of all intensities across both channels and all cycles
    # # BASELINE = bases_data[bases_data['channel'].isin(['Cy3', 'Cy5'])]['intensity'].quantile(0.10)
    # #
    # # # Print for inspection
    # # for cycle, thrs in sorted(cycle_thresholds.items()):
    # #     print(f"Cycle {cycle}  Cy3: {thrs['Cy3']}  Cy5: {thrs['Cy5']}")
    # # print("BASELINE:", BASELINE)
    # #
    # # bases_data = decode_2color_to_bases(bases_data, cycle_thresholds, BASELINE)

    if bases_data.empty or (
        correction_only_in_cells and len(bases_data.query("cell > 0")) == 0
    ):
        # Get number of cycles from the most common value in the cycle column
        # Default to 0 if no cycles are found
        cycles = len(set(bases_data["cycle"])) if not bases_data.empty else 0

        # Create base columns that are always present
        base_columns = [
            "read",
            "cell",
            "i",
            "j",
            "tile",
            "well",
            "barcode",
        ]

        # Dynamically generate Q-score columns based on number of cycles
        q_columns = [f"Q_{i}" for i in range(cycles)]

        # Add Q_min and peak columns
        final_columns = base_columns + q_columns + ["Q_min", "peak"]

        return pd.DataFrame(columns=final_columns)

    cycles = len(set(bases_data["cycle"]))
    channels = len(set(bases_data["channel"]))

    # Choose the appropriate method for read calling
    if method == "median":
        if normalize_bases_first:
            # Clean up and normalize base intensities, then perform median calling
            df_reads = (
                bases_data.pipe(clean_up_bases)
                .pipe(normalize_bases)
                .pipe(
                    do_median_call,
                    cycles,
                    channels=channels,
                    correction_only_in_cells=correction_only_in_cells,
                )
            )
        else:
            # Clean up bases and perform median calling without normalization
            df_reads = bases_data.pipe(clean_up_bases).pipe(
                do_median_call,
                cycles,
                channels=channels,
                correction_only_in_cells=correction_only_in_cells,
            )
    elif method == "percentile":
        # Clean up bases and perform percentile calling
        df_reads = bases_data.pipe(clean_up_bases).pipe(
            do_percentile_call,
            cycles=cycles,
            channels=channels,
            correction_only_in_cells=correction_only_in_cells,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median' or 'percentile'.")

    # Include peaks scores if available
    if peaks_data is not None:
        i, j = df_reads[["i", "j"]].values.T
        df_reads["peak"] = peaks_data[i, j]

    return df_reads


def clean_up_bases(df_bases):
    """Sort DataFrame df_bases for pre-processing before dataframe_to_values.

    Args:
        df_bases (pandas.DataFrame): DataFrame containing raw base signal intensities.

    Returns:
        pandas.DataFrame: Sorted DataFrame.
    """
    # Sort DataFrame based on multiple columns
    return df_bases.sort_values([WELL, TILE, CELL, READ, CYCLE, CHANNEL])


def do_median_call(
    df_bases,
    cycles=12,
    channels=4,
    correction_quartile=0,
    correction_only_in_cells=False,
    correction_by_cycle=False,
):
    """Call reads from raw base signal using median correction.

    Args:
        df_bases (pandas.DataFrame): DataFrame containing raw base signal intensities.
        cycles (int): Number of sequencing cycles.
        channels (int): Number of sequencing channels.
        correction_quartile (int): Quartile used for correction.
        correction_only_in_cells (bool): Flag specifying whether correction is based on reads within cells or all reads.
        correction_by_cycle (bool): Flag specifying if correction should be done by cycle.

    Returns:
        pandas.DataFrame: DataFrame containing the called reads.
    """

    def correction(df, channels, correction_quartile, correction_only_in_cells):
        # Define the correction function
        if correction_only_in_cells:
            # Obtain transformation matrix W based on reads within cells
            X_ = dataframe_to_values(df.query("cell > 0"))
            _, W = transform_medians(
                X_.reshape(-1, channels), correction_quartile=correction_quartile
            )
            # Apply transformation to all data
            X = dataframe_to_values(df)
            Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
        else:
            # Apply correction to all data
            X = dataframe_to_values(df)
            Y, W = transform_medians(
                X.reshape(-1, channels), correction_quartile=correction_quartile
            )
        return Y, W

    # Apply correction either by cycle or to the entire dataset
    if correction_by_cycle:
        # Apply correction cycle by cycle
        Y = np.empty(df_bases.pipe(len), dtype=df_bases.dtypes["intensity"]).reshape(
            -1, channels
        )
        for cycle, (_, df_cycle) in enumerate(df_bases.groupby("cycle")):
            Y[cycle::cycles, :], _ = correction(
                df_cycle, channels, correction_quartile, correction_only_in_cells
            )
    else:
        # Apply correction to the entire dataset
        Y, W = correction(
            df_bases, channels, correction_quartile, correction_only_in_cells
        )

    # Call barcodes
    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads


def do_percentile_call(
    df_bases,
    cycles=12,
    channels=4,
    correction_only_in_cells=False,
):
    """Call reads from raw base signal using percentile-based correction.

    Args:
        df_bases (pandas.DataFrame): DataFrame containing raw base signal intensities.
        cycles (int): Number of sequencing cycles.
        channels (int): Number of sequencing channels.
        correction_only_in_cells (bool): Flag specifying whether correction is based on reads within cells or all reads.

    Returns:
        pandas.DataFrame: DataFrame containing the called reads.
    """
    if correction_only_in_cells:
        # First obtain transformation matrix W
        X_ = dataframe_to_values(df_bases.query("cell > 0"))
        _, W = transform_percentiles(X_.reshape(-1, channels))

        # Then apply to all data
        X = dataframe_to_values(df_bases)
        Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
    else:
        X = dataframe_to_values(df_bases)
        Y, W = transform_percentiles(X.reshape(-1, channels))

    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads


def dataframe_to_values(df, value="intensity"):
    """Convert a sorted DataFrame containing intensity values into a 3D NumPy array.

    Args:
        df (pandas.DataFrame): DataFrame containing intensity values.
        value (str): Column name containing the intensity values.

    Returns:
        numpy.ndarray: 3D NumPy array representing intensity values with dimensions N x cycles x channels.
    """
    # Calculate the number of cycles
    cycles = df[CYCLE].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)

    # Calculate the number of channels
    n_channels = len(df[CHANNEL].value_counts())

    # Reshape intensity values into a 3D array
    x = np.array(df[value]).reshape(-1, n_cycles, n_channels)

    return x


def transform_medians(X, correction_quartile=0):
    """Compute a linear transformation matrix based on the median values of maximum points along each dimension of X.

    Args:
        X (numpy.ndarray): Input array.
        correction_quartile (float): Quartile used for correction.

    Returns:
        numpy.ndarray: Transformed array Y.
        numpy.ndarray: Transformation matrix W.
    """

    def get_medians(X, correction_quartile):
        arr = []
        for i in range(X.shape[1]):
            max_spots = X[X.argmax(axis=1) == i]
            try:
                arr.append(
                    np.median(
                        max_spots[
                            max_spots[:, i]
                            >= np.quantile(max_spots, axis=0, q=correction_quartile)[i]
                        ],
                        axis=0,
                    )
                )
            except:
                arr.append(np.median(max_spots, axis=0))
        M = np.array(arr)
        return M

    # Compute medians and construct matrix M
    M = get_medians(X, correction_quartile).T
    # Normalize matrix M
    M = M / M.sum(axis=0)
    # Compute the inverse of M to obtain the transformation matrix W
    W = np.linalg.inv(M)
    # Apply transformation to X
    Y = W.dot(X.T).T.astype(int)
    return Y, W


def transform_percentiles(X):
    """For each dimension, find points where that dimension is >=95th percentile intensity.

    Use median of those points to define new axes.
    Describe with linear transformation W so that W * X = Y.

    Args:
        X (numpy.ndarray): Input array of intensity values.

    Returns:
        Y (numpy.ndarray): Transformed array.
        W (numpy.ndarray): Transformation matrix.
    """

    def get_percentiles(X):
        arr = []
        for i in range(X.shape[1]):
            # Calculate relative intensities by dividing by rowsums
            rowsums = np.sum(X, axis=1)[:, np.newaxis]
            X_rel = X / rowsums

            # Find 95th percentile of relative intensities for channel i
            perc = np.nanpercentile(X_rel[:, i], 95)

            # Select spots where channel i has high relative intensity
            high = X[X_rel[:, i] >= perc]

            # Take median of those high-intensity spots
            arr += [np.median(high, axis=0)]

        M = np.array(arr)
        return M

    # Get percentile-based matrix
    M = get_percentiles(X).T

    # Normalize columns
    M = M / M.sum(axis=0)

    # Compute inverse to get transformation matrix
    W = np.linalg.inv(M)

    # Apply transformation
    Y = W.dot(X.T).T.astype(int)

    return Y, W


def call_barcodes(df_bases, Y, cycles=12, channels=4):
    """Assign barcode sequences to reads based on the transformed base signal obtained from sequencing data.

    Args:
        df_bases (pandas DataFrame): DataFrame containing base signal information for each read.
        Y (numpy array): Transformed base signal reshaped into a suitable format for the calling process.
        cycles (int): Number of sequencing cycles.
        channels (int): Number of sequencing channels.

    Returns:
        df_reads (pandas DataFrame): DataFrame with assigned barcode sequences and quality scores for each read.
    """
    # Extract unique bases
    bases = sorted(set(df_bases[CHANNEL]))

    # Check for weird bases
    if any(len(x) != 1 for x in bases):
        raise ValueError("supplied weird bases: {0}".format(bases))

    # Remove duplicate entries and create a copy for storing barcode calls
    df_reads = df_bases.drop_duplicates([WELL, TILE, READ]).copy()

    # Call barcodes based on the transformed base signal
    df_reads[BARCODE] = call_bases_fast(Y.reshape(-1, cycles, channels), bases)

    # Calculate quality scores for each read
    Q = quality(Y.reshape(-1, cycles, channels))

    # Store quality scores in DataFrame
    for i in range(len(Q[0])):
        df_reads["Q_%d" % i] = Q[:, i]

    # Assign minimum quality score for each read
    df_reads = df_reads.assign(Q_min=lambda x: x.filter(regex="Q_\d+").min(axis=1))

    # Drop unnecessary columns
    df_reads = df_reads.drop([CYCLE, CHANNEL, INTENSITY], axis=1)

    return df_reads


def call_bases_fast(values, bases):
    """Call bases based on the maximum intensity value for each cycle/channel combination.

    Args:
        values (numpy array): 3D array containing intensity values for each cycle, channel, and base.
        bases (str): String containing the base symbols corresponding to each channel.

    Returns:
        calls (list of str): List of called bases for each read.
    """
    # Check dimensions and base length
    assert values.ndim == 3
    assert values.shape[2] == len(bases)

    # Determine the index of the maximum intensity value for each cycle/channel
    calls = values.argmax(axis=2)

    # Map the index to the corresponding base symbol
    calls = np.array(list(bases))[calls]

    # Combine the base symbols for each cycle to form the called bases for each read
    return ["".join(x) for x in calls]


def quality(X):
    """Calculate quality scores for base calling.

    Quality score is based on the highest and second-highest intensity channels.
    Calculated as 1 - [log2(2 + second) / log2(2 + first)].

    Args:
        X (numpy array): Array containing intensity values.

    Returns:
        Q (numpy array): Array containing quality scores.
    """
    # Sort the intensity values and convert to float
    X = np.abs(np.sort(X, axis=-1).astype(float))

    # Calculate the quality scores
    Q = 1 - np.log(2 + X[..., -2]) / np.log(2 + X[..., -1])

    # Clip the quality scores to the range [0, 1]
    Q = (Q * 2).clip(0, 1)

    return Q


def normalize_bases(df):
    """Normalize the channel intensities by the median brightness of each channel in all spots.

    Args:
        df (pandas.DataFrame): DataFrame containing spot intensity data.

    Returns:
        pandas.DataFrame: DataFrame with normalized intensity values.
    """
    # Calculate median brightness of each channel
    df_medians = df.groupby("channel").intensity.median()

    # Normalize intensity values by dividing by respective channel median
    df_out = df.copy()
    df_out.intensity = df.apply(lambda x: x.intensity / df_medians[x.channel], axis=1)

    return df_out


def plot_normalization_comparison(
    df_bases,
    base_pairs=[("C", "A"), ("T", "G")],
    filter_to_compared_bases=True,
    base_order=["G", "T", "A", "C"],
):
    """Compare raw, median and percentile normalization with all cycles combined.

    Args:
        df_bases: DataFrame containing raw base intensities from extract_bases
        base_pairs: List of tuples representing base pairs to compare
        filter_to_compared_bases: If True, only show points where one of the compared bases was called
        base_order: List specifying the order of bases in the channel dimension (default: ["G", "T", "A", "C"])

    """
    # First, clean up the bases data
    df_bases_clean = clean_up_bases(df_bases)

    # Create figure with subplots - 3 rows (normalization methods) x 2 columns (base pairs)
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    # Get channel indices based on the specified base order
    channel_to_idx = {base: idx for idx, base in enumerate(base_order)}

    # Define base colors - keep the original colors
    base_colors = {"G": "purple", "T": "cyan", "A": "green", "C": "red"}

    # Process the raw data
    for bp_idx, (base1, base2) in enumerate(base_pairs):
        # Filter the data for the relevant base pairs
        df_base1 = df_bases_clean[df_bases_clean["channel"] == base1][
            ["read", "cycle", "intensity"]
        ]
        df_base2 = df_bases_clean[df_bases_clean["channel"] == base2][
            ["read", "cycle", "intensity"]
        ]

        # Merge the two dataframes to get paired intensities
        df_merged = pd.merge(
            df_base1,
            df_base2,
            on=["read", "cycle"],
            suffixes=("_" + base1, "_" + base2),
        )

        # Get max intensities for base calling coloring
        all_bases = base_order.copy()  # Use the custom base order
        df_max = None

        for base in all_bases:
            df_temp = df_bases_clean[df_bases_clean["channel"] == base][
                ["read", "cycle", "intensity"]
            ]
            df_temp = df_temp.rename(columns={"intensity": "intensity_" + base})

            if df_max is None:
                df_max = df_temp
            else:
                df_max = pd.merge(df_max, df_temp, on=["read", "cycle"])

        # Determine which base has max intensity for each read/cycle
        for base in all_bases:
            if base == all_bases[0]:
                df_max["max_base"] = base
                df_max["max_intensity"] = df_max["intensity_" + base]
            else:
                mask = df_max["intensity_" + base] > df_max["max_intensity"]
                df_max.loc[mask, "max_base"] = base
                df_max.loc[mask, "max_intensity"] = df_max["intensity_" + base]

        # Merge with the paired data
        df_merged = pd.merge(
            df_merged, df_max[["read", "cycle", "max_base"]], on=["read", "cycle"]
        )

        # Plot raw data (first row)
        ax = axes[0, bp_idx]

        # Filter to only show the compared bases if requested
        if filter_to_compared_bases:
            plot_bases = [base1, base2]
        else:
            plot_bases = all_bases

        for base in plot_bases:
            mask = df_merged["max_base"] == base
            if mask.any():
                ax.scatter(
                    df_merged.loc[mask, "intensity_" + base2],
                    df_merged.loc[mask, "intensity_" + base1],
                    color=base_colors[base],
                    alpha=0.5,
                    s=10,
                    label=f"'{base}' base call",
                )

        title_suffix = " (filtered)" if filter_to_compared_bases else ""
        ax.set_title(f"All Cycles: {base1}-{base2} (Raw){title_suffix}")
        ax.set_xlabel(base2)
        ax.set_ylabel(base1)

        # Now process for median and percentile normalization
        # First get all base intensities for each read/cycle
        data_all = df_bases_clean.pivot_table(
            index=["read", "cycle"],
            columns="channel",
            values="intensity",
            aggfunc="first",
        ).reset_index()

        # Make sure all bases are present in data_all
        for base in all_bases:
            if base not in data_all.columns:
                data_all[base] = 0

        # Convert to numpy array for normalization functions (ensure correct order)
        X = data_all[all_bases].values

        # Apply median normalization
        # First normalize bases
        df_bases_norm = normalize_bases(df_bases_clean)
        X_norm = dataframe_to_values(df_bases_norm)
        Y_median, W_median = transform_medians(X_norm.reshape(-1, len(all_bases)))

        # Apply percentile normalization
        Y_percentile, W_percentile = transform_percentiles(
            X.reshape(-1, len(all_bases))
        )

        # Add normalized values back to the dataframe
        data_all["median_" + base1] = Y_median[:, channel_to_idx[base1]]
        data_all["median_" + base2] = Y_median[:, channel_to_idx[base2]]
        data_all["percentile_" + base1] = Y_percentile[:, channel_to_idx[base1]]
        data_all["percentile_" + base2] = Y_percentile[:, channel_to_idx[base2]]

        # Determine max channel after normalization
        for method in ["median", "percentile"]:
            values = Y_median if method == "median" else Y_percentile
            max_indices = np.argmax(values, axis=1)
            data_all[method + "_max_base"] = [all_bases[i] for i in max_indices]

        # Plot median normalized data (second row)
        ax = axes[1, bp_idx]

        # Filter to only show the compared bases if requested
        if filter_to_compared_bases:
            plot_bases = [base1, base2]
        else:
            plot_bases = all_bases

        for base in plot_bases:
            mask = data_all["median_max_base"] == base
            if mask.any():
                ax.scatter(
                    data_all.loc[mask, "median_" + base2],
                    data_all.loc[mask, "median_" + base1],
                    color=base_colors[base],
                    alpha=0.5,
                    s=10,
                    label=f"'{base}' base call",
                )

        # Add diagonal reference line
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.7)

        ax.set_title(f"All Cycles: {base1}-{base2} (Median){title_suffix}")
        ax.set_xlabel(base2)
        ax.set_ylabel(base1)

        # Plot percentile normalized data (third row)
        ax = axes[2, bp_idx]

        # Filter to only show the compared bases if requested
        if filter_to_compared_bases:
            plot_bases = [base1, base2]
        else:
            plot_bases = all_bases

        for base in plot_bases:
            mask = data_all["percentile_max_base"] == base
            if mask.any():
                ax.scatter(
                    data_all.loc[mask, "percentile_" + base2],
                    data_all.loc[mask, "percentile_" + base1],
                    color=base_colors[base],
                    alpha=0.5,
                    s=10,
                    label=f"'{base}' base call",
                )

        # Add diagonal reference line
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.7)

        ax.set_title(f"All Cycles: {base1}-{base2} (Percentile){title_suffix}")
        ax.set_xlabel(base2)
        ax.set_ylabel(base1)

    # Create a shared legend at the bottom
    # Choose which bases to show in the legend based on filter setting
    if filter_to_compared_bases:
        # Get unique bases from all base pairs
        legend_bases = set()
        for base1, base2 in base_pairs:
            legend_bases.add(base1)
            legend_bases.add(base2)
        legend_bases = sorted(legend_bases)
    else:
        legend_bases = sorted(all_bases)

    # Create legend elements
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=base_colors[base],
            markersize=10,
            label=f"'{base}' base call",
        )
        for base in legend_bases
    ]

    # Add diagonal line to legend if present in plots
    legend_elements.append(
        Line2D([0], [0], linestyle="--", color="k", label="Diagonal reference (y=x)")
    )

    # Add legend at the bottom
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_elements),
        fontsize=12,
    )

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
