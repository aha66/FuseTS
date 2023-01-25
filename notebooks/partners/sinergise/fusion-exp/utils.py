import json
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import GPy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from am.tseo.markers.mowing import MowingInvalidCastCols, MowingOutputCols, get_mowing_events
from pydantic import BaseModel, Field
from scipy.signal import savgol_filter

GP_KERNELS = {
    "ICM_Mat32": GPy.util.multioutput.ICM(input_dim=1, num_outputs=2, kernel=GPy.kern.Matern32(input_dim=1)),
    "ICM_Bias+Mat32": GPy.util.multioutput.ICM(
        input_dim=1, num_outputs=2, kernel=(GPy.kern.Bias(input_dim=1) + GPy.kern.Matern32(input_dim=1))
    ),
    "LCM_Bias+Mat32": GPy.util.multioutput.LCM(
        input_dim=1, num_outputs=2, kernels_list=[GPy.kern.Bias(input_dim=1), GPy.kern.Matern32(input_dim=1)]
    ),
}

# constrain LCM bias correlation to 0
GP_KERNELS["LCM_Bias+Mat32"]["ICM0.B.W"].constrain_fixed(0)


class ARDSchema(BaseModel):
    s1_path: str = Field(description="Path to S1 data")
    s2_path: str = Field(description="Path to S2 data")
    otsc_path: str = Field(description="Path to OTSC labels data")

    s2_valid_query_mode: Literal["loose", "strict"] = Field(
        "loose", description='Valid query mode for filtering S2 time series ("loose" or "strict")'
    )

    min_foi_size: int = Field(description="Minimum size of the parcel")
    min_foi_nobs: int = Field(description="Minimum amount of valid optical observations")
    lu_label: int = Field(description="LU Label to keep")


def load_ard_data(config: ARDSchema) -> Tuple[pd.DataFrame, pd.DataFrame]:

    labels = pd.read_excel(config.otsc_path).set_index("Field_id").dropna(subset=["Comment"])
    labels = labels[
        (labels.groupby("Field_id").Comment.transform("count") == 1)  # single comment per FOI
        & (labels["Comment"].str.lower().isin(["mowed", "unmowed"]))  # only mowed and unmowed
        & (labels[" Land use code"] == config.lu_label)  # only meadows
    ].copy()

    # load S1 6D Coherence and filter
    df_s1 = pd.read_parquet(
        config.s1_path,
        columns=["TIMESTAMP", "POLY_ID", "VV", "VH"],
        filters=[("LU_LABEL", "=", "PR"), ("MASK", "=", 1.0)],
    )

    # load S2 and filter
    s2_valid_query = "(CLP <= 0.54 or CLM <= 0.99) and OUT_PROBA <= 0.5"
    if config.s2_valid_query_mode == "strict":
        s2_valid_query = "(CLP <= 0.99 or CLM <= 0.91) and OUT_PROBA < 0.125"

    df_s2 = (
        pd.read_parquet(
            config.s2_path,
            columns=["TIMESTAMP", "POLY_ID", "NDVI", "CLP", "CLM", "OUT_PROBA", "LU_LABEL"],
            filters=[
                ("LU_LABEL", "=", str(config.lu_label)),
                ("N_PIXEL", ">", config.min_foi_size),  # keep only meadows and large FOIs
            ],
        )
        .query(s2_valid_query)
        .drop(columns=["CLP", "CLM", "OUT_PROBA", "LU_LABEL"])
    )

    # add DOY
    df_s1["DOY"] = df_s1.TIMESTAMP.dt.dayofyear
    df_s2["DOY"] = df_s2.TIMESTAMP.dt.dayofyear

    # keep only FOIs with at least 15 optical observations
    s2t = df_s2.groupby("POLY_ID")["TIMESTAMP"].transform("count")
    df_s2 = df_s2[(s2t > config.min_foi_nobs)]

    # intersect available FOIs
    pids = list(set(df_s1.POLY_ID.unique()) & set(df_s2.POLY_ID.unique()))
    df_s1 = df_s1[df_s1.POLY_ID.isin(pids)].copy()
    df_s2 = df_s2[df_s2.POLY_ID.isin(pids)].copy()

    # calculate RVI
    df_s1["RVI"] = (df_s1.VH + df_s1.VH) / (df_s1.VV + df_s1.VH)

    # append mowing label info
    labeled_fois = labels.index.unique()
    df_s1["MOWING_LABEL"] = df_s1.POLY_ID.map(
        lambda pid: -1 if pid not in labeled_fois else 1 if labels.loc[pid].Comment.lower() == "mowed" else 2
    )
    df_s2["MOWING_LABEL"] = df_s2.POLY_ID.map(
        lambda pid: -1 if pid not in labeled_fois else 1 if labels.loc[pid].Comment.lower() == "mowed" else 2
    )

    return df_s1, df_s2


def prepare_timeseries(series: pd.Series, window: Optional[int], order: int = 1) -> pd.Series:
    series = series.drop(columns="POLY_ID").set_index("TIMESTAMP")
    dates = series.index.values
    series = series.resample("1D").interpolate()
    series = series.apply(savgol_filter, window_length=window, polyorder=order)
    return series.loc[dates]


def create_groups(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    group_size: int,
    n_groups: int,
    sw: Optional[int],
    normalize: bool,
    labels: List[int] = [-1],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:

    df_s1 = df_s1.copy()
    df_s2 = df_s2.copy()

    if normalize:
        for col in ["VV", "VH", "RVI"]:
            m = df_s1.groupby("POLY_ID")[col].mean().mean()
            s = df_s1.groupby("POLY_ID")[col].std().mean()
            df_s1[col] = (df_s1[col] - m) / s

        col = "NDVI"
        m = df_s2.groupby("POLY_ID")[col].mean().mean()
        s = df_s2.groupby("POLY_ID")[col].std().mean()
        df_s2[col] = (df_s2[col] - m) / s

    pids = df_s1.POLY_ID.unique()
    pid_list = np.random.choice(pids, group_size * n_groups, replace=False)
    s1p_coh = df_s1[(df_s1.POLY_ID.isin(pid_list)) & (df_s1.MOWING_LABEL.isin(labels))].sort_values(
        ["POLY_ID", "TIMESTAMP"]
    )
    s2p = df_s2[(df_s2.POLY_ID.isin(pid_list)) & (df_s2.MOWING_LABEL.isin(labels))].sort_values(
        ["POLY_ID", "TIMESTAMP"]
    )
    if sw is not None:
        s1p_coh = s1p_coh.groupby("POLY_ID", group_keys=True).apply(prepare_timeseries, window=sw).reset_index()

    return s1p_coh, s2p, pid_list.reshape(n_groups, group_size)


def get_group(
    idx: int, dfp_s1: pd.DataFrame, dfp_s2: pd.DataFrame, pid_array: np.ndarray, var: str
) -> Tuple[List[np.ndarray], ...]:
    subset = pid_array[idx]
    sub_s1, sub_s2 = (
        dfp_s1[dfp_s1.POLY_ID.isin(subset)].copy(),
        dfp_s2[dfp_s2.POLY_ID.isin(subset)].copy(),
    )

    sub_s1["N_GROUP"] = sub_s1.groupby("POLY_ID").ngroup()
    sub_s1["NX"] = sub_s1.TIMESTAMP.dt.dayofyear + sub_s1["N_GROUP"] * 365

    sub_s2["N_GROUP"] = sub_s2.groupby("POLY_ID").ngroup()
    sub_s2["NX"] = sub_s2.TIMESTAMP.dt.dayofyear + sub_s2["N_GROUP"] * 365

    X1 = sub_s1.NX.values[..., np.newaxis]
    Y1 = sub_s1[var].values[..., np.newaxis]
    X2 = sub_s2.NX.values[..., np.newaxis]
    Y2 = sub_s2.NDVI.values[..., np.newaxis]

    return [X1, X2], [Y1, Y2]


class OptimizationSchema(BaseModel):
    num_restarts: int = Field(10, description="How many times to run a single run. Returns the best run")
    num_processes: int = Field(1, description="Whether to use paralelization for retries (> 1) or not (default)")
    optimizer: Optional[str] = Field(description="Which optimizer to use")


def optimize_single(
    x: List[np.ndarray], y: List[np.ndarray], kernel_name: str, config: OptimizationSchema
) -> Tuple[Any, Dict[str, Any]]:

    m = GPy.models.GPCoregionalizedRegression(x, y, kernel=GP_KERNELS[kernel_name].copy())
    params_to_ignore = [name for name in m.parameter_names() if m[name].is_fixed]

    settings = dict(messages=False, robust=True, verbose=False, **dict(config))
    if config.num_restarts == 1:
        m.optimize_restarts(parallel=False, **settings)
    else:
        m.optimize_restarts(parallel=True, **settings)

    params = {}
    for name in set(m.parameter_names()) - set(params_to_ignore):
        if any(name.endswith(substr) for substr in ["B.W", "B.kappa"]):
            params[name] = m[name].values.tolist()
        else:
            params[name] = m[name].values.tolist()[0]

    params["obj_func"] = m.objective_function()

    return m, params


class ExperimentSchema(BaseModel):
    s1_var: str = Field(description="S1 variable to work with")
    s1_smoothing: Optional[int] = Field(description="How much Savitzky-Golay smoothing to apply on S1 data")
    normalize: bool = Field(
        description="Whether to normalize the time series for the GP process (global mean/std normalization)"
    )
    group_size: int = Field(1, description="How many FOIs to stack in a single time series")
    n_groups: int = Field(1, description="How many FOI stacks to collect")
    kernel_name: str = Field(description="Name of the predefined kernel structure")
    opt_config: OptimizationSchema

    save_to: Optional[str] = Field(description="Path where to dump the parameters of the experiments and the config")


def run_experiment(df_s1: pd.DataFrame, df_s2: pd.DataFrame, config: ExperimentSchema) -> List[Dict[str, Any]]:
    s1p_coh, s2p, pid_array = create_groups(
        df_s1, df_s2, config.group_size, config.n_groups, normalize=config.normalize, sw=config.s1_smoothing
    )

    start = time.time()
    params_list = []
    for gdx in range(config.n_groups):
        x, y = get_group(gdx, s1p_coh, s2p, pid_array, config.s1_var)
        _, params = optimize_single(x, y, config.kernel_name, config.opt_config)
        params_list.append(params)

    end = time.time() - start
    metainfo = {
        "tot_time_s": end,
        "tot_time_per_group_s": end / config.n_groups,
        "tot_time_per_foi_s": end / config.n_groups / config.group_size,
        "tot_n_foi": config.n_groups * config.group_size,
    }

    exp_name = f"exp_{config.s1_var}_sw_{config.s1_smoothing}_norm{int(config.normalize)}_g_{config.group_size}"
    exp_name += f"_{config.n_groups}_opt_{config.opt_config.optimizer}_k_{config.kernel_name}"

    if config.save_to is not None:
        exp_path = os.path.join(config.save_to, exp_name)
        os.makedirs(exp_path, exist_ok=True)
        json.dump(params_list, open(os.path.join(exp_path, "parameters.json"), "w"), indent=4)
        json.dump(config.dict(), open(os.path.join(exp_path, "config.json"), "w"), indent=4)
        json.dump(metainfo, open(os.path.join(exp_path, "meta.json"), "w"), indent=4)

    return params_list


def plot_2outputs(m, var="VH", ax1=None, ax2=None, color="C0", suffix="", label=None, pid=None):
    x1, x2 = m.X[m.X[:, 1] == 0, 0], m.X[m.X[:, 1] == 1, 0]
    y1, y2 = m.Y[m.X[:, 1] == 0, 0], m.Y[m.X[:, 1] == 1, 0]

    ylim = np.array([[y1.min(), y1.max()], [y2.min(), y2.max()]])
    ylim += 0.4 * np.stack([-np.diff(ylim, axis=1), np.diff(ylim, axis=1)], axis=-1).squeeze()

    if ax1 is None and ax2 is None:
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 8), nrows=2)
        fig.patch.set_alpha(1)

    # # Output 1
    ax1.set_title(f"COH_6D_{var}" + suffix + f", POLY_ID: {pid}" if pid is not None else "")
    ax1.scatter(m.X[m.X[:, 1] == 0, 0], m.Y[m.X[:, 1] == 0], color="k", marker="x")
    m.plot_mean(fixed_inputs=[(1, 0)], ax=ax1, color=color, label=label)
    m.plot_confidence(fixed_inputs=[(1, 0)], ax=ax1, color=color, alpha=0.2, label=None)
    ax1.legend()

    # Output 2
    ax2.set_title("NDVI" + suffix + f", POLY_ID: {pid}" if pid is not None else "")
    ax2.scatter(m.X[m.X[:, 1] == 1, 0], m.Y[m.X[:, 1] == 1], color="k", marker="x")
    m.plot_mean(fixed_inputs=[(1, 1)], ax=ax2, color=color, label=label)
    m.plot_confidence(fixed_inputs=[(1, 1)], ax=ax2, color=color, alpha=0.2, label=None)
    ax2.legend()

    return ax1, ax2


def run_mowing(foi_df: pd.DataFrame) -> List[Dict]:
    slope_thr = -0.007
    delta_thr = 0.1
    drop_r = 0.5
    grow_r = 1.0

    delta_thr, grow_thr, drop_thr, slope_thr = delta_thr, delta_thr * grow_r, delta_thr * drop_r, slope_thr

    result_list = get_mowing_events(
        pd.to_datetime(foi_df.TIMESTAMP.values),
        foi_df.NDVI.values,
        {**MowingOutputCols.defaults, "POLY_ID": foi_df.POLY_ID.unique()[0]},
        drop_thr=drop_thr,
        grow_thr=grow_thr,
        slope_thr=slope_thr,
        delta_thr=delta_thr,
    )

    return result_list


def postproc_mw_events(result_list: List[Dict]) -> pd.DataFrame:
    postproc_queries = [
        "STATUS == 0",
        "not (num_observations == 3 and duration + days_between_start_and_threshold < 9)",
        "not (ndvi_start < 0.506 and bottom_timestamp<'2020-05-01')",
        "event_threshold > '2020-04-01'",
    ]

    result = pd.DataFrame(result_list, columns=MowingOutputCols()).reset_index(drop=True)
    for col in ["event_start", "event_threshold", "event_end", "bottom_timestamp"]:
        result[col] = result[col].fillna(pd.NaT)

    # filter mowing events
    start, end = "2020-04-01", "2020-11-01"
    filtered = result.query(f"event_end >= '{start}' and event_threshold <= '{end}'").copy()

    for filter_query in postproc_queries:
        filtered = filtered.query(filter_query).copy()

    event_mask = result.index.isin(filtered.index)
    NULL_DATA = MowingInvalidCastCols.defaults
    result.loc[~event_mask, NULL_DATA.keys()] = NULL_DATA.values()

    valid_events = result[~result["event_start"].isna()]
    fois_with_valid = valid_events["POLY_ID"].unique()
    columns = list(set(result.columns) - {"MARKER_ID", "STATUS"})
    invalid_events = result[~result["POLY_ID"].isin(fois_with_valid)].drop_duplicates(subset=columns, keep="first")

    result = pd.concat([valid_events, invalid_events]).copy()
    result = result[MowingOutputCols()]
    result = result.astype(MowingOutputCols.dtypes).reset_index(drop=True).drop(columns=["H3HEX", "MARKER_ID"])

    return result
