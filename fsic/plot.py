import typing as ty

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

ACTIVE_COLOR = "orange"
PASSIVE_COLOR = "blue"
palette = {"active": ACTIVE_COLOR, "passive": PASSIVE_COLOR}


def _plot(**kwargs):
    if len(kwargs["data"]) == 0:
        return None
    sns.set(style='ticks',font_scale=0.7)
    ax = sns.lineplot(
        # color=ACTIVE_COLOR
        **kwargs
    )

    # xticker = ticker.EngFormatter(sep="", places=1)
    # ax.xaxis.set_major_formatter(
    #     ticker.FuncFormatter(
    #         lambda x, pos: xticker.format_eng(
    #             (kwargs["data"].timesteps + 1).unique()[pos]
    #         )
    #     )
    # )

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
    return ax


def absolute(
    df: pd.DataFrame,
    y_axis: str = "length",
    separate_by: ty.Optional[ty.Dict[str, str]] = None,
):
    g = sns.FacetGrid(
        df.reset_index().replace({"active": {1.0: "active", 0.0: "passive"}}),
        **(separate_by or dict(col="buffer_size")),
    )
    g.map_dataframe(
        _plot,
        x="timesteps",
        y=y_axis,
        hue="active",
        palette=palette,
    )
    g.set_ylabels(label="average_length")
    # g.add_legend()


def percentage_optimal(
    df: pd.DataFrame,
    greater_than_eq_is_optimal: float,
    y_axis: str = "length",
    separate_by: ty.Optional[ty.Dict[str, str]] = None,
):
    separate_by = separate_by or dict(col="buffer_size", row="train_interval")
    combined_df = None
    for x in df[separate_by["col"]].unique():
        for y in df[separate_by["row"]].unique():
            _df = df[((df[separate_by["col"]] == x) &
                      (df[separate_by["row"]] == y))]
    
            for a in [True]:
            # for a in [True,False]:
                _adf = _df[_df["active"] == a]

                _bdf = (
                    _adf.groupby("timesteps")
                    .apply(lambda d: (d[y_axis].ge(greater_than_eq_is_optimal).sum()) / (d.shape[0]))
                    .to_frame("percentage_optimal")
                )
                
                for col in _adf.columns:
                    if col not in _bdf.columns and len(_adf[col].unique()) == 1:
                        _bdf[col] = _adf[col].iloc[0]

                print("_bdf")
                print(_bdf.columns)

                combined_df = (
                    _bdf
                    if combined_df is None
                    else pd.concat([combined_df, _bdf.copy()])
                )
    assert combined_df is not None

    print(combined_df.columns)
    print(combined_df.head())

    g = sns.FacetGrid(
        combined_df.reset_index().replace(
            {"active": {1.0: "active", 0.0: "passive"}}),
        **separate_by,
    )
    g.map_dataframe(
        _plot,
        x="timesteps",
        y="percentage_optimal",
        hue="active",
        palette=palette,
    )
    # g.add_legend()

def success_rate(
    df: pd.DataFrame,
    greater_than_eq_is_optimal: float,
    y_axis: str = "failure",
    separate_by: ty.Optional[ty.Dict[str, str]] = None,
):
    separate_by = separate_by or dict(col="buffer_size", row="train_interval")
    combined_df = None
    for x in df[separate_by["col"]].unique():
        for y in df[separate_by["row"]].unique():
            _df = df[((df[separate_by["col"]] == x) &
                      (df[separate_by["row"]] == y))]
    
            for a in [True]:
            # for a in [True,False]:
                _adf = _df[_df["active"] == a]

                _bdf = (
                    _adf.groupby("timesteps")
                    .apply(lambda d: (d[y_axis].eq(False).sum()) / d.shape[0])
                    .to_frame("success_rate")
                )

                for col in _adf.columns:
                    if col not in _bdf.columns and len(_adf[col].unique()) == 1:
                        _bdf[col] = _adf[col].iloc[0]
                combined_df = (
                    _bdf
                    if combined_df is None
                    else pd.concat([combined_df, _bdf.copy()])
                )
    assert combined_df is not None


    combined_df = combined_df.reset_index().replace(
            {"active": {True: 1.0, False: 0.0}})

    print(combined_df.columns)
    print(combined_df.head())


    g = sns.FacetGrid(
        combined_df.reset_index().replace(
            {"active": {1.0: "active", 0.0: "passive"}}),
        **separate_by,
    )
    g.map_dataframe(
        _plot,
        x="timesteps",
        y="success_rate",
        hue="active",
        palette=palette,
    )

    g.set(ylim=(0,1))
    # g.add_legend()


def episode_lengths(df: pd.DataFrame,
    df_info=None,
    y_axis: str = "length",
    separate_by: ty.Optional[ty.Dict[str, str]] = None
    ):
    plt.style.use('_mpl-gallery')

    # make data
    x = df["timesteps"]
    y1 = df[f"{y_axis}_max"]
    y2 = df[f"{y_axis}_min"]

    # plot
    fig = plt.figure(dpi=400)
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.tick_params(axis='both', which='major', labelsize=4)
    # fig, ax = plt.subplots(dpi=120)

    if df_info is not None:
        ax.set_title(f'gamma: {df_info.iloc[0]["gamma"]} | min_samples_leaf: {df_info.iloc[0]["min_samples_leaf"]}',
        fontsize=5)

    ax.fill_between(x, y1, y2=y2, alpha=.2, linewidth=0, color="orange")
    # ax.set_xticklabels(df["timesteps"].unique(), fontsize=5,rotation=45, label="timesteps")
    # ax.set_yticks(range(0,3001,500), fontsize=5)

    ax.set_xlabel("timesteps", fontsize=5)
    ax.set_ylabel(y_axis, fontsize=5)
    ax.plot(x, df[f"{y_axis}_mean"], linewidth=1, color="orange")

    return ax

def successful_episodes(df: pd.DataFrame,
    df_info=None,
    y_axis: str = "success_rate",
    separate_by: ty.Optional[ty.Dict[str, str]] = None
    ):
    g = sns.FacetGrid(
        df.reset_index().replace(
            {"active": {1.0: "active", 0.0: "passive"}}),
        **separate_by,
    )
    g.map_dataframe(
        _plot,
        x="timesteps",
        y=y_axis,
        hue="active",
        palette=palette,
    )
    # g.add_legend()

def plot_eval_rewards(path: str, evaldf:pd.DataFrame, ts:int, eval_num:int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    eval = df[df.timestep==ts]
    eval = eval.iloc[eval_num]

    plt.style.use('_mpl-gallery')

    # make data
    x = range(len(eval.rewards))

    # plot
    fig = plt.figure(dpi=400)
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.tick_params(axis='both', which='major', labelsize=4)

    ax.plot(x, eval["rewards"], color="orange", linewidth=0.1)
    # ax.set_xticklabels(df["timesteps"].unique(), fontsize=5,rotation=45, label="timesteps")
    # ax.set_yticks(range(0,3001,500), fontsize=5)

    ax.set_xlabel("eval_length", fontsize=5)
    ax.set_ylabel("reward", fontsize=5)