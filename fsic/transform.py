import typing as ty

import pandas as pd
import numpy as np


def get_unique_indices(df: pd.DataFrame, index_id="uid") -> ty.Set[ty.Any]:
    return set(df[index_id].unique())


def add_column_from_map(
    df: pd.DataFrame, map: ty.Callable, new_column: str, index_id: str = "uid"
):
    index = get_unique_indices(df, index_id)
    index_map = {idx: map(idx) for idx in index}
    _df = df.copy()
    _df.insert(
        loc=0,
        column=new_column,
        value=df[index_id].map(index_map),
    )
    return _df


def get_experiment(resolution: ty.Dict, uid) -> ty.Dict:
    if uid not in resolution.keys():
        raise KeyError()
    v = resolution[uid]
    return v["meta"]["experiment"]


def add_active(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(df, lambda x: res[x]["training_pass"] == 0, "active")


def add_active_seed(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(df, lambda x: res[x]["meta"]["seed"], "active_seed")


def add_own_seed(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df, lambda x: res[x]["meta"]["seed"] +
        res[x]["training_pass"], "own_seed"
    )


def add_buffer_size(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["buffer_size"],
        "buffer_size",
    )


def add_batch_size(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["batch_size"],
        "batch_size",
    )


def add_train_interval(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["train_interval"],
        "train_interval",
    )


def add_learning_rate(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["learning_rate"],
        "learning_rate",
    )


def add_optimizer(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["optimizer"],
        "optimizer",
    )


def add_gradient_steps(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["gradient_steps"],
        "gradient_steps",
    )


def add_epsilon(df: pd.DataFrame, res: ty.Dict):
    try:
        return add_column_from_map(
            df,
            lambda x: get_experiment(res, x)["config"]["train"]["exploration"][
                "Exponential"
            ]["init"],
            "epsilon",
        )
    except:
        return add_column_from_map(
            df,
            lambda x: get_experiment(res, x)["config"]["train"]["exploration"][
                "Linear"
            ]["start"],
            "epsilon",
        )

def add_gamma(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["train"]["gamma"],
        "gamma",
    )

def add_eval_iv(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["eval"]["eval_interval"],
        "eval_interval",
    )

def add_num_evals(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["eval"]["n_concurrent"],
        "n_concurrent_evals",
    )

def add_forest_leaves(df: pd.DataFrame, res: ty.Dict):
    return add_column_from_map(
        df,
        lambda x: get_experiment(res, x)["config"]["forest"]["min_samples_leaf"],
        "min_samples_leaf",
    )


def augment_all(df: pd.DataFrame, resolution: ty.Dict) -> pd.DataFrame:
    fns = [
        add_active,
        add_active_seed,
        add_own_seed,
        add_buffer_size,
        add_batch_size,
        add_train_interval,
        add_learning_rate,
        add_optimizer,
        add_gradient_steps,
        add_epsilon,
        add_gamma,
        add_eval_iv,
        add_num_evals,
        add_forest_leaves,
        # Add additional column augmentation functions here
    ]
    for fn in fns:
        df = fn(df, resolution)
    return df


def aggregate(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return df.groupby(["uid", "pass_idx", "timesteps"])


def aggregate_mean(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df).mean()


def aggregate_median(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df).median()

def aggregate_list(df:pd.DataFrame)-> pd.DataFrame:
    # print(df)
    return pd.DataFrame(aggregate(df).length.apply(list))
    # return pd.DataFrame(aggregate(df).length.values)
    


def aggregate_single(df: pd.DataFrame) -> pd.DataFrame:
    newdf = []
    for ts in df.timesteps.unique():
        data = df[df.timesteps == ts]
        # print(df.length0.mean())
        rate = len(data)
        success = len(data[data.failure == False])
        newdf.append({'timesteps':ts, 'success':success, 'success_rate':success / rate, 'length_mean':data.length.mean(), 
        'length_min':data.length.min(), 'length_max':data.length.max(), 'length':data.length,
        'reward_mean':data.reward.mean(), 
        'reward_min':data.reward.min(), 'reward_max':data.reward.max(), 'reward':data.reward,
        'train_interval':df.iloc[0].train_interval,'buffer_size':df.iloc[0].buffer_size,
        'active':1.0})
    return pd.DataFrame(newdf)

def get_lengths(df: pd.DataFrame):
    newdf = []
    for ts in df.timesteps.unique():
        data = df[df.timesteps == ts]
        newdf.append(data.length.values)
    return newdf

def get_rewards(path: str, evaldf:pd.DataFrame) -> pd.DataFrame:
    newdf = []
    df = pd.read_parquet(path)
    for eval in df.iloc:
        newdf.append({'timesteps':eval.timestep, 'rewards_min':eval.rewards.min(),
        'rewards_max': eval.rewards.max(), 'rewards_mean': eval.rewards.mean(), 'rewards':eval.rewards,
        # 'rewards_median': eval.rewards.median(),
        'train_interval':evaldf.iloc[0].train_interval,
        'buffer_size':evaldf.iloc[0].buffer_size,
        'active':1.0})

    single_episodes = pd.DataFrame(newdf)

    newdf = []
    for ts in single_episodes.timesteps.unique():
        data = single_episodes[single_episodes.timesteps == ts]
        newdf.append({'timesteps':ts,
        'reward_min':data.rewards_min.min(), 'reward_max':data.rewards_max.max(), 'reward_mean':data.rewards_mean.mean(),
        'train_interval':evaldf.iloc[0].train_interval,'buffer_size':evaldf.iloc[0].buffer_size,
        'active':1.0})
    
    return single_episodes, pd.DataFrame(newdf)

# def get_rewards(path: str, evaldf:pd.DataFrame, ts:int, eval_num:int) -> pd.DataFrame:
    # newdf = []
    # df = pd.read_parquet(path)
    # eval = df[df.timesteps==ts]
    # eval = eval.iloc[eval_num]
    # for eval in df.iloc:
    #     newdf.append({'timesteps':eval.timestep, 'rewards_min':eval.rewards.min(),
    #     'rewards_max': eval.rewards.max(), 'rewards_mean': eval.rewards.mean(), 'rewards':eval.rewards,
    #     # 'rewards_median': eval.rewards.median(),
    #     'train_interval':evaldf.iloc[0].train_interval,
    #     'buffer_size':evaldf.iloc[0].buffer_size,
    #     'active':1.0})

    # single_episodes = pd.DataFrame(newdf)

    # newdf = []
    # for ts in single_episodes.timesteps.unique():
    #     data = single_episodes[single_episodes.timesteps == ts]
    #     newdf.append({'timesteps':ts,
    #     'reward_min':data.rewards_min.min(), 'reward_max':data.rewards_max.max(), 'reward_mean':data.rewards_mean.mean(),
    #     'train_interval':evaldf.iloc[0].train_interval,'buffer_size':evaldf.iloc[0].buffer_size,
    #     'active':1.0})
    
    # return single_episodes, pd.DataFrame(newdf)