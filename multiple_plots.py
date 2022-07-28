from tkinter import E
import matplotlib.pyplot as plt
import os

from fsic import convert, io, merge, plot, query, transform

PATH_TO_EXPERIMENTS = "C:\\Users\Z004HK5F\Desktop\Bachelor\doubleRegressor\LGBM\min_samples=5"
for experiment in os.listdir(PATH_TO_EXPERIMENTS):
    # Query / filter experiments by some callback.
    # Here we filter all experiment which have a experiment) name starting with a given prefix.
    if os.path.isdir(f"{PATH_TO_EXPERIMENTS}/{experiment}") and not os.path.exists(f"{PATH_TO_EXPERIMENTS}/{experiment}/evals_length.png") and os.path.exists(f"{PATH_TO_EXPERIMENTS}/{experiment}/meta.yaml"):
        experiment_paths = [f"{PATH_TO_EXPERIMENTS}/{experiment}"]   
    else:
        print(f"Skipping {experiment}")
        continue
    print(experiment_paths)

    # Convert raw dataframes into pairs of dataframes (more flexible) and resolution tables.
    # The resolution table contains information, like meta-data, on the experiments.
    # Later we will add this information, e.g., buffer capacity, as a column to the dataframe.
    converted = convert.convert_iter(experiment_paths)

    # Merge all selected experiments into a single dataframe
    df, res = merge.merge_iter(converted)

    # Add information from the resolution table (the contained meta-data) as rows to the dataframe.
    # `transform.augment_all()` is a helper that does this for all supported fields.
    # Needs to be updated when other fields are added!
    df = transform.augment_all(df, res)
    # Aggregate over all trials for every trainings (uid) at each timestep, while respecting the active/passive value.
    # This does not aggregate all trainings!
    df2 = transform.aggregate_mean(df)
    df3 = transform.aggregate_single(df)
    # df3 = transform.aggregate_list(df)
    

    # Save and load the dataframe & resolution table, it will be saved as two files, `.parquet` and `.res`, respectively.
    EXPORT_PATH = "./demo_data"
    
    plot.successful_episodes(df3,df, y_axis="success_rate", separate_by=dict(col="buffer_size", row="train_interval")
    )
    plt.savefig(f"{PATH_TO_EXPERIMENTS}/{experiment}/successful_evals.png", dpi=400)

    plot.episode_lengths(df3,df, y_axis="length", separate_by=dict(col="buffer_size", row="train_interval")
    )
    plt.savefig(f"{PATH_TO_EXPERIMENTS}/{experiment}/evals_length.png", dpi=400)

    plot.episode_lengths(df3,df, y_axis="reward", separate_by=dict(col="buffer_size", row="train_interval")
    )
    plt.savefig(f"{PATH_TO_EXPERIMENTS}/{experiment}/cumulated_reward.png", dpi=400)

    if os.path.isfile(f"{PATH_TO_EXPERIMENTS}/{experiment}/rewards.parquet"):
        rewards_per_eval, rewardsdf = transform.get_rewards(f"{PATH_TO_EXPERIMENTS}/{experiment}/rewards.parquet", df)

        plot.episode_lengths(rewardsdf,df, y_axis="reward", separate_by=dict(col="buffer_size", row="train_interval")
        )
        plt.savefig(f"{PATH_TO_EXPERIMENTS}/{experiment}/mean_reward.png", dpi=400)



