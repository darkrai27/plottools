from tkinter import E
import matplotlib.pyplot as plt
import os
import argparse

from fsic import convert, io, merge, plot, query, transform


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="path to rewards file")
parser.add_argument("--eval_num", type=int, required=True, help="evaluation number at timestep t to plot")
parser.add_argument("--ts", type=int, required=True, help="timestep for the evaluation to plot")

opts = parser.parse_args()

assert os.path.isdir(opts.path)

print(opts.path)
eval_file = f"{opts.path}/eval.parquet"
rewards_file = f"{opts.path}/rewards.parquet"
assert os.path.isfile(eval_file)
assert os.path.isfile(rewards_file)

# Convert raw dataframes into pairs of dataframes (more flexible) and resolution tables.
# The resolution table contains information, like meta-data, on the experiments.
# Later we will add this information, e.g., buffer capacity, as a column to the dataframe.
converted = convert.convert_iter([opts.path])

# Merge all selected experiments into a single dataframe
df, res = merge.merge_iter(converted)

plot.plot_eval_rewards(rewards_file, df, opts.ts, opts.eval_num)


plt.savefig(f"{opts.path}/timestep={opts.ts}eval={opts.eval_num}_rewards.png", dpi=400)