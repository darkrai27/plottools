import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

initial_punishment = 0
gamma = 0.99
punishment_per_step = -2
terminal_punishment = initial_punishment
steps = []
punishments = []
data = []
for t in range(0, 751):
    data.append({t:terminal_punishment})
    steps.append(t)
    punishments.append(terminal_punishment)
    terminal_punishment = (
                            punishment_per_step +
                            gamma * terminal_punishment
                        )

# fig = plt.figure(dpi=500)
# ax = fig.add_axes([0.2,0.2,0.7,0.7])
data = pd.DataFrame({'timesteps':steps,'terminal_punishment':punishments})
print(data.head())
# sns.lineplot(data=data, x='timesteps', y='terminal_punishment')
ax = sns.FacetGrid(data.reset_index())
ax.map(sns.lineplot, "timesteps", "terminal_punishment")

# ax.plot(steps, punishments)
# ax.set_xlabel("train_steps")
# ax.set_ylabel("terminal_punishment")
plt.savefig("terminal_asynt.png", dpi=350)