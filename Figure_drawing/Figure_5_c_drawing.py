import matplotlib.pyplot as plt
import numpy as np

# Set a random seed for reproducibility
np.random.seed(0)

# X-axis values
x = np.array([2, 4, 6, 8])

# Y-axis values for the five subplots
y1 = np.random.rand(4) * 10  # MAE
y2 = np.random.rand(4) * 10  # RMSE
y3 = np.random.rand(4) * 100  # MAPE
y4 = np.random.rand(4) * 1e8  # FLOPs
y4_2 = np.random.rand(4) * 1e6  # Param
y5 = np.random.rand(4) * 1e7  # Latency
y5_2 = np.random.rand(4) * 1e5  # Peak Mem

# Style adjustments
plt.style.use('default')
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.edgecolor"] = 'black'
plt.rcParams["axes.linewidth"] = 1

fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)

# Plot data with black line color
axs[0].plot(x, y1, marker='D', color='black')
axs[1].plot(x, y2, marker='o', color='black')
axs[2].plot(x, y3, marker='s', color='black')

# Bar charts with two y-axis
axs3_2 = axs[3].twinx()  # instantiate a second y-axis that shares the same x-axis
axs4_2 = axs[4].twinx()  # instantiate a second y-axis that shares the same x-axis

axs[3].bar(x - 2, y4, width=4, color='darkgrey', label='FLOPs')
axs3_2.bar(x + 2, y4_2, width=4, color='lightgrey', label='Param')

axs[4].bar(x - 2, y5, width=4, color='darkgrey', label='Latency')
axs4_2.bar(x + 2, y5_2, width=4, color='lightgrey', label='Peak Mem')

# Set y labels
axs[0].set_ylabel('MAE')
axs[1].set_ylabel('RMSE')
axs[2].set_ylabel('MAPE (%)')
axs[3].set_ylabel('FLOPs(M)')
axs3_2.set_ylabel('Param(K)')
axs[4].set_ylabel('Latency(s)')
axs4_2.set_ylabel('Peak Mem(Mb)')

# Set x labels
axs[4].set_xlabel('Attention Block Number L_S')

# Add legends for bar plots
axs[3].legend(loc='upper left')
axs3_2.legend(loc='upper right')
axs[4].legend(loc='upper left')
axs4_2.legend(loc='upper right')

# Adjust space between plots
fig.subplots_adjust(hspace=0.4)

plt.show()
