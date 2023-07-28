import matplotlib.pyplot as plt
import numpy as np

# Set a random seed for reproducibility
np.random.seed(0)

# X-axis values
x = np.array([1,2,4,8])
markers = ['o', 's', 'v', '^']

# Y-axis values for the four subplots
y1 = [np.random.rand(5) * 10 for _ in range(4)]  # RRSE for 4 horizons
y2 = [np.random.rand(5) * 10 for _ in range(4)]  # CORR for 4 horizons
y3 = np.random.rand(5) * 1e8  # FLOPs
y3_2 = np.random.rand(5) * 1e6  # Param
y4 = np.random.rand(5) * 1e7  # Latency
y4_2 = np.random.rand(5) * 1e5  # Peak Mem

# Style adjustments
plt.style.use('default')
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.edgecolor"] = 'black'
plt.rcParams["axes.linewidth"] = 1

fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

# Plot data with black line color
for i in range(4):
    axs[0].plot(x, y1[i], marker=markers[i], color='black')
    axs[1].plot(x, y2[i], marker=markers[i], color='black')

# Bar charts with two y-axis
axs2_2 = axs[2].twinx()  # instantiate a second y-axis that shares the same x-axis
axs3_2 = axs[3].twinx()  # instantiate a second y-axis that shares the same x-axis

axs[2].bar(x - 1, y3, width=2, color='darkgrey')
axs2_2.bar(x + 1, y3_2, width=2, color='lightgrey')

axs[3].bar(x - 1, y4, width=2, color='darkgrey')
axs3_2.bar(x + 1, y4_2, width=2, color='lightgrey')

# Set y labels
axs[0].set_ylabel('RRSE')
axs[1].set_ylabel('CORR')
axs[2].set_ylabel('FLOPs(M)')
axs2_2.set_ylabel('Param(K)')  # we already handled the x-label with ax1
axs[3].set_ylabel('Latency(s)')
axs3_2.set_ylabel('Peak Mem(Mb)')  # we already handled the x-label with ax1

# Set x labels
axs[3].set_xlabel('Attention Block Number L_S')

# Adjust space between plots
fig.subplots_adjust(hspace=0.4)

plt.show()
