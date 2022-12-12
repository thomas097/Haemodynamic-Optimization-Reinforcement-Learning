import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(13)

plt.rcParams["font.family"] = "Times New Roman"
hatching = ['/', '\\']
colors = ['C1', 'C0']



def generate_fake_data(n, freqs, alphas, starts=None):
    if starts is None:
        starts = np.random.random(len(freqs)) * np.pi
    x = np.cumsum(np.random.random(n) ** 2)
    x = x / np.max(x) * 10
    y = np.mean([a * np.sin(i * x + s) for s, a, i in zip(starts, alphas, freqs)], axis=0)
    return x - np.max(x), y + np.random.normal(0, 0.1, y.shape)

data = []
for height in range(2):
    freqs = np.random.uniform(1, 3, 3)
    x, y = generate_fake_data(50, freqs=freqs, alphas=[0.2, 0.5, 0.3])
    data.append((x, y))

plt.figure()
ax = plt.subplot(2, 1, 1)
for i, (x, y) in enumerate(data):
    plt.scatter(x, y + i, s=50, color=colors[i], alpha=0.7, edgecolors="black")

ax.set_xlabel('')
ax.set_xticks([])
ax.set_yticks([0, 1],
           ['Heart Rate', 'Sys. Blood Pressure'][::-1])
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.yaxis.set_label_position("right")
ax.set_ylabel('Raw')
ax.yaxis.tick_left()

# Add the patch to the Axes
k = -0.24
plt.plot([-3.9+k, -3.9+k], [-.5, 1.55], c='k')
plt.plot([-3.5+k, -3.5+k], [-.5, 1.55], c='k')


ax2 = plt.subplot(2, 1, 2)
for i, (x, y) in enumerate(data):
    t = np.arange(-20, 1) / 2
    y = np.interp(t, x, y) + i
    plt.scatter(t, y, s=50, alpha=0.7, color=colors[i], edgecolors="black")
    
ax2.set_xlabel('Time step')
ax2.set_xticks([-10, -7.5, -5, -2.5, 0], [i * 5 for i in range(5)])
ax2.set_yticks([0, 1],
           ['Heart Rate', 'Sys. Blood Pressure'][::-1])
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Resampled')
ax2.yaxis.tick_left()

ax2.arrow(-4, 1.4, 0, -.1, width=0.1, color='k', head_length=0.2)


plt.tight_layout()
plt.show()
