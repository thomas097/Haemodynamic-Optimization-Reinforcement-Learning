import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(1)

plt.rcParams["font.family"] = "Times New Roman"
colors = ['C1', 'C0']



def generate_fake_data(n, freqs, alphas, starts=None):
    if starts is None:
        starts = np.random.random(len(freqs)) * np.pi
    x = np.cumsum(np.random.random(n) ** 2)
    x = x / np.max(x) * 10
    y = np.mean([a * np.sin(i * x + s) for s, a, i in zip(starts, alphas, freqs)], axis=0)
    return x - np.max(x), y + np.random.normal(0, 0.1, y.shape)

#
#  Create fake dataset with two features: heart rate and systolic blood pressure
#

data = []
for height in range(2):
    freqs = np.random.uniform(1, 3, 3)
    x, y = generate_fake_data(50, freqs=freqs, alphas=[0.2, 0.5, 0.3])

    # drop some measurements
    if height == 1:
        y[8:17] = np.NaN
    
    data.append((x, y))

#
# Plot aggregation window
#
plt.figure()
ax = plt.subplot(2, 1, 1)

rect1 = patches.Rectangle((-4.225, -2), .45, 5, linewidth=1, edgecolor='k', hatch='\\\\\\',
                          facecolor='grey', alpha=0.1, zorder=-99)
rect2 = patches.Rectangle((-4.225, -2), .45, 5, linewidth=1, edgecolor='k', fill=False,
                          alpha=.5, facecolor='none', zorder=-50)
ax.add_patch(rect1)
ax.add_patch(rect2)

#
#  Plot raw data
#

for i, (x, y) in enumerate(data):
    plt.scatter(x, y + i, s=50, color=colors[i], alpha=0.7, edgecolors="black")

ax.set_xlabel('')
ax.set_xticks([])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Heart Rate', 'Sys. Blood Pressure'][::-1])
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.yaxis.set_label_position("right")
ax.set_ylabel('Raw')
ax.yaxis.tick_left()
ax.set_xlim(-11, 1)
ax.set_ylim(-0.6, 1.8)


#
# Plot aggregation window
#
ax2 = plt.subplot(2, 1, 2)

rect1 = patches.Rectangle((-4.225, -2), .45, 5, linewidth=1, edgecolor='k', hatch='\\\\\\',
                          facecolor='grey', alpha=0.1, zorder=-99)
rect2 = patches.Rectangle((-4.225, -2), .45, 5, linewidth=1, edgecolor='k', fill=False,
                          alpha=.5, facecolor='none', zorder=-50)
ax2.add_patch(rect1)
ax2.add_patch(rect2)


#
#  Resample data into fixed bins and plot in bottom window
#

# https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
def pandas_fill(arr):
    df = pd.DataFrame(arr)
    df.fillna(method='ffill', axis=0, inplace=True)
    return df.values.flatten()

for i, (x, y) in enumerate(data):
    y2 = pandas_fill(y)
    
    t = np.arange(-20, 1) / 2
    y2 = np.interp(t, x, y2) + i

    c = colors[i]
    clr = ['grey' if (j > 4) and (j < 9) and i == 1 else c for j in range(len(t))]
    
    plt.scatter(t, y2, s=50, alpha=0.7, color=clr, edgecolors="black")
    

ax2.text(-6.75, 1.25, ha='center', s='LVCF')
#ax2.text(-7.05, 1.2, ha='center', s='{', stretch=1000, fontsize=60, rotation=-90)
    
ax2.set_xlabel('Time step')
ax2.set_xticks([-10, -7.5, -5, -2.5, 0])
ax2.set_xticklabels([i * 5 for i in range(5)])
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Heart Rate', 'Sys. Blood Pressure'][::-1])
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Resampled')
ax2.yaxis.tick_left()
ax2.set_xlim(-11, 1)
ax2.set_ylim(-0.5, 1.8)

#
#  Arrow
#
ax.arrow(-4, -0.5, 0, -2, width=0.1, color='k', head_length=0.2)
ax2.arrow(-4, 1.8, 0, -.05, width=0.1, color='k', head_length=0.2)


plt.tight_layout()
plt.show()
