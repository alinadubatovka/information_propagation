import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf(0.025, n - 1)
    return h


plt.rcParams["axes.titlesize"] = 'xx-large'
plt.rcParams["font.size"] = 24
plt.rcParams["hatch.linewidth"] = 5
plt.rcParams["patch.edgecolor"] = "C0"
plt.rcParams["figure.edgecolor"] = "C0"


def compute_point(N, L, mode, dataset):
    with open('%s/%s_init/N_%d_L_%d_%s_20_epochs_30_trials.csv' % (dataset, mode, N, L, mode), 'r',
              newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        rows = []
        for row in csv_reader:
            rows.append([float(v) for v in row])
        data = np.array(rows)
    return np.mean(data[:, -2]), confidence_interval(data[:, -2])


dataset = "mnist"

plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.09, right=0.99, bottom=0.1, top=0.97, wspace=0.13)
L_range = list(range(0, 10))
L_plot_range = np.array(L_range) + 1

ax = plt.subplot(121)
for mode in ["new", "orthogonal", "old"]:
    for N in [100, 300, 500]:

        means = []
        vars = []

        for L in L_range:
            mean, var = compute_point(dataset, N, L, mode)
            means.append(np.mean(mean))
            vars.append(var)

        means = np.array(means)
        vars = np.array(vars)
        mode_name = {
            "new": "GSM",
            "old": "He",
            "orthogonal": "Ortho"
        }[mode]

        label = "%d, %s" % (N, mode_name)

        ax.fill_between(L_plot_range, means - vars, means + vars, linewidth=1, linestyle='-', alpha=0.1)
        ax.plot(L_plot_range, means, "--", lw=4, label=label)

ax.set_xticks(np.array(L_range) + 1)
ax.set_xlabel("Number of hidden layers")
ax.grid(b=True, which="major", axis='both', lw=0.3)

handles, labels = ax.get_legend_handles_labels()
ax.set_ylabel("Accuracy")

ax = plt.subplot(122)

for mode in ["new", "orthogonal"]:  # , "old"]:
    for N in [100, 300, 500]:

        means = []
        vars = []

        for L in L_range:
            mean, var = compute_point(dataset, N, L, mode)
            means.append(np.mean(mean))
            vars.append(var)

        means = np.array(means)
        vars = np.array(vars)

        label = "N = %d, %s init" % (N, mode if mode != "orthogonal" else "orth")
        ax.fill_between(L_plot_range, means - vars, means + vars, linewidth=1, linestyle='-', alpha=0.1)
        ax.plot(L_plot_range, means, "--", lw=4, label=label)

ax.set_xticks(np.array(L_range) + 1)
ax.set_xlabel("Number of hidden layers")
ax.grid(b=True, which="major", axis='both', lw=0.3)

ax.legend(handles[:], labels[:], prop={'size': 18}, ncol=2, title="N, initialization", fancybox=True, framealpha=0)
plt.show()
