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

dataset = "mnist"
sample_size = 30

plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.09, right=0.99, bottom=0.1, top=0.97, wspace=0.13)
L_range = list(range(0, 10))
L_plot_range = np.array(L_range) + 1

ax = plt.subplots()
for mode in ["new", "orthogonal", "old"]:
    mode_name = {
        "new": "GSM",
        "old": "He",
        "orthogonal": "Ortho"
    }[mode]

    for N in [100, 300, 500]:
        means = []
        vars = []

        for L in L_range:
            data = np.genfromtxt('%s/%s_init/N_%d_L_%d_%s_%d_trials.csv' % (dataset, mode, N, L, mode, sample_size), delimiter=' ')
            means.append(np.mean(data[:, -2]))
            vars.append(confidence_interval(data[:, -2]))

        means = np.array(means)
        vars = np.array(vars)

        label = "%d, %s" % (N, mode_name)
        ax.fill_between(L_plot_range, means - vars, means + vars, linewidth=1, linestyle='-', alpha=0.1)
        ax.plot(L_plot_range, means, "--", lw=4, label=label)

ax.set_xticks(np.array(L_range) + 1)
ax.set_xlabel("Number of hidden layers")
ax.grid(b=True, which="major", axis='both', lw=0.3)

handles, labels = ax.get_legend_handles_labels()
ax.set_ylabel("Accuracy")
ax.legend(handles[:], labels[:], prop={'size': 18}, ncol=2, title="N, initialization", fancybox=True, framealpha=0)
plt.show()
