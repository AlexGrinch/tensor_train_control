import numpy as np
import matplotlib.pyplot as plt


def plot_boi(
    x, y, tau, ax,
    x_scale=1, y_scale=1, y_range=None,
    color="black", label="algo"
):

    y = np.array(y) * y_scale
    x = np.array(x) * x_scale

    means = np.cumsum(y, dtype=np.float)
    means[tau:] = means[tau:] - means[:-tau]
    means = means[tau-1:] / tau

    stdevs = np.cumsum(np.square(y), dtype=np.float)
    stdevs[tau:] = stdevs[tau:] - stdevs[:-tau]
    stdevs = np.maximum(stdevs[tau-1:] / tau - np.square(means), 0)
    stdevs = np.sqrt(stdevs) / 2

    lower = means - stdevs
    upper = means + stdevs

    x = np.cumsum(x, dtype=np.float)
    x[tau:] = x[tau:] - x[:-tau]
    x = x[tau-1:] / tau

    ax.plot(x, means, color=color, label=label)
    ax.fill_between(
        x, lower, means, alpha=0.2,
        where=lower <= means, facecolor=color
    )
    ax.fill_between(
        x, upper, means, alpha=0.2,
        where=upper >= means, facecolor=color
    )

    if y_range is not None:
        ax.set_ylim(y_range)
        
        
def plot_everything(
    paths, titles, colors, labels, tau,
    y_range=[-0.6, 1.2], save=None
):
    fig, ax = plt.subplots(3, 2, figsize=(18, 15))
    for i in range(6):

        ax[i//2, i%2].grid()
        ax[i//2, i%2].spines['right'].set_visible(False)
        ax[i//2, i%2].spines['top'].set_visible(False)
        for tick in ax[i//2, i%2].xaxis.get_major_ticks():
            tick.label.set_fontsize(18) 
        for tick in ax[i//2, i%2].yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        if y_range is not None:
            ax[i//2, i%2].set_yticks([-0.5, 0.0, 0.5, 1.0])
        ax[i//2, i%2].set_title(titles[i], fontsize=18)

        data = np.load(paths[i])["test"]
        algos = [data[0, :], data[1, :], data[2, :]]
        for j in range(3):
            returns = data[j, :]
            plot_boi(
                np.arange(len(returns)), returns,
                tau=tau, ax=ax[i//2, i%2],
                color=colors[j], label=labels[j],
                x_scale=1/200, y_range=y_range
            )

        if i == 0:
            ax[i//2, i%2].legend(fontsize=20)
            
    if save is not None:
        fig.savefig(save, dpi=100, bbox_inches="tight")


def plot_two_lines(
        x, ys, tau, figsize=(12, 6),
        colors=["dodgerblue", "mediumseagreen"],
        labels=["Quantilee", "Standard"]):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = plot_max(x, tau)
    for i in range(len(ys)):
        means = np.cumsum(ys[i], dtype=np.float)
        stdevs = np.cumsum(np.square(ys[i]), dtype=np.float)

        means[tau:] = means[tau:] - means[:-tau]
        means = means[tau-1:] / tau

        stdevs[tau:] = stdevs[tau:] - stdevs[:-tau]
        stdevs = stdevs[tau-1:] / tau - np.square(means)
        stdevs = np.sqrt(stdevs) / 2

        lower = means - stdevs
        upper = means + stdevs

        

        ax.plot(x, means, color=colors[i], label=labels[i])
        ax.fill_between(x, lower, means, alpha=0.2, where=lower <= means, facecolor=colors[i])
        ax.fill_between(x, upper, means, alpha=0.2, where=upper >= means, facecolor=colors[i])
        ax.legend()

    plt.grid()


def plot_learning_curve(path, tau=100, figsize=(8, 5), x_scale=1000000, color='dodgerblue'):
    r = np.load(path+"/learning_curve.npz")['r']
    f_train = np.load(path + "/learning_curve.npz")["f"]/x_scale
    
    plot_means_and_stdevs(f_train, r, tau=tau, figsize=figsize, color=color)


def plot_reward_distribution(path, from_episode=0, num_episodes=100, bins=None):
    r = np.load(path+"/learning_curve.npz")['r']
    plt.hist(r[from_episode:from_episode+num_episodes], bins=bins,
             color='dodgerblue', edgecolor='black', linewidth=1.5)
    plt.grid()


def compare_two_distributions(path, from_episode1=0, from_episode2=100, 
                              num_episodes=100, bins=None):
    r = np.load(path+"/learning_curve.npz")['r']
    
    hist1 = r[from_episode1:from_episode1+num_episodes]
    hist2 = r[from_episode2:from_episode2+num_episodes]
    hist12 = np.concatenate((hist1, hist2))
    
    left = np.min(hist12)
    right = np.max(hist12)
    
    plt.hist(hist1, bins=bins, range=[left, right], color='dodgerblue', 
             edgecolor='black', linewidth=1.5, alpha=0.7)
    plt.hist(hist2, bins=bins, range=[left, right],color='mediumseagreen', 
             edgecolor='black', linewidth=1.5, alpha=0.7)
    plt.grid()


def plot_performance(path, eta=10, x_scale=1000000):
    r_train = np.load(path + "/learning_curve.npz")["r"]
    f_train = np.load(path + "/learning_curve.npz")["f"]/x_scale
    x = plot_average(f_train, eta)
    y = plot_average(r_train, eta)
    plt.plot(x, y)
    plt.grid()


def plot_lifetimes(path, eta=10, x_scale=1000000):
    l_train = np.load(path + "/learning_curve.npz")["l"]
    f_train = np.load(path + "/learning_curve.npz")["f"]/x_scale
    x = plot_average(f_train, eta)
    y = plot_average(l_train, eta)
    plt.plot(x, y)
    plt.grid()


def plot_compare(paths, eta, x_scale=1000000, figsize=(8, 5)):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    r, f = [], []
    for i in range(len(paths)):
        r = np.load(paths[i] + "/learning_curve.npz")["r"]
        f = np.load(paths[i] + "/learning_curve.npz")["f"]/x_scale
        x = plot_max(f, eta)
        y = plot_average(r, eta)
        ax.plot(x, y, label=paths[i].split("/")[1], lw=3)
        ax.legend()
    ax.grid()


def plot_average2(paths, eta, x_scale=1000000, figsize=(8, 5)):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    r, f, ys = [], [], []
    for i in range(len(paths)):
        r = np.load(paths[i] + "/learning_curve.npz")["r"]
        f = np.load(paths[i] + "/learning_curve.npz")["f"]
        
        y = gen_equal(r, f)
        x = np.arange(0, 1800000, 100)
        
        x = plot_max(x, eta)
        y = plot_average(y, eta)
        ys.append(y)
    s = np.zeros_like(y)
    for i in range(len(paths)):
        s += ys[i]
    s = s / len(paths)
    ax.plot(x/x_scale, s, lw=3)
    ax.grid()
    return s


def plot_compare_lifetimes(paths, eta, x_scale=1000000, figsize=(8, 5)):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(len(paths)):
        f = np.load(paths[i] + "/learning_curve.npz")["f"]
        x = plot_max(f[:-1], eta)
        y = plot_average(f[1:] - f[:-1], eta)
        ax.plot(x, y, label=paths[i].split("/")[1])
        ax.legend()
    ax.grid()


def plot_compare_equal(paths, eta, x_scale=1000000):
    
    min_l = np.inf
    for i in range(len(paths)):
        l = len(np.load(paths[i] + "/learning_curve.npz")["r"])
        if l < min_l:
            min_l = l
    min_l = int(min_l)
    
    r, f = [], []
    for i in range(len(paths)):
        r = np.load(paths[i] + "/learning_curve.npz")["r"]
        f = np.load(paths[i] + "/learning_curve.npz")["f"]/x_scale
        x = plot_average(f, eta)
        y = plot_average(r, eta)
        plt.plot(x[:min_l], y[:min_l])
    plt.grid()


def plot_q_values(env, q_values, figsize=(15, 5)):
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    q_table = np.zeros((env.w, env.h, 4))

    idx = 0
    for i in range(env.w):
        for j in range(env.h):
            valid, s = env.set_pos((i, j))
            if valid:
                q_table[i, j] = q_values[idx]
                idx += 1

    for a in range(4):
        img = np.rot90(q_table[:,:,a])
        ax[a].imshow(img, cmap='gray', vmin=-1, vmax=1)
