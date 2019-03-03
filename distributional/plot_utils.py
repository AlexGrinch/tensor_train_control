import numpy as np
import matplotlib.pyplot as plt


def plot_average(r, eta):
    ret = np.cumsum(r, dtype=np.float)
    ret[eta:] = ret[eta:] - ret[:-eta] 
    y = ret[eta - 1:] / eta
    return y


def plot_max(r, eta):
    return r[eta-1:]


def plot_means_and_stdevs(x, y, tau, figsize=(8, 5), color='dodgerblue'):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    means = np.cumsum(y, dtype=np.float)
    stdevs = np.cumsum(np.square(y), dtype=np.float)
    
    means[tau:] = means[tau:] - means[:-tau]
    means = means[tau-1:] / tau
    
    stdevs[tau:] = stdevs[tau:] - stdevs[:-tau]
    stdevs = stdevs[tau-1:] / tau - np.square(means)
    stdevs = np.sqrt(stdevs)
    
    lower = means - stdevs
    upper = means + stdevs
    
    x = plot_max(x, tau)
    
    ax.plot(x, means, color=color)
    ax.fill_between(x, lower, means, alpha=0.2, where=lower <= means, facecolor=color)
    ax.fill_between(x, upper, means, alpha=0.2, where=upper >= means, facecolor=color)
    plt.ylim([-1.1, 1.6])
    
    ax.grid()


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
