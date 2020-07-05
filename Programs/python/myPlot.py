import matplotlib.patheffects as PathEffects
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import itertools

# Here is a utility function used to display the transformed dataset.
# The color of each point refers to the actual digit (of course,
# this information was not used by the dimensionality reduction algorithm).
# For general classification problem (not MNIST digit recognition), colors
# contain the class labels
def scatter2D(x, colors, markers, n_colors=10, title='None'):
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    mkr = ['o', '*', '<', 's', '^']

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_colors))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    n_markers = np.max(markers) + 1
    k = 0

    # Cycling the markers
    for m in range(n_markers):
        sc = ax.scatter(x[markers == m, 0], x[markers == m, 1], lw=0, s=40,
                          c=palette[colors[markers == m].astype(np.int)], marker=mkr[k])
        k = k + 1
        if k == len(mkr):
            k = 0

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    labels = np.unique(colors).astype(int)
    for i in labels:
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i+1), fontsize=17)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    if title is not None:
        plt.title(title)
    return f, ax, sc, txts


def scatter3D(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a 3D scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = Axes3D(f)
    sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-25, 25)
    ax.axis('tight')

    return f, ax, sc


def plot_output(out, labels):
    plt.figure()
    y0 = out[labels == 0, 0]
    x0 = range(y0.shape[0])
    y1 = out[labels == 1, 1]
    x1 = range(len(x0), len(x0) + y1.shape[0])
    plt.scatter(x0, y0, marker='o', color='r')
    plt.scatter(x1, y1, marker='*', color='b')


# Display 25 images in a 5x5 grid
def show_imgs(imgs):
    cnt = 0
    r, c = 5, 5
    fi, ax = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):
            ax[i, j].imshow(imgs[cnt, :, :, 0], cmap='gray')
            ax[i, j].axis('off')
            cnt += 1
    plt.show()


# Plot histograms of the the two outputs of a binary classifier
def plot_hist(x, nbins=50):
    # the histogram of the data
    h1, bins = np.histogram(x[:, 0], bins=nbins, normed=1)
    plt.plot(bins[0:len(bins)-1], h1, 'r', linewidth=1)
    h2, bins = np.histogram(x[:, 1], bins=nbins, normed=1)
    plt.plot(bins[0:len(bins)-1], h2, 'b', linewidth=1)
    plt.ylabel('Probability')


if __name__ == '__main__':
    # Test plot_hist()
    N = 100000
    mu1, sigma1 = 10, 15
    mu2, sigma2 = -10, 10
    x1 = np.asarray(mu1 + sigma1 * np.random.randn(N))
    x2 = np.asarray(mu2 + sigma2 * np.random.randn(N))
    x = np.hstack([x1.reshape((N, 1)), x2.reshape(N,1)])
    plot_hist(x, 100)

    # Test scatter2D()
    N = 100
    x = np.vstack([np.random.normal(1,1,(N,2)), np.random.normal(-1,1,(N,2))])
    y = [0] * N + [1] * N
    y = np.asarray(y)
    markers = y
    scatter2D(x, y, markers, n_colors=2, title='No. of points = %d' % (N *2 ))
    plt. show()