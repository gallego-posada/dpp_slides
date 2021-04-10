from imports import *
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.axes_grid1 import ImageGrid

def custom_cmap(color_list):
    return mpl.colors.LinearSegmentedColormap.from_list("", color_list)

def gaussian_kernel(X, s=1, exponent=2):
    # X is a NxD Matrix
    # s is scale parameter
    
    pairwise_sq_dists = squareform(pdist(X, 'euclidean'))
    K = np.exp(-pairwise_sq_dists**exponent / s**2)
    
    return K

def cosine_kernel(X):
    # X is a NxD Matrix
    return cosine_similarity(X)

def img_feature_extractor(img, window_width=25, feat_window_width=12, plot_keypts=False):
    keypts = []
    for r in range(2*window_width, img.shape[0] - 2*window_width, window_width):
        for c in range(2*window_width, img.shape[1] - 2*window_width, window_width):
            keypts.append([r, c])
    keypts = np.array(keypts)
    
    if plot_keypts:
        cmap = 'gray' if len(img.shape) < 3 else None
        plt.imshow(img, cmap=cmap)
        plt.plot(keypts[:, 1], keypts[:, 0], '.', c='white')
        plt.axis('off')
        plt.show()

    feats = np.array(extract_window_at_locs(img, keypts, feat_window_width, True))
    
    return keypts, feats

def extract_window_at_locs(img, keypoints, window_width, flatten=False):
    
    res = []
    
    for r, c in keypoints:
        r, c = int(r), int(c)
        window = img[r-window_width:r+window_width, c-window_width:c+window_width, ...]
        if flatten:
            window = window.flatten()
        res.append(window)
    
    return res

def plot_img_grid(img_list, figsize=(8, 4), title=None):
    fig = plt.figure(figsize=figsize)
    
    if len(img_list) % 5 == 0:
        nrows_ncols = (len(img_list) // 5, 5)
    elif len(img_list) % 2 == 0 and len(img_list) / 2 < 8:
        nrows_ncols = (2, len(img_list) // 2)
    else:
        nrows_ncols = (int(np.sqrt(len(img_list))), int(np.sqrt(len(img_list))))
        
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nrows_ncols, axes_pad=0.1)

    for ax, im in zip(grid, img_list):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')
        
    if title:
        plt.suptitle(title)

def plot_kernel_with_values(L, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.matshow(L, cmap=custom_cmap(['black', CLR['dblue'], CLR['gold']]))
    
    for (i, j), z in np.ndenumerate(L):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white')

    plt.title('Det: %.3f' % np.linalg.det(L))
    
    if ax is None:
        plt.show()

def run_mean_experiment(exp, X, sampling_fn, num_samples, num_trials):
    
    est_means = []
    
    for _ in range(num_trials):
        sample = sampling_fn(num_samples)
        est_mean = np.mean(X[sample, :], axis=0, keepdims=True)
        est_means.append(est_mean)
    
    array = np.array(est_means)
    var = np.mean(np.linalg.norm(array.squeeze() - array.mean() , axis=1)**2)
    
    exp.storage.update({'array': array, 'var': var})
    
    
class Experiment():
     def __init__(self, name):
        self.name = name
        self.storage = {}
        
