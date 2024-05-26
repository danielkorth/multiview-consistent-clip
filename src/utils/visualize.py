import matplotlib.pyplot as plt


def save_matrix_png(sim, path, type: str = 'mean'):
    plt.figure(figsize=(10, 10))
    plt.imshow(sim, cmap='hot', interpolation='nearest')
    if type == 'mean':
        plt.title('Mean Cosine Similarity')
        plt.clim(0.5, 1)
    else: 
        plt.title('Std Cosine Similarity')
        plt.clim(0, 0.4) 
    plt.colorbar()
    plt.savefig(path)