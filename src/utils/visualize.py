import matplotlib.pyplot as plt
from pathlib import Path


def plot_object_similarity(x, path: str, title: str = "CHANGE ME", type: str = 'mean'):
    plt.figure(figsize=(6, 6))
    plt.imshow(x, cmap='jet', interpolation='nearest')
    plt.xticks(ticks=range(0, 36, 12), labels=range(1, 37, 12))
    plt.yticks(ticks=range(0, 36, 12), labels=range(1, 37, 12))
    if type == 'mean':
        plt.title(title)
        plt.clim(0, 1) 
    else: 
        plt.title(title)
        plt.clim(0, 0.5) 
    
    plt.colorbar(shrink=0.81)

    plt.savefig(Path(path) / f"{title.lower().replace(' ', '_')}.png", dpi=300, transparent=True)


def plot_all_similarity(x, path: str, title: str = "CHANGE ME", type: str = 'mean'):
    plt.figure(figsize=(6, 6))
    plt.imshow(x, cmap='jet', interpolation='nearest')
    plt.xticks(ticks=[100, 3600], labels=['text', 'images'])
    plt.yticks(ticks=[100, 3600], labels=['text', 'images'])
    if type == 'mean':
        plt.title(title)
        plt.clim(0, 1) 
    else: 
        plt.title(title)
        plt.clim(0, 0.5)
    
    plt.colorbar(shrink=0.81)

    plt.savefig(Path(path) / f"{title.lower().replace(' ', '_')}.png", dpi=300, transparent=True)