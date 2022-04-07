import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_area_distribution(model,xtrain,ytrain,title,xlabel,ylabel,colors=('red', 'green'),alpha = 0.75):
    X_set, y_set = xtrain, ytrain
    X1, X2 = np.meshgrid(np.arange( start = X_set[:, 0].min() - 1, 
                                    stop = X_set[:, 0].max() + 1, 
                                    step = 0.01),
                        np.arange(  start = X_set[:, 1].min() - 1, 
                                    stop = X_set[:, 1].max() + 1, 
                                    step = 0.01)
    )
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = alpha,
                 cmap = ListedColormap(colors))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(colors)(i), label = j)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()