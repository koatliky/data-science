print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn import linear_model, decomposition, datasets


mnist = fetch_mldata("MNIST original", data_home='C:/Python34/scikit_learn_data/' )
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
inxs = np.random.randint(y.shape[0], size=1000)
y = y[inxs]
X = X[inxs,:]
X_train, X_test = X[:20000], X[20000:]
y_train, y_test = y[:20000], y[20000:]
logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pca.fit(X)
components = np.array(pca.explained_variance_)
y_values = pca.transform(X)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_.cumsum(), linewidth=2)
plt.axis('tight')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
