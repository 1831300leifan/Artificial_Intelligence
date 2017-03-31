import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    datafile = 'data/ex2data1.txt'
    #!head $datafile
    cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) #Read in comma separated data
    ##Form the usual "X" matrix and "y" vector
    X = np.transpose(np.array(cols[:-1]))
    y = np.transpose(np.array(cols[-1:])).ravel()
    m = y.size # number of training examples
    ##Insert the usual column of 1's into the "X" matrix
    # X = np.insert(X,0,1,axis=1)

    return X, y

def plotData(X, plt):
    pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
    neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])

    plt.figure(figsize=(10,6))
    plt.plot(pos[:,1],pos[:,2],'k+',label='Admitted')
    plt.plot(neg[:,1],neg[:,2],'yo',label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.grid(True)

X, y = load_data()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)
print(clf.coef_)

xx, yy = np.mgrid[np.min(X[:, 0]):np.max(X[:, 0]):2., np.min(X[:, 0]):np.max(X[:, 0]):2.]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(X[:,0], X[:, 1], c=y[:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(30, 90), ylim=(30, 90),
       xlabel="$X_1$", ylabel="$X_2$")

plt.show()