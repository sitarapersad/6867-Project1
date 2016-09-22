import matplotlib.pyplot as plt
import pylab as pl

def getData(ifPlotData=True):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('curvefittingp2.txt')

    X = data[0,:]
    Y = data[1,:]

    if ifPlotData:
        plt.plot(X,Y,'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return (X,Y)

def computePolynomialWeight(X, Y, M):
    '''
    @param: X - n x 1 array of n 1-dimensional data points
    @param: Y - n x 1 array of corresponding y vals
    @param: M - maximum order of polynomial basis

    @return: MLE for the weight vector
    '''

    # generate the phi vector based on the maximum order
    # 
