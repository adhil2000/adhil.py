import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def main(degrees):
    paramFits = []

    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    #in poly.txt.
    diamonds = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    X = diamonds['High Temp (°F)']
    Y = diamonds['Total']

    X = X.to_numpy()
    Y = Y.to_numpy()

    col1 = [float(x) for x in X]
    col2 = [float(y.replace(',', '')) for y in Y]
    #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.
    for n in degrees:
        X_param = feature_matrix(col1, n)
        B_param = least_squares(X_param, col2)
        paramFits.append(B_param)

    return paramFits


#Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
#samples in x.
#Input: x as a list of the independent variable samples, and d as an integer.
#Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
#for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    X = [[i ** j for j in range(d, -1, -1)] for i in x]
    return X


#Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
#Input: X as a list of features for each sample, and y as a list of target variable samples.
#Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    #fill in
    #Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    B = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
    return B

def getGraph(col1, col2,x_var,y_var):

    degrees = [1]
    paramFits = main(degrees)
    #print(paramFits)
    # R-Squared Value
    correlation_matrix = np.corrcoef(col1, col2)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    print('The R-Squared Value for ' + x_var + ' vs ' + y_var + ' is {}' .format(r_squared))
    # Plot
    plt.scatter(col1, col2, color='black', label='data')
    col1.sort()
    for parameters in paramFits:
        d = len(parameters) - 1
        X = np.array(feature_matrix(col1, d))
        Y = np.dot(X, parameters)
        plt.plot(col1, Y, label='d = ' + str(d))
    plt.title(x_var + ' vs ' + y_var)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    diamonds = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    # Total Bridge Bicycle Traffic
    total = diamonds['Total']
    total = [float(i.replace(',', '')) for i in total.to_numpy()]

    # Precipitation vs Total Traffic
    pr = diamonds['Precipitation']
    pr = [i.replace('47 (S)', '0') for i in pr]
    pr = [i.replace('T', '0') for i in pr]
    pr = [float(i) for i in pr]
    getGraph(pr, total, 'Total Traffic', 'Precipitation')
    corr, _ = spearmanr(total, pr)
    print('Spearman"s Correlation for Total Traffic vs Precipitation: %.3f' % corr)

    # Precipitation vs Average Bridge Traffic
    bb = diamonds['Brooklyn Bridge']
    bb = [float(i.replace(',', '')) for i in bb.to_numpy()]
    mb = diamonds['Manhattan Bridge']
    mb = [float(i.replace(',', '')) for i in mb.to_numpy()]
    wb = diamonds['Williamsburg Bridge']
    wb = [float(i.replace(',', '')) for i in wb.to_numpy()]
    qb = diamonds['Queensboro Bridge']
    qb = [float(i.replace(',', '')) for i in qb.to_numpy()]

    avg_b = []
    for i in range(len(total)):
        avg_b.append((bb[i] + mb[i] + wb[i] + qb[i]) / 4)
    getGraph(avg_b, pr, 'Average Traffic', 'Precipitation')
    corr1, _ = spearmanr(pr, total)
    print('Spearman"s Correlation for Average Traffic vs Precipitation: %.3f' % corr1)
