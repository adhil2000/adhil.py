import numpy as np
import matplotlib.pyplot as plt
#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []

    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    #in poly.txt.
    myFile = open(datapath, 'r')
    col1, col2 = [], []
    for rows in myFile:
        [x, y] = rows.split()
        col1.append(float(x))
        col2.append(float(y))
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

if __name__ == '__main__':
    datapath = 'poly.txt'
    #degrees = [2, 4]

    myFile = open(datapath, 'r')
    col1, col2 = [], []
    for rows in myFile:
        [x, y] = rows.split()
        col1.append(float(x))
        col2.append(float(y))

    degrees = [1, 2, 3, 4, 5]
    paramFits = main(datapath, degrees)
    print(paramFits)

    plt.scatter(col1, col2, color='black', label='data')
    col1.sort()
    for parameters in paramFits:
        d = len(parameters) - 1
        X = np.array(feature_matrix(col1, d))
        Y = np.dot(X, parameters)
        plt.plot(col1, Y, label='d = ' + str(d))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
