import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt


def main():
    #Importing dataset
    diamonds = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')

    #Feature and target matrices
    #Uncomment accordingly for each Target

    #Target: Queensboro Bridge
    #X = diamonds[['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge']]
    #y = diamonds[['Queensboro Bridge']]
    #name = 'Queensboro Bridge'

    #Target: Williamsburg Bridge
    #X = diamonds[['Brooklyn Bridge','Manhattan Bridge','Queensboro Bridge']]
    #y = diamonds[['Williamsburg Bridge']]
    #name = 'Williamsburg Bridge'

    #Target: Manhattan Bridge
    #X = diamonds[['Brooklyn Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']]
    #y = diamonds[['Manhattan Bridge']]
    #name = 'Manhattan Bridge'

    #Target: Brooklyn Bridge
    X = diamonds[['Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']]
    y = diamonds[['Brooklyn Bridge']]
    name = 'Brooklyn Bridge'

    #Training and testing split, with 25% of the data reserved as the test set
    X = X.to_numpy()
    y = y.to_numpy()

    X = [[float(x.replace(',', '')) for x in z] for z in X]
    y = [[float(i.replace(',', '')) for i in z] for z in y]

    X = np.array(X)
    y = np.array(y)

    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    #Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(-1, 2, num=101) #fill in

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)
        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)
        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Plot the MSE as a function of lmbda
    #fill in
    plt.plot(lmbda, MSE)
    plt.title(name + ': λ vs MSE')
    plt.xlabel('Lambda λ')
    plt.ylabel('Mean Squared Error', fontsize =8)
    plt.grid()
    plt.show()

    #Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print('Best lambda Tested: ' + str(lmda_best))
    print('Yielded MSE Value: ' + str(MSE_best))

    return model_best


#Function that normalizes features in training set to zero mean and unit variance.
#Input: training data X_train
#Output: the normalized version of the feature matrix: X, the mean of each column in
#training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):

    #fill in
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X = (X_train - mean) / std
    return X, mean, std


#Function that normalizes testing set according to mean and std of training set
#Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
#column in training set: trn_std
#Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):
    #fill in
    shape_X = np.shape(X_test)
    X = np.empty(shape_X)
    size = len(X_test[0])
    for i in range(size):
        X[:, i] = (X_test[:, i] - trn_mean[i]) / trn_std[i]

    return X


#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):
    #fill in
    model = linear_model.Ridge(alpha=l, fit_intercept=True)
    model.fit(X, y)

    return model


#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def error(X,y,model):
    #Fill in
    pred_y = model.predict(X)
    sqr = (y - pred_y) ** 2
    mse = np.mean(sqr)

    return mse

if __name__ == '__main__':
    model_best = main()
    #We use the following functions to obtain the model parameters instead of model_best.get_params()
    print('Model Best Coefficients:')
    print(model_best.coef_)
    print('Model Best Intercept:')
    print(model_best.intercept_)

