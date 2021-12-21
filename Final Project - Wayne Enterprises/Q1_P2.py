import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.stats import norm



def getStats(inputFile):
    # Importing dataset
    diamonds = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    X = diamonds[inputFile]
    data = [float(x.replace(',', '')) for x in X]
    size = len(data)
    avg = np.mean(data)
    sd = np.std(data, ddof=1) / m.sqrt(size)
    z = (avg - 0.75) / sd
    p = 2 * norm.cdf(-abs(z))
    print("Data Values for " + inputFile + ":")
    print("Sample Size: {}".format(size))
    print("Sample Mean: {}".format(avg))
    print("Standard Error: {}".format(sd))
    print("Standard Score: {}".format(z))
    # print("P-Value: {}".format(p))

    plt.hist(data, bins=50, range=(min(data),max(data)),align='left', color='skyblue', edgecolor='black',linewidth=0.5)
    plt.ylabel('Frequency')
    plt.xlabel('# of ' + inputFile + ' Bicyclists')
    plt.show()
    return


def getGraphs():
    # Importing dataset
    diamonds = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    X = diamonds['Total']
    data1 = [float(x.replace(',', '')) for x in X]
    plt.hist(data1)
    plt.show()

    Y = diamonds['High Temp (°F)']
    Z = diamonds['Low Temp (°F)']
    list1 = [float(x) for x in Y]
    list2 = [float(x) for x in Z]
    data = []

    zip_object = zip(list1, list2)
    for list1_i, list2_i in zip_object:
        data.append((list1_i - list2_i) / 2)

    plt.scatter(data, data1)
    plt.show()
    return


if __name__ == '__main__':
    getStats('Brooklyn Bridge')
    getStats('Manhattan Bridge')
    getStats('Williamsburg Bridge')
    getStats('Queensboro Bridge')
    # getGraphs()
