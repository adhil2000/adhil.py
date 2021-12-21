import numpy as np
import matplotlib.pyplot as plt

def norm_histogram(hist):
    return [i/sum(hist) for i in hist]

def compute_j(histo, width):
    """
    takes histogram of counts, uses norm_histogram to convert to probabilties, it then calculates compute_j for one bin width
    :param histo: list 
    :param width: float
    :return: float
    """
    hist_prob = norm_histogram(histo)
    n_points = sum(histo)
    tot_sum = sum([x ** 2 for x in hist_prob])
    J = (2.0 - (n_points + 1) * tot_sum) / (width * (n_points - 1))
    return J

    pass


def sweep_n(data, minimum, maximum, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep

    :param data: list
    :param minimum: int
    :param maximum: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """
    bounds = (minimum, maximum)
    range_diff = maximum - minimum
    js = [compute_j(plt.hist(data, k, bounds)[0], range_diff / k) for k in range(min_bins, max_bins + 1)]
    return js
    pass


def find_min(l):
    """
    generic function that takes a list of numbers and returns smallest number in that list its index.
    return optimal value and the index of the optimal value as a tuple.

    :param l: list
    :return: tuple
    """
    minIndex = 0
    minVal = l[minIndex]
    for index, val in zip(range(1, len(l)), l[1:]):
        if (val < minVal):
            minVal = val
            minIndex = index
    return (minVal, minIndex)

pass


if __name__ == '__main__':
    data = np.loadtxt('input.txt')  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
