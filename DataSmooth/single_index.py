from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv("average.csv",sep=',', header='infer')


# 单指数平滑
def exponential_smoothing(series, alpha):
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


def plotExponentialSmoothing(series, alphas):
    """
        Plots exponential smoothing with different alphas

        series - dataset with timestamps
        alphas - list of floats, smoothing parameters

    """
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(15, 7))
        for alpha in alphas:
            # plt.plot(exponential_smoothing(series, alpha), "g", label="Alpha {}".format(alpha))
            plt.plot(exponential_smoothing(series, alpha), "g", label="smooth".format(alpha))
        plt.plot(series.values, "r", label="real")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);


plotExponentialSmoothing(data['1300K_Small'], [0.05])

plt.show()