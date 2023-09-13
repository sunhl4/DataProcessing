from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv("average.csv",sep=',', header='infer')
# 滑动窗口估计,发现数据变化趋势
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
    """
    rolling_mean = series.rolling(window=window).mean()
    print(rolling_mean[:30])
    plt.figure(figsize=(13, 5))
    #plt.title("差值法".format(window))
    plt.plot(series, "r", label="real")
    plt.plot(rolling_mean, "g", label="smooth")
    # plt.plot(rolling_mean, "g", label="Rolling mean trend")
    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        # Having the intervals, find abnormal values
        # if plot_anomalies:
        #     anomalies = pd.DataFrame(index=series.index)
        #     anomalies['series<lower_bond'] = series[series < lower_bond]
        #     anomalies['series>upper_bond'] = series[series > upper_bond]
        #     plt.plot(anomalies, "ro", markersize=10)
#    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)


plotMovingAverage(data['1300K_Small'], 30, plot_intervals=False, scale=2, plot_anomalies=True)
plt.show()