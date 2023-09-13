import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
readData = pd.read_csv("average.csv",sep=',', header='infer')   #读取csv数据
#ydata = jnp.array(readData["2-1000-rescall"])      #获取dataFrame中的第3列，并将此转换为list
y = np.array(readData["1300K_Small"])
#获取dataFrame中的第3列，并将此转换为list
x = np.array(readData["frame"])
x_smooth = np.linspace(x.min(), x.max(), 200)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
y_smooth = make_interp_spline(x, y)(x_smooth)
#print(len(y_smooth))
plt.plot(x, y, "r-",label="real")
plt.plot(x_smooth, y_smooth, "g-", label="smooth")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
