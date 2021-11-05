import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
data1 = pd.read_csv('number.csv')
data2 = pd.read_csv('reward.csv')
data11,data22 = np.array(data1),np.array(data2)
datalist1 = np.array(data11[:,1])
datalist2 = np.array(data22[:,1])

datalist22 = []
for i in range(0,1000,10):
  datalist22.append(sum(datalist2[i:i+10])/10)

input1 = datalist1
output1 = datalist22

plt.figure(2)
plt.plot(input1,output1,linewidth=1)
plt.legend()
plt.xlabel("Episode",fontsize=14)
plt.ylabel("Reward",fontsize=14)
plt.grid()
plt.show()
