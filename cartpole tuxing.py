import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
data1 = pd.read_csv('datatest1.csv')
data2 = pd.read_csv('datatest2.csv')
data11,data22 = np.array(data1),np.array(data2)
datalist1 = np.array(data11[:,1])
datalist2 = np.array(data22[:,1])

input = datalist1
output = datalist2
plt.plot(input,output,linewidth=1,label='PREDQN + DDQN +Dueling DQN')
plt.legend()
plt.xlabel("Episode",fontsize=14)
plt.ylabel("Ave_Rewards",fontsize=14)
plt.grid()
plt.show()
