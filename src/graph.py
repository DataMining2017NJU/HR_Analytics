import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

local_path = "/Users/lixuefei/Desktop/file/DM/work/HR_comma_sep.csv"

dataset = pd.read_csv(local_path)#names=data_name,
g = sns.heatmap(dataset.corr(),annot=True,cmap="RdYlGn",xticklabels=True,yticklabels=True)

#plt.figure(f)
ax = plt.gca()

for label in ax.xaxis.get_ticklabels():
    label.set_rotation(45)
for label in ax.yaxis.get_ticklabels():
    label.set_rotation(45)

plt.show()




