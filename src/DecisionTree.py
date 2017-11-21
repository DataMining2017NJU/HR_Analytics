import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pydot
#import pydotplus as pydot
#from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
#from sklearn.model_selection import StratifiedKFold

from PIL import Image

from IPython.display import Image,display
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
from math import log

#import matplotlib.pyplot as plt

#satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,left,promotion_last_5years,sales,salary
local_path = "/Users/lixuefei/Documents/GitHub/HR_Analysis/HR_comma_sep.csv"
data_name = ['satisfaction_level','last_evaluation','number_project',"average_montly_hours","time_spend_company","Work_accident","left","promotion_last_5years","sales","salary"]

dataset = pd.read_csv(local_path,header = None,names = data_name)#columns=data_name names=data_name,

#print(type(dataset))
X_train = dataset.drop(['left', 'sales','salary'],axis = 1) #inplace=True .convert_objects(convert_numeric=True).dtypes

#print(X_train)

Y_train = dataset["left"]#.convert_objects(convert_numeric=True).dtypes

train_features = X_train.columns

#kfold = StratifiedKFold(Y_train,n_folds=10,random_state=2)
# = StratifiedKFold(Y_train,n_splits=10,random_state=2)
kfold = StratifiedKFold(Y_train,n_folds=10,random_state=2)
DTC = DecisionTreeClassifier(max_depth=3)

cv_results = cross_val_score(DTC,X_train, Y_train, cv=kfold, scoring="accuracy")
#print(type(cv_results))

print(cv_results.mean())
#print(type(DTC))
DTC.fit(X_train,Y_train)
dot_data = StringIO()
tree.export_graphviz(DTC, out_file=dot_data,feature_names=train_features,filled=True, rounded=True,special_characters=True)

graph = pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())#[0]
graph.set_lwidth(400)
graph.set_lheight(300)

#img = Image(graph.create_png(prog='dot'))
img = Image(graph.create_png())#prog='dot'
display(img)
graph.write_png("out.png")
# # treat the dot output string as an image file
# sio = StringIO()
# sio.write(png_str)
# sio.seek(0)
# img = mpimg.imread(sio)
#
# # plot the image
# imgplot = plt.imshow(img, aspect='equal')
# plt.show(block=False)
'''
def calcshan(dataset):
    lenDataSet=len(dataset)
    p={}
    H=0.0
    for data in dataset:
        currentLabel=data[-1]  #获取类别标签
        if currentLabel not in p.keys():  #若字典中不存在该类别标签，即创建
            p[currentLabel]=0
        p[currentLabel]+=1    #递增类别标签的值
    for key in p:
        px=float(p[key])/float(lenDataSet)  #计算某个标签的概率
        H-=px*log(px,2)  #计算信息熵
    return H

print calcshan(dataset)

def spiltData(dataset,axis,value):    #dataSet为要划分的数据集,axis为给定的特征，value为给定特征的具体值
    subDataSet=[]
    for data in dataset:
        subData=[]
        if data[axis]==value:
            subData=data[:axis]  #取出data中第0到axis-1个数进subData;
            subData.extend(data[axis+1:])  #取出data中第axis+1到最后一个数进subData;这两行代码相当于把第axis个数从数据集中剔除掉
            subDataSet.append(subData) #此处要注意expend和append的区别
    return subDataSet

def splitDataSet(dataset, axis, value):
    retDataSet = []
    for eachLine in dataset:
        if eachLine[axis] == value:
            reducedFeatVec = eachLine[:axis]
            reducedFeatVec.extend(eachLine[axis+1:]) # 划分数据子集，子集排除了用来划分的特征的数值
            retDataSet.append(reducedFeatVec)
    return retDataSet

print splitDataSet(dataset,0,0)


def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcshan(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset, i, value)
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * calcshan(dataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

print chooseBestFeatureToSplit(dataset)  '''