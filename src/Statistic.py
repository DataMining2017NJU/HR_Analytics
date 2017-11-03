from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#data_name = ['satisfaction_level','last_evaluation','number_project',"average_montly_hours","time_spend_company","Work_accident","left","promotion_last_5years"]
local_path = "/Users/lixuefei/Desktop/file/DM/work/HR_comma_sep.csv"
dataset = pd.read_csv(local_path)#names=data_name,

mode_s_l = dataset["satisfaction_level"].mode()
mode_l_e = dataset["last_evaluation"].mode()
mode_n_p = dataset["number_project"].mode()
mode_a_m_h = dataset["average_montly_hours"].mode()
mode_t_s_c = dataset["time_spend_company"].mode()
mode_W_a = dataset["Work_accident"].mode()
mode_l = dataset["left"].mode()
mode_p_l_5 = dataset["promotion_last_5years"].mode()
print("_______")
#print(mode_s_l,mode_l_e,mode_n_p,mode_a_m_h,mode_t_s_c,mode_W_a,mode_l,mode_p_l_5)
print(mode_a_m_h)
median_s_l = dataset["satisfaction_level"].median()
median_l_e = dataset["last_evaluation"].median()
median_n_p = dataset["number_project"].median()
median_a_m_h = dataset["average_montly_hours"].median()
median_t_s_c = dataset["time_spend_company"].median()
median_W_a = dataset["Work_accident"].median()
median_l = dataset["left"].median()
median_p_l_5 = dataset["promotion_last_5years"].median()

print("_______")
#print(median_s_l,median_l_e,median_n_p,median_a_m_h,median_t_s_c,median_W_a,median_l,median_p_l_5)


'''
fig, axs = plt.subplots(ncols=2,figsize=(12,6))
g = sns.countplot(dataset["satisfaction_level"], ax=axs[0])
plt.setp(g.get_xticklabels(), rotation=45)
g = sns.countplot(dataset["salary"], ax=axs[1])
plt.tight_layout()
plt.show()
plt.gcf().clear()

#print "hello"
'''