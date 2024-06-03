import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# 解决图标题中文乱码问题
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

filename = 'data/附件.xlsx'
info = pd.read_excel(filename, sheet_name='表单1')
chem = pd.read_excel(filename, sheet_name='表单2')
chem.fillna(0, inplace=True)
chem['累加和'] = chem.iloc[:, 1:].apply(lambda x: x.sum(), axis=1)
chem = chem[chem['累加和'] <= 105]
chem = chem[chem['累加和'] >= 85]
chem['文物编号'] = chem['文物采样点'].apply(lambda x: int(x[:2]))
chem['采样点'] = chem['文物采样点'].apply(lambda x: x[2:])
# chem = chem[chem['采样点'] != '严重风化点']
chem.index = list(range(len(chem)))
chem['类型'] = [0] * len(chem)
chem['表面风化'] = [0] * len(chem)
for i in chem.index:
    chem['类型'][i] = info[info['文物编号'] == chem['文物编号'][i]]['类型'].tolist()[0]
    chem['表面风化'][i] = info[info['文物编号'] == chem['文物编号'][i]]['表面风化'].tolist()[0]

highK = chem[chem['类型'] == '高钾']

# 原始数据序列
X = highK.iloc[:, 1:-5].values
# 无量纲化处理
X = MinMaxScaler().fit_transform(X)
# 求差值序列
X = abs((X - X[:, 0].reshape(len(X), 1))[:, 1:])
M_delata = X.max()
m_delta = X.min()
rho = 0.5
# 求关联系数xi
Xi = (m_delta + rho * M_delata) / (X + rho * M_delata)
gamma = Xi.mean(axis=0)
print(gamma)


highK = highK.iloc[:, 1:-5].to_numpy()

# 0.002~1 区间归一化
highK_mm = highK.copy()
ymin = 0.002
ymax = 1
for j in range(0, 14):
    d_max = max(highK[:, j])
    d_min = min(highK[:, j])
    highK_mm[:, j] = (ymax - ymin) * (highK[:, j] - d_min) / (d_max - d_min) + ymin

# print(highK_mm)
for i in [0, 1, 2, -2]:
    plt.plot(range(len(highK_mm)), highK_mm[:, i], '.-')
plt.xlabel('化学成分')
plt.legend([str(i) for i in [0, 1, 2, -2]])
plt.title('灰色关联分析')
plt.show()

# 得到其他列和参考列相等的绝对值
for i in range(1, 14):
    highK_mm[:, i] = np.abs(highK_mm[:, i] - highK_mm[:, 0])

# 得到绝对值矩阵的全局最大值和最小值
highK_mm = highK_mm[:, 1:14]
d_max = np.max(highK_mm)
d_min = np.min(highK_mm)
# 定义分辨系数
a = 0.5
# 计算灰色关联矩阵
gra_mat = (d_min + a * d_max) / (highK_mm + a * d_max)
gra_dreg = np.mean(gra_mat, axis=0)
print(gra_dreg)

# #导入数据
# data=pd.read_excel('D:\桌面\huiseguanlian.xlsx')
# # print(data)
# #提取变量名 x1 -- x7
# label_need=data.keys()[1:]
# # print(label_need)
# #提取上面变量名下的数据
# data1=data[label_need].values
# print(data1)
# #0.002~1区间归一化
# [m,n]=data1.shape #得到行数和列数
# data2=data1.astype('float')
# data3=data2
# ymin=0.002
# ymax=1
# for j in range(0,n):
#     d_max=max(data2[:,j])
#     d_min=min(data2[:,j])
#     data3[:,j]=(ymax-ymin)*(data2[:,j]-d_min)/(d_max-d_min)+ymin
# print(data3)
# # 绘制 x1,x4,x5,x6,x7 的折线图
# t=range(2007,2014)
# plt.plot(t,data3[:,0],'*-',c='red')
# for i in range(4):
#     plt.plot(t,data3[:,2+i],'.-')
# plt.xlabel('year')
# plt.legend(['x1','x4','x5','x6','x7'])
# plt.title('灰色关联分析')
# # 得到其他列和参考列相等的绝对值
# for i in range(3,7):
#     data3[:,i]=np.abs(data3[:,i]-data3[:,0])
# #得到绝对值矩阵的全局最大值和最小值
# data4=data3[:,3:7]
# d_max=np.max(data4)
# d_min=np.min(data4)
# a=0.5 #定义分辨系数
# # 计算灰色关联矩阵
# data4=(d_min+a*d_max)/(data4+a*d_max)
# xishu=np.mean(data4, axis=0)
# print(' x4,x5,x6,x7 与 x1之间的灰色关联度分别为：')
# print(xishu)