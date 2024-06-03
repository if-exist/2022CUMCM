import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.globals import ThemeType

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.style.use('ggplot')


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
    # if '未风化点' in chem['采样点'][i]:
    #     chem['表面风化'][i] = '无风化'

weather = pd.get_dummies(chem['表面风化'])
chem = pd.concat([chem, weather], axis=1)
highK = chem[chem['类型'] == '高钾']
highK_X = highK.iloc[:, list(range(1, 15)) + [-2, -1]]
print(highK_X)
# 从sklearn.cluster中导入KMeans模型
from sklearn.cluster import KMeans
# 初始化KMeans模型，并设置聚类中心数量为10。
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
# 逐条判断每个测试图像所属的聚类中心。
y_predict = kmeans.predict(X_test)

"""使用ARI进行K-means聚类性能评估"""
# 从sklearn导入度量函数库metrics。
from sklearn import metrics
# 使用ARI进行KMeans聚类性能评估。
print(metrics.adjusted_rand_score(y_test, y_predict))

PbBa = chem[chem['类型'] == '铅钡']


# 获取数据集
# highK_X, labels_true = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4, random_state=0)
# 数据预处理
highK_X = StandardScaler().fit_transform(highK_X)
# 执行DBSCAN算法
db = DBSCAN(eps=0.5, min_samples=5).fit(highK_X)  # eps 邻域半径，min_samples 邻域样本数量阈值
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# 标记核心对象，用于作图
core_samples_mask[db.core_sample_indices_] = True  # core_samples_indices:核心对象在训练数据中的索引
# 算法得出的的聚类标签，-1表示样本点时噪声点，其余值表示样本点所属的类
labels = db.labels_
# 获取聚类数量
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
# 绘图
plt.style.use("ggplot")
# 黑色用作标记噪声点
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))  # 颜色分配
i = -1
# 标记样式，x表示噪声点
marker = ['v', '^', 'o', 'x']
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
    class_member_mask = (labels == k)
    i += 1
    if i >= len(unique_labels):
        i = 0
    # 绘制核心对象
    xy = highK_X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker[i], markerfacecolor=col, markeredgecolor='k', markersize=14)
    # 绘制非核心对象
    xy = highK_X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], marker[i], markerfacecolor=col, markeredgecolor='k', markersize=6)
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()

search_param(highK_X)