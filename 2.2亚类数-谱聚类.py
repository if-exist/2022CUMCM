import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
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

# 从sklearn.cluster中导入SpectralClustering模型
from sklearn.cluster import SpectralClustering
# 初始化SpectralClustering模型，并设置聚类中心数量为10。
sc = SpectralClustering(n_clusters=5)
sc.fit(highK_X)


"""利用轮廓系数评价不同类簇数量的K-means聚类实例"""
# 从sklearn.metrics导入silhouette_score用于计算轮廓系数。
from sklearn.metrics import silhouette_score


clusters = range(2, 10)
sc_scores = []
for t in clusters:
    sc_model = SpectralClustering(n_clusters=t).fit(highK_X)
    sc_score = silhouette_score(highK_X, sc_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)

# 绘制轮廓系数与不同类簇数量的关系曲线
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')

plt.show()

PbBa = chem[chem['类型'] == '铅钡']
