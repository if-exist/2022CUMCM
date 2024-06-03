"""聚类分析：通过肘部法和轮廓系数法确定聚类数目的最佳值"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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
chem.index = list(range(len(chem)))
chem['类型'] = [0] * len(chem)
chem['表面风化'] = [0] * len(chem)
for i in chem.index:
    chem['类型'][i] = info[info['文物编号'] == chem['文物编号'][i]]['类型'].tolist()[0]
    chem['表面风化'][i] = info[info['文物编号'] == chem['文物编号'][i]]['表面风化'].tolist()[0]


def cluster_analyse(X, K_max, title):
    X = StandardScaler().fit_transform(X)
    K = range(1, K_max)
    meandistortions = []
    from scipy.spatial.distance import cdist

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=9)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)))

    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.savefig(r'image\肘部法' + title + '.png', dpi=600)

    """利用轮廓系数评价不同类簇数量的K-means聚类实例"""
    from sklearn.metrics import silhouette_score

    clusters = range(2, K_max)
    sc_scores = []
    for t in clusters:
        kmeans_model = KMeans(n_clusters=t, random_state=9).fit(X)
        sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)

    # 绘制轮廓系数与不同类簇数量的关系曲线
    plt.figure()
    plt.plot(clusters, sc_scores, '*-')
    plt.xlabel('k')
    plt.ylabel('轮廓系数')
    plt.savefig(r'image\轮廓系数法' + title + '.png', dpi=600)
    plt.show()


highK = chem[chem['类型'] == '高钾']
highK_X = highK.iloc[:, list(range(1, 15))]
cluster_analyse(highK_X, K_max=10, title='高钾玻璃')
print(highK_X)
highK_X = StandardScaler().fit_transform(highK_X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(highK_X)
highK['亚类'] = kmeans.labels_
print(highK)


PbBa = chem[chem['类型'] == '铅钡']
PbBa_X = PbBa.iloc[:, list(range(1, 15))]
cluster_analyse(PbBa_X, K_max=8, title="铅钡玻璃")
print(PbBa_X)
PbBa_X = StandardScaler().fit_transform(PbBa_X)
kmeans = KMeans(n_clusters=8)
kmeans.fit(PbBa_X)
PbBa['亚类'] = kmeans.labels_
print(PbBa_X)

