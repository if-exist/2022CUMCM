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


# 调参确定eps和min_samples的最佳值
def show_heatmap(value, x_label, y_label, title):
    c = (
        HeatMap(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
        .add_xaxis(x_label)
        .add_yaxis(
            "",
            y_label,
            value,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=title,
                pos_left="center",
            ),
            visualmap_opts=opts.VisualMapOpts(
                # is_show=False,
                max_=max(i[2] for i in value),
                min_=min(i[2] for i in value),
                range_color=['rgb(255,255,217)', 'rgb(237,248,177)', 'rgb(199,233,180)', 'rgb(127,205,187)',
                             'rgb(65,182,196)', 'rgb(29,145,192)', 'rgb(34,94,168)', 'rgb(37,52,148)', 'rgb(8,29,88)'],
                pos_right="right",
                pos_top="center",
            ),
            xaxis_opts=opts.AxisOpts(
                name="eps",
                name_location="middle",
                name_gap=25,
                name_textstyle_opts=opts.TextStyleOpts(
                    # color="#d14a61",
                    font_size=16,
                    font_weight="bold",
                )
            ),
            yaxis_opts=opts.AxisOpts(
                name="min_samples",
                name_location="middle",
                name_gap=35,
                name_textstyle_opts=opts.TextStyleOpts(
                    # color="#675bba",
                    font_size=16,
                    font_weight="bold",
                )
            ),
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        pixel_ratio=6,  # 设置保存图片的分辨率
                        background_color='white',  # 设置导出图片的背景颜色
                    )
                )
            ),
        )
        .render(title + ".html")
    )


# 根据轮廓系数进行调参
def search_param(X):
    sc_scores, cd_scores = [], []
    eps_range = np.linspace(0.025, 0.45, 18)
    min_samples_range = range(2, 18)
    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)  # eps 邻域半径，默认0.5，min_samples 邻域样本数量阈值，默认5
            if len(set(db.labels_)) <= 1:
                print(eps)
                sc_scores.append(0)
                cd_scores.append(0)
            else:
                sc_score = silhouette_score(X, db.labels_, metric='euclidean')
                sc_scores.append(sc_score)
                cd_score = calinski_harabasz_score(X, db.labels_)
                cd_scores.append(cd_score)
    print(sc_scores)
    sc_value, cd_value = [], []
    k = 0
    for i in range(len(eps_range)):
        for j in range(len(min_samples_range)):
            sc_value.append([i, j, round(sc_scores[k], 2)])
            cd_value.append([i, j, round(sc_scores[k], 2)])
            k += 1
    x_label = [str(round(i, 3)) for i in eps_range]
    y_label = [str(i) for i in min_samples_range]
    show_heatmap(sc_value, x_label, y_label, "output/不同eps和min_samples取值下的轮廓系数")
    show_heatmap(cd_value, x_label, y_label, "output/不同eps和min_samples取值下的Calinski-Harabaz指数")


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


PbBa = chem[chem['类型'] == '铅钡']


# 获取数据集
# highK_X, labels_true = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4, random_state=0)
# 数据预处理
# highK_X = StandardScaler().fit_transform(highK_X)
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