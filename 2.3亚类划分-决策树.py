import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType
import warnings
warnings.filterwarnings('ignore')


def plot_line(y, y_label, title):
    c = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
        .add_xaxis(xaxis_data=[str(i) + "%" for i in range(1, 41)])
        .add_yaxis(
            series_name='',
            y_axis=y,
            label_opts=opts.LabelOpts(is_show=False),
            is_smooth=True
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name='随机波动幅度'),
            yaxis_opts=opts.AxisOpts(
                name=y_label,
                type_="value",
                splitline_opts=opts.SplitLineOpts(is_show=True),
                min_=0.5,
                max_=1
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
        .render('output/决策树' + title + y_label + '.html')
    )


def sens_analyse(model, X, y_true, title):
    scores = []
    pred_probas = []
    for _ in range(1000):
        score_round = []
        proba_round = []
        for i in range(1, 41):
            X_wave = X * (np.random.uniform(low=(100 - i) / 100, high=(100 + i) / 100, size=(len(X), 14)))
            score_round.append(model.score(X_wave, y_true))
            proba_round.append(model.predict_proba(X_wave[[5], :]))
        scores.append(score_round)
        pred_probas.append(proba_round)
    scores = np.array(scores).mean(axis=0)
    scores = [round(score, 2) for score in scores]
    pred_probas = np.array(pred_probas).mean(axis=0)
    pred_probas = [round(max(proba[0]), 2) for proba in pred_probas]

    print(scores)
    print(pred_probas)

    plot_line(scores, '精确度', title)
    plot_line(pred_probas, '预测概率值', title)


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

weather = pd.get_dummies(chem['表面风化'])
chem = pd.concat([chem, weather], axis=1)

highK = chem[chem['类型'] == '高钾']
highK_X = highK.iloc[:, list(range(1, 15))]
highK_X = StandardScaler().fit_transform(highK_X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(highK_X)
highK['亚类'] = kmeans.labels_


highK_X = highK.iloc[:, list(range(1, 15))]
highK_y = highK['亚类']
# 初始化决策树分类器
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(highK_X, highK_y)
plot_tree(dtc,
          feature_names=highK_X.columns,
          class_names=['亚类Ⅰ', '亚类Ⅱ', '亚类Ⅲ', '亚类Ⅳ', '亚类Ⅴ', '亚类Ⅵ'],
          filled=True,
          rounded=True)
plt.savefig('image/高钾玻璃决策树.png', dpi=600)
highK_X = highK_X.to_numpy()
sens_analyse(dtc, highK_X, highK['亚类'], "高钾玻璃")


PbBa = chem[chem['类型'] == '铅钡']
PbBa_X = PbBa.iloc[:, list(range(1, 15))]
PbBa_X = StandardScaler().fit_transform(PbBa_X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(PbBa_X)
PbBa['亚类'] = kmeans.labels_
PbBa_X = PbBa.iloc[:, list(range(1, 15))]
PbBa_y = PbBa['亚类']
# 初始化决策树分类器
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(PbBa_X, PbBa_y)
plot_tree(dtc,
          feature_names=PbBa_X.columns,
          class_names=['亚类Ⅰ', '亚类Ⅱ', '亚类Ⅲ', '亚类Ⅳ', '亚类Ⅴ', '亚类Ⅵ'],
          filled=True,
          rounded=True
          )
plt.savefig('image/铅钡玻璃决策树.png', dpi=600)


