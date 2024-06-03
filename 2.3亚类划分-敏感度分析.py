"""根据选定的最佳k值训练kmeans模型，赋给每个样本一个亚类标签，并以此训练决策树模型，得到具体的亚类划分方法，并对训练后的决策树模型进行敏感度测试"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
import plotly.graph_objects as go
from PIL import Image
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType
import warnings
warnings.filterwarnings('ignore')


def plot_line_accu(y, y_label, title):
    x = [str(i) + "%" for i in range(1, 41)]

    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines',
    )

    layout = go.Layout(
        xaxis=dict(
            title='随机波动幅度',
            ticks='outside',
        ),
        yaxis=dict(
            title=y_label,
            ticks='outside',
        ),
    )

    fig = go.Figure(data=trace, layout=layout)
    fig.show()
    filename = "image/决策树" + title + y_label + ".png"
    fig.write_image(filename, scale=6)
    Image.open(filename).show()


def plot_line_prob(y1, y2, y_label1, y_label2, title):
    c = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
        .add_xaxis(xaxis_data=[str(i) + "%" for i in range(1, 41)])
        .add_yaxis(
            series_name=y_label1,
            y_axis=y1,
            label_opts=opts.LabelOpts(is_show=False),
            is_smooth=True
        )
        .add_yaxis(
            series_name=y_label2,
            y_axis=y2,
            label_opts=opts.LabelOpts(is_show=False),
            is_smooth=True
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name='随机波动幅度'),
            yaxis_opts=opts.AxisOpts(
                name="预测概率值",
                type_="value",
                splitline_opts=opts.SplitLineOpts(is_show=True),
                min_=0,
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
        .render('output/决策树' + title + '预测概率值.html')
    )


def sens_analyse(model, X, y_true, y_label1, y_label2, title):
    scores = []
    pred_probas1 = []
    pred_probas2 = []
    for _ in range(1000):
        score_round = []
        proba_round1 = []
        proba_round2 = []
        for i in range(1, 41):
            X_wave = X * (np.random.uniform(low=(100 - i) / 100, high=(100 + i) / 100, size=(len(X), 14)))
            score_round.append(model.score(X_wave, y_true))
            proba_round1.append(model.predict_proba(X_wave[[3], :]))
            proba_round2.append(model.predict_proba(X_wave[[7], :]))
        scores.append(score_round)
        pred_probas1.append(proba_round1)
        pred_probas2.append(proba_round2)
    scores = np.array(scores).mean(axis=0)
    scores = [round(score, 2) for score in scores]
    pred_probas1 = np.array(pred_probas1).mean(axis=0)
    pred_probas2 = np.array(pred_probas2).mean(axis=0)
    pred_probas1 = [round(max(proba[0]), 2) for proba in pred_probas1]
    pred_probas2 = [round(max(proba[0]), 2) for proba in pred_probas2]

    print(scores)

    plot_line_accu(scores, '精确度', title)
    plot_line_prob(pred_probas1, pred_probas2, y_label1, y_label2, title)


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


highK = chem[chem['类型'] == '高钾']
highK_X = highK.iloc[:, list(range(1, 15))]
highK_X = StandardScaler().fit_transform(highK_X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(highK_X)
highK['亚类'] = kmeans.labels_
print(highK.loc[:, ['二氧化硅(SiO2)', '二氧化硫(SO2)', '氧化钙(CaO)', '氧化钡(BaO)', '氧化镁(MgO)', '亚类']])

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
sens_analyse(dtc, highK_X, highK['亚类'], '文物编号4', '文物编号7', "高钾玻璃")


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

PbBa_X = PbBa_X.to_numpy()
sens_analyse(dtc, PbBa_X, PbBa['亚类'], '文物编号11', '文物编号24', "铅钡玻璃")

