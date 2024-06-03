"""绘制不同类别玻璃化学成分的小提琴图和饼图，计算各个化学成分的基本统计量"""
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.globals import ThemeType
import plotly.graph_objects as go
import pandas as pd
from PIL import Image


# 绘制小提琴图
def plot_violin(df1x, df1y, df2x, df2y, title):
    trace1 = go.Violin(
        x=df1x,
        y=df1y,
        legendgroup='Yes',
        scalegroup='Yes',
        name='风化',
        side='negative',
        box_visible=True,
        meanline_visible=True,
    )

    trace2 = go.Violin(
        x=df2x,
        y=df2y,
        legendgroup='No',
        scalegroup='No',
        name='无风化',
        side='positive',
        box_visible=True,  # 显示箱线图
        meanline_visible=True,  # 显示mean线
    )

    layout = go.Layout(
        title=dict(
            x=0.5,
            yanchor='top',
        ),
        xaxis=dict(
            title='化学成分',
            ticks='outside',
        ),
        yaxis=dict(
            title='含量',
            ticks='outside',
        ),
        violingap=0,
        violinmode='overlay',
    )

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)

    fig.show()
    filename = "image/" + title + ".png"
    fig.write_image(filename, scale=6)
    Image.open(filename).show()


# 绘制饼图
def plot_pie(x_data, y_data, title=''):
    x_data = x_data
    y_data = [round(data, 2) for data in y_data]
    data_pair = [list(z) for z in zip(x_data, y_data)]  # 生成由包含对应数据的元组组成的列表
    data_pair.sort(key=lambda x: x[1])  # 按y_data中的值进行排序
    (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
        .add(
            series_name="",
            data_pair=data_pair,
            radius=["40%", "55%"],  # 环
            center=["50%", "55%"],
            label_opts=opts.LabelOpts(
                position="outside",
                formatter="{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
                background_color="#eee",
                border_color="#aaa",
                border_width=1,
                border_radius=4,
                # 富文本
                rich={
                    "hr": {
                        "borderColor": "#aaa",
                        "width": "100%",
                        "borderWidth": 0.5,
                        "height": 0,
                    },
                    "b": {"fontSize": 12, "lineHeight": 33},
                    "per": {
                        "color": "#eee",
                        "backgroundColor": "#334455",
                        "padding": [2, 4],
                        "borderRadius": 2,
                    },
                },
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                pos_left="center",
                pos_top="40",
                title_textstyle_opts=opts.TextStyleOpts(color="#2c343c"),
                ),
            legend_opts=opts.LegendOpts(
                is_show=False,
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
        .render("output/" + title + "pie.html")
    )


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

filename = 'data/附件.xlsx'
info = pd.read_excel(filename, sheet_name='表单1')
chem = pd.read_excel(filename, sheet_name='表单2')
# 用0填充空值
chem.fillna(0, inplace=True)
# 筛选有效数据
chem['累加和'] = chem.iloc[:, 1:].apply(lambda x: x.sum(), axis=1)
chem = chem[chem['累加和'] <= 105]
chem = chem[chem['累加和'] >= 85]
chem['文物编号'] = chem['文物采样点'].apply(lambda x: int(x[:2]))
chem.index = list(range(len(chem)))
chem['类型'] = [0] * len(chem)
chem['表面风化'] = [0] * len(chem)
for i in chem.index:
    chem['类型'][i] = info[info['文物编号'] == chem['文物编号'][i]]['类型'].tolist()[0]
    chem['表面风化'][i] = info[info['文物编号'] == chem['文物编号'][i]]['表面风化'].tolist()[0]

PbBa = chem[chem['类型'] == '铅钡']
PbBa_weat = PbBa[PbBa['表面风化'] == '风化']
PbBa_unweat = PbBa[PbBa['表面风化'] == '无风化']


dy_weat = PbBa_weat.iloc[:, 1:-4].to_numpy().T.ravel()
dx_weat = []
for col in PbBa_weat.columns[1:-4]:
    dx_weat += [col] * len(PbBa_weat)


print(PbBa_unweat)
dy_unweat = PbBa_unweat.iloc[:, 1:-4].to_numpy().T.ravel()
dx_unweat = []
for col in PbBa_unweat.columns[1:-4]:
    dx_unweat += [col] * len(PbBa_unweat)

plot_violin(dx_weat, dy_weat, dx_unweat, dy_unweat, '铅钡玻璃violin')
print(PbBa_weat)
print(PbBa_unweat)
for col in PbBa.columns[1:-4]:
    print("\n风化")
    print(PbBa_weat[col].describe())
    print("\n无风化")
    print(PbBa_unweat[col].describe())
    print("\n-------------------------------")

PbBa_weat_mean = PbBa_weat.iloc[:, 1:-4].mean()
print("铅钡玻璃风化均值：", PbBa_weat_mean)
print("均值为0：", PbBa_weat_mean[PbBa_weat_mean == 0].index)
PbBa_weat_mean = PbBa_weat_mean[PbBa_weat_mean > 0]
plot_pie(PbBa_weat_mean.index, PbBa_weat_mean.values, title='风化铅钡玻璃')

PbBa_unweat_mean = PbBa_unweat.iloc[:, 1:-4].mean()
print("铅钡玻璃未风化均值：", PbBa_unweat_mean)
print("均值为0：", PbBa_unweat_mean[PbBa_unweat_mean == 0].index)
PbBa_unweat_mean = PbBa_unweat_mean[PbBa_unweat_mean > 0]
plot_pie(PbBa_unweat_mean.index, PbBa_unweat_mean.values, title='无风化铅钡玻璃')


