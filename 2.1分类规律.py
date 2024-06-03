"""分别绘制高钾玻璃和铅钡玻璃的化学成分箱线图，均值饼图和均值柱状图"""
from pyecharts.charts import Boxplot, Bar, Pie
import pandas as pd
from pyecharts import options as opts
from pyecharts.globals import ThemeType


def plot_box(x_label, y_data1, y_data2, title):
    c = Boxplot(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
    c.add_xaxis(x_label)
    c.add_yaxis("高钾玻璃", c.prepare_data(y_data1))
    c.add_yaxis("铅钡玻璃", c.prepare_data(y_data2))
    c.set_global_opts(
        title_opts=opts.TitleOpts(
            title="",
            pos_left="center",
        ),
        xaxis_opts=opts.AxisOpts(
            name="化学成分",
            name_location="middle",
            name_gap=25,
            name_textstyle_opts=opts.TextStyleOpts(
                font_size=16,
                font_weight="bold",
            )
        ),
        yaxis_opts=opts.AxisOpts(
            name="含量",
            name_location="middle",
            name_gap=35,
            name_textstyle_opts=opts.TextStyleOpts(
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
    c.render("output/" + title + "boxplot.html")


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


def plot_bar(x_label, y_data1, y_data2, title):
    (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE))
        .add_xaxis(x_label)
        # 同一系列的柱间距离，默认为类目间距的 20%，可设固定值.直方图中设置为0.单柱形图间距为"40%"为宜
        .add_yaxis("高钾玻璃", y_data1, category_gap="20%")
        .add_yaxis("铅钡玻璃", y_data2, category_gap="20%")
        .set_series_opts(label_opts=opts.LabelOpts(position="top"))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                name="化学成分",
                name_location="middle",
                name_gap=25,
                name_textstyle_opts=opts.TextStyleOpts(
                    # color="#d14a61",
                    font_size=16,
                    font_weight="bold",
                )
            ),
            yaxis_opts=opts.AxisOpts(
                name="含量",
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
        # 翻转坐标轴
        # .reversal_axis()
        .render("output/" + title + "bar.html")
    )


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
chem.index = list(range(len(chem)))
chem['类型'] = [0] * len(chem)
chem['表面风化'] = [0] * len(chem)
for i in chem.index:
    chem['类型'][i] = info[info['文物编号'] == chem['文物编号'][i]]['类型'].tolist()[0]
    chem['表面风化'][i] = info[info['文物编号'] == chem['文物编号'][i]]['表面风化'].tolist()[0]

highK = chem[chem['类型'] == '高钾']
PbBa = chem[chem['类型'] == '铅钡']
x_label1 = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3']
highK_data1 = highK.iloc[:, 1:8].to_numpy().T
PbBa_data1 = PbBa.iloc[:, 1:8].to_numpy().T
plot_box(x_label1, highK_data1, PbBa_data1, "前七")

x_label2 = ['CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
highK_data2 = highK.iloc[:, 8:15].to_numpy().T
PbBa_data2 = PbBa.iloc[:, 8:15].to_numpy().T
plot_box(x_label2, highK_data2, PbBa_data2, "后七")

highK = chem[chem['类型'] == '高钾']
highK = highK.iloc[:, 1:15].mean()
plot_pie(highK.index, highK.values, "高钾玻璃")

PbBa = chem[chem['类型'] == '铅钡']
PbBa = PbBa.iloc[:, 1:15].mean()
plot_pie(PbBa.index, PbBa.values, "铅钡玻璃")


highK = chem[chem['类型'] == '高钾']
PbBa = chem[chem['类型'] == '铅钡']
x_label1 = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3']
highK_data1 = highK.iloc[:, 1:8].mean().apply(lambda x: round(x, 2)).to_list()
PbBa_data1 = PbBa.iloc[:, 1:8].mean().apply(lambda x: round(x, 2)).to_list()
plot_bar(x_label1, highK_data1, PbBa_data1, "前七")


x_label2 = ['CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
highK_data2 = highK.iloc[:, 8:15].mean().apply(lambda x: round(x, 2)).to_list()
PbBa_data2 = PbBa.iloc[:, 8:15].mean().apply(lambda x: round(x, 2)).to_list()
plot_bar(x_label2, highK_data2, PbBa_data2, "后七")