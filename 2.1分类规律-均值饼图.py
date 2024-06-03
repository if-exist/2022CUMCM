import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.globals import ThemeType
import plotly.graph_objects as go


def plot_pie(x_data, y_data, title=''):
    x_data = x_data
    y_data = [round(data, 2) for data in y_data]
    data_pair = [list(z) for z in zip(x_data, y_data)]  # 生成由包含对应数据的元组组成的列表
    data_pair.sort(key=lambda x: x[1])  # 按y_data中的值进行排序
    (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))  # bg_color="#2c343c"  # 设置大小和背景颜色
        .add(
            series_name="",  # 在网页中当鼠标指向时显示系列名称，在静态图中直接设置为空，此行可直接写成"",
            data_pair=data_pair,
            # 饼图的半径，数组的第一项是内半径，第二项是外半径。默认设置成百分比，相对于容器高宽中较小的一项的一半
            # radius="55%",  # 圆
            radius=["40%", "55%"],  # 环
            center=["50%", "55%"],  # 饼图的圆心坐标，数组的第一项是横坐标，第二项是纵坐标。默认设置成百分比，设置成百分比时第一项是相对于容器宽度，第二项是相对于容器高度
            # rosetype="radius",  # 是否展示成南丁格尔图（通过半径区分数据大小），有'radius'（扇区圆心角展现数据的百分比）和'area'(所有扇区圆心角相同)两种模式。
            label_opts=opts.LabelOpts(
                position="outside",
                # 关于formatter:标签内容格式器，在Pie中各变量指{a}（系列名称），{b}（数据项名称），{c}（数值）, {d}（百分比）
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
                # title=title,
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

highK = chem[chem['类型'] == '高钾']
highK = highK.iloc[:, 1:15].mean()
plot_pie(highK.index, highK.values, "高钾玻璃")

PbBa = chem[chem['类型'] == '铅钡']
PbBa = PbBa.iloc[:, 1:15].mean()
plot_pie(PbBa.index, PbBa.values, "铅钡玻璃")