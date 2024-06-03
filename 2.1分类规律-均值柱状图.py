from pyecharts.charts import Bar
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.globals import ThemeType
from pyecharts import options as opts


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


