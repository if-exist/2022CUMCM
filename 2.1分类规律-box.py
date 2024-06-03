from pyecharts.charts import Boxplot
import pandas as pd
import matplotlib.pyplot as plt
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
PbBa = chem[chem['类型'] == '铅钡']
x_label1 = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3']
highK_data1 = highK.iloc[:, 1:8].to_numpy().T
PbBa_data1 = PbBa.iloc[:, 1:8].to_numpy().T
plot_box(x_label1, highK_data1, PbBa_data1, "前七")


x_label2 = ['CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
highK_data2 = highK.iloc[:, 8:15].to_numpy().T
PbBa_data2 = PbBa.iloc[:, 8:15].to_numpy().T
plot_box(x_label2, highK_data2, PbBa_data2, "后七")



