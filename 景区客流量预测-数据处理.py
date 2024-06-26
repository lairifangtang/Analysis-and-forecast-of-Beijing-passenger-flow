import pandas as pd
import numpy as np
import csv
import re

# data = pd.read_csv('../大数据/数据/flow_realtime.csv')
# data.info()
# data.head()
# print(data.columns)
'''
# 将csv文件内数据读出
data=pd.read_csv('../大数据/数据/flow_realtime.csv')
hotel = [1, 0, 9, 27, 13, 14, 1, 5, 19, 40, 58, 22, 13, 54, 17, 51]
test = [x for x in range(3109324)]
data['周边1公里酒店数量']=test                                  #注明列名，就可以直接添加新列
data.to_csv('../大数据/数据/flow_realtime.csv',index=False)         #把数据写入数据集，index=False表示不加索引'''
'''
data = pd.read_csv('../大数据/数据/flow_realtime.csv')
data = np.array(data)
for i in range(1, 10):
    print(data[i][2])'''
'''
cdata = open(r'../大数据/数据/flow_realtime.csv', mode='r', encoding='utf-8')
clist = csv.reader(cdata)
cdict = csv.DictReader(cdata)
f = 0
for i in clist:
    print(i)
    f = f + 1
    if f == 3:
        break
'''

f = 0

# 打开初始数据
origin_data = open(r'../大数据/数据/flow_realtime.csv', mode='r', encoding='utf-8')
origin_list = csv.reader(origin_data)

with open(r'../大数据/数据/data.csv', mode='w+', newline='', encoding='utf8') as cf:
    wf = csv.writer(cf)
    # 写入表头
    wf.writerow(["序号", "景区名称", "sno", "数据获取时间", "实时客流人数", "景区舒适度指数", "周边1公里酒店数量",
                 "周边1公里地铁站数量", "景区评分", "最大瞬时承载量", "每日最大承载量", "年", "月", "日", "小时",
                 "分钟"])
    data = {'703': [1, 0, 4.9, 2.7, 10.8],
            '704': [0, 0, 4.9, 3.5, 10.4],
            '706': [9, 1, 4.9, 8, 18],
            '707': [27, 0, 4.9, 0.8, 8],
            '708': [13, 1, 4.5, 3, 5],
            '751': [14, 1, 4.9, 5.8, 17.5],
            '753': [1, 0, 4.9, 2.1, 5.5],
            '756': [5, 0, 4.6, 0.26, 2.82],
            '758': [19, 2, 4.8, 10.8, 30.2],
            '761': [40, 1, 4.5, 4.5, 6.2],
            '762': [58, 0, 4.9, 1, 3.5],
            '763': [22, 0, 4.9, 2.1, 8.4],
            '764': [13, 2, 4.9, 3.7, 15],
            '767': [54, 1, 4.6, 0.8, 3.5],
            '769': [17, 1, 4.8, 10, 30],
            '770': [51, 2, 4.9, 7.7, 27.1],
            '773': [8, 3, 4.5, 5, 25],
            '777': [18, 2, 4.6, 0.3, 1.5],
            '778': [13, 2, 4.5, 2, 10],
            '779': [4, 0, 4.5, 1, 5],
            '780': [0, 0, 4.7, 2, 10],
            '783': [16, 2, 4.6, 2, 8],
            '805': [0, 0, 4.3, 0.2, 0.5],
            '806': [23, 2, 4.6, 5, 25],
            '807': [10, 2, 4.5, 0.5, 2],
            '808': [5, 2, 4.5, 3, 10],
            '810': [7, 2, 4.6, 0.5, 3],
            '815': [15, 2, 4.7, 1, 5],
            '817': [2, 0, 4.5, 0.1, 0.5],
            '818': [3, 0, 4.6, 0.15, 0.3],
            '819': [5, 0, 4.4, 0.5, 6],
            '820': [2, 0, 4.5, 0.05, 0.2],
            '821': [6, 0, 4.8, 1.6, 4.9],
            '822': [9, 0, 4.7, 1.8, 3.5],
            '901': [82, 3, 4.9, 0.3, 0.8],
            '902': [181, 4, 4.8, 0.13, 1.01],
            '903': [205, 3, 4.9, 1.5, 7],
            '904': [63, 24.9, 0.6, 0.8],
            '905': [56, 3, 4.7, 0.5, 0.9],
            '907': [36, 3, 4.9, 0.1, 0.6],
            '908': [157, 3, 4.9, 5.3, 31.7],
            '909': [9, 0, 4.5, 3.8, 6.5],
            '910': [12, 1, 4.9, 6.9, 13.9],
            '911': [31, 1, 4.8, 0.1, 1.1],
            '912': [6, 0, 4.2, 1.1, 1.5],
            '913': [9, 1, 4.8, 0.5, 0.7],
            '914': [58, 2, 4.8, 0.3, 1.1],
            '915': [2, 0, 4.8, 5, 10],
            '916': [0, 0, 4.7, 0.7, 2.8],
            '917': [1, 0, 4.6, 0.4, 0.6],
            '918': [1, 0, 4.1, 0.2, 1.1],
            '919': [9, 0, 5, 3.7, 11],
            '920': [0, 0, 4.6, 0.2, 2],
            '921': [0, 0, 4.7, 2.11, 6.57],
            '922': [14, 2, 4.6, 0.12, 0.5],
            '923': [0, 0, 4.5, 5.5, 7.8],
            '924': [1, 0, 4.6, 5, 10],
            '925': [0, 0, 4, 0.94, 4],
            '926': [9, 0, 4.5, 0.5, 0.2],
            '927': [8, 0, 4.3, 0.2, 1.6],
            '928': [10, 0, 4.2, 0.5, 2],
            '929': [1, 0, 4.7, 0.3, 0.8],
            '930': [0, 0, 4.7, 3, 7],
            '931': [0, 0, 4.5, 0.2, 0.8],
            '932': [10, 0, 4.3, 1.1, 2],
            '933': [4, 0, 4.9, 0.8, 1.6],
            '934': [0, 0, 4.7, 1, 1.5],
            '935': [0, 0, 4.4, 0.9, 1.5],
            '936': [3, 0, 4.8, 0.8, 2],
            '937': [31, 0, 4.8, 0.8, 4.2],
            '938': [83, 2, 4.9, 0.4, 5.1],
            '939': [0, 0, 4.7, 0.8, 1.2],
            '940': [0, 0, 4.2, 1, 2],
            '941': [1, 0, 4.7, 0.2, 0.6],
            '942': [2, 0, 4.7, 0.8, 3],
            '943': [1, 0, 4.6, 0.6, 1],
            '944': [0, 0, 4.5, 0.5, 0.85],
            '945': [2, 0, 3.4, 0.8, 3.5],
            '946': [2, 2, 4.9, 21.7, 50],
            '947': [2, 2, 4.9, 21.7, 50],
            '948': [3, 0, 4.7, 8, 11],
            '949': [1, 0, 4.9, 0.4, 1.8],
            '950': [1, 0, 4.5, 0.5, 7.8],
            '951': [9, 0, 4.6, 0.3, 1.1]
            }
    for lis in origin_list:
        if f == 0:
            f = f + 1
            continue
        # print(data[lis[2]])
        time_list = []
        time_group = re.match(r'(.*)/(.*)/(.*) (.*):(.*):(.*)', lis[3])
        time_list.append(time_group.group(3))
        time_list.append(time_group.group(2))
        time_list.append(time_group.group(1))
        time_list.append(time_group.group(4))
        time_list.append(time_group.group(5))
        lis = lis + data[lis[2]] + time_list
        wf.writerow(lis)
