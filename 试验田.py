import pandas as pd
import re

'''#a和b的长度必须保持一致，否则报错
a = [x for x in range(5)]
b = [x for x in range(5,10)]
c = [x for x in range(10,15)]
d = [x for x in range(20,25)]
#字典中的key值即为csv中列名
data = {'aaa':a,'bbb':b,'d':c, 'fdfd':d}
dataframe = pd.DataFrame(data)
dataframe.to_csv('../大数据/数据/test.csv',columns = ['aaa','bbb','d','fdfd'], mode='a', header=None)
data = pd.read_csv('../大数据/数据/test.csv')
data.info()'''
'''
list = ['sss', 'ff', 'aa']
dic = {'sss':1, 'ff':3, 'aa':[45, 34]}
print(dic[list[2]])'''

string = "27/11/2018 17:45:00"
time = re.match(r'(.*)/(.*)/(.*) (.*):(.*):(.*)', string)
print(time.group(6))
