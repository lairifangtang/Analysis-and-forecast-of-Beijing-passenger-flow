import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')

# 获取数据
# data = pd.read_csv('../大数据/数据/data.csv')

# 打印查看数据
# data.info()
# print(data.head())

# 进行数据切片
'''
data = data[
    ['sno', '实时客流人数', '景区舒适度指数', '周边1公里酒店数量', '周边1公里地铁站数量', '景区评分', '最大瞬时承载量',
     '每日最大承载量', '年', '月', '日', '小时', '分钟']]
data = data.loc[::, 'sno':'每日最大承载量']
print(data)
print(data.corr()['实时客流人数'].sort_values())'''

'''
X, y = data[data.columns.delete(-1)], data['实时客流人数']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)'''

''' 选择相关性最高三个属性进行线性回归
print(data.corr()['实时客流人数'].abs().sort_values(ascending=False).head(4))
X, y = data[data.columns.delete(-1)], data['实时客流人数']
X2 = np.array(data[['最大瞬时承载量','景区舒适度指数','sno']])
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, random_state=1,test_size=0.2)
linear_model2 = LinearRegression()
linear_model2.fit(X2_train,y_train)
print(linear_model2.intercept_)
print(linear_model2.coef_)
line2_pre = linear_model2.predict(X2_test)  #预测值
print('SCORE:{:.4f}'.format(linear_model2.score(X2_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, line2_pre))))#RMSE(标准误差)
'''


# 对前1000个数据进行多变量研究
def multivariate_analysis(num):
    # 多变量分析
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    sns.pairplot(data[["景区舒适度指数", "景区评分", "每日最大承载量", "最大瞬时承载量", "实时客流人数"]].loc[:num])
    plt.show()

    # 绘制热力图

    # # 将数据转换为DataFrame格式
    # df = pd.DataFrame.from_dict(data,orient='columns')
    # # 计算相关性矩阵
    # corr_matrix = df.corr(numeric_only=True)  # 显式指定numeric_only参数为True

    # # 将字体设置为中文字体
    # sns.set(font='SimHei')
    # plt.figure(figsize=(12,8))
    # sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='PuBu')
    
    # # 显示热力图
    # plt.show()


# 数据处理
def data_processing(data, sample_num):
    # 进行数据切片
    data = data[
        ['sno', '实时客流人数', '景区舒适度指数', '周边1公里酒店数量', '周边1公里地铁站数量', '景区评分',
         '最大瞬时承载量',
         '每日最大承载量', '年', '月', '日', '小时', '分钟']]
    # 随机取样
    data = data.sample(n=sample_num)
    return data


# 划分训练测试集
def split_train_test(data, test_size, random_state):
    X, y = data[data.columns.delete(-1)], data['实时客流人数']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# 数据标准化
def data_standardization(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import StandardScaler
    ss_x = StandardScaler()
    X_train = ss_x.fit_transform(X_train)
    X_test = ss_x.transform(X_test)
    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = ss_y.transform(y_test.values.reshape(-1, 1))

    return X_train, X_test, y_train, y_test


# 线性回归
def linear_regression():
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    # 回归系数
    coef = linear_model.coef_
    # 预测值
    y_predict_train_linear = linear_model.predict(X_train)
    y_predict_test_linear = linear_model.predict(X_test)
    # 模型评分
    print('线性回归：')
    print('训练集SCORE:{:.4f}'.format(linear_model.score(X_train, y_train)))
    print('测试集SCORE:{:.4f}'.format(linear_model.score(X_test, y_test)))
    # RMSE(标准误差)
    print('训练集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train_linear))))
    print('测试集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test_linear))))
    # 打印回归系数
    df_coef = pd.DataFrame()
    df_coef['Title'] = data.columns.delete(-1)
    df_coef['Coef'] = coef[0]
    print(df_coef)

    # 真实值与预测值的折线图对比
    plt.rcParams['font.family'] = 'SimHei'
    plt.plot(y_test, label='真实值', zorder=1)
    plt.plot(y_predict_test_linear + np.random.normal(0, 0.01, y_test.shape), label='预测值', zorder=2)
    

    # 设置图例和标题
    plt.legend()
    plt.title('linear回归预测值与真实值对比')
    plt.show()

# 优化线性回归
def linear_regression_plus(test_size, random_state):
    from sklearn.linear_model import LinearRegression
    # 选择相关性最高的三个属性进行预测
    print(data.corr()['实时客流人数'].abs().sort_values(ascending=False).head(4))
    X, y = data[data.columns.delete(-1)], data['实时客流人数']
    new_data = data[['最大瞬时承载量', '景区舒适度指数', 'sno']]
    X2 = np.array(new_data)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=test_size, random_state=random_state)

    linear_model2 = LinearRegression()
    linear_model2.fit(X2_train, y2_train)
    # 回归系数
    coef = linear_model2.coef_
    # 预测值
    y_predict_train_linear = linear_model2.predict(X2_train)
    y_predict_test_linear = linear_model2.predict(X2_test)
    # 模型评分
    print('线性回归：')
    print('训练集SCORE:{:.4f}'.format(linear_model2.score(X2_train, y2_train)))
    print('测试集SCORE:{:.4f}'.format(linear_model2.score(X2_test, y2_test)))
    # RMSE(标准误差)
    print('训练集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y2_train, y_predict_train_linear))))
    print('测试集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y2_test, y_predict_test_linear))))
    # 打印回归系数
    df_coef = pd.DataFrame()
    df_coef['Title'] = new_data.columns
    df_coef['Coef'] = coef
    print(df_coef)

    # 评价模型图
    plt.scatter(y2_test, y_predict_test_linear,label='y')
    plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], 'k--', lw=4,label='predicted')
    plt.show()

    # 真实值与预测值的折线图对比
    # plt.rcParams['font.family'] = 'SimHei'
    # plt.plot(y2_test, label='真实值')
    # plt.plot(y_predict_test_linear, label='预测值')

    # # 设置图例和标题
    # plt.legend()
    # plt.title('ElasticNet回归预测值与真实值对比')
    # plt.show()

# 梯度提升
def gradient_boosting_regression():
    from sklearn import ensemble
    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,'learning_rate': 0.01, 'loss': 'ls'}
    # clf = ensemble.GradientBoostingRegressor(**params)
    gbr = ensemble.GradientBoostingRegressor()
    gbr.fit(X_train, np.ravel(y_train))
    # 预测值
    y_predict_train_gbr = gbr.predict(X_train)
    y_predict_test_gbr = gbr.predict(X_test)
    # 模型评分
    print('梯度提升:')
    print('训练集SCORE:{:.4f}'.format(gbr.score(X_train, y_train)))
    print('测试集SCORE:{:.4f}'.format(gbr.score(X_test, y_test)))
    # RMSE(标准误差)
    print('训练集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train_gbr))))
    print('测试集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test_gbr))))

    # x_hat = np.linspace(y_test.min() - 1, y_test.max() + 1, num=100)
    # x_hat.shape = -1, 1
    # y_hat = y_predict_test_gbr
    # lin = gbr.get_params('linear')
    
    # z是下面画图中的zorder参数的值，是指该线在图中的级别，数值越大，级别越高，
    # 在多线交叉时会显示在最上面，也就是会压住其他的线显示在最前面，这里是设置二阶拟合的级别最高

    # if hasattr(lin, 'l1_ratio_'):
    #     label += '，L1 ratio=%.2f' % lin.l1_ratio_
    # plt.plot(x_hat, y_hat, lw=5, alpha=1)

    # 真实值与预测值的折线图对比
    plt.rcParams['font.family'] = 'SimHei'
    plt.plot(y_test, label='真实值')
    plt.plot(y_predict_test_gbr, label='预测值')

    # 设置图例和标题
    plt.legend()
    plt.title('gbr回归预测值与真实值对比')
    plt.show()


# Lasso回归
def lasso_regression(alpha, max_iter):
    from sklearn.linear_model import Lasso
    # 正则化参数alpha，运行迭代最大次数max_iter
    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_train, y_train)
    # 预测值
    y_predict_train_lasso = lasso.predict(X_train)
    y_predict_test_lasso = lasso.predict(X_test)
    # 模型评分
    print('Lasso回归:')
    print('训练集SCORE:{:.4f}'.format(lasso.score(X_train, y_train)))
    print('测试集SCORE:{:.4f}'.format(lasso.score(X_test, y_test)))
    # RMSE(标准误差)
    print('训练集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train_lasso))))
    print('测试集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test_lasso))))

    # 真实值与预测值的折线图对比
    plt.rcParams['font.family'] = 'SimHei'
    plt.plot(y_test, label='真实值')
    plt.plot(y_predict_test_lasso, label='预测值')

    # 设置图例和标题
    plt.legend()
    plt.title('lasso回归预测值与真实值对比')
    plt.show()
    # 评价模型图
    # plt.scatter(y_test, y_predict_test_lasso,label='y')
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4,label='predicted')
    # plt.show()


# ElasticNet回归
def elastic_net():
    from sklearn.linear_model import ElasticNet
    enet = ElasticNet()
    enet.fit(X_train, y_train)
    # 预测值
    y_predict_train_enet = enet.predict(X_train)
    y_predict_test_enet = enet.predict(X_test)
    # 模型评分
    print('ElasticNet回归:')
    print('训练集SCORE:{:.4f}'.format(enet.score(X_train, y_train)))
    print('测试集SCORE:{:.4f}'.format(enet.score(X_test, y_test)))
    # RMSE(标准误差)
    print('训练集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train_enet))))
    print('测试集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test_enet))))


    # x_hat = np.linspace(y_test.min() - 1, y_test.max() + 1, num=100)
    # x_hat.shape = -1, 1
    # y_hat = y_predict_test_enet
    # lin = enet.get_params('linear')
    

    # if hasattr(lin, 'l1_ratio_'):
    #     label += '，L1 ratio=%.2f' % lin.l1_ratio_
    # # plt.plot(x_hat, y_hat, lw=5, alpha=1)
    plt.rcParams['font.family'] = 'SimHei'
    plt.plot(y_test, label='真实值')
    plt.plot(y_predict_test_enet, label='预测值')

    # 设置图例和标题
    plt.legend()
    plt.title('ElasticNet回归预测值与真实值对比')

    # 评价模型图
    # plt.scatter(y_test, y_predict_test_enet,label='y')
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4,label='predicted')
    plt.show()


# 支持向量回归（SVR）
def svr_model(kernel):
    from sklearn.svm import SVR
    svr = SVR(kernel=kernel)
    svr.fit(X_train, np.ravel(y_train))
    # 预测值
    y_predict_train_svr = svr.predict(X_train)
    y_predict_test_svr = svr.predict(X_test)
    # score(): Returns the coefficient of determination R^2 of the prediction.
    # 模型评分
    print('支持向量回归:')
    print(kernel, ' 训练集SCORE:{:.4f}'.format(svr.score(X_train, y_train)))
    print(kernel, ' 测试集SCORE:{:.4f}'.format(svr.score(X_test, y_test)))
    # RMSE(标准误差)
    print(kernel, ' 训练集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train_svr))))
    print(kernel, ' 测试集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test_svr))))

    # 真实值与预测值的折线图对比
    plt.rcParams['font.family'] = 'SimHei'
    plt.plot(y_test, label='真实值')
    plt.plot(y_predict_test_svr, label='预测值')

    # 设置图例和标题
    plt.legend()
    plt.title('SVR回归预测值与真实值对比')
    plt.show()
    # 评价模型图
    # plt.scatter(y_test, y_predict_test_svr,label='y')
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4,label='predicted')
    # plt.show()


# 决策树回归
def decision_tree(max_depth):
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor(max_depth=max_depth)
    tree_reg.fit(X_train, y_train)
    # 预测值
    y_predict_train_tree_reg = tree_reg.predict(X_train)
    y_predict_test_tree_reg = tree_reg.predict(X_test)
    # 模型评分
    print('决策树回归:')
    print('训练集SCORE:{:.4f}'.format(tree_reg.score(X_train, y_train)))
    print('测试集SCORE:{:.4f}'.format(tree_reg.score(X_test, y_test)))
    # RMSE(标准误差)
    print('训练集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_train, y_predict_train_tree_reg))))
    print('测试集RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, y_predict_test_tree_reg))))

    # 真实值与预测值的折线图对比
    plt.rcParams['font.family'] = 'SimHei'
    plt.plot(y_test, label='真实值')
    plt.plot(y_predict_test_tree_reg, label='预测值')

    # 设置图例和标题
    plt.legend()
    plt.title('tree_reg回归预测值与真实值对比')
    plt.show()
    # 评价模型图
    # plt.scatter(y_test, y_predict_test_tree_reg,label='y')
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4,label='predicted')
    # plt.show()


data_path = '../大数据/数据/data.csv'

# 获取数据
data = pd.read_csv(data_path)

# 数据分析
# data.info()
# data.head()
# multivariate_analysis(num=500)
print(data.corr()['实时客流人数'].abs().sort_values(ascending=False))

# 数据处理
data = data_processing(data=data, sample_num=1000)

# 划分训练测试集
X_train, X_test, y_train, y_test = split_train_test(data=data, test_size=0.2, random_state=0)

# 数据标准化
X_train, X_test, y_train, y_test = data_standardization(X_train, X_test, y_train, y_test)


# 模型选择

# 线性回归
linear_regression()

# 优化线性回归
# linear_regression_plus(test_size=0.2, random_state=10)


# 梯度提升回归
# gradient_boosting_regression()

# Lasso回归
# lasso_regression(0.2, 1000)

# ElasticNet回归
# elastic_net()

# SVR
# svr_model(kernel='linear')    # 线性核函数
# svr_model(kernel='poly')      # 多项式核函数
# svr_model(kernel='rbf')       # 径向基函数

# 决策树回归
# decision_tree(max_depth=3)
