from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from tqdm.tk import trange
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# original_data = pd.read_csv(r'F:\GAN工作\数据集\MD-Train-721.txt',sep='\s+',header=None).to_numpy()

# data = pd.read_csv(r'F:\Training-MP.txt',sep='\s+',header=None).fillna(0)
# data.columns = ['Pd', 'Dw', 'I', 'a', 'K', 'P', 'D', 'c', 'De']#5变量
data = pd.read_csv(r'F:\GAN工作\数据集\real+gan1442.txt',sep='\s+',header=None).fillna(0)
data.columns = ['I', 'Pd', 'Rs', 'Pb', 'T', 'uw', 'Aext', 'nc', 'κ', 'ps', 'εacc', 'εtot', 'K']
feats = [feat for feat in data.columns if feat not in ["K"]]
X = data[feats]
y = data["K"]
# y = np.log10(data["K"])

# X_test = pd.read_csv(r'F:\Test-MP.txt', sep='\s+', header=None, encoding='utf-8').to_numpy()[:, :-1]
# Y_test = pd.read_csv(r'F:\Test-MP.txt', sep='\s+', header=None, encoding='utf-8').to_numpy()[:, -1]
# X_test = pd.read_csv(r'F:\GAN\dataset\测试集归一化.txt', sep='\s+', header=None, encoding='utf-8').to_numpy()[:, :-1]
# Y_test = pd.read_csv(r'F:\GAN\dataset\测试集归一化.txt', sep='\s+', header=None, encoding='utf-8').to_numpy()[:, -1]
# Y_test = np.log10(Y_test)
####还原归一化

def drawing(x, y, K, database):
    plt.figure()  # 创建一个新的图形
    if database == "train":
        plt.plot(x, x, c='C3', label='train')
        fmt = 'o'
    elif database == "vail":
        plt.plot(x, x, c='C3', label='vali')
        fmt = '*'
    else:
        plt.plot(x, x, c='C3', label='test')
        fmt = '+'
    plt.scatter(x, y, marker=fmt, color='g')  # 画误差棒
    plt.title(f'第{K+1}次 + {database} 数据图')
    plt.legend()  # 显示图例
    plt.show()



"""只有K 的值"""
# def X_denormalized(X_normalized, original_data):
#     min_vals = original_data[:, -1].min()#列的最小值
#     # print('min_vals:', min_vals)
#     max_vals = original_data[:, -1].max()#列的最大值
#     # print('max_vals:', max_vals)
#     denormalized_X = X_normalized* (max_vals - min_vals) + min_vals
#     return denormalized_X
# k = 10**X_denormalized(y, original_data)

def plot_fig(num, model, n, data1, data2, data3, dpi=500):
    #num表示你第几次的结果
    import shap
    import matplotlib.pyplot as plt
    explainer = shap.TreeExplainer(model)
    # ['I', 'Pd', 'Rs', 'Pb', 'T', 'K']
    if (num - 1) == n:
        shap_values = explainer.shap_values(data1)
        shap.initjs()
        Y_pred_test = model.predict(data2, num_iteration=model.best_iteration)
        score_test = r2_score(data3, Y_pred_test)
        print("第" + str(round(n+1)) + "次的预测R2:", score_test)
        fig = plt.figure()
        shap.summary_plot(shap_values, data1, plot_type='bar')  # 绘制shap直方图
        fig.savefig(r'F:\GAN工作\图形\shap-hist.png', dpi=dpi, fmt='png')
        fig = plt.figure()
        shap.summary_plot(shap_values[:, :], data1.iloc[:, :])  # 绘制shap散点图
        fig.savefig(r'F:\GAN工作\图形\shap.png', dpi=dpi, fmt='png')
        #['I', 'Pd', 'Rs', 'Pb', 'T', 'uw', 'Aext', 'nc', 'κ', 'ps', 'εacc', 'εtot']
        shap.plots.partial_dependence("I", model.predict, data1, ice=False)
        shap.plots.partial_dependence("Pd", model.predict, data1, ice=False)
        shap.plots.partial_dependence("Rs", model.predict, data1, ice=False)
        shap.plots.partial_dependence("Pb", model.predict, data1, ice=False)
        shap.plots.partial_dependence("T", model.predict, data1, ice=False)
        shap.plots.partial_dependence("uw", model.predict, data1, ice=False)
        shap.plots.partial_dependence("Aext", model.predict, data1, ice=False)
        shap.plots.partial_dependence("nc", model.predict, data1, ice=False)
        shap.plots.partial_dependence("κ", model.predict, data1, ice=False)
        shap.plots.partial_dependence("ps", model.predict, data1, ice=False)
        shap.plots.partial_dependence("εacc", model.predict, data1, ice=False)
        shap.plots.partial_dependence("εtot", model.predict, data1, ice=False)
        # shap_fig(shap_values, data1) ,
        print("绘制完毕！")


#%%
Train = []
Vail = []
Test = []
R = []
'参数设置'
n_splits = 5 #设置几倍交叉验证
# X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, y, test_size=0.2, random_state=20)#45, test_size=0.2,

for i in range(1):

    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, y, test_size=0.2, random_state=72)#72
    l=(i+1)/100
    params = {'num_leaves':30,  #每棵树的最多叶子数
          'min_data_in_leaf':20,#2
          'objective': 'regression',  #目标函数
          'max_depth':9,  #每棵树的最大深度，防止过拟合。2
          'learning_rate':0.1,  #学习率，初始状态建议选择较大的学习率，设置为0.1.  0.0005 0.15
          # "min_sum_hessian_in_leaf": 1e-3,
          "boosting": "gbdt",  #设置提升类型，学习器模型算法。‘gbdt’： 表示传统的梯度提升决策树。
          "feature_fraction":0.6,  #每次新建一棵树时，随机使用多少的特征 建树的特征选择比例0.8 9
          "bagging_freq":4,  # 每建立多少棵树，就进行一次bagging。2
          "bagging_fraction":1,
          "bagging_seed":3, #一个整数，表示bagging 的随机数种子5 19
          "lambda_l1": 0.0, #一个浮点数，表示L1正则化系数。默认为0 0.6 0.4
          "lambda_l2": 0.0,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mae',
          #'subsample':1,
          "random_state":None, }


    '''以下交叉验证'''
    # kf = KFold(n_splits=n_splits, shuffle=False,random_state=None)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=62)#64
    # for train_index, vail_index in kf.split(X):
    for k, (train_index, vail_index) in enumerate(kf.split(X_train1)):
        L = len(train_index) / len(vail_index)#输出训练集与测试集的比值
        print("训练集与测试集的比值：", L)
        X_train = X_train1.iloc[train_index]
        y_train = Y_train1.iloc[train_index]
        X_vail = X_train1.iloc[vail_index]
        y_vail = Y_train1.iloc[vail_index]
        # print(vail_index)

        lgb_train = lgb.Dataset(X_train, y_train)  ## 将数据保存到LightGBM二进制文件将使加载更快
        lgb_eval = lgb.Dataset(X_vail, y_vail, reference=lgb_train)  # 创建验证数据
        lgbm = lgb.train(params,
                         lgb_train,
                         num_boost_round=10000,  # 迭代次数
                         valid_sets=lgb_eval,
                         verbose_eval=500,  # 一棵棵树
                         early_stopping_rounds=500
                         )

        lgbm.save_model(r'F:/GAN_model_1/LightGBM_model_1442_{}.txt'.format(k + 1))

        Y_pred_train = lgbm.predict(X_train, num_iteration=lgbm.best_iteration)#验证结果
        # drawing(y_train, Y_pred_train, k, "train")
        # Y_pred_train_1 = 10**X_denormalized(Y_pred_train, original_data)
        # y_train_1 = 10**X_denormalized(y_train, original_data)
        score_train = r2_score(y_train, Y_pred_train)
        rmse_train = mean_squared_error(y_train, Y_pred_train)**0.5
        print("Train第" + str(round(k+1)) + "交叉分数R2:", score_train)
        data_train = np.column_stack((y_train, Y_pred_train))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM-train-pred-1442-{}.txt'.format(k + 1), data_train, fmt='%.4e')

        Y_pred_vail = lgbm.predict(X_vail, num_iteration=lgbm.best_iteration)#验证结果
        # drawing(y_vail, Y_pred_vail, k, "vail")
        # Y_pred_vail_1 = 10**X_denormalized(Y_pred_vail, original_data)
        # y_vail_1 = 10**X_denormalized(y_vail, original_data)
        score_vail = r2_score(y_vail, Y_pred_vail)
        rmse_vail = mean_squared_error(y_vail, Y_pred_vail)**0.5
        print("Vail第" + str(round(k+1)) + "交叉分数R2:", score_vail)
        data_vail = np.column_stack((y_vail, Y_pred_vail))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM-vail-pred-1442-{}.txt'.format(k + 1), data_vail, fmt='%.4e')

        Y_pred_test = lgbm.predict(X_test1, num_iteration=lgbm.best_iteration)#验证结果
        # drawing(Y_test1, Y_pred_test, k, "test")
        # Y_pred_test_1 = 10**X_denormalized(Y_pred_test, original_data)
        # Y_test1_1 = 10**X_denormalized(Y_test1, original_data)
        score_test = r2_score(Y_test1, Y_pred_test)
        rmse_test1 = mean_squared_error(Y_test1, Y_pred_test)**0.5
        print("Test第" + str(round(k+1)) + "交叉分数R2:", score_test)
        data_test = np.column_stack((Y_test1, Y_pred_test))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM-test-pred-1442-{}.txt'.format(k + 1), data_test, fmt='%.4e')


        # plot_fig(1, lgbm, k, X_train, X_test1, Y_test1)



        print("-----------------------------------------")

        if k == 0:
            pred_test = Y_pred_test
        else:
            pred_test = np.column_stack((pred_test, Y_pred_test))

        if k == 0:
            train_score = score_train
            train_rmse = rmse_train
        else:
            train_score = np.column_stack((train_score, score_train))
            train_rmse = np.column_stack((train_rmse, rmse_train))
        if k == 0:
            vail_score = score_vail
            vail_rmse = rmse_vail
        else:
            vail_score = np.column_stack((vail_score, score_vail))
            vail_rmse = np.column_stack((vail_rmse, rmse_vail))
        if k == 0:
            test_score = score_test
            test_rmse = rmse_test1
        else:
            test_score = np.column_stack((test_score, score_test))
            test_rmse = np.column_stack((test_rmse, rmse_test1))
    #     Train.append(score_train)
    #     Vail.append(score_vail)
    #     Test.append(score_test)
    # Train = np.array(Train)
    # Vail = np.array(Vail)
    # Test = np.array(Test)
    print("Mean R2 Score of train: {:.4f}".format(np.mean(train_score)))
    print("Mean R2 Score of vail: {:.4f}".format(np.mean(vail_score)))
    print("Mean R2 Score of test: {:.4f}".format(np.mean(test_score)))
    print("Mean RMSE Score of train: {:.4e}".format(np.mean(train_rmse)))
    print("Mean RMSE Score of vail: {:.4e}".format(np.mean(vail_rmse)))
    print("Mean RMSE Score of test: {:.4e}".format(np.mean(test_rmse)))

    np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM_Train_R2_1442.txt', np.column_stack((train_score, np.mean(train_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM_Vail_R2_1442.txt', np.column_stack((vail_score, np.mean(vail_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM_Test_R2_1442.txt', np.column_stack((test_score, np.mean(test_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM_Train_rmse_1442.txt', np.column_stack((train_rmse, np.mean(train_rmse, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM_Vail_rmse_1442.txt', np.column_stack((vail_rmse, np.mean(vail_rmse, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\lgbm_1442\LGBM_Test_rmse_1442.txt', np.column_stack((test_rmse, np.mean(test_rmse, axis=1))), fmt='%15E')
    # np.savetxt(r'F:\GAN\LGBM\pred_test.txt', np.column_stack((Y_test, pred_test, np.mean(pred_test, axis=1))),
    #            fmt='%15E')
    R.append(np.mean(test_score))
R = np.array(R)
e = np.column_stack((R.reshape(-1,1)))

np.savetxt(r'F:/LightGBM-r2-循环-5.txt', e, fmt='%15E')#'%.4f'/'%15E'


mean_pred = np.mean(pred_test, axis=1)
pred_error = np.std(pred_test, axis=1)

#交叉验证求平均值后的结果
import matplotlib.pyplot as plt
plt.plot(Y_test1, Y_test1, c='C3', label=r'$b_{\rm test}$')
plt.errorbar(Y_test1, mean_pred, pred_error, fmt='o',ecolor='g', color='g', elinewidth=2, capsize=4)#画误差棒
# plt.show()




#%%
import lightgbm as lgb
import pandas as pd
model_path = r'F:/GAN_model_1/LightGBM_model_1442_1.txt'

# # 加载模型
bst = lgb.Booster(model_file=model_path)


Vali_data = pd.read_csv(r'F:\GAN工作\数据集\Validation set.txt',sep='\s+',header=None).to_numpy()

Vali_x = Vali_data[:, :-1]
Vali_y = Vali_data[:, -1]

Vali_y_pred = bst.predict(Vali_x)

# 输出预测结果的R²分数
rmse = mean_squared_error(Vali_y, Vali_y_pred)**0.5
r2 = r2_score(Vali_y, Vali_y_pred)
print("LGBM_Vali_1442预测结果的MSE分数：", rmse)
print("LGBM_Vali_1442预测结果的R²分数：", r2)


a = np.column_stack((Vali_y, Vali_y_pred))

np.savetxt(r'f:/GAN工作/代码保存文件重做版/lgbm_1442/LightGBM-验证泛化能力-1442.txt', a, fmt='%.4E')#保存完a a就没用了



