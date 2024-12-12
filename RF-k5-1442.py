from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from tqdm.tk import trange

import warnings
warnings.filterwarnings('ignore')


# data = pd.read_csv(r'F:\Training-MP.txt',sep='\s+',header=None).fillna(0)
# data.columns = ['Pd', 'Dw', 'I', 'a', 'K', 'P', 'D', 'c', 'De']#5变量
data = pd.read_csv(r'F:\GAN工作\数据集\real+gan1442.txt',sep='\s+',header=None).fillna(0)
data.columns = ['I', 'Pd', 'Rs', 'Pb', 'T', 'uw', 'Aext', 'nc', 'κ', 'ps', 'εacc', 'εtot', 'K']
feats = [feat for feat in data.columns if feat not in ["K"]]
X = data[feats]
y = data["K"]
# y = np.log10(data["K"])
"""只有K 的值"""

#%%
Train = []
Vail = []
Test = []
R = []
'参数设置'
n_splits = 5 #设置几倍交叉验证

for i in range(1):
    l=(i+1)/100
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, y, test_size=0.2, random_state=9)
    rfr = RandomForestRegressor(bootstrap=True,
                                criterion="mse",#重要参数
                                max_depth=None,#None 10 主要参数
                                max_features='auto',#'auto','sqrt',浮点数
                                min_samples_leaf=1,#1  主要参数3
                                min_samples_split=2,#2  主要参数5
                                min_weight_fraction_leaf=0,  #主要参数0-0.5
                                n_estimators=80,  #70  主要参数
                                random_state=92,#89
                                )

    '''以下交叉验证'''
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=40)#64
    # for train_index, vail_index in kf.split(X):
    for k, (train_index, vail_index) in enumerate(kf.split(X_train1)):
        L = len(train_index) / len(vail_index)#输出训练集与测试集的比值
        # print("训练集与测试集的比值：", L)
        X_train = X_train1.iloc[train_index]
        y_train = Y_train1.iloc[train_index]
        X_vail = X_train1.iloc[vail_index]
        y_vail = Y_train1.iloc[vail_index]
        # print(vail_index)
        # print(len(train_index))
        # print(len(vail_index))

        rfr = rfr.fit(X_train, y_train)

        import joblib

        joblib.dump(rfr, r'F:\GAN_model_1\RF_model_1442_{}.pkl'.format(k + 1))

        Y_pred_train = rfr.predict(X_train)#验证结果

        score_train = r2_score(y_train, Y_pred_train)
        rmse_train = mean_squared_error(y_train, Y_pred_train) ** 0.5
        print("Train第" + str(round(k + 1)) + "交叉分数R2:", score_train)
        data_train = np.column_stack((y_train, Y_pred_train))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF-train-pred-1442-{}.txt'.format(k+1), data_train, fmt='%.4e')

        Y_pred_vail = rfr.predict(X_vail)#验证结果

        score_vail = r2_score(y_vail, Y_pred_vail)
        rmse_vail = mean_squared_error(y_vail, Y_pred_vail) ** 0.5
        print("Vail第" + str(round(k + 1)) + "交叉分数R2:", score_vail)
        data_vail = np.column_stack((y_vail, Y_pred_vail))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF-vail-pred-1442-{}.txt'.format(k+1), data_vail, fmt='%.4e')

        Y_pred_test = rfr.predict(X_test1)#验证结果

        score_test = r2_score(Y_test1, Y_pred_test)
        rmse_test1 = mean_squared_error(Y_test1, Y_pred_test) ** 0.5
        print("Test第" + str(round(k + 1)) + "交叉分数R2:", score_test)
        data_test = np.column_stack((Y_test1, Y_pred_test))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF-test-pred-1442-{}.txt'.format(k + 1), data_test, fmt='%.4e')

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

    np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF_Train_R2_1442.txt', np.column_stack((train_score, np.mean(train_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF_Vail_R2_1442.txt', np.column_stack((vail_score, np.mean(vail_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF_Test_R2_1442.txt', np.column_stack((test_score, np.mean(test_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF_Train_rmse_1442.txt', np.column_stack((train_rmse, np.mean(train_rmse, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF_Vail_rmse_1442.txt', np.column_stack((vail_rmse, np.mean(vail_rmse, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\rf_1442\RF_Test_rmse_1442.txt', np.column_stack((test_rmse, np.mean(test_rmse, axis=1))), fmt='%15E')
    ## np.savetxt(r'F:\GAN\RF\pred_test.txt', np.column_stack((Y_test, pred_test, np.mean(pred_test, axis=1))),
    ##            fmt='%15E')

    R.append(np.mean(test_score))
R = np.array(R)
e = np.column_stack((R.reshape(-1, 1)))

np.savetxt(r'F:/RF-r2-1442.txt', e, fmt='%15E')#'%.4f'/'%15E'

mean_pred = np.mean(pred_test, axis=1)
pred_error = np.std(pred_test, axis=1)

#交叉验证求平均值后的结果
import matplotlib.pyplot as plt
plt.plot(Y_test1, Y_test1, c='C3', label=r'$b_{\rm test}$')
plt.errorbar(Y_test1, mean_pred, pred_error, fmt='o',ecolor='g', color='g', elinewidth=2, capsize=4)#画误差棒
plt.show()



#%%
from joblib import load

model_path = r'F:\GAN_model_1\RF_model_1442_4.pkl'
rfr_model = load(model_path)

Vali_data = pd.read_csv(r'F:\GAN工作\数据集\Validation set.txt',sep='\s+',header=None).to_numpy()

Vali_x = Vali_data[:, :-1]
Vali_y = Vali_data[:, -1]


Vali_y_pred = rfr_model.predict(Vali_x)


# 输出预测结果的R²分数
rmse = mean_squared_error(Vali_y, Vali_y_pred)**0.5
r2 = r2_score(Vali_y, Vali_y_pred)
print("Vali_1442预测结果的RMSE分数：", rmse)
print("Vali_1442预测结果的R²分数：", r2)


a = np.column_stack((Vali_y, Vali_y_pred))
#
np.savetxt(r'f:\GAN工作\代码保存文件重做版\rf_1442\RF-验证泛化能力-1442.txt', a, fmt='%.4E')#保存完a a就没用了