import numpy as np#numpy用来存储和处理大型多维矩阵，支持大量的维度数组与矩阵运算，这个方式使用numpy的函数时，需要以np.开头。
import keras#Keras是一个模型库，是为开发深度学习模型提供了高层次的构建模块
from sklearn.model_selection import train_test_split#将数组或矩阵分割成随机的测试集和训练集
import pandas as pd#Pandas纳入了大量库和一些标准的数据模型，提供了大量能使我们快速便捷地处理数据的函数和方法。
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.layers.advanced_activations import PReLU, ReLU
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from tqdm.tk import trange
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv(r'F:\GAN工作\数据集\real+gan1442.txt',sep='\s+',header=None).fillna(0)
data.columns = ['I', 'Pd', 'Rs', 'Pb', 'T', 'uw', 'Aext', 'nc', 'κ', 'ps', 'εacc', 'εtot', 'K']
# data.columns = ['Aext','I','m','Pd','a','If','n','Dw','De']
feats = [feat for feat in data.columns if feat not in ["K"]]
X = data[feats]
y = data["K"]
# y = np.log10(data["K"])

#%%
Train = []
Vail = []
Test = []
R = []
'参数设置'
n_splits = 5 #设置几倍交叉验证
X_train1,X_test1,Y_train1,Y_test1 = train_test_split(X, y, test_size=0.2, random_state=40)#45, test_size=0.2,

for i in range(1):

    model = Sequential()  # 开始构建模型

    model = keras.Sequential()
    model.add(Dense(128, input_dim=X.shape[1]))
    # model.add(BatchNormalization(axis=1))  # axis: 整数，指定要规范化的轴，通常为特征轴。一般会设axis=1。
    model.add(PReLU())
    model.add(Dropout(0.2))  # 0.2是一个很好的起点，太低的概率产生的作用有限，太高的概率可能导致网络的训练不充分。
    #64
    model.add(Dense(128))#64
    # model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(Dropout(0.2))
    #


    model.add(Dense(1))  # 输出层

    model.compile(loss=keras.losses.mean_squared_error, optimizer=Adam(lr=0.005), metrics=['mae'])
    model.summary()  # 输出模型各层的参数状况

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=40)
    # for train_index, vail_index in kf.split(X):
    for k, (train_index, vail_index) in enumerate(kf.split(X_train1)):
        L = len(train_index) / len(vail_index)#输出训练集与测试集的比值
        print("训练集与测试集的比值：", L)
        X_train = X_train1.iloc[train_index]
        y_train = Y_train1.iloc[train_index]
        X_vail = X_train1.iloc[vail_index]
        y_vail = Y_train1.iloc[vail_index]
        # print(vail_index)

        callbacks = [keras.callbacks.ModelCheckpoint(filepath=r'f:\GAN_model\ANN_best_model_1442_{}.h5'.format(k + 1), save_best_only=True)]

        class TestCallback(keras.callbacks.Callback):

            # 回调函数是一个函数的合集，会在训练的阶段中所使用。可以使用回调函数来查看训练模型的内在状态和统计。
            def __init__(self, test_data):
                self.test_data = test_data

            # self.valueName valueName：表示self对象，即实例的变量
            def on_epoch_end(self, epoch, logs={}):
                x, y = self.test_data  # self.name = name的意思就是把外部传来的参数name的值赋值给自己的属性变量self.name。
                loss, acc = self.model.evaluate(x, y, verbose=0)
                print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

        history = model.fit(X_train, y_train, epochs=10000, batch_size=128, verbose=2, validation_data=(X_vail, y_vail), callbacks=callbacks)

        model.load_weights(r'F:\GAN_model\ANN_best_model_1442_{}.h5'.format(k + 1))
# %%
        from keras.models import load_model

        model = load_model(r'f:\GAN_model\ANN_best_model_1442_5.h5')
        Y_pred_train = model.predict(X_train)


        score_train = r2_score(y_train, Y_pred_train)
        rmse_train = mean_squared_error(y_train, Y_pred_train) ** 0.5
        print("Train第" + str(round(k + 1)) + "交叉分数R2:", score_train)
        data_train = np.column_stack((y_train, Y_pred_train))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN-train-pred-1442-{}.txt'.format(k + 1), data_train, fmt='%.4e')

        Y_pred_vail = model.predict(X_vail)
        # Y_pred_vail_1 = 10 ** X_denormalized(Y_pred_vail, original_data)
        # y_vail_1 = 10 ** X_denormalized(y_vail, original_data)
        score_vail = r2_score(y_vail, Y_pred_vail)
        rmse_vail = mean_squared_error(y_vail, Y_pred_vail) ** 0.5
        print("Vail第" + str(round(k + 1)) + "交叉分数R2:", score_vail)
        data_vail = np.column_stack((y_vail, Y_pred_vail))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN-vail-pred-1442-{}.txt'.format(k + 1), data_vail, fmt='%.4e')

        Y_pred_test = model.predict(X_test1)

        score_test = r2_score(Y_test1, Y_pred_test)
        rmse_test1 = mean_squared_error(Y_test1, Y_pred_test) ** 0.5
        print("Test第" + str(round(k + 1)) + "交叉分数R2:", score_test)
        data_test = np.column_stack((Y_test1, Y_pred_test))
        np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN-test-pred-1442-{}.txt'.format(k + 1), data_test, fmt='%.4e')


        print("-----------------------------------------")


        if k == 0:
            pred_test = Y_pred_test
        else:
            pred_test = np.column_stack((pred_test, Y_pred_test))

        if k == 0:#R2
            train_score = score_train
        else:
            train_score = np.column_stack((train_score, score_train))

        if k == 0:#R2
            vail_score = score_vail
        else:
            vail_score = np.column_stack((vail_score, score_vail))

        if k == 0:#R2
            test_score = score_test
        else:
            test_score = np.column_stack((test_score, score_test))

        if k == 0:
            test_rmse = rmse_test1
        else:
            test_rmse = np.column_stack((test_rmse, rmse_test1))

        if k == 0:
            vail_rmse = rmse_vail
        else:
            vail_rmse = np.column_stack((vail_rmse, rmse_vail))

        if k == 0:
            train_rmse = rmse_train
        else:
            train_rmse = np.column_stack((train_rmse, rmse_train))


    print("Mean R2 Score of train: {:.4f}".format(np.mean(train_score)))
    print("Mean R2 Score of vail: {:.4f}".format(np.mean(vail_score)))
    print("Mean R2 Score of test: {:.4f}".format(np.mean(test_score)))


    np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN_Train_R2_1442.txt', np.column_stack((train_score, np.mean(train_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN_Vail_R2_1442.txt', np.column_stack((vail_score, np.mean(vail_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN_Test_R2_1442.txt', np.column_stack((test_score, np.mean(test_score, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN_Train_rmse_1442.txt', np.column_stack((train_rmse, np.mean(train_rmse, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN_Vail_rmse_1442.txt', np.column_stack((vail_rmse, np.mean(vail_rmse, axis=1))), fmt='%15E')
    np.savetxt(r'F:\GAN工作\代码保存文件重做版\ann_1442\ANN_Test_rmse_1442.txt', np.column_stack((test_rmse, np.mean(test_rmse, axis=1))), fmt='%15E')
    ### np.savetxt(r'F:\GAN\ANN\pred_test.txt', np.column_stack((Y_test1, pred_test, np.mean(pred_test, axis=1))), fmt='%15E')
    R.append(np.mean(test_score))
R = np.array(R)
e = np.column_stack((R.reshape(-1,1)))

# np.savetxt(r'F:/ann-r2-循环-5.txt', e, fmt='%15E')#'%.4f'/'%15E'

mean_pred = np.mean(pred_test, axis=1)
pred_error = np.std(pred_test, axis=1)

#交叉验证求平均值后的结果
import matplotlib.pyplot as plt
plt.plot(Y_test1, Y_test1, c='C3', label=r'$b_{\rm test}$')
plt.errorbar(Y_test1, mean_pred, pred_error, fmt='o',ecolor='g', color='g', elinewidth=2, capsize=4)#画误差棒
plt.show()

#%%

from keras.models import load_model

model = load_model(r'f:\GAN_model\ANN_best_model_1442_5.h5')
Vali_data = pd.read_csv(r'F:\GAN工作\数据集\Validation set.txt',sep='\s+',header=None).to_numpy()

Vali_x = Vali_data[:, :-1]
Vali_y = Vali_data[:, -1]


Vali_y_pred = model.predict(Vali_x)


# 输出预测结果的R²分数
rmse = mean_squared_error(Vali_y, Vali_y_pred)**0.5
r2 = r2_score(Vali_y, Vali_y_pred)
print("Vali_1442预测结果的MSE分数：", rmse)
print("Vali_1442预测结果的R²分数：", r2)


a = np.column_stack((Vali_y, Vali_y_pred))

np.savetxt(r'f:\GAN工作\代码保存文件重做版\ann_1442\ANN-泛化能力验证-1442.txt', a, fmt='%.4E')#保存完a a就没用了