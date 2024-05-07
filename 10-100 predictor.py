
import numpy as np
import pandas as pd

# training dataset loading
dataset = pd.read_excel(r'D:\program(all)\new\MNIST\train_10-100.xlsx',na_filter = False) # take care the NA sequence problem
dataset_test=pd.read_excel(r'D:\program(all)\new\test_10-100.xlsx',na_filter = False)
y_train = dataset['Label']
y_train = np.array(y_train) # transformed as np.array for CNN model
print(y_train)
y_test = dataset_test['Label']
y_test = np.array(y_test) # transformed as np.array for CNN model
print(y_test)

# assign the dataset
X_train_data_name = 'train_10-100_after_protT5.npy'
X_train_data = np.load(X_train_data_name)

X_test_data_name = 'test_10-100_after_protT5.npy'
X_test_data = np.load(X_test_data_name)

X_train = np.array(X_train_data)
X_test = np.array(X_test_data)

hc_CTD_train_data_name = 'CTD_train_10-100_after.csv'
hc_CTD_train_data = pd.read_csv(hc_CTD_train_data_name,header=0, index_col = 0,delimiter=',')

hc_CTD_test_data_name = 'CTD_test_10-100_after.csv'
hc_CTD_test_data = pd.read_csv(hc_CTD_test_data_name,header=0, index_col = 0,delimiter=',')

hc_CKS_train_data_name = 'CKS_train_10-100_after.csv'
hc_CKS_train_data = pd.read_csv(hc_CKS_train_data_name,header=0, index_col = 0,delimiter=',')

hc_CKS_test_data_name = 'CKS_test_10-100_after.csv'
hc_CKS_test_data = pd.read_csv(hc_CKS_test_data_name,header=0, index_col = 0,delimiter=',')

hc_CTD_train_data = np.array(hc_CTD_train_data)
hc_CTD_test_data = np.array(hc_CTD_test_data)
hc_CKS_train = np.array(hc_CKS_train_data)
hc_CKS_test = np.array(hc_CKS_test_data)

# normalize the X data range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train) # normalize X to 0-1 range
X_test = scaler.transform(X_test)
scaler.fit(hc_CTD_train_data)
hc_CTD_train_data = scaler.transform(hc_CTD_train_data) # normalize X to 0-1 range
hc_CTD_test_data = scaler.transform(hc_CTD_test_data)
scaler.fit(hc_CKS_train)
hc_CKS_train = scaler.transform(hc_CKS_train) # normalize X to 0-1 range
hc_CKS_test = scaler.transform(hc_CKS_test)
from keras.layers import LSTM, Bidirectional, GRU, Convolution2D, concatenate
X_train=concatenate([X_train,hc_CKS_train,hc_CTD_train_data])
X_test=concatenate([X_test,hc_CKS_test ,hc_CTD_test_data])

from keras import backend as K
from keras.layers import Input, Dense
import tensorflow as tf
def swish(x):
    return x * K.sigmoid(10*x)


def glu_unit(inputs, units):
    x = Dense(units)(inputs)
    x1 = Dense(units)(inputs)
    x = swish(x)
    # x1 = tf.keras.activations.gelu(x1)
    return x * x1
import xgboost as xgb
# 使用 XGBoost 训练模型
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# 获取特征的重要性
importance = model.feature_importances_

# 按照特征重要性排序并选择前500个最重要特征
sorted_index = np.argsort(importance)[::-1]
top_features = sorted_index[:500]
X_train = X_train[:, top_features]
X_test = X_test[:, top_features]

def ESM_CNN(X_train, y_train,X_test,y_test):

  from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Conv1D
  from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, AveragePooling1D, MaxPooling1D,Bidirectional,LSTM
  from keras.models import Sequential,Model
  from keras.optimizers import SGD
  from keras.callbacks import ModelCheckpoint,LearningRateScheduler, EarlyStopping
  from keras.initializers import he_normal
  from keras.layers import Input, Dense
  inputShape=(1447,1)
  input = Input(inputShape)
  x = Conv1D(128,(5),strides = (2),name='layer_conv1',padding='same',kernel_initializer=he_normal(seed=None))(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling1D((2), name='MaxPool1',padding="same")(x)
  x = Dropout(0.15)(x)
  x = glu_unit(x, 64)
  x = Flatten()(x)
  x = Dense(128,activation = 'relu',name='fc1')(x)
  x = Dropout(0.15)(x)
  x = Dense(32,activation = 'relu',name='fc2')(x)
  x = Dropout(0.15)(x)
  x = BatchNormalization()(x)
  x = Dense(2,activation = 'softmax',name='fc3')(x)
  model = Model(inputs = input,outputs = x,name='Predict')
  # define SGD optimizer
  momentum = 0.5
  sgd = SGD(learning_rate=0.01, momentum=momentum, nesterov=False)
  # compile the model
  model.compile(loss='sparse_categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
  # learning deccay setting
  import math
  def step_decay(epoch): # gradually decrease the learning rate
      initial_lrate=0.1
      drop=0.6
      epochs_drop = 3.0
      lrate= initial_lrate * math.pow(drop,    # math.pow base raised to a power
            math.floor((1+epoch)/epochs_drop)) # math.floor Round numbers down to the nearest integer
      return lrate
  lrate = LearningRateScheduler(step_decay)

  # early stop setting
  early_stop = EarlyStopping(monitor='val_accuracy', patience = 20,restore_best_weights = True)

  # summary the callbacks_list
  callbacks_list = [ lrate , early_stop]

  model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=200,callbacks=callbacks_list,batch_size = 64, verbose=1)
  return model, model_history

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

# result collection list
ACC_collecton = []
BACC_collecton = []
Sn_collecton = []
Sp_collecton = []
MCC_collecton = []
AUC_collecton = []
model, model_history = ESM_CNN(X_train, y_train, X_test , y_test)
predicted_class= []
predicted_protability = model.predict(X_test,batch_size=1)
for i in range(predicted_protability.shape[0]):
  index = np.where(predicted_protability[i] == np.amax(predicted_protability[i]))[0][0]
  predicted_class.append(index)
predicted_class = np.array(predicted_class)
y_true = y_test
predicted_class1 = []
predicted_protability1 = model.predict(X_test, batch_size=1)
predicted_class1.append(predicted_protability1)
predicted_class1 = np.array(predicted_class1)
predicted_class1 = tf.reshape(predicted_class1, shape=(114,2))

y_true = y_test
from sklearn.metrics import confusion_matrix
import math
TP, FP, FN, TN = confusion_matrix(y_true, predicted_class).ravel() # shape [ [True-Positive, False-positive], [False-negative, True-negative] ]
ACC = (TP+TN)/(TP+TN+FP+FN)
Sn=TP/(TP+FN)
Sp=TN/(TN+FP)
ACC_collecton.append(ACC)
Sn_collecton.append(TP/(TP+FN))
Sp_collecton.append(TN/(TN+FP))
MCC = (TP*TN-FP*FN)/math.pow(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),0.5)
MCC_collecton.append(MCC)
BACC_collecton.append(0.5*TP/(TP+FN)+0.5*TN/(TN+FP))
from sklearn.metrics import roc_auc_score
AUC = roc_auc_score(y_test, predicted_protability[:,1])
AUC_collecton.append(AUC)
print(TP, FP, FN, TN,ACC,Sn,Sp,MCC,AUC)
from sklearn.metrics import precision_score, recall_score, f1_score

# 计算精确度
precision = precision_score(y_true, predicted_class)
print(f'Precision: {precision}')

# 计算召回率
recall = recall_score(y_true, predicted_class)
print(f'Recall: {recall}')

# 计算F1得分
f1 = f1_score(y_true, predicted_class)
print(f'F1 Score: {f1}')

