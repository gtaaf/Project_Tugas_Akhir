import pandas as pd
import numpy as np
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score


#Masukan Data
data = pd.read_csv(r'C:\Users\Gusti Aditya AF\Documents\Kuliah\TA\data\Data Siap.csv'
                   r'')

#test AdFuller
kols = data.columns
D = []
for col in kols :
    result = adfuller(data[col].dropna())
    print(col,result[1])

#Differencing
kol =['Week1','Week2','Week3','Week4','HB','HDS','HMG','HGP','Target']
kol = data[kol]
D = pd.DataFrame(D)
for col in kol :
    D[col]= data[col].diff().values
    print(D)

kolom = D.columns
for col in kolom :
    result = adfuller(D[col].dropna())
    print(col,result[1])


#Menghitung nilai 0 hasil differencing
D.replace(0,np.nan,inplace=True)
print(D.isna().sum())

#Mengambil Parameter
data_ch =['Week1','Week2','Week3','Week4']
#data_a = ['Inflasi','HBP','HBM','HDA','HTA']
#data_df=pd.concat([D[data_ch],data[data_a]],axis=1)
data_df = D[data_ch]

#Membuat Index
def create_ts_data(dataset, lookback=1, predicted_col=1):
    temp = dataset.copy()
    temp["id"] = range(1, len(temp) + 1)
    temp = temp.iloc[:-lookback, :]
    temp.set_index('id', inplace=True)
    predicted_value = dataset.copy()
    predicted_value = predicted_value.iloc[lookback:, predicted_col]
    predicted_value.columns = ["Predcited"]
    predicted_value = pd.DataFrame(predicted_value)

    predicted_value["id"] = range(1, len(predicted_value) + 1)
    predicted_value.set_index('id', inplace=True)
    final_df = pd.concat([temp, predicted_value], axis=1)
    return final_df


reframed_df = create_ts_data(data_df, 1, 3)
reframed_df.fillna(0, inplace=True)

#PENENTUAN COLUMN TARGET
reframed_df.columns = ['Week1','Week2','Week3','Week4','Target']#'Inflasi','HBP','HBM','HDA','HTA','Target']#'Inflasi','HBP','HBM','HDA','HTA',]

#NORMALISASI DATA DENGAN PERUBAHAN SKALA DATA
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(reframed_df)
scaled= pd.DataFrame(scaled)
#MEMBAGI DATA TRAINING DAN TESTING SERTA PARAMETER INPUT DAN PARAMETER OUTPUT
values = scaled.values
training_sample = math.ceil(len(reframed_df) * 0.9)

train = values[:training_sample, :]
test = values[training_sample:, :]
Xtrain, ytrain = train[:, :-1], train[:, -1]
Xtest, ytest = test[:, :-1], test[:, -1]

X_train = Xtrain.reshape((Xtrain.shape[0], 1, Xtrain.shape[1]))
X_test = Xtest.reshape((Xtest.shape[0], 1, Xtest.shape[1]))

#MEMBUAT MODEL
model = Sequential()
model.add(GRU(125, return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(100, return_sequences=True))
model.add(GRU(75, return_sequences=True))
model.add(GRU(50, return_sequences=True))
model.add(GRU(25))
model.add(Dense(1))

model.summary()
model.compile(loss='mae', optimizer='adam')

history = model.fit(X_train, ytrain, epochs=60, batch_size=3, shuffle=False, verbose=0, validation_data=(X_test,ytest))

#plot hasil training
pyplot.plot(history.history['loss'], label='GRU loss', color='brown')
pyplot.plot(history.history['val_loss'], label='GRU val_loss', color='green')
pyplot.title('plot hasil training')
pyplot.legend()
pyplot.show()

#DEF EVALUATE
def evaluate_prediction(predictions, actual, model_name):
    errors = actual - predictions
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    r2 = r2_score(actual, predictions)
    smape = 1/len(actual)*np.sum(2*np.abs(predictions-actual)/(np.abs(actual)+np.abs(predictions))*100)
    mape = np.mean(np.abs((np.array(predictions)-np.array(actual)))/np.array(actual))*100

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('Mean Square Error: {:.4f}'.format(mse))
    print('R-Square : {:.4f}'.format(r2))
    print('SMAPE :{:.4f}'.format(smape))
    print('MAPE :{:.4f}'.format(mape))


#DEF PLOT
def plot_future(prediction, model_name, y_test):
    pyplot.figure(figsize=(10,6))
    range_future = len(prediction)

    pyplot.plot(np.arange(range_future), np.array(y_test), label='Test data')
    pyplot.plot(np.arange(range_future),np.array(prediction),label='Prediction')

    pyplot.title('Test vs Predict'+model_name)
    pyplot.legend(loc='upper left')
    pyplot.show()

#TESTING
predic = model.predict(X_test)
prediksi = pd.DataFrame(predic)

#Mengembalikan shape scaled
test_data=pd.DataFrame(Xtest)

#Menggabungkan Data Input Asli Dan Hasil Prediksi
data_prediksi = pd.concat([test_data,prediksi],axis=1)

#Reverse Data Prediksi
prediksi_data = scaler.inverse_transform(data_prediksi)
prediksi_data=pd.DataFrame(prediksi_data)
prediksi_data.columns=['Week1','Week2','Week3','Week4','Target']#,'Inflasi','HBP','HBM','HDA','HTA']
#prediksi_data=prediksi_data.filter(items=["Week1","Week2","Week3","Week4","Target"])

#Prepare Reverse Differencing
data2=['Week1','Week2','Week3','Week4','Target']
data1=data[data2]
nilai = data1.values
ukuran = math.ceil(len(data1)*0.9)
data_uji = nilai[ukuran:,:]
data_uji=pd.DataFrame(data_uji)
data_uji.columns=['Week1','Week2','Week3','Week4','Target']


#prediksi_data = pd.DataFrame(prediksi_data)
#prediksi_data.columns=['Week1','Week2','Week3','Week4','Target']

kolom_pred = prediksi_data.columns
kolom_act = data_uji.columns
Z=[]
for col in kolom_pred:
    diff_result = data_uji[col]+prediksi_data[col]
    Z.append(diff_result)
data_rev = pd.concat(Z,axis=1)

last_data_rev = int(data_rev['Target'].iloc[len(data_rev.index)-1])

data_rev=np.array(data_rev).astype(int)
data_uji=np.array(data_uji)
print(data_rev)
print(data_uji)


evaluate_prediction(data_rev[:,-1],data_uji[:,-1] , 'GRU')
plot_future(data_rev[:,-1], 'GRU',data_uji[:,-1])

print(last_data_rev)