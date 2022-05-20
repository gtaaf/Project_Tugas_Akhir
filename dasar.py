from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask_mysqldb import MySQL

app = Flask(__name__)

model = tf.keras.models.load_model('my_model.h5')

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '10032000GTAaf'
app.config['MYSQL_DB'] = 'db_TA'

mysql=MySQL(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    #data[4] = request.form['e']
    #data[5] = request.form['f']
    #data[6] = request.form['g']
    #data[7] = request.form['h']
    #data[8] = request.form['i']
    #data[9] = request.form['j']
    #data[10] = request.form['k']
    #data[11] = request.form['l']
    #data[12] = request.form['m']

    #data_df = arr
    #data_df = pd.DataFrame(data_df)

    #data_df = arr.diff()

    # data_ds=data[13,12,11,6,8]
    # data_df=pd.concat([data_ds,data_df],axis=1)

    arr = np.array([[data1, data2, data3, data4]])
    arr = arr.reshape((arr.shape[0], 1, arr.shape[1]))
    #arr = pd.DataFrame(arr)

    pred = model.predict(arr)

    # def evaluate_prediction(predictions, actual, model_name):
    #    errors = predictions - actual
    #    mse = np.square(errors).mean()
    #    rmse = np.sqrt(mse)
    #    mae = np.abs(errors).mean()
    #    r2= r2_score(ytest, prediksi)

    #    print(model_name + ':')
    #    print('Mean Absolute Error: {:.4f}'.format(mae))
    #    print('Root Mean Square Error: {:.4f}'.format(rmse))
    #    print('Mean Square Error: {:.4f}'.format(mse))
    #    print('R-Square : {:.4f}'.format(r2))

    # evaluate_prediction(prediksi,ytest,'GRU')
    return render_template("after.html", arr=pred)


if __name__ == "__main__":
    app.run(debug=True)
