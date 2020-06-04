import numpy as np
import sqlite3
import pandas as pd  
import joblib
import time
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_boston

filename = 'lr_model.sav'
databaseFile="predict_database.db"

app = Flask(__name__, static_url_path = "/img", static_folder = "img")

################ Log of predictions ###############################
def clearLog():
    con=sqlite3.connect(databaseFile)
    c = con.cursor()
    c.execute("delete from tb_predict_log;")
    con.commit()
    con.close()

def getConnection():
    con=sqlite3.connect(databaseFile)
    return con
   
def createTable(con):
    try:
        c = con.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS tb_predict_log
                 (time, ip, features, predict)""")
    except Exception as e:
        pass

def insertLog(con, ip, features, predict):
    c = con.cursor()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    #print("INSERT INTO tb_predict_log values('"+st+"', '"+ip+"', '"+features+"', '"+predict+"')")
    c.execute("INSERT INTO tb_predict_log values('"+st+"', '"+ip+"', '"+features+"', '"+predict+"')")
    con.commit()
    con.close()

def queryLog():
    try:
        con=getConnection()
        result=pd.read_sql_query("select * from tb_predict_log;",con)
        return result
    except:
        return "Log not found ..."
################ Log of predictions ###############################

def runModel():
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    boston.isnull().sum()

    X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
    Y = boston['MEDV']


    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)


    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    # save the model to disk
    
    joblib.dump(lin_model, filename)

    # model evaluation for training set
    y_train_predict = lin_model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    r2 = r2_score(Y_train, y_train_predict)

    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

    # model evaluation for testing set
    y_test_predict = lin_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2 = r2_score(Y_test, y_test_predict)

    str_html = "The model performance for testing set</br>"
    str_html += "RMSE is {}".format(rmse)
    return str_html

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/GetPredictions")
def calc():
    return render_template('GetPredictions.html')

@app.route('/Run')
def model():
    retorno = runModel()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    retorno += "</br></br></br><b>Last execution: </b>" + st 
    return retorno

@app.route('/Logs')
def logPredicts():
    result = queryLog()
    strhtml = "<h1>Predictions Report</h1></br><table border=1>"
    strhtml += "<tr><td>Datetime</td><td>IP</td><td>Features[LSTAT | RM]</td><td>Predict Value</td></tr>"
    for index, row in result.iterrows():
        strhtml += "<tr><td>"+row[0]+"</td><td>"+row[1]+"</td><td>"+row[2]+"</td><td>"+row[3]+"</td></tr>"
    strhtml += "</table>"
    return strhtml

@app.route('/GetPredict', methods=['GET'])
def getPredict():
    #prediction ##############################
    valor1 = request.args.get('valor1', 0)
    valor2 = request.args.get('valor2', 0)
    df_valida = pd.DataFrame([[valor1, valor2]])
    loaded_model = joblib.load(filename)
    result = loaded_model.predict(df_valida)

    #save it in the log #########################
    con=getConnection()
    v_features = str(valor1) + " | " + str(valor2)

    insertLog(con, '10.0.1.1',  v_features , str(result))
    #return the result to user
    retorno_pred = jsonify({'predict': str(result).replace("[","").replace("]","")}) 
    return retorno_pred

@app.route('/ClearLog')
def clear():
    clearLog()
    return "Log is cleaned."


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')