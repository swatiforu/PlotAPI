from firebase import firebase
import pandas as pd
import numpy as np
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

fb = firebase.FirebaseApplication('https://historicaldatafyp-default-rtdb.firebaseio.com/', None)

a = fb.get('historicaldatafyp-default-rtdb/Plots/-Mas01Cu0i8scu0dN_By', '')
df = pd.DataFrame(a)

@app.route('/historical/', methods=['GET'])
def historical():
  area = request.args['area']
  ndf = df[df['Location'] == area]
  k = ndf.Date.unique()
  k.sort()
  t = datetime.datetime.strptime(k[0], '%Y-%m-%d')
  dates = []
  while t.date()!=datetime.date.today():
    dates.append(str(t.date()))
    t = t+datetime.timedelta(days=1)
  ppm = []
  for i in dates:
    vals = ndf[ndf['Date'] == i]
    vals['PPM'] = vals['Price']/pd.to_numeric(vals['Area'])
    ppm.append(vals.PPM.sum()/len(vals))
  ppm = pd.Series(ppm)
  data = pd.DataFrame(ppm.interpolate(), columns = ['ppm'])
  data['Date'] = dates
  val = data.values[-60:]
  tt = dict()
  for i in range(len(val)):
    key = val[i][1]
    price = val[i][0]
    tt[key] = price
  return tt

@app.route('/predictions/', methods=['GET'])
def getprediction():
  area = request.args['area']
  ndf = df[df['Location'] == area]
  k = ndf.Date.unique()
  k.sort()
  t = datetime.datetime.strptime(k[0], '%Y-%m-%d')
  dates = []
  while t.date()!=datetime.date.today():
    dates.append(str(t.date()))
    t = t+datetime.timedelta(days=1)
  ppm = []
  for i in dates:
    vals = ndf[ndf['Date'] == i]
    vals['PPM'] = vals['Price']/pd.to_numeric(vals['Area'])
    ppm.append(vals.PPM.sum()/len(vals))
  ppm = pd.Series(ppm)
  data = pd.DataFrame(ppm.interpolate(), columns = ['ppm'])
  data['Date'] = dates
  scaler1 = MinMaxScaler(feature_range=(0,1))
  scaler2 = MinMaxScaler(feature_range=(0,1))
  X = []
  y = []
  window = 60
  data = data.ppm.values.copy()
  for i in range(len(data)-60):
    X.append(np.asarray(data[i:i+60]))
    y.append(np.asarray([data[i+60]]))
  X = np.asarray(X)
  y = np.asarray(y)
  X_scaled = scaler1.fit_transform(X)
  y_scaled = scaler2.fit_transform(y)
  X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1] , 1))
  model = load_model('PlotModels/'+area+'.h5')
  dat = X_scaled[-1].copy()
  nd = []
  preds = []
  dt = datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
  for i in range(7):
    val = model.predict(np.asarray([dat]))
    preds.append(val)
    dt = dt+datetime.timedelta(days=1)
    nd.append(str(dt.date()))
    for j in range(len(dat)-1):
      dat[j] = np.asarray([dat[j+1]])
    dat[-1] = val
  preds = np.array(preds)
  p = scaler2.inverse_transform(preds.reshape((preds.shape[0], preds.shape[1])))
  return pd.DataFrame(list(p.reshape((p.shape[0]*p.shape[1]))), columns = ['values'], index = nd)['values'].T.to_dict()

@app.route('/predictions/', methods=['GET'])
def investmentPlot():
  area = request.args['area']
  time = request.args['time']
  if time == '3 Month':
    return predictionInvestment(area, 90)
  elif time == '6 Month':
    return predictionInvestment(area, 180)
  elif time == '1 Year':
    return predictionInvestment(area, 365)
  elif time == '3 Year':
    return predictionInvestment(area, 365*3)
  elif time == '5 Year':
    return predictionInvestment(area, 365*5)

def predictionInvestment(area, time):
  ndf = df[df['Location'] == area]
  k = ndf.Date.unique()
  k.sort()
  t = datetime.datetime.strptime(k[0], '%Y-%m-%d')
  dates = []
  while t.date()!=datetime.date.today():
    dates.append(str(t.date()))
    t = t+datetime.timedelta(days=1)
  ppm = []
  for i in dates:
    vals = ndf[ndf['Date'] == i]
    vals['PPM'] = vals['Price']/pd.to_numeric(vals['Area'])
    ppm.append(vals.PPM.sum()/len(vals))
  ppm = pd.Series(ppm)
  data = pd.DataFrame(ppm.interpolate(), columns = ['ppm'])
  data['Date'] = dates
  scaler1 = MinMaxScaler(feature_range=(0,1))
  scaler2 = MinMaxScaler(feature_range=(0,1))
  X = []
  y = []
  window = 60
  data = data.ppm.values.copy()
  for i in range(len(data)-60):
    X.append(np.asarray(data[i:i+60]))
    y.append(np.asarray([data[i+60]]))
  X = np.asarray(X)
  y = np.asarray(y)
  X_scaled = scaler1.fit_transform(X)
  y_scaled = scaler2.fit_transform(y)
  X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1] , 1))
  model = load_model('PlotModels/'+area+'.h5')
  dat = X_scaled[-1].copy()
  nd = []
  preds = []
  dt = datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
  for i in range(time):
    val = model.predict(np.asarray([dat]))
    preds.append(val)
    dt = dt+datetime.timedelta(days=1)
    nd.append(str(dt.date()))
    for j in range(len(dat)-1):
      dat[j] = np.asarray([dat[j+1]])
    dat[-1] = val
  preds = np.array(preds)
  p = scaler2.inverse_transform(preds.reshape((preds.shape[0], preds.shape[1])))
  return pd.DataFrame(list(p.reshape((p.shape[0]*p.shape[1]))), columns = ['values'], index = nd)['values'].T.to_dict()
