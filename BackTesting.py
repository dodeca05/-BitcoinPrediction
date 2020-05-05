import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import copy
money=100
fee=0.0005
buyPrice=0

buyX=[]
buyY=[]
sellX=[]
sellY=[]

deadLineX=[]
deadLineY=[]

sbuyX=[]
sbuyY=[]
ssellX=[]
ssellY=[]


def GetCandle(interval=60, count=200, date=None):
    url = "https://api.upbit.com/v1/candles/minutes/" + str(interval)
    if date == None:
        querystring = {"market": "KRW-BTC", "count": str(count)}
    else:
        querystring = {"market": "KRW-BTC", "to": str(date), "count": str(count)}

    response = requests.request("GET", url, params=querystring)

    json_data = json.loads(response.text)[::-1]
    return json_data


model=keras.models.load_model('BIT_PREDICT_6.h5')

#시간 지정
#date="2020-03-09 23:30:00"
#candle=GetCandle(date=date)
candle=GetCandle()#실시간

def predict(data):
    Xdata = np.array(data)
    maxV = max(Xdata)
    minV = min(Xdata)
    Xdata -= minV
    Xdata /= (maxV - minV)
    Xdata -= 0.5
    result = model.predict(Xdata.reshape(1, 50))[0][0]
    result += 0.5
    result *= (maxV - minV)
    result += minV
    return result
for i in range(200):
    candle[i]=(candle[i]['low_price']+candle[i]['high_price'])/2
predictdata=copy.deepcopy(candle)
for i in range(51,200):
    beforePredict=predict(candle[i-51:i-1])
    nowPrecit=predict(candle[i-50:i])
    predictdata[i]=nowPrecit


    if candle[i-1]<beforePredict and candle[i]<candle[i-1] and candle[i]<nowPrecit:
        sellX.append(i)
        sellY.append(candle[i])
        if buyPrice != 0 and (candle[i] > buyPrice * (1 + fee) * (1 + fee) or buyPrice * 0.99 > candle[i]):
            money = money * (1 - fee) / buyPrice * candle[i] * (1 - fee)
            buyPrice = 0
            ssellX.append(i)
            ssellY.append(candle[i])



    elif candle[i-1]>beforePredict and candle[i]>candle[i-1] and candle[i]>nowPrecit:
        buyX.append(i)
        buyY.append(candle[i])
        if buyPrice == 0:
            buyPrice = candle[i]
            sbuyX.append(i)
            sbuyY.append(candle[i])



if buyPrice!=0:
    money=money/buyPrice*candle[-1]
print("수익률 = ",money-100,"%")

ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.plot(candle,"k-")
plt.plot(predictdata,"k:")

plt.plot(buyX,buyY,"gx")
plt.plot(sellX,sellY,"rx")
plt.plot(sbuyX,sbuyY,"go")
plt.plot(ssellX,ssellY,"ro")

plt.show()
