import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
import copy
import time


####1.데이터 처리
def CreateData(data,seq_len,shuffle=False):
    stdLst=[]

    for i in range(seq_len,len(data)):
        temp=copy.deepcopy(data[i-seq_len:i])
        temp=np.array(temp)
        maxV=max(temp)
        minV=min(temp)
        temp-=minV
        temp/=(maxV-minV)
        temp-=0.5
        if((maxV-minV)==0):
            print("ZeroDivision!!")
            print(data[i-seq_len:i+1])
            continue
        tempY=data[i]
        tempY-=minV
        tempY/=(maxV-minV)
        tempY-= 0.5
        stdLst.append([temp[:],tempY,[minV,maxV]])
    if shuffle==True:
        random.shuffle(stdLst)
    x_data=[]
    y_data=[]
    base_lst=[]
    for temp in stdLst:
        x_data.append(temp[0])
        y_data.append(temp[1])
        base_lst.append(temp[2])
    return np.array(x_data),np.array(y_data),np.array(base_lst)




data=pd.read_csv("./data.csv")
high_prices = data['high_price'].values
low_prices = data['low_price'].values
mid_prices = (high_prices + low_prices) / 2
seq_len = 50

row = int(round(mid_prices.shape[0] * 0.9))
train = mid_prices[:row]
test=mid_prices[row:]

x_train,y_train,z_train=CreateData(train,seq_len,True)
print(x_train.shape)
x_test1,y_test1,z_test1=CreateData(test,seq_len,True)
x_test2,y_test2,z_test2=CreateData(test,seq_len,False)
#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))

### 모델구성


model=Sequential()
model.add(Dense(64,input_dim=seq_len,activation='relu'))
model.add(Dropout(0.3))
for i in range(4):
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.3))
model.add(Dense(1,activation='tanh'))
model.compile(loss='mean_squared_error', optimizer=Adam(1e-6))
print(model.summary())

lossHist=[]
epochs=100000
for i in range(epochs):
    print(i+1,"/",epochs)
    hist = model.fit(x_train, y_train, epochs=1, batch_size=64,shuffle=False)
    lossHist.append(hist.history['loss'][-1])
    if i%250==0:
        model.save("BIT_PREDICT_6.h5")
        valScore = model.evaluate(x_test1, y_test1, batch_size=64)

        print('Validataion Score: ', valScore)
        time.sleep(10)
        model.reset_states()
        plt.plot(lossHist)

        plt.show()

        test_result=model.predict(x_test2).reshape(-1)
        pltDataP=[]
        pltDataY=[]
        for i in range(100):
            minV=z_test2[i][0]
            maxV=z_test2[i][1]
            pltDataY.append(minV+(maxV-minV)*((y_test2[i]+0.5)))
            pltDataP.append(minV+(maxV-minV)*((test_result[i]+0.5)))
        plt.plot(pltDataP[:100],"b-")
        plt.plot(pltDataY[:100],"r-")
        print(pltDataP[:100])
        print(pltDataY[:100])
        error=abs(y_test2[:100]-test_result[:100])


        plt.show()
        #continue
        # 실전 테스트
        x_test =copy.deepcopy(x_test2[1])

        predictLen = 10
        predictData = []
        temp_std=[z_test2[0]]
        for i in range(predictLen):
            result=model.predict(x_test.reshape(1,50))
            print(result)
            predictData.append((result[0][0]+0.5)*(temp_std[-1][1]-temp_std[-1][0])+temp_std[-1][0])
            tempX=np.hstack([x_test[1:],result[0]])
            tempX=(tempX+0.5)*(temp_std[-1][1]-temp_std[-1][0])+temp_std[-1][0]
            print("AtempX>",tempX)
            maxV=max(tempX)
            minV=min(tempX)
            tempX=tempX-minV
            print("BtempX>", tempX)
            tempX/=(maxV-minV)
            print("CtempX>", tempX)
            tempX-=0.5
            print("DtempX>", tempX)
            testX=tempX

            temp_std.append([minV,maxV])
        plt.plot(predictData,"bo")
        plt.plot(pltDataY,"ro")
        plt.show()




