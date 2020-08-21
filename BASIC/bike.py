import csv
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp

#以天为单位的数据
with open('../data/bike_day.csv','r') as f:
    #读取器
    reader=csv.reader(f)
    x,y=[],[]
    #这里不需要转置码？因为row表示一行，其中各项已经自然分割，成为了一个列表
    for row in reader:
        '''
        print(row[0])
        print('***********')
        print(row[1])
        print('***********')
        print(row[-1])
        '''
        #csv在获得每行时已经用，做了分割
        #前两行一个序号，一个日期，输入的部分
        x.append(row[2:13])
        #输出的部分
        y.append(row[-1])
feature_names_dy=np.array(x[0])
x=np.array(x[1:],dtype=float)
y=np.array(y[1:],dtype=float)

x,y=su.shuffle(x,y,random_state=7)
#训练集的大小
train_size=int(len(x)*0.9)
train_x,test_x,train_y,test_y=x[:train_size],x[train_size:],\
                        y[:train_size],y[train_size:]
model=se.RandomForestRegressor(max_depth=10,n_estimators=1000,min_samples_split=2)
model.fit(train_x,train_y)
fi_dy=model.feature_importances_    
pred_test_y=model.predict(test_x)
# print(sm.r2_score(test_y,pred_test_y))   

#以小时为单位的数据
with open('../data/bike_hour.csv','r') as f:
    #读取器
    reader=csv.reader(f)
    x,y=[],[]
    #这里不需要转置码？因为row表示一行，其中各项已经自然分割，成为了一个列表
    for row in reader:
        #前两行一个序号，一个日期，输入的部分
        x.append(row[2:13])
        #输出的部分
        y.append(row[-1])
feature_names_hr=np.array(x[0])
x=np.array(x[1:],dtype=float)
y=np.array(y[1:],dtype=float)

x,y=su.shuffle(x,y,random_state=7)
#训练集的大小
train_size=int(len(x)*0.9)
train_x,test_x,train_y,test_y=x[:train_size],x[train_size:],\
                        y[:train_size],y[train_size:]
model=se.RandomForestRegressor(max_depth=10,n_estimators=1000,min_samples_split=2)
model.fit(train_x,train_y)
fi_hr=model.feature_importances_    
pred_test_y=model.predict(test_x)
# print(sm.r2_score(test_y,pred_test_y)) 
mp.figure('Bike',facecolor='lightgray')
mp.subplot(211)
mp.title('Day',fontsize=16)
mp.ylabel('Importance',fontsize=12)
mp.tick_params(labelsize=8)
mp.grid(axis='y',linestyle=':')
sorted_indices=fi_dy.argsort()[::-1]
pos=np.arange(sorted_indices.size)
mp.bar(pos,fi_dy[sorted_indices],facecolor='deepskyblue')
mp.xticks(pos,feature_names_dy[sorted_indices],rotation=30) 

mp.subplot(212)
mp.title('Hour',fontsize=16)
mp.ylabel('Importance',fontsize=12)
mp.tick_params(labelsize=8)
mp.grid(axis='y',linestyle=':')
sorted_indices=fi_hr.argsort()[::-1]
pos=np.arange(sorted_indices.size)
mp.bar(pos,fi_hr[sorted_indices],facecolor='lightcoral')
mp.xticks(pos,feature_names_hr[sorted_indices],rotation=30) 
mp.show() 