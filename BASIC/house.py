#数据集
import sklearn.datasets as sd
#小工具
import sklearn.utils as su
#决策树
import sklearn.tree as st
#集合算法：比如随机森林和激励，在该模块中
import sklearn.ensemble as se
#评估模块metrics
import sklearn.metrics as sm
housing=sd.load_boston()
'''
print(housing.feature_names)
print(housing.data.shape)
print(housing.target)
'''
#洗牌，打乱，random_state混乱地程度
x,y=su.shuffle(housing.data,housing.target,
                random_state=7)
train_size=int(len(x)*0.8)
train_x,test_x,train_y,test_y=x[:train_size],x[train_size:],\
                            y[:train_size],y[train_size:]
#构建决策树回归器
model=st.DecisionTreeRegressor(max_depth=4)
#训练数据
model.fit(train_x,train_y)
#预测数据
pred_test_y=model.predict(test_x)
#评估正确率
print(sm.r2_score(test_y,pred_test_y))

#由决策树组成的正向激励回归器
#n_estimators评估器数目,400棵决策树----b
model=se.AdaBoostRegressor(st.DecisionTreeRegressor(max_depth=4),
        n_estimators=400)
model.fit(train_x,train_y)
pred_test_y=model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))

