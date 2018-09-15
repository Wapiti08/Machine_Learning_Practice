from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

#load the datasets
iris=datasets.load_iris()
#train the data
gnt=GaussianNB()

#predict the data
y_pred=gnt.fit(iris.data,iris.target).predict(iris.data)
print(iris.target,'\n')
print(y_pred)
#print the result
print("the total data is %d,and the rate of mistakes is %d"%(iris.data.shape[0],(y_pred!=iris.target).sum()))