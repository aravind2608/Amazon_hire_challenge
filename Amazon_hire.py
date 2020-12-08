
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("G:\\Amazon\\train.csv")
df=df.dropna()

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
df['customer_active_segment']= label_encoder.fit_transform(df['customer_active_segment']) 
df['customer_active_segment'].unique() 

df['X1']= label_encoder.fit_transform(df['X1']) 
df['X1'].unique() 


X_mean=df.drop(['customer_category'],axis=1)

import matplotlib.pyplot as plt
import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = X_mean.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

X=np.array(df.iloc[:,[1,2,3,5,6,7,8,9,10]])
Y=np.array(df.iloc[:,11])

from sklearn.feature_selection import VarianceThreshold
var_thres=VarianceThreshold(1)
var_thres.fit(X)

print(var_thres.get_support())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.15,random_state=22)
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

poly_reg = PolynomialFeatures(degree = 3,interaction_only=False, include_bias=True)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X_train) 
#scaled_Y = scaler.fit_transform(y_train)


X_poly = poly_reg.fit_transform(scaled_X)
poly_reg.fit(scaled_X, y_train)


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
dec=RandomForestClassifier(max_depth=5)
dec.fit(X_poly, y_train)

Scaled_testX=scaler.fit_transform(X_test)
y_pred=dec.predict(poly_reg.fit_transform(Scaled_testX))
from sklearn.metrics import accuracy_score
print(dec.score(poly_reg.fit_transform(Scaled_testX), y_test))
#print(dec.score(scaled_X,y_train))

training_accuracy=[]
testing_accuracy=[]
for depth in range(1,16):
    clf=RandomForestClassifier(max_depth=depth)
    clf.fit(X_poly,y_train)
    training_accuracy.append(clf.score(X_poly,y_train))
    testing_accuracy.append(clf.score(poly_reg.fit_transform(Scaled_testX),y_test))
    
corpus=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

plt.plot(corpus,training_accuracy)
plt.plot(corpus,testing_accuracy)
plt.legend('train_acc','test_acc')






df1=pd.read_csv("G:\\Amazon\\test.csv")
df1=df1.fillna(df1.mean())


from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 
df1['customer_active_segment']= label_encoder.fit_transform(df1['customer_active_segment']) 
df1['customer_active_segment'].unique() 

df1['X1']= label_encoder.fit_transform(df1['X1']) 
df1['X1'].unique()

test_X=np.array(df1.iloc[:,[1,2,3,5,6,7,8,9,10]])
test_scaled_X = scaler.fit_transform(test_X) 
#scaled_Y = scaler.fit_transform(y_train)


test_X_poly = poly_reg.fit_transform(test_scaled_X)
#poly_reg.fit(scaled_X, y_train)



Scaled_test_X=scaler.fit_transform(test_X)
testing_Y=dec.predict(poly_reg.fit_transform(Scaled_test_X))






df2=pd.DataFrame()
df2['customer_id']=df1['customer_id']
df2['customer_category']=testing_Y
df2.to_csv("G:\submit.csv",index=False)        
