import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('CarPrice_Assignment.csv',encoding="gbk")

train_x=data[["doornumber",'carbody','wheelbase','carlength','carwidth','carheight','curbweight', \
    "fueltype","aspiration",'drivewheel','enginelocation','enginetype','cylindernumber', \
    'enginesize','fuelsystem','boreratio','stroke','compressionratio','horsepower','peakrpm', \
    'citympg','highwaympg','price','symboling']]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_x["fueltype"] = le.fit_transform(train_x["fueltype"])
train_x["aspiration"] = le.fit_transform(train_x["aspiration"])
train_x["carbody"] = le.fit_transform(train_x["carbody"])
train_x["drivewheel"] = le.fit_transform(train_x["drivewheel"])
train_x["enginelocation"] = le.fit_transform(train_x["enginelocation"])
train_x["enginetype"] = le.fit_transform(train_x["enginetype"])
train_x["fuelsystem"] = le.fit_transform(train_x["fuelsystem"])
train_x["doornumber"] = le.fit_transform(train_x["doornumber"])
train_x["cylindernumber"] = train_x["cylindernumber"].map({"two":2, "three":3,'four':4, 'five':5, 'six':6,'eight':8,'twelve':12})


min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
pca=PCA(n_components=24)
pca.fit(train_x)
x=range(1,25)
y=pca.explained_variance_ratio_
z=[]
ss=0
for i in range(24):
    ss+=y[i]
    z.append(ss)

fig,ax1 = plt.subplots()
line1, = ax1.plot(x,y,'b*-')
ax1.set_ylabel("explained_variance_ratio")
ax1.set_ylim(0,0.45)
ax1.set_xlim(0,25)
ax1.grid()
ax2 = ax1.twinx()
line2,=ax2.plot(x,z,'g*-')
ax2.set_ylabel("accumulate_explained_variance_ratio")
ax2.set_ylim(0,1.3)
plt.legend([line1, line2], ['explained_variance_ratio', 'accumulate_explained_variance_ratio'],loc='upper left')
#plt.show()


pca=PCA(n_components=4)
train_x_pca = pd.DataFrame(pca.fit_transform(train_x))
#train_x_pca.to_csv('train_x_pca.csv')

min_max_scaler=preprocessing.MinMaxScaler()
train_x1=min_max_scaler.fit_transform(train_x_pca)


sse=[]
for k in range(1, 25):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x1)
    sse.append(kmeans.inertia_)# 计算inertia簇内误差平方和
x = range(1, 25)
fig, ax3 = plt.subplots()
ax3.plot(x, sse, 'o-')
ax3.set_xlabel('K')
ax3.set_ylabel('SSE')
#plt.show()

kmeans = KMeans(n_clusters=30)
kmeans.fit(train_x1)
predict_y = kmeans.predict(train_x1)
result = pd.concat([data,pd.DataFrame(predict_y)],axis=1)
result.rename({0:u'聚类结果'},axis=1,inplace=True)
result.to_csv("car_result.csv",index=False,encoding="gbk")

#层次聚类
fig=plt.subplots()
linkage_matrix = ward(train_x1)
dendrogram(linkage_matrix)
plt.show()


