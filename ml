import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score,accuracy_score, classification_report,ConfusionMatrixDisplay

df=pd.read_csv("C:\\Users\\rakhi\\Desktop\\Datasets\\diabetes.csv")
df

x=df.drop(['Outcome'],axis=1)
y=df['Outcome']

x

y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train

x_test

knn=KNeighborsClassifier(n_neighbors=11) 
knn.fit(x_train,y_train)

knn_preds=knn.predict(x_test)

knn_a=accuracy_score(y_test,knn_preds)
print(knn_a)

 ConfusionMatrixDisplay.from_predictions(y_test,knn_preds)

sns.countplot(df)
 

sns.barplot(x=df['Glucose'],y=df['Insulin'])

plt.scatter(x=df['Glucose'],y=df['Age'])

plt.plot(df['Glucose'],df['Age'])#lineplot

plt.hist(df['Age'],bins=90)

plt.boxplot(df)

df['Glucose'].value_counts().plot(kind='pie',autopct='%.2f')

sns.violinplot(x=df['Glucose'],y=df['Age'])

sns.catplot(x=df['Glucose'],y=df['Age'] )

sns.distplot(df['Age'] )

sns.heatmap(df,annot=True)

sorted_data = df.sort_values(by='Glucose', ascending=False)[:3]
plt.pie(sorted_data['Glucose'], autopct='%.2f' )

unsorted_data = df[:3]
plt.pie(unsorted_data['Glucose'], autopct='%.2f')

data1 = df[:3]
sns.violinplot(data1['Glucose'] )


data1 = df.head(15)
plt.bar(data1['Glucose'],data1['BMI'])

sorted_data = df.sort_values(by='BMI', ascending=False)
data = sorted_data.head(15)
plt.bar(data['Glucose'], data['BMI'])

#sorted_data = df.sort_values(by='BMI', ascending=False)[:5]
#data1 = df[:5]
#plt.plot(data1['BMI'])

sorted_data = df.sort_values(by='BMI', ascending=False)[:20]
data = sorted_data.head(20)
plt.plot(data['Glucose'], data['BMI']) 

sorted_data = df.sort_values(by='BMI', ascending=False)[:15]
data = sorted_data.head(15)
plt.bar(data['Glucose'], data['BMI'])

df.plot(kind = 'box',subplots=True,layout=(7,2), figsize=(25,30))

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'Insulin')

df.plot(kind = 'box',subplots=True,layout=(7,2), figsize =(25,30))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) 

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)

lr_pred=lr.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
 

r2=r2_score(y_test,lr_pred)
print(r2)

rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print(rmse)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
knn_pred=lr.predict(x_test)

knn_pred=knn.predict(x_test)

cm1=confusion_matrix(y_test,knn_pred)
print(cm1)

from sklearn.svm import SVC

svc=SVC(kernel='poly')
#svc=SVC(kernel='sigmoid')
#svc=SVC(kernel='Linear')
#svc=SVC(kernel='rbf')
svc.fit(x_train,y_train)
svc_pred=svc.predict(x_test)

svc_pred=svc.predict(x_test)

cm2=confusion_matrix(y_test,svc_pred)
print(cm2)

ConfusionMatrixDisplay.from_predictions(y_test, knn_pred)

from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(hidden_layer_sizes=(100,50,100),max_iter=1000,activation='relu',random_state=0)
mlp.fit(x_train,y_train)
mlp_pred=mlp.predict(x_test)

mlp_pred=mlp.predict(x_test)

cm3=confusion_matrix(y_test,mlp_pred)
print(cm3)

 df.corr() .style.background_gradient(cmap='BuGn')

from sklearn.cluster import KMeans

inertia = []

for i in range(1, 6):  # You can adjust the range as needed
    cluster = KMeans(n_clusters=i, random_state=42)
    cluster.fit(x)
    inertia.append(cluster.inertia_)

plt.plot(range(1, 6), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


kmeans = KMeans(n_clusters = 3, random_state = 42)
y_kmeans = kmeans.fit_predict(x)
y_kmeans


from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

 
