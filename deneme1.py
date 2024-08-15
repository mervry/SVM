# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:06:12 2021

@author: URAY
"""
import pandas as pd
veriler=pd.read_csv("deneme.csv")
'''
x=independent
y=dependent
'''
Y=veriler.iloc[:,2:3]
#kategorikten nümeriğe geçiş

outlook=veriler.iloc[:,0:1].values #dataframe üzerinde işlem yapamayız o yüzden array a dönüştürmeliyiz onun için de values kullanıyoruz
'''
dummy variable,
encoding-label encoding-one hot coding #sklearn kütüphanesinden incele
'''
from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
outlook[:,0]=lr.fit_transform(outlook[:,0])
'''
makine 0 1 2 olarak etiketledi datayı ama tekrardan bi encoding yapmalıyız çünkü sistem bunu 
0 büyüktür 1, 1 büyüktür 2 den şeklinde etiketlediklerini de o şekilde kodluyor yani mesela bu data 
için sunny büyüktür rainy ve overcast şeklinde düşünüyor. bunu düzeltmek için tekrar kodlama yapıyoruz
'''
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
'''
array den dataframe geçmeliyiz tekrardan
'''
outlook_1=pd.DataFrame(data=outlook, index=range(14), columns=["overcast", "rainy", "sunny"])

'''
ödev-3
windy, play, label encoder ile dönüşüm yapalım
ikisi birlikte, ayrı ayrı
'''

windy_play=veriler.iloc[:,3:5]
windy_play_2=windy_play.apply(lr.fit_transform)

#yeni veri seti kümesini oluşturma


df1=pd.concat((outlook_1,veriler.iloc[:,1:2]), axis=1) #axis kodunu index sıralamasına uygun sıralarsın diye kullanıyoruz
X=pd.concat([df1,windy_play_2], axis=1)
#Y yukarda
veriler_yeni=pd.concat([X, Y], axis=1)

#train-test aşaması

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,Y, test_size=0.33, random_state=0)

#algoritma seçme aşaması
#linear regression/multi-linear regression ve svm-svr öğrendik. Sırayla yapalım

#linear regression

from sklearn.linear_model import LinearRegression
lin=LinearRegression()

lin.fit(x_train, y_train)
tahmin1=lin.predict(x_test)
print(tahmin1)

#evaluation
'''
from sklearn.metrics import r2_score
print(r2_score(y_test, tahmin1))
#sonuç çok kötü, linear reg data setimize uygun değil

#SVM- support vector regression deneyelim

from sklearn.svm import SVR
svr=SVR(kernel="linear")

svr.fit(x_train, y_train)
tahmin2=svr.predict(x_test)
print(tahmin2)
'''
 
