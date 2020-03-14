# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:49:45 2019

@author: 將軍
"""

import pandas as pd

#讀檔
train_file=open('training_label.txt',encoding = 'utf8')
test_file=open('testing_label.txt',encoding = 'utf8')
train=train_file.read()
test=test_file.read()

################################################################
#處理test
test_String=['']*90
j=0
for i in range(90):
    while j<len(test):
        if (test[j]=='0' or test[j]=='1'):
            test_String[i]=test_String[i]+test[j]
            j=j+1
        elif (test[j]=='#'):
            j=j+1
        elif (test[j]=='\n'):
            if (j!=len(test)-1):
                if (test[j+1]=='\n'):
                    j=j+1
                else:
                    j=j+1
                    break
            else:
                break
        else:
            test_String[i]=test_String[i]+test[j]
            j=j+1

#test_label存每個字串的label
test_label=['']*90
for i in range(90):
    test_label[i]=test_String[i][0]  
    

#把test_String前面的0或1去掉(剩下字串)    
for i in range(90):
    if (test_String[i][0]=='0'):
        test_String[i]=test_String[i].replace('0','')
    else:
        test_String[i]=test_String[i].replace('1','')
#l3儲存label和字串
l3=[['']*2 for i in range(90)]
for i in range(90):
    l3[i][0]=test_label[i]
for i in range(90):
    l3[i][1]=test_String[i]

subject=['label','String']
df_test=pd.DataFrame(l3,columns=subject)

#把test_label變成int(好放入模型)
for i in range(90):
    test_label[i]=int(test_label[i])
         
#################################################################
#處理train
train_l=train.split('\n')
train_l2=['']*10000
for i in range(10000):
    train_l2[i]=train_l[i].split('+++$+++')
subject=['label','String']
df_train=pd.DataFrame(train_l2,columns=subject)
train_String=df_train['String']
train_label=df_train['label']

            
##################################################################
#去除stopwords

from nltk.corpus import stopwords
words=stopwords.words('english')

for i in range(90):
    test_String[i]=' '.join([word for word in test_String[i].split() if word not in words])

for i in range(10000):
    train_String[i]=' '.join([word for word in train_String[i].split() if word not in words])



###################################################################
#將文字轉換成向量，像是常見的方法 tf-idf、word2vec
#把train_String和test_String合起來(一起做向量,feature才會一樣，最後才能姜test data放入模型)
String=['']*10090
for i in range(10090):
    if i <10000:
        String[i]=train_String[i]
    else:
        String[i]=test_String[i-10000]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
String_vec=vectorizer.fit_transform(String)
String_array = String_vec.toarray()

train_vec=String_array[0:10000]
test_vec=String_array[10000:10090]



##################################################################
#AdaBoost建模
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
ada=clf.fit(train_vec,train_label)

#XGBboost建模
from xgboost import XGBClassifier
xgbc=XGBClassifier()
xgb=xgbc.fit(train_vec,train_label)

###################################################################
#利用"testing_label.txt"的資料對所建立的模型進行測試，並計算Accuracy、Precision、Recall、F-measure
ada_test=ada.predict(test_vec)
ada_test_ar=[0]*90
for i in range(90):
    ada_test_ar[i]=ada_test[i]
    ada_test_ar[i]=int(ada_test_ar[i])
xgb_test=xgb.predict(test_vec)
xgb_test_ar=[0]*90
for i in range(90):
    xgb_test_ar[i]=xgb_test[i]
    xgb_test_ar[i]=int(xgb_test_ar[i])



from sklearn.metrics import classification_report
print('adaboost報告:\n',classification_report(test_label,ada_test_ar))

print('xgbboost報告:\n',classification_report(test_label,xgb_test_ar))


