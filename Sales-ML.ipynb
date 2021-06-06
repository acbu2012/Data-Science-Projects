#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# In[6]:


data = pd.read_csv('C:/MLprojects/Superstore sales prediction/train.csv')
data.info()


# In[10]:


#Preprocessing
def encode_dates(df,column):
    df= df.copy()
    df[column]=pd.to_datetime(df[column])
    df[column +'_year']=df[column].apply(lambda x:x.year)
    df[column +'_month']=df[column].apply(lambda x:x.month)
    df[column +'_day']=df[column].apply(lambda x:x.day)
    df=df.drop(column, axis=1)
    return df


def onehot_encode(df,column):
    df=df.copy()
    dummies=pd.get_dummies(df[column], prefix=column)
    df=pd.concat([df,dummies], axis=1)
    df=df.drop(column, axis=1)
    return df

def preprocess_inputs(df):
    df=df.copy()
    # drop the unncecessary columns, axis=1 is column axis
    df=df.drop(['Row ID','Country', 'Customer Name', 'Product Name'], axis=1)
    
    #drop cusotmer specific feature columns
    df=df.drop(['Order ID', 'Customer ID'], axis=1)
    
    #extract date features
    df=encode_dates(df,column='Order Date')
    df=encode_dates(df,column='Ship Date')
    
    #one-hot encode categorical features
    for column in ['Ship Mode','Segment','City','State','Postal Code','Region','Product ID','Category','Sub-Category']:
        df = onehot_encode(df, column=column)
        
    #split df into x and y
    y=df['Sales']
    X=df.drop('Sales', axis=1)
    
    #Train-test split
    X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.7,shuffle=True,random_state=1)
    
    
    #scale the data
    scaler=StandardScaler()
    scaler.fit(X_train)
   # X_train=scaler.transform(X_train)
   # X_test=scaler.transform(X_test)
    X_train=pd.DataFrame(scaler.transform(X_train),columns=X.columns)
    X_test=pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test=preprocess_inputs(data)

X_train

y_train

#scale the data
X_train.describe()
X_train.shape


# In[28]:



#Training (Two hidden layer NN)
import tensorflow as tf
import keras
inputs= tf.keras.Input(shape=(X_train.shape[1],))
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model =  tf.keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())


# In[ ]:


#COmpile model

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train,y_train, validation_split=0.2, batch_size=32, epochs=100,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau()])


# In[22]:


#Results
test_loss = model.evaluate(X_test, y_test,verbose=0)
print ("Test Loss: {:0.5f}".format(test_loss))


# In[26]:


y_pred = np.squeeze(model.predict(X_test))
test_r2=r2_score(y_test, y_pred)

print("Test R^2 Score: {:0.5f}".format(test_r2))


# In[ ]:


# rscore should be close change the epoch to 256 instead of 128

