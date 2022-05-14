#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

df_red = pd.read_csv('./dataset/winequality-red.csv', delimiter=';')
df_white = pd.read_csv('./dataset/winequality-white.csv', delimiter=';')


# In[9]:


df_red.head()


# In[10]:


df_white.head()


# In[12]:


df_red.info()


# In[14]:


df_red.isnull().sum()


# In[20]:


df = pd.concat([df_red,df_white])


# In[21]:


df.head()


# In[23]:


df_scaled = df

from sklearn.preprocessing import StandardScaler

scale_cols = df_scaled[df_scaled.columns.difference(['quality'])].columns

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[scale_cols])

df_scaled = pd.DataFrame(df_scaled, columns=scale_cols)

df_scaled['quality'] = df['quality'].values 


# In[25]:


df_scaled.head()


# In[27]:


feature_df = df_scaled[df_scaled.columns.difference(['quality'])]

label_df = df_scaled['quality']


# In[33]:


feature_df.shape


# In[28]:


feature_np = feature_df.to_numpy().astype('float32')
label_np = label_df.to_numpy().astype('float32')


# In[30]:


import numpy as np

s = np.arange(len(feature_np))

np.random.shuffle(s)

feature_np = feature_np[s]
label_np = label_np[s]


# In[43]:


split = 0.15

test_num = int(split*len(label_np))

x_test = feature_np[0:test_num]
y_test = label_np[0:test_num]

x_train = feature_np[test_num:]
y_train = label_np[test_num:]


# ### linear regression

# In[88]:


import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input

model = Sequential()

model.add(Dense(1, activation='linear', input_shape=(11,)))


# In[89]:


model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mae'] )


# In[90]:


from datetime import datetime

start_time = datetime.now()

hist = model.fit(x_train, y_train, epochs=400, validation_data = (x_test, y_test))

end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)


# In[91]:


import matplotlib.pyplot as plt

plt.title('loss trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()


# In[92]:


pred = model.predict(x_test[-5:])

print(pred.flatten())
print(y_test[-5:])


# ### 은닉층 128노드, 활성함수 sigmoid, optimizer SDG, 손실함수 mse

# In[60]:


model = Sequential()

model.add(Dense(128, activation='sigmoid', input_shape=(11,)))

model.add(Dense(1, activation='linear')) 


# In[61]:


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2), 
              loss='mse', metrics=['mae'])


# In[62]:


from datetime import datetime

start_time = datetime.now()

hist = model.fit(x_train, y_train, epochs=400, validation_data = (x_test, y_test))

end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)


# In[63]:


plt.title('loss trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()


# In[64]:


pred = model.predict(x_test[-5:])

print(pred.flatten())
print(y_test[-5:])


# ### 은닉층 128노드, 활성함수 sigmoid, optimizer Adam, 손실함수 mse

# In[65]:


model = Sequential()

model.add(Dense(128, activation='sigmoid', input_shape=(11,)))

model.add(Dense(1, activation='linear')) 

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='mse', metrics=['mae'])


# In[66]:


start_time = datetime.now()

hist = model.fit(x_train, y_train, epochs=400, validation_data = (x_test, y_test))

end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)


# In[67]:


plt.title('loss trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()


# In[68]:


pred = model.predict(x_test[-5:])

print(pred.flatten())
print(y_test[-5:])


# ### 은닉층 128노드, 활성함수 relu, optimizer Adam, 손실함수 mse

# In[69]:


model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(11,)))

model.add(Dense(1, activation='linear')) 

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='mse', metrics=['mae'])


# In[70]:


start_time = datetime.now()

hist = model.fit(x_train, y_train, epochs=400, validation_data = (x_test, y_test))

end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)


# In[71]:


plt.title('loss trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()


# In[72]:


pred = model.predict(x_test[-5:])

print(pred.flatten())
print(y_test[-5:])

