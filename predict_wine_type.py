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


# In[11]:


df_red.isnull().sum()


# In[13]:


df_red_scaled = df_red

from sklearn.preprocessing import StandardScaler

scale_cols = df_red_scaled.columns

scaler = StandardScaler()
df_red_scaled = scaler.fit_transform(df_red[scale_cols])

df_red_scaled = pd.DataFrame(df_red_scaled, columns=scale_cols)


# In[14]:


df_white_scaled = df_white

from sklearn.preprocessing import StandardScaler

scale_cols = df_white_scaled.columns

scaler = StandardScaler()
df_white_scaled = scaler.fit_transform(df_white[scale_cols])

df_white_scaled = pd.DataFrame(df_white_scaled, columns=scale_cols)


# In[15]:


df_red_scaled['type'] = 'red'
df_white_scaled['type'] = 'white'


# In[16]:


df_red_scaled.head()


# In[17]:


df_white_scaled.head()


# In[18]:


df = pd.concat([df_red_scaled,df_white_scaled])


# In[19]:


df.shape


# In[20]:


df['type'] = df['type'].replace('red', 0)
df['type'] = df['type'].replace('white', 1)


# In[22]:


feature_df = df[df.columns.difference(['type'])]
label_df = df['type']


# In[23]:


feature_np = feature_df.to_numpy().astype('float32')
label_np = label_df.to_numpy().astype('float32')


# In[50]:


import numpy as np

s = np.arange(len(feature_np))

np.random.shuffle(s)

feature_np = feature_np[s]
label_np = label_np[s]


# In[51]:


split = 0.15

test_num = int(split*len(label_np))

x_test = feature_np[0:test_num]
y_test = label_np[0:test_num]

x_train = feature_np[test_num:]
y_train = label_np[test_num:]


# In[52]:


print(y_test[:20])


# ### linear regression

# In[53]:


import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input

model = Sequential()

model.add(Dense(1, activation='linear', input_shape=(x_train.shape[-1],)))


# In[54]:


model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mae'] )


# In[55]:


from datetime import datetime

start_time = datetime.now()

hist = model.fit(x_train, y_train, epochs=400, validation_data = (x_test, y_test))

end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)


# In[56]:


import matplotlib.pyplot as plt

plt.title('loss trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()


# In[57]:


pred = model.predict(x_test[-5:])

print(pred.flatten())
print(y_test[-5:])


# ### 은닉층 128노드, 활성함수 sigmoid, optimizer SDG, 손실함수 mse

# In[58]:


model = Sequential()

model.add(Dense(128, activation='sigmoid', input_shape=(x_train.shape[-1],)))

model.add(Dense(1, activation='linear')) 


# In[59]:


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2), 
              loss='mse', metrics=['mae'])


# In[60]:


from datetime import datetime

start_time = datetime.now()

hist = model.fit(x_train, y_train, epochs=400, validation_data = (x_test, y_test))

end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)


# In[61]:


plt.title('loss trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()


# In[62]:


pred = model.predict(x_test[-5:])

print(pred.flatten())
print(y_test[-5:])


# ### 은닉층 128노드, 활성함수 sigmoid, optimizer Adam, 손실함수 mse

# In[64]:


model = Sequential()

model.add(Dense(128, activation='sigmoid', input_shape=(x_train.shape[-1],)))

model.add(Dense(1, activation='linear')) 

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='mse', metrics=['mae'])


# In[65]:


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

model.add(Dense(128, activation='relu', input_shape=(x_train.shape[-1],)))

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


# In[73]:


pred = model.predict(x_test[-5:])

print(pred.flatten())
print(y_test[-5:])

