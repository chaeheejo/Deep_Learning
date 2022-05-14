#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib
import pandas as pd
from matplotlib import pyplot as plt


# In[24]:


df = pd.read_csv('./dataset/kaggle_diabetes.csv')
df.head()


# In[25]:


df.hist()

plt.tight_layout()
plt.show()


# In[26]:


df['BloodPressure'].hist()

plt.tight_layout()
plt.show()


# In[27]:


df.info()


# In[28]:


df.describe()


# In[29]:


df.isnull().sum()


# In[30]:


for col in df.columns:
    missing_rows = df.loc[df[col]==0].shape[0]
    print(col + ": "+ str(missing_rows))


# In[31]:


import numpy as np

df['Glucose'] = df['Glucose'].replace(0,np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0,np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0,np.nan)
df['Insulin'] = df['Insulin'].replace(0,np.nan)
df['BMI'] = df['BMI'].replace(0,np.nan)

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())


# In[32]:


for col in df.columns:
    missing_rows = df.loc[df[col]==0].shape[0]
    print(col + ": "+ str(missing_rows))


# In[33]:


df_scaled = df

df_scaled.describe()


# In[35]:


#표준화 진행

from sklearn.preprocessing import StandardScaler

scale_cols = df_scaled[df_scaled.columns.difference(['Outcome'])].columns

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[scale_cols])

print(type(df_scaled))

df_scaled = pd.DataFrame(df_scaled, columns=scale_cols)

df_scaled['Outcome'] = df['Outcome'].values  # 원본 DataFrame 보존


# In[36]:


feature_df = df_scaled[df_scaled.columns.difference(['Outcome'])]

label_df = df_scaled['Outcome']

print(feature_df)


# In[37]:


feature_np = feature_df.to_numpy().astype('float32')
label_np = label_df.to_numpy().astype('float32')

print(feature_np.shape, label_np.shape)


# In[38]:


s = np.arange(len(feature_np))

np.random.shuffle(s)

feature_np = feature_np[s]
label_np = label_np[s]


# In[39]:


split = 0.15

test_num = int(split*len(label_np))

x_test = feature_np[0:test_num]
y_test = label_np[0:test_num]

x_train = feature_np[test_num:]
y_train = label_np[test_num:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[40]:


import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input

model = Sequential()

model.add(Dense(1, input_shape=(8, ),activation='sigmoid'))  


# In[41]:


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2), 
              loss='binary_crossentropy', metrics=['accuracy'])


# In[42]:


from datetime import datetime

start_time = datetime.now()

hist = model.fit(x_train, y_train, epochs=400, validation_data = (x_test, y_test))

end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)


# In[43]:


model.evaluate(x_test, y_test)


# In[44]:


import matplotlib.pyplot as plt

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()


# In[45]:


import matplotlib.pyplot as plt

plt.title('accuracy trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')
plt.legend(loc='best')

plt.show()


# In[ ]:




