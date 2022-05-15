#!/usr/bin/env python
# coding: utf-8

# In[62]:


import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical


# In[63]:


(x_train, t_train), (x_test, t_test) = mnist.load_data()


# In[64]:


import matplotlib.pyplot as plt

plt.figure(figsize = (6,6))

for index in range(25):
    plt.subplot(5,5,index+1)
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')
    
plt.show()


# In[65]:


x_train = x_train/255.0
x_test = x_test/255.0

print('train max : ', x_train[0].max(), 'train min : ', x_train[0].min())
print('test max : ', x_test[0].max(), 'test min : ', x_test[0].min())


# ### one-hot encoding(to_categorical) and categorical_crossentropy

# In[44]:


#one-hot 인코딩을 하던 안하던 다항분류는 출력층 노드 개수 : 정답의 개수 출력층 활성함수 : softmax
#softmax로부터 십진수를 뽑아내기 위해선 argmax를 해야 함
#sparse categorical는 원핫인코딩을 내부적으로 컴퓨터가 자동으로 해줌
#원핫인코딩을 직접 하려면 categorical_crossentropy


# In[66]:


#one-hot encoding

t_train = to_categorical(t_train, 10)
t_test = to_categorical(t_test, 10)


# In[67]:


model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'softmax'))


# In[68]:


from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'] )


# In[69]:


hist = model.fit(x_train, t_train, epochs=20, validation_split=0.2)


# In[70]:


model.evaluate(x_test, t_test)


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


plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')

plt.legend(loc='best')

plt.show()


# ### auto one-hot encoding and sparse_categorical_crossentropy

# In[55]:


model = Sequential()

model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# In[56]:


from tensorflow.keras.optimizers import SGD

model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'] )


# In[57]:


hist = model.fit(x_train, t_train, epochs=20, validation_split=0.2)


# In[58]:


plt.title('loss trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.show()


# In[59]:


plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')

plt.legend(loc='best')

plt.show()


# In[ ]:




