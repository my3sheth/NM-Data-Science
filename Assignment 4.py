import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('//content//Fake News Data.csv',engine='python',error_bad_lines=False)

df.head()

df.shape

df.isnull().sum()

df=df.dropna()

X=df.drop('label',axis=1)
Y=df['label']

X['text'][1]

import nltk
nltk.download('all')

all_words=[]
for words in X['text']:
    all_words.append(words)
print(len(all_words))

all_words=set(all_words)

from wordcloud import WordCloud, STOPWORDS
stopwords=set(STOPWORDS)
wc=WordCloud(stopwords=stopwords,width=800,height=800,background_color='White',min_font_size=10)

word_string=' '.join(all_words)
wc.generate(word_string)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wc)

X.reset_index(inplace=True)

from keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM
from keras.layers import Dense

voc_size=2000

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(X)):
  review=re.sub('[^a-zA-Z]',' ',X['text'][i])
  review=review.lower()
  review=review.split()

  review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
  review=' '.join(review)
  corpus.append(review)

corpus

corpus[1]

onehot_repr=[]
for words in corpus:
  en=one_hot(words,voc_size)
  onehot_repr.append(en)

onehot_repr[1]

sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length )
print(embedded_docs)

embedded_docs[1]

embedded_docs[0]

embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

len(embedded_docs),Y.shape

X_final=np.array(embedded_docs)
Y_final=np.array(Y)

X_final.shape,Y_final.shape

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_final,Y_final,test_size=0.3,random_state=42)

model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=10,batch_size=64)

Y_pred=model.predict(X_test)

Y_pred

Y_pred=np.where(Y_pred>0.5,1,0)

from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,Y_pred)

import seaborn as sns
sns.heatmap(confusion_matrix(Y_test,Y_pred),annot=True,fmt='g')

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)

from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))

user=X['text'][0]
en=one_hot(user,voc_size)
en=list(en)
print(en)
sent_length=20
embedded_docs=pad_sequences([en],padding='post',maxlen=sent_length)
print(embedded_docs)
ans=model.predict(embedded_docs)
if ans>0.5:
  print('This news is reliable.')
else:
  print('This news is fake.')

user=X['text'][5]
en=one_hot(user,voc_size)
en=list(en)
print(en)
sent_length=20
embedded_docs=pad_sequences([en],padding='post',maxlen=sent_length)
print(embedded_docs)
ans=model.predict(embedded_docs)
if ans>0.5:
  print('This news is reliable.')
else:
  print('This news is fake.')
