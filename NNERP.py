import numpy as np
import pandas as pd
from mapsdict import GET_DICT
import tensorflow
from keras import Sequential,Model,Input,optimizers
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional
from keras.utils import plot_model
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)
input_dim=len(list(set(GET_DICT.data['Word'].to_list())))+1
output_dim= 64
input_length=max([len(s) for s in GET_DICT.data_group['Word_idx'].tolist()])
n_tag=len(GET_DICT.tag2idx)

def get_bilstm_lstm_model():
    model=Sequential()
    #Embed layer
    model.add(Embedding(input_dim=input_dim,output_dim=output_dim,input_shape=(input_length,)))
    # bidirectional layer
    model.add(Bidirectional(LSTM(units=output_dim,return_sequences=True,dropout=0.2,recurrent_dropout=0.2),merge_mode='concat'))
    #LSTM
    model.add(LSTM(units=output_dim,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))
    #timedis
    model.add(TimeDistributed(Dense(n_tag,activation='relu')))
    #optimiser
    #adam=optimizers.Adam(lr=0.0005,beta_1=0.9,beta_2=0.999)
    #Compile
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model
def train_model(X,y, model):
    loss=list()
    for i in range(25):
        hist=model.fit(X,y,batch_size=1000,verbose=1,epochs=1,validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss

results = pd.DataFrame()
model_bilstm_lstm = get_bilstm_lstm_model()
#plot_model(model_bilstm_lstm)
results['with_add_lstm'] = train_model(GET_DICT.train_tokens, np.array(GET_DICT.train_tags), model_bilstm_lstm)
