#%% imports
import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from RNN_data_preprocessing import *

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# import Eksplorativna_analiza

#%%

class SimpleRNN:
    def __init__(self, input_shape, output_len, loss='mean_squared_error'):
        self.input_shape = input_shape
        self.output_len = output_len
        self.loss = loss
        self.model = keras.models.Sequential()
        self._addLayers()
        self._compileModel()
        
    def _addLayers(self):
        self.model.add(keras.layers.GRU(name='GRU_layer',
            units=10, ##output shape of this layer
            input_shape=self.input_shape
            ))
        self.model.add(keras.layers.Dense(self.output_len))
    
    def _compileModel(self):
        self.model.compile(loss=self.loss
                           , optimizer="adam"
                           , metrics=[keras.metrics.MeanSquaredError()])



#%% function definition
def GetSimpleRNNWithTrainingHistory(df=None,epochs=3, timeseries_batch_size= 32, timeseries_sequence_length= 5, batch_size=32 ):
    ''' returns the  'SimpleRNN' object with its training history 
    'timeseries_sequence_length' can be set to practicaly anything since we use the time difference in minutes between instances'''


    ##%% preprocessing 
    if (df == None):
        import Eksplorativna_analiza
        df = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
    
    df = df.loc[ (df['DATE OCC'].dt.year >= 2023) 
                & (df['DATE OCC'].dt.year <= 2024) ]
    
    df = GetExactTimeOfCrimeOccurrence(df,new_column_name = 'DATE TIME OCC')
    df = GetExactTimeOfCrimeOccurrenceInMinutes(df,new_column_name = 'TIME OCC minutes')
    df = GetExactTimeOfCrimeOccurrenceInMinutesSinceEpoch(df,new_column_name = 'TOTAL TIME OCC minutes')
    df = GetTimeDifferenceInMinutes(df,new_column_name = 'TIME DIFFERENCE minutes')
    
    input_columns = ['TIME DIFFERENCE minutes','YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    target_columns = ['LAT','LON']
    
    # ##ADDED
    # label_encoder = LabelEncoder()
    # df['AREA'] = label_encoder.fit_transform(df['AREA'])
    # df['Crm Cd'] = label_encoder.fit_transform(df['Crm Cd'])
    # # Skaliranje podataka
    # scaler_X = StandardScaler()
    # df[input_columns] = scaler_X.fit_transform(df[input_columns])
    
    # scaler_y = StandardScaler()
    # df[target_columns] = scaler_y.fit_transform(df[target_columns])
    # ##-------
    
    ##%% train and validation split
    train_to_validation_split = 0.9
    train_df = df.loc[ (df['DATE OCC'].dt.year >= 2023) 
                      & (df['DATE OCC'].dt.year <= 2024) ]
    
    validation_df = train_df[ int(len(train_df) * train_to_validation_split) : ]
    train_df = train_df[ 0 : int(len(train_df) * train_to_validation_split) ]
    
    
    ##%% create time series
    ## these 2 paramaters are now function paramaters
    # timeseries_sequence_length = 5
    # timeseries_batch_size = 10
    
    
    train_data = tf.keras.preprocessing.timeseries_dataset_from_array(
        data= train_df[input_columns],
        targets= train_df[target_columns],
        sequence_length= timeseries_sequence_length,
        batch_size= timeseries_batch_size 
        )
    
    validation_data = tf.keras.preprocessing.timeseries_dataset_from_array(
        data= validation_df[input_columns],
        targets= validation_df[target_columns],
        sequence_length= timeseries_sequence_length,
        batch_size= timeseries_batch_size 
        )
    
    # print(train_data)
    # for batch in train_data:
    #     inputs, targets = batch
    #     print('inputs=',inputs)
    #     print('targets=',targets )
    # print(train_data )
    
    
    
    ##%% create and compile
    # del rnn
    rnn = SimpleRNN(input_shape= (timeseries_sequence_length,len(input_columns)) 
                    ,output_len=len(target_columns )  
                    )
    print(rnn.model.summary())
    
    ##%% train
    training_history = rnn.model.fit(train_data
                                     , epochs= epochs
                                     , shuffle= False
                                     , validation_data= validation_data
                                     , batch_size=batch_size) 
    
    return rnn,training_history 

#%%
def GetSimpleRNNEvaluation(rnn_model,df = None,timeseries_batch_size= 32,timeseries_sequence_length= 5,batch_size=32):
    
    if (df == None):
        import Eksplorativna_analiza
        df = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()

    df = df.loc[ (df['DATE OCC'].dt.year == 2022) 
                & (df['DATE OCC'].dt.month == 1) ]
    
    df = GetExactTimeOfCrimeOccurrence(df,new_column_name = 'DATE TIME OCC')
    df = GetExactTimeOfCrimeOccurrenceInMinutes(df,new_column_name = 'TIME OCC minutes')
    df = GetExactTimeOfCrimeOccurrenceInMinutesSinceEpoch(df,new_column_name = 'TOTAL TIME OCC minutes')
    df = GetTimeDifferenceInMinutes(df,new_column_name = 'TIME DIFFERENCE minutes')
    
    input_columns = ['TIME DIFFERENCE minutes','YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    target_columns = ['LAT','LON']
    
    # ##ADDED
    # label_encoder = LabelEncoder()
    # df['AREA'] = label_encoder.fit_transform(df['AREA'])
    # df['Crm Cd'] = label_encoder.fit_transform(df['Crm Cd'])
    # # Skaliranje podataka
    # scaler_X = StandardScaler()
    # df[input_columns] = scaler_X.fit_transform(df[input_columns])
    
    # scaler_y = StandardScaler()
    # df[target_columns] = scaler_y.fit_transform(df[target_columns])
    # ##-------
    
    ## create time series
    # timeseries_sequence_length = timepoints_per_day 
    
    test_data_input = tf.keras.preprocessing.timeseries_dataset_from_array(
        data= df[input_columns],
        targets= df[target_columns],
        sequence_length= timeseries_sequence_length,
        batch_size= timeseries_batch_size 
        )
    
    evaluation_result = rnn_model.evaluate(x=test_data_input
                                           ,batch_size= batch_size)
    return evaluation_result 

#%% 

def trainSimpleRNN(save_model=False,save_history=False,epochs=10,batch_size=32,timeseries_batch_size= 32):
    
    rnn,training_history = GetSimpleRNNWithTrainingHistory(epochs= epochs
                                                            ,batch_size= batch_size
                                                            ,timeseries_batch_size= timeseries_batch_size
                                                            ,timeseries_sequence_length= 5)
    
    if (save_model):        
        rnn.model.save("saved_models/SimpleRNN.keras")
    
    if (save_history):
        training_history_df = pd.DataFrame(training_history.history)
        training_history_fname = 'saved_models/SimpleRNN_history.csv'
        with open(training_history_fname, mode='w') as f:
            training_history_df.to_csv(f)
    return rnn, training_history



def evalueateSimpleRNN(rnn_model= None, save_history=False, timeseries_batch_size= 32,batch_size=32):    

    if (rnn_model== None):
        rnn_model = keras.models.load_model('saved_models/SimpleRNN.keras')    
    evaluation_result = GetSimpleRNNEvaluation(rnn_model
                                               ,timeseries_batch_size= timeseries_batch_size
                                               ,batch_size= batch_size
                                               ,timeseries_sequence_length= 5
                                               )
    
    if (save_history):
        evaluation_result_dictionary = {'loss': evaluation_result[0], 'mse': evaluation_result[1] }
        evaluation_result_df = pd.DataFrame(data=evaluation_result_dictionary,dtype=float,index=[0])
        evaluation_result_fname = 'saved_models/SimpleRNN_evaluation_result.csv'
        with open(evaluation_result_fname, mode='w') as f:
            evaluation_result_df.to_csv(f)
    
    return evaluation_result

    
#%% 
def testSimpleRNNTimeDifference(save_model=False,save_history=False,epochs=10,batch_size=32,timeseries_batch_size= 32):
    rnn, training_history = trainSimpleRNN(save_model=save_model
                                            ,save_history=save_history
                                            ,epochs= epochs
                                            ,batch_size= batch_size
                                            ,timeseries_batch_size= timeseries_batch_size
                                            )
    evaluation_result = evalueateSimpleRNN(rnn_model= rnn.model
                                           ,save_history= save_history
                                           ,timeseries_batch_size= timeseries_batch_size
                                           ,batch_size= batch_size
                                           )

#%% main function
if __name__ == '__main__':
    rnn, training_history = trainSimpleRNN(save_model=False
                                            ,save_history=False
                                            ,epochs=10
                                            ,batch_size=32
                                            ,timeseries_batch_size= 32
                                            )
    evaluation_result = evalueateSimpleRNN(rnn_model= rnn.model
                                           ,save_history=True
                                           ,timeseries_batch_size= 32
                                           ,batch_size=32
                                           )
