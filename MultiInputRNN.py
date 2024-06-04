#%% imports
import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from RNN_data_preprocessing import *

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


#%% class declaration

class MultiInputRNN:
    def __init__(self, RNN_input_shape, Dense_input_shape, output_len, loss='mean_squared_error'):
        self.RNN_input_shape = RNN_input_shape
        self.Dense_input_shape = Dense_input_shape
        self.output_len = output_len
        self.loss = loss
        self.model = None
        self.RNNInputLayer = None
        self.RNNLayer = None
        self.DenseInputLayer = None
        self.DenseLayer = None
        self.ConcatLayer = None
        self.DenseOutputLayer = None
        self._addLayers()
        self._compileModel()
        
    def _addLayers(self):
        self.RNNInputLayer = keras.layers.Input(shape=self.RNN_input_shape)
        self.RNNLayer = keras.layers.GRU(units=10) (self.RNNInputLayer )
        
        self.DenseInputLayer = keras.layers.Input(shape=self.Dense_input_shape)
        self.DenseLayer = keras.layers.Dense(8,activation='linear')(self.DenseInputLayer )
        
        ## trainable=True by default
        self.ConcatLayer = keras.layers.Concatenate(trainable=True)([self.DenseLayer ,self.RNNLayer ])
        self.DenseOutputLayer = keras.layers.Dense(self.output_len,activation='linear')(self.ConcatLayer )
        
        self.model = keras.models.Model(
            inputs=[self.RNNInputLayer, self.DenseInputLayer]
            ,outputs=self.DenseOutputLayer 
            )
    
    def _compileModel(self):
        self.model.compile(loss=self.loss
                           , optimizer="adam"
                           , metrics=[keras.metrics.MeanSquaredError()])
    
        

#%% function definition
def GetMultiInputRNNWithTrainingHistory(df = None,epochs=3, timeseries_batch_size= 32, timepoints_per_day= 24, batch_size= 32 ):
    
    if df == None:
        df = GetDataFrameWithMask()
        # print(df)
        ## NOTE: KEEP THE ''DATE TIME OCC' in its original state since we will pass it to the RNN without skipping it with a mask
        # df.loc[df['AVAILABLE MASK'] == 0] = 0 ## set the mask values to be 0
        # print(df)
    
    # df = GetExactTimeOfCrimeOccurrenceInMinutesSinceEpoch(df,new_column_name = 'TOTAL TIME OCC minutes')
    # df = GetTimeDifferenceInMinutes(df,new_column_name = 'TIME DIFFERENCE minutes')

    rnn_input_columns = ['YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    rnn_target_columns = ['LAT','LON']
    dense_input_columns = ['YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    dense_target_columns = rnn_target_columns ## must be the same as target columns
    

    ## %% train and validation split
    train_to_validation_split = 0.9
    train_df = df
    
    validation_df = train_df[ int(len(train_df) * train_to_validation_split) : ]
    train_df = train_df[ 0 : int(len(train_df) * train_to_validation_split) ]
    
    
    ##%% create time series
    ## these 2 paramaters are now function paramaters
    # timepoints_per_day = 24 ## from above so we look at the previous day
    timeseries_sequence_length = timepoints_per_day 
    # timeseries_batch_size = timepoints_per_day
    
    #####################
    ## RNN INPUT (can't put timeseries object, they must be tensors)
    
    ## X value for RNN training
    rnn_train_data_input = np.array(train_df[rnn_input_columns ]).copy()
    
    ## the dimensions of the input
    dim1_train_data =  int(rnn_train_data_input.size  / (timepoints_per_day*len(rnn_input_columns))) ##how many instances we have
    dim2 = timepoints_per_day     ##sequence length
    dim3 = len(rnn_input_columns) ##number of fields/columns
    
    rnn_train_data_input = GetTimeSeriesArrayFromArray(rnn_train_data_input
                                                       ,timepoints_per_day=timepoints_per_day
                                                       ,timeseries_batch_size=timeseries_batch_size)
    
    ## Y value for RNN training
    rnn_train_data_output = np.array(train_df[rnn_target_columns ][dim2:]).copy()
    ##the last missing column should be the first in the series, I can't be bothered with it
    rnn_train_data_output = np.append(rnn_train_data_output, train_df[rnn_target_columns][0:1],axis=0 ) 
    
    rnn_train_data_output = GetTimeSeriesArrayFromArray(rnn_train_data_output 
                                                       ,timepoints_per_day=1
                                                       ,timeseries_batch_size=timeseries_batch_size)
    
    ## shape: (number_of_windows, 1, num_of_coumns) => (number_of_windows, num_of_coumns)
    rnn_train_data_output = rnn_train_data_output.reshape((rnn_train_data_output.shape[0], rnn_train_data_output.shape[2]))
    
    ##Get the RNN validation input and output data
    ## X value for RNN validation
    rnn_validation_data_input = np.array(validation_df[rnn_input_columns ]).copy()
    
    rnn_validation_data_input = GetTimeSeriesArrayFromArray(rnn_validation_data_input 
                                                       ,timepoints_per_day=timepoints_per_day
                                                       ,timeseries_batch_size=timeseries_batch_size)
    
    
    
    rnn_validation_data_output = np.array(validation_df[rnn_target_columns][dim2:]).copy()
    ##the last missing column should be the first in the series, I can't be bothered with it
    rnn_validation_data_output = np.append(rnn_validation_data_output, validation_df[rnn_target_columns][0:1],axis=0 ) 
    rnn_validation_data_output = GetTimeSeriesArrayFromArray(rnn_validation_data_output 
                                                       ,timepoints_per_day=1
                                                       ,timeseries_batch_size=timeseries_batch_size)
    rnn_validation_data_output = rnn_validation_data_output .reshape((rnn_validation_data_output.shape[0], rnn_validation_data_output.shape[2]))
    #####################
    ## DENSE LAYER INPUT
    
    ## X value for Dense NN training
    dense_train_data_input = np.array(train_df[dense_input_columns ][dim2:] ) 
    ##the last missing column should be the first in the series, I can't be bothered with it
    dense_train_data_input = np.append(dense_train_data_input, train_df[dense_input_columns][0:1],axis=0 ) 
    dense_train_data_input  = GetTimeSeriesArrayFromArray(dense_train_data_input 
                                                       ,timepoints_per_day=1
                                                       ,timeseries_batch_size=timeseries_batch_size)
    
    ## shape: (number_of_windows, 1, num_of_coumns) => (number_of_windows, num_of_coumns)
    dense_train_data_input = dense_train_data_input.reshape((dense_train_data_input.shape[0], dense_train_data_input.shape[2]))
    
    ## X value for Dense NN validation
    dense_validation_data_input = np.array(validation_df[dense_input_columns][dim2:])
    ##the last missing column should be the first in the series, I can't be bothered with it
    dense_validation_data_input = np.append(dense_validation_data_input,validation_df[dense_input_columns][0:1],axis=0 )
    dense_validation_data_input = GetTimeSeriesArrayFromArray(dense_validation_data_input 
                                                       ,timepoints_per_day=1
                                                       ,timeseries_batch_size=timeseries_batch_size)
    
    ## shape: (number_of_windows, 1, num_of_coumns) => (number_of_windows, num_of_coumns)
    dense_validation_data_input = dense_validation_data_input.reshape( (dense_validation_data_input.shape[0],dense_validation_data_input.shape[2]) )
    
    
    ##NOTE: 
    ## Y value for Dense NN validation and training is 'rnn_train_data_input' and 'rnn_validation_data_output'
    
    
    ##########################
    ##%% create and compile the model
    multi_rnn = MultiInputRNN(RNN_input_shape= ( timeseries_sequence_length,len(rnn_input_columns ))
                              , Dense_input_shape= (len(dense_input_columns ))
                              , output_len=len(dense_target_columns )  
                          )
    print(multi_rnn.model.summary())
    
    
    ##########################
    ##%% train
    training_history = multi_rnn.model.fit( x=[rnn_train_data_input ,dense_train_data_input ]
                                            , y= rnn_train_data_output 
                                            , epochs= epochs
                                            , shuffle= False
                                            , validation_data= ( [ rnn_validation_data_input,dense_validation_data_input] , rnn_validation_data_output) 
                                            # , batch_size= batch_size
                                            )
        
    return multi_rnn ,training_history 

#%%
def GetMultiInputRNNEvaluation(multi_input_rnn_model,df = None,timeseries_batch_size= 32, timepoints_per_day= 24 ,batch_size=32):
    
    if df == None:
        df = GetDataFrameWithMaskForYear2022()
        ## NOTE: KEEP THE ''DATE TIME OCC' in its original state since we will pass it to the RNN without skipping it with a mask
        # df.loc[df['AVAILABLE MASK'] == 0] = 0 ## set the mask values to be 0
    

    rnn_input_columns = ['YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    rnn_target_columns = ['LAT','LON']
    dense_input_columns = ['YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    dense_target_columns = rnn_target_columns ## must be the same as target columns
    
    
    #####################
    ## RNN LAYER INPUT
    
    ## X value for RNN
    rnn_data_input = np.array(df[rnn_input_columns ]).copy()
    
    ## the dimensions of the input
    dim1_train_data =  int(rnn_data_input.size  / (timepoints_per_day*len(rnn_input_columns))) ##how many instances we have
    dim2 = timepoints_per_day     ##sequence length
    dim3 = len(rnn_input_columns) ##number of fields/columns
    
    rnn_data_input = GetTimeSeriesArrayFromArray(rnn_data_input 
                                                       ,timepoints_per_day=timepoints_per_day
                                                       ,timeseries_batch_size=timeseries_batch_size)
    
    ## Y value for RNN (same for the dense layer)
    rnn_data_output = np.array(df[rnn_target_columns ][dim2:]).copy()
    ##the last missing column should be the first in the series, I can't be bothered with it
    rnn_data_output = np.append(rnn_data_output , df[rnn_target_columns][0:1],axis=0 ) 
    
    rnn_data_output = GetTimeSeriesArrayFromArray(rnn_data_output 
                                                       ,timepoints_per_day=1
                                                       ,timeseries_batch_size=timeseries_batch_size)
    
    ## shape: (number_of_windows, 1, num_of_coumns) => (number_of_windows, num_of_coumns)
    rnn_data_output = rnn_data_output.reshape((rnn_data_output.shape[0], rnn_data_output.shape[2]))
    
    
    #####################
    ## DENSE LAYER INPUT
    
    ## X value for Dense NN training
    dense_data_input = np.array(df[dense_input_columns ][dim2:] ) 
    ##the last missing column should be the first in the series, I can't be bothered with it
    dense_data_input = np.append(dense_data_input, df[dense_input_columns][0:1],axis=0 ) 
    dense_data_input = GetTimeSeriesArrayFromArray(dense_data_input  
                                                       ,timepoints_per_day=1
                                                       ,timeseries_batch_size=timeseries_batch_size)
    
    ## shape: (number_of_windows, 1, num_of_coumns) => (number_of_windows, num_of_coumns)
    dense_data_input = dense_data_input.reshape((dense_data_input.shape[0], dense_data_input.shape[2]))
    
    
    #####################
    ## EVALUATE IT
    ##TODO:check if you can pass it
    evaluation_result = multi_input_rnn_model.evaluate(x=[rnn_data_input,dense_data_input]
                                                  ,y = rnn_data_output
                                                  ,batch_size= batch_size)
    return evaluation_result 


#%% 
def trainMultiInputRNN(save_model=False,save_history=False,epochs=10,batch_size=32,timeseries_batch_size= 32):
    multi_input_rnn, training_history = GetMultiInputRNNWithTrainingHistory(epochs= epochs
                                                                       ,batch_size= batch_size
                                                                       ,timeseries_batch_size= timeseries_batch_size)

    if (save_model):    
        multi_input_rnn.model.save("saved_models/MultiInputRNN.keras")
    if (save_history):
        training_history_df = pd.DataFrame(training_history.history)
        training_history_fname = 'saved_models/MultiInputRNN_history.csv'
        with open(training_history_fname, mode='w') as f:
            training_history_df.to_csv(f)
    
    return multi_input_rnn, training_history 

def evalueateMultiInputRNN(multi_input_rnn_model= None, save_history=False, timeseries_batch_size= 32, timepoints_per_day= 24 ,batch_size=32):
    
    if (multi_input_rnn_model== None):
        multi_input_rnn_model= keras.models.load_model('saved_models/MultiInputRNN.keras')    
    evaluation_result = GetMultiInputRNNEvaluation(multi_input_rnn_model
                                                   ,timeseries_batch_size= timeseries_batch_size
                                                   ,timepoints_per_day= timepoints_per_day
                                                   ,batch_size= batch_size)
    
    if (save_history):
        evaluation_result_dictionary = {'loss': evaluation_result[0], 'mse': evaluation_result[1] }
        evaluation_result_df = pd.DataFrame(data=evaluation_result_dictionary,dtype=float,index=[0])
        evaluation_result_fname = 'saved_models/MultiInputRNN_evaluation_result.csv'
        with open(evaluation_result_fname, mode='w') as f:
            evaluation_result_df.to_csv(f)
    
    return evaluation_result


#%% 
def testMultiInputRNN(save_model=False,save_history=False,epochs=100,batch_size=32,timeseries_batch_size= 32):
    multi_rnn, training_hitstory = trainMultiInputRNN(
        save_model=save_model
        ,save_history=save_history
        ,epochs=epochs
        ,batch_size=batch_size
        ,timeseries_batch_size= timeseries_batch_size
        )
    evaluation_result = evalueateMultiInputRNN(
        multi_rnn.model
        ,save_history= save_history
        ,timeseries_batch_size= timeseries_batch_size
        ,timepoints_per_day= 24 
        ,batch_size= batch_size
        )


#%% main function
if __name__ == '__main__':
    # import Eksplorativna_analiza
    # data = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
    multi_rnn, training_hitstory = trainMultiInputRNN(save_model=True
                                                      ,save_history=True
                                                      ,epochs=100
                                                      ,batch_size=32
                                                      ,timeseries_batch_size= 32)
    evaluation_result = evalueateMultiInputRNN(None##multi_rnn.model
                                               ,save_history=True
                                               ,timeseries_batch_size= 32
                                               ,timepoints_per_day= 24 
                                               ,batch_size=32)
