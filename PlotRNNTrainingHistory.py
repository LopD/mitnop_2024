# import Eksplorativna_analiza
# tmp_df = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()

#%%
import pandas as pd
import matplotlib.pyplot as plt
# from Neuronska_mreza import  prikazi_istoriju_ucenja

#%%
def plotLossHistory(model_training_history,model_name= ""):
    plt.plot(model_training_history['loss'])
    plt.title('trening \''+model_name+ "\' modela")
    plt.ylabel('preciznost (gubitak detalja)')
    plt.xlabel('epohe')
    plt.show()

def plotMultiInputRNN():
    history = None
    try:
        history = pd.read_csv('saved_models/MultiInputRNN_history.csv')
        plotLossHistory(history, model_name='Multi input RNN')
    except:
        print('\'saved_models/MultiInputRNN_history.csv\' was not found!')


def plotMaskedRNN():
    history = None
    try:
        history = pd.read_csv('saved_models/SimpleMaskedRNN_history.csv')
        plotLossHistory(history, model_name='Masked RNN')
    except:
        print('\'saved_models/SimpleMaskedRNN_history.csv\' was not found!')

def plotSimpleRNN():
    history = None
    try:
        history = pd.read_csv('saved_models/SimpleRNN_history.csv')
        plotLossHistory(history, model_name='Simple time difference RNN')
    except:
        print('\'saved_models/SimpleRNN_history.csv\' was not found!')


def plotRNNModelsLoss():
    plotSimpleRNN()
    plotMaskedRNN()
    plotMultiInputRNN()
    
    
#%%
if __name__ == '__main__':
    plotRNNModelsLoss()