#%% Importi

import Eksplorativna_analiza as ea
import Neuronska_mreza as nm
from tensorflow.keras.models import save_model
from Neuronska_mreza import prikazi_istoriju_ucenja

#%%
def testiraj_single_layer_NN(epochs=10):
    import single_layer_NN
    singleLayerNN,singleLayerNN_training_history = single_layer_NN.GetSimple1LayerNNWithTrainingHistory(epochs=epochs)
    prikazi_istoriju_ucenja(singleLayerNN_training_history )

#%%
def testiraj_simple_RNN_time_difference(epochs=10):
    import simple_RNN_time_difference
    rnn,training_history = simple_RNN_time_difference.GetSimpleRNNWithTrainingHistory(epochs=epochs)
    prikazi_istoriju_ucenja(training_history )

#%%
def testiraj_MaskedRNN(epochs=10):
    import MaskedRNN
    rnn_masked, training_history = MaskedRNN.GetSimpleMaskedRNNWithTrainingHistory(epochs=epochs)
    prikazi_istoriju_ucenja(training_history )

#%%
def testiraj_MultiInputRNN(epochs=10):
    import MultiInputRNN
    multi_rnn, training_hitstory = MultiInputRNN.GetSimpleMaskedRNNWithTrainingHistory(epochs=epochs)
    prikazi_istoriju_ucenja(training_hitstory  )

#%%
def testiraj_Neuronsku_mrezu(epochs=10,predict=False,save=False):
    file_name = 'Crime_Data_from_2020_to_Present.csv'
    data = ea.izvrsi_eksplorativnu_analizu(file_name=file_name)

    # print(data)

    # Generisanje heat mape
    #TODO: vrati ovo
    # output_file_name = 'heatmap_of_LA.html'
    # ea.generisi_heat_mapu(data=data, top_n=20, output_file=output_file_name)
## %%    
    # Priprema podataka
    pripremljeni_podaci = nm.priprema_podataka(data)
## %% 
    # Kreiranje i treniranje modela
    model, scaler_X, scaler_y, X_test, y_test, history = nm.kreiraj_i_treniraj_model(pripremljeni_podaci,epochs=epochs)
## %%
    # Čuvanje modela
    if save:
        save_model(model, 'crime_prediction_model.h5')  
## %% 
    # Predikcija i evaluacija
    y_pred_rescaled = None 
    y_test_rescaled = None
    if predict:
        y_pred_rescaled, y_test_rescaled = nm.predikcija(model, scaler_y, X_test, y_test)
##%%  prikaz preciznosti kroz ucenje
    prikazi_istoriju_ucenja(history)
    

#%% Main funkcija - poziva zasebne fajlove od kojih je svaki vezan za jedan deo istraživanja
if __name__ == '__main__':
    broj_epoha_za_guste_NN = 3
    broj_epoha_za_RNN = 3
    broj_epoha_za_vremenskeserije_RNN = 3
    
##%% prikazuje i testira viseslojnu neuronsku mrezu 
    testiraj_Neuronsku_mrezu(epochs=broj_epoha_za_guste_NN )

##%% prikaz preciznost kroz ucenje 1slojne
    testiraj_single_layer_NN(epochs=broj_epoha_za_guste_NN )
    
##%%
    testiraj_simple_RNN_time_difference(epochs=broj_epoha_za_RNN )

##%%
    testiraj_MaskedRNN(epochs=broj_epoha_za_vremenskeserije_RNN  )
    
    testiraj_MultiInputRNN(epochs=100)

