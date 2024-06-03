#%% Importi

import Eksplorativna_analiza as ea
import Neuronska_mreza as nm
import folium
from folium.plugins import HeatMap
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


#%% Main funkcija - poziva zasebne fajlove od kojih je svaki vezan za jedan deo istraživanja
def testiraj_Neuronsku_mrezu(epochs=10):

    data = ea.izvrsi_eksplorativnu_analizu()
##%% 
    generis_heat_mapu = False
    # Generisanje heat mape
    if generis_heat_mapu:
        output_file_name = 'heatmap_of_LA.html'
        ea.generisi_heat_mapu(data=data, top_n=20, output_file=output_file_name)
# <<<<<<< HEAD
##%%  
    # Priprema podataka
    pripremljeni_podaci = nm.priprema_podataka(data)
##%%     
    # Kreiranje i treniranje modela
    model, scaler_X, scaler_y, X_test, y_test, history = nm.kreiraj_i_treniraj_model(pripremljeni_podaci,epochs=epochs)
##%%    
    # Čuvanje modela
    save_model(model, 'crime_prediction_model.keras')
##%%    
    # Predikcija i evaluacija
    y_pred_rescaled, y_test_rescaled = nm.predikcija(model, scaler_y, X_test, y_test)
    
    # Kreiranje mape za prikaz predviđenih zločina
    m = folium.Map([34.0522, -118.2437], zoom_start=12)  # Koordinate Los Angelesa

    # Dodavanje sloja koji prikazuje celu površinu Los Angelesa
    folium.Choropleth(
        geo_data='los-angeles.geojson',  # GeoJSON koji sadrži granice Los Angelesa
        fill_opacity=0.1,
        line_opacity=0.3,
    ).add_to(m)

    # Pretvaranje predviđenih koordinata u format pogodan za HeatMap
    predikcije = []
    for pred in y_pred_rescaled:
        predikcije.append([pred[0], pred[1]])

    # Generisanje HeatMap za predviđene zločine
    heat_map = HeatMap(predikcije, radius=10, blur=15, max_zoom=1, min_opacity=0.5, name='Predicted Crimes')
    heat_map.add_to(m)

    # Čuvanje mape kao HTML
    m.save('predicted_crime_locations.html')
# =======
## %%    
#     # Priprema podataka
#     pripremljeni_podaci = nm.priprema_podataka(data)
# ## %% 
#     # Kreiranje i treniranje modela
#     model, scaler_X, scaler_y, X_test, y_test, history = nm.kreiraj_i_treniraj_model(pripremljeni_podaci,epochs=epochs)
# ## %%
#     # Čuvanje modela
#     if save:
#         save_model(model, 'crime_prediction_model.h5')  
# ## %% 
#     # Predikcija i evaluacija
#     y_pred_rescaled = None 
#     y_test_rescaled = None
#     if predict:
#         y_pred_rescaled, y_test_rescaled = nm.predikcija(model, scaler_y, X_test, y_test)
# ##%%  prikaz preciznosti kroz ucenje
    prikazi_istoriju_ucenja(history)
    

#%% Main funkcija - poziva zasebne fajlove od kojih je svaki vezan za jedan deo istraživanja
if __name__ == '__main__':
    broj_epoha_za_guste_NN = 3
    broj_epoha_za_RNN = 3
    broj_epoha_za_vremenskeserije_RNN = 3
    
##%% prikazuje i testira viseslojnu neuronsku mrezu 
    # testiraj_Neuronsku_mrezu(epochs=broj_epoha_za_guste_NN )

##%% prikaz preciznost kroz ucenje 1slojne
    testiraj_single_layer_NN(epochs=broj_epoha_za_guste_NN )
    
##%%
    # testiraj_simple_RNN_time_difference(epochs=broj_epoha_za_RNN )

##%%
    # testiraj_MaskedRNN(epochs=broj_epoha_za_vremenskeserije_RNN  )
    
    # testiraj_MultiInputRNN(epochs=100)

