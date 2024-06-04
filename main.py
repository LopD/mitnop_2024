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


#%% testing RNN 
def testRNNModels(plotTrainingHistory=False,save_model=False,save_history=False):
    RNN_epochs = 100
    RNN_batch_size = 32
    timeseries_batch_size = 32
    
    import simple_RNN_time_difference
    simple_RNN_time_difference.testSimpleRNNTimeDifference(
        save_model= save_model 
        ,save_history= save_history 
        ,epochs=RNN_epochs 
        ,batch_size=RNN_batch_size 
        ,timeseries_batch_size= 32
        )
    

    import MaskedRNN
    MaskedRNN.testSimpleMaskedRNN(
        save_model= save_model 
        ,save_history= save_history 
        ,epochs= RNN_epochs 
        ,batch_size= RNN_batch_size 
        ,timeseries_batch_size= timeseries_batch_size 
        )
    
    import MultiInputRNN
    MultiInputRNN.testMultiInputRNN(
        save_model= save_model 
        ,save_history= save_history 
        ,epochs= RNN_epochs 
        ,batch_size= RNN_batch_size 
        ,timeseries_batch_size= timeseries_batch_size 
        )
    
    if (plotTrainingHistory):
        import PlotRNNTrainingHistory
        PlotRNNTrainingHistory.plotRNNModelsLoss()


#%% Main funkcija - poziva zasebne fajlove od kojih je svaki vezan za jedan deo istraživanja
if __name__ == '__main__':
    testRNNModels(plotTrainingHistory=True,save_model=True,save_history=True)


