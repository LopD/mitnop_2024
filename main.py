#%% Importi

import Eksplorativna_analiza as ea
import Neuronska_mreza as nm
from tensorflow.keras.models import save_model
#%% Main funkcija - poziva zasebne fajlove od kojih je svaki vezan za jedan deo istraživanja
if __name__ == '__main__':
    file_name = 'Crime_Data_from_2020_to_Present.csv'
    data = ea.izvrsi_eksplorativnu_analizu(file_name=file_name)

    print(data)

    # Generisanje heat mape
    output_file_name = 'heatmap_of_LA.html'
    ea.generisi_heat_mapu(data=data, top_n=20, output_file=output_file_name)
# %%    
    # Priprema podataka
    pripremljeni_podaci = nm.priprema_podataka(data)
# %% 
    # Kreiranje i treniranje modela
    model, scaler_X, scaler_y, X_test, y_test, history = nm.kreiraj_i_treniraj_model(pripremljeni_podaci)
# %%
    # Čuvanje modela
    save_model(model, 'crime_prediction_model.h5')
# %% 
    # Predikcija i evaluacija
    y_pred_rescaled, y_test_rescaled = nm.predikcija(model, scaler_y, X_test, y_test)