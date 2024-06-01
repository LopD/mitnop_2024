#%% Importi

import Eksplorativna_analiza

#%% Main funkcija - poziva zasebne fajlove od kojih je svaki vezan za jedan deo istraživanja
if __name__ == '__main__':
    
    ## Dobavljamo df uz izvršavanje eksplorativne analize - df je dataframe koji koristimo za obuku mreza
    df = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
    
    ## Kreiramo heat map-u koja prikazuje raspodelu zločina na teritoriji LA    
    Eksplorativna_analiza.generisi_heat_mapu(df)
    