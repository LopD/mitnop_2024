#%% MITNOP_project
# Name: Prediction of crimes

# Common tasks:
# • Data filtering and preprocessing 
# • Intercomparison of neural networks (depending on which 
#   we received as a task) and temporal data model

# Savo Savić:
# 1. Exploratory analysis and creation of a temporal data model
# 2. Creation of a multi-layer neural network
# 3. Visualization of the temporal data model

#%% Importi

import os
import pandas as pd
import matplotlib.pyplot as plt



def izvrsi_eksplorativnu_analizu(plot=False):
    '''Vraca dataframe sa kolonama koje ce se koristiti pri obuci neuronskih mreza'''    
#%% Čitanje podataka iz .CSV fajla
    
    df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')
    
    print("Prvih nekoliko redova skupa podataka:")
    print(df.head())
    
    print("Osnovna statistika numeričkih kolona:")
    print(df.describe())
    
    print("Osnovne informacije o skupu podataka:")
    print(df.info())
    
#%% Provera postojanja nedostajućih vrednosti u skupu
    
    missing_values = df.isna().sum()
    
    print(missing_values)
    
#%% Uklanjanje uočenih nedostajućih vrednosti
    
    columns_for_drop = ['Mocodes', 'Vict Sex', 'Vict Descent',
                        'Premis Cd', 'Premis Desc', 'Weapon Used Cd',
                        'Weapon Desc', 'Crm Cd 1', 'Crm Cd 2',
                        'Crm Cd 3', 'Crm Cd 4', 'Cross Street']
    
    data = df.drop(columns = columns_for_drop)
    
    missing_values = data.isna().sum()
    
    print(missing_values)
    
    print(df)
    
#%% Brisanje redova gdje su koordinate ne postojece 
    df = df.drop(df.loc[ (df['LAT'] == df['LON'])  & (df['LAT'] == 0) ].index)

    print(df)
    
#%% Parsiranje datuma i vremena u skupu podataka
    
    try:
        data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')
    except ValueError:
        data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], errors='coerce')
    
    data['YEAR OCC'] = data['DATE OCC'].dt.year
    data['MONTH OCC'] = data['DATE OCC'].dt.month
    data['DAY OCC'] = data['DATE OCC'].dt.day
    data['HOUR OCC'] = data['TIME OCC'].apply(lambda x: int(str(x).zfill(4)[:2]))
    data['QUARTER OCC'] = data['DATE OCC'].dt.to_period('Q')
    
#%% Identifikacija ključnih parametara za kreiranje istarživanja

    # Provera broja različitih vrednosti za kolonu 'AREA'
    area_unique_count = data['AREA'].nunique()
    print(f"Broj različitih vrednosti u koloni 'AREA': {area_unique_count}")
    
    # Provera broja različitih vrednosti za kolonu 'LOCATION'
    location_unique_count = data['LOCATION'].nunique()
    print(f"Broj različitih vrednosti u koloni 'LOCATION': {location_unique_count}")
    
    # Provera broja različitih vrednosti za kolonu 'Rpt Dist No'
    rpt_dist_unique_count = data['Rpt Dist No'].nunique()
    print(f"Broj različitih vrednosti u koloni 'Rpt Dist No': {rpt_dist_unique_count}")
    
    # Provera broja različitih vrednosti za kolonu 'Crm Cd'
    crmcd_dist_unique_count = data['Crm Cd'].nunique()
    print(f"Broj različitih vrednosti u koloni 'Crm Cd': {crmcd_dist_unique_count}")
    
    if (plot):
#%% Distribucija zločina prema tipu zločina #1
        
        # Provera broja različitih vrednosti za kolonu 'Crm Cd Desc'
        crime_type_counts = df['Crm Cd Desc'].value_counts()
        print(crime_type_counts.head(15))
        
        plt.figure(figsize=(15, 8))
        crime_type_counts.plot(kind='bar')
        plt.title('Distribucija zločina prema tipu zločina')
        plt.xlabel('Tip krivičnog dela')
        plt.ylabel('Broj krivičnih dela')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
#%% Distribucija zločina prema tipu zločina #2
        
        # Nakon prikazivanja podataka uočeno je da je spektar tipova 
        # zločina u odnosu na frekvenciju njihovog desavanja u gradu 
        # tolike teritorije preveliki pa je posmatranje distribucije
        # zločina prema tipu bazirano na one bitne i najučestalije
        
        # Izaberite nekoliko najčešćih vrsta krivičnih dela za plotovanje
        top_crime_types = crime_type_counts.head(20)
        
        plt.figure(figsize=(15, 10))
        top_crime_types.plot(kind='bar')
        plt.title('Distribucija zločina prema tipu zločina')
        plt.xlabel('Tip krivičnog dela')
        plt.ylabel('Broj krivičnih dela')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
#%% Distribucija zločina prema oblasti u kojoj su se desili
        
        plt.figure(figsize=(12, 8))
        data['AREA NAME'].value_counts().plot(kind='bar')
        plt.title('Distribucija zločina prema oblasti u kojoj su se desili')
        plt.xlabel('Oblast dešavanja')
        plt.ylabel('Broj krivičnih dela')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
#%% Distribucija zločina prema godini dešavanja
        
        plt.figure(figsize=(10, 6))
        data['YEAR OCC'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribucija zločina prema godini dešavanja')
        plt.xlabel('Godina dešavanja')
        plt.ylabel('Broj krivičnih dela')
        plt.show()
        
#%% Distribucija zločina prema mesecu dešavanja
        
        plt.figure(figsize=(10, 6))
        data['MONTH OCC'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribucija zločina prema mesecu dešavanja')
        plt.xlabel('Mesec dešavanja')
        plt.ylabel('Broj krivičnih dela')
        plt.show()
        
#%% Distribucija zločina prema vremena dešavanja u toku dana
    
        plt.figure(figsize=(10, 6))
        hour_counts = data['HOUR OCC'].value_counts().sort_index()
        hour_labels = [f"{hour:02}:00" for hour in hour_counts.index]
        hour_counts.plot(kind='bar')
        plt.title('Distribucija zločina prema vremena dešavanja u toku dana')
        plt.xlabel('Sat dešavanja')
        plt.ylabel('Broj krivičnih dela')
        plt.xticks(range(len(hour_labels)), hour_labels, rotation=45)
        plt.tight_layout()
        plt.show()
    
    return data

# %% Dodatna obrada skupa podataka i generisanje heat mape

def filter_top_crimes(df, top_n=20):
    '''Filters the dataframe to only include the top N crime types by frequency'''
    top_crime_types = df['Crm Cd Desc'].value_counts().head(top_n).index
    return df[df['Crm Cd Desc'].isin(top_crime_types)]

def generisi_heat_mapu(df, output_file='heatmap_of_LA.html'):
    '''Generates a heatmap of crimes with customized parameters and saves it to an HTML file.'''
    
    import folium as folium
    from folium.plugins import MarkerCluster, HeatMap
    
    df = filter_top_crimes(df)
    
    # Kreiranje baze za toplotnu mapu
    # Postavljanjem koordinata na ovu vrednost centriramo mapu na prikaz Los Angeles-a
    mapa = folium.Map(location=[34.0522, -118.2437], zoom_start=10)
    
    # Pripremanje podataka za prikaz - bazirano na geog. dužini i širini
    heat_data = [[row['LAT'], row['LON']] for index, row in df.iterrows()]

    # Definisan veci spektar boja kako bi se na krupnom planu
    # podaci iole normalno prikazali
    gradient = {
        0.2: 'blue',
        0.4: 'lime',
        0.6: 'yellow',
        0.8: 'orange',
        1.0: 'red'
    }

    # Dodaje se sloj za prikaz na mapi sa custom podesavanjima radi boljeg prikaza
    HeatMap(heat_data, radius=10, blur=5, max_zoom=1, min_opacity=0.3, gradient=gradient).add_to(mapa)
    
    # Dodavanje markera za obeležavanje oblasti grada
    area_groups = df.groupby(['AREA', 'AREA NAME']).first().reset_index()
    for _, row in area_groups.iterrows():
        folium.Marker(
            location=[row['LAT'], row['LON']],
            popup=f"Area {row['AREA']}: {row['AREA NAME']}",
            icon=folium.Icon(color='blue')
            ).add_to(mapa)
    
    # Definise se apsolutna putanja koju koristimo za čuvanje mape
    absolute_path = os.path.join(os.getcwd(), output_file)
    
    # Čivanje toplotne mape u obliku HTML fajla
    mapa.save(absolute_path)
    print(f"Heat map saved to {absolute_path}")
    