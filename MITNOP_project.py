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

import pandas as pd
import matplotlib.pyplot as plt

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

#%% Parsiranje datuma i vremena u skupu podataka

try:
    data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')
except ValueError:
    data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], errors='coerce')

data['YEAR OCC'] = data['DATE OCC'].dt.year
data['MONTH OCC'] = data['DATE OCC'].dt.month
data['HOUR OCC'] = data['TIME OCC'].apply(lambda x: int(str(x).zfill(4)[:2]))
data['QUARTER OCC'] = data['DATE OCC'].dt.to_period('Q')

#%% Identifikacija ključnih parametara za kreiranje istarživanja

# Provera broja različitih vrednosti za kolonu 'AREA'
area_unique_count = data['AREA'].nunique()
print("Broj različitih vrednosti u koloni 'AREA': ", data['AREA'].nunique())

# Provera broja različitih vrednosti za kolonu 'LOCATION'
location_unique_count = data['LOCATION'].nunique()
print(f"Broj različitih vrednosti u koloni 'LOCATION': {location_unique_count}")

# Provera broja različitih vrednosti za kolonu 'Rpt Dist No'
rpt_dist_unique_count = data['Rpt Dist No'].nunique()
print(f"Broj različitih vrednosti u koloni 'Rpt Dist No': {rpt_dist_unique_count}")

# Provera broja različitih vrednosti za kolonu 'Crm Cd'
crmcd_dist_unique_count = data['Crm Cd'].nunique()
print(f"Broj različitih vrednosti u koloni 'Crm Cd': {crmcd_dist_unique_count}")

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