
#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% testing:
#df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')
# print(df.head(10))
# print(df.index)
# columns_to_keep = ['DR_NO'
#                    ,'Date Rptd'
#                    ,'DATE OCC'
#                    ,'TIME OCC' #time of crime occurrence
#                    ,'AREA' #area ID
#                    ,'LAT' #lattitude
#                    ,'LON' #longitiude
#                    ]
# df = df[columns_to_keep]


# test_df['LAT'].scatter()
# print(test_df.columns)
# plt.scatter(test_df.index, test_df['LAT'])
# plt.show()



# start = dt.datetime.now()
# sim_df['date'] = start + sim_df['cum_days'].map(dt.timedelta)

#%% load data frame

df = pd.read_csv('Crime_Data_from_2020_to_Present.csv'
                 )

#%% dropping columns
#NOTE: none of the date fields had any description so I'm not sure what they represent
columns_to_keep = ['DR_NO'
                   ,'Date Rptd'#date of report? 
                   ,'DATE OCC' #date of occurence?
                   ,'TIME OCC' #time of crime occurrence military time as int64
                               #example: 1800 is 18:00
                   ,'AREA' #area ID
                   ,'LAT' #lattitude
                   ,'LON' #longitiude
                   ]

print(df.dtypes)

# df.index = pd.to_datetime(df['TIME OCC'], format='$')


#%% divide into train and test
train_df , test_df = train_test_split(df, test_size=0.2)

#%% functions
def SelectWhereLATIsNullOrLONIsNull(df):
    df_lat_lon_missing_values = df[df['LAT'] == df['LON'] ]
    df_lat_lon_missing_values = df_lat_lon_isNULL_mask[df_lat_lon_isNULL_mask['LAT'] == 0]
    return df_lat_lon_missing_values 

def GetExactTimeOfCrimeOccurrence():
    df_TIME_Hours =  (test_df['TIME OCC'] / 100).astype(np.int64)
    # print('hours=',df_TIME_Hours, df_TIME_Hours.dtypes)
    df_TIME_Minutes = ((test_df['TIME OCC'] %60)).astype(np.int64) 
    # print('minutes=',df_TIME_Minutes, df_TIME_Minutes.dtypes)
    
    df_TimeStamp = pd.to_datetime(test_df['DATE OCC'], format='%m/%d/%Y %H:%M:%S AM') 
    # df_TimeDelta = pd.Series(pd.to_timedelta(np.arange(5), unit="d"))
    df_TimeDelta = pd.Series(pd.to_timedelta(df_TIME_Hours,unit='hours'))
    df_TimeDelta += pd.Series(pd.to_timedelta(df_TIME_Minutes,unit='minutes'))
    df_TimeDelta += df_TimeStamp 
    
    # print(df_TimeDelta) 
    return df_TimeDelta
