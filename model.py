import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

df=pd.read_csv('airline_passenger_satisfaction.csv')
#dropping usuless coloumns
df.drop('Unnamed: 0',axis=1,inplace=True)

#preprocessing
df["arrival_delay_in_minutes"].fillna(df["arrival_delay_in_minutes"].median() , inplace =True)

#encoding and transforming
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['Gender']=encoder.fit_transform(df['Gender'])
df['customer_type']=encoder.fit_transform(df['customer_type'])
df['type_of_travel']=encoder.fit_transform(df['type_of_travel'])
df['customer_class']=encoder.fit_transform(df['customer_class'])
df['satisfaction']=encoder.fit_transform(df['satisfaction'])

#import relevant libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#features extraction

x = df[['Gender', 'customer_type','age', 'type_of_travel', 'customer_class',
       'flight_distance', 'inflight_wifi_service',
       'departure_arrival_time_convenient', 'ease_of_online_booking',
       'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort',
       'inflight_entertainment', 'onboard_service', 'leg_room_service',
       'baggage_handling', 'checkin_service', 'inflight_service',
       'cleanliness', 'departure_delay_in_minutes', 'arrival_delay_in_minutes']]
y = df['satisfaction']



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0,stratify=y)  #splitting data with test size of 25%
scaler = StandardScaler().fit(x_train)
logreg = LogisticRegression( max_iter=5000)   #build our logistic model
logreg.fit(x_train, y_train)  #fitting training data
y_pred  = logreg.predict(x_test)    #testing model’s performance
print("Accuracy={:.2f}".format(logreg.score(x_test, y_test)))

pickle.dump(logreg,open('model.pkl','wb'))

