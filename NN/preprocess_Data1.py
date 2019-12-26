import numpy as np
import pandas as pd

data = pd.read_csv('FinalFlight.csv')

#print(admit)
# Make dummy variables for rank

#data = pd.concat([admit, pd.get_dummies(admit['DISTANCE'], prefix='DISTANCE')], axis=1)
# YO chahe halera garnu paryoo data= admit raknu paryoo distance laii dummy banayo vanne dheraii column badxaaa ani tala ko drop garnu parennaa

#print(data)
#data = data.drop('DISTANCE', axis=1)

#print(data)

# Standarize features
#data = admit 

for field in ['DEP_TIME' ,'SurfaceTemperatureFahrenheit','SurfaceDewpointTemperatureFahrenheit','WindChillTemperatureFahrenheit','WindSpeedMph','WindDirectionDegrees','RelativeHumidityPercent','SurfaceAirPressureMillibars','DAY_OF_MONTH','OP_CARRIER_AIRLINE_ID','CloudCoveragePercent','MONTH','ORIGIN_AIRPORT_ID','inputLongitude','inputLatitude']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:, field] = (data[field] - mean) / std  # Computing the Z score

#print(data)
# Split off random 10% of the data for testing

np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data, test_data = data.iloc[sample], data.drop(sample)


# Split into features and targets

features, targets = data.drop('cat_response', axis=1), data['cat_response']
features_test, targets_test = test_data.drop('cat_response', axis=1), test_data['cat_response']

#print(data['admit'])
#print(features)
#print(features_test)
print(np.shape(features))