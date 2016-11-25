# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:33:57 2016

@author: kiranbv
"""

## This program implements the Grupo Bimbo Inventory Demand challenge in
## kaggle

# Importing important modules
import os
import numpy
import scipy
import pandas
import sklearn

# Change working directory
wd_present = os.getcwd()
os.chdir('c:\\grupo_bimbo_challenge')


# changing data types of training and testing
train_types = {'Agencia_ID':numpy.uint16, 'Ruta_SAK':numpy.uint16, 'Cliente_ID':numpy.uint32, 
               'Producto_ID':numpy.uint16, 'Demanda_uni_equil':numpy.uint32}

test_types = {'Agencia_ID':numpy.uint16, 'Ruta_SAK':numpy.uint16, 'Cliente_ID':numpy.uint32, 
              'Producto_ID':numpy.uint16, 'id':numpy.uint32}


# -----------------------------------------------------------------------------
# Getting the data
# Read the training and testing files
training_data_full = pandas.read_csv('train.csv',usecols=train_types.keys(), dtype=train_types)
testing_data_full = pandas.read_csv('test.csv',usecols=test_types.keys(), dtype=test_types)
#training_data_copy = training_data_full1.copy(deep=True)
#testing_data_copy = testing_data_full.copy(deep=True)


# column names in training and testing
training_data_full.columns
testing_data_full.columns
size_training = training_data_full.shape
size_testing = testing_data_full.shape



# reading the client and product table data
client_data = pandas.read_csv('cliente_tabla.csv')
product_data = pandas.read_csv('producto_tabla.csv')
townstate_data = pandas.read_csv('town_state.csv')

# getting the training and testing data sets
#newdf_tr = pandas.DataFrame(index=range(0,size_training[0]),columns=['NombreCliente','NombreProducto','Town','State'], dtype='str')
#newdf_te = pandas.DataFrame(index=range(0,size_testing[0]),columns=['NombreCliente','NombreProducto','Town','State'], dtype='str')

test_ids = testing_data_full['id'].values
target = training_data_full['Demanda_uni_equil'].values
training_data = training_data_full.drop(['Demanda_uni_equil'],axis=1).values

testing_data = testing_data_full.drop(['id'],axis=1).values
size_tr = training_data.shape
size_te = testing_data.shape


# first simple learning system without using client,product or city data
from sklearn import linear_model
## lasso model



# ridge model
ridge_model = linear_model.Ridge(alpha=1) 
ridge_model.fit(training_data,target)
output_model = ridge_model.predict(testing_data)


# svm regression
#from sklearn import svm
#svm_model = svm.SVR(kernel='linear',C=0.8, gamma=0.1)
#svm_model.fit(training_data,target)
#output_model = svm_model.predict(testing_data)


# random forest regressor
from sklearn.ensemble.forest import RandomForestRegressor
random_forest_model = RandomForestRegressor()
random_forest_model.fit(training_data,target)
output_model = random_forest_model.predict(testing_data)


# gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
grad_boosting_model =  GradientBoostingRegressor(loss='ls',
                                n_estimators=100, max_depth=5,
                                learning_rate=.3,subsample=1)
grad_boosting_model.fit(training_data,target)
output_model = grad_boosting_model.predict(testing_data)



# xgboost regressor
#import xgboost
#xgboost_model = xgboost.XGBRegressor(missing=numpy.nan, max_depth=3, n_estimators=50, learning_rate=0.03, nthread=4, seed=4242)
#xgboost_model.fit(training_data,target)
#output_model = xgboost_model.predict(testing_data)





# writing to output file
output = {'id':test_ids.astype(int),
          
          'Demanda_uni_equil':output_model
          }
outputdf = pandas.DataFrame(output)
outputdf = outputdf[['id','Demanda_uni_equil']]
siz = outputdf.shape
outputdf.ix[outputdf.Demanda_uni_equil<=0,'Demanda_uni_equil']=0.1

# Write to a csv file
outputdf.to_csv('grupo_bimbo_submission.csv',sep = ',',index=False)
