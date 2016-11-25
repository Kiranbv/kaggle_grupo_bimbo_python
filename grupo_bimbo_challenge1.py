# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:59:32 2016

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


# Here data is read in chunks and processed. The output from each chunk is then averaged. 
# first read testing data and other files.
testing_data_full = pandas.read_csv('test.csv')
testing_data_copy = testing_data_full.copy(deep=True)
test_ids = testing_data_copy['id'].values
size_testing = testing_data_full.shape
output_model = numpy.zeros(size_testing[0])

# reading the client and product table data.
client_data = pandas.read_csv('cliente_tabla.csv')
product_data = pandas.read_csv('producto_tabla.csv')
townstate_data = pandas.read_csv('town_state.csv')
count = 0;

reader= pandas.read_csv('train.csv',chunksize=10000000)
for training_data_full in reader:
    type(training_data_full)
    training_data_copy = training_data_full.copy(deep=True)
      
    # column names in training and testing
    size_training = training_data_full.shape
    
    
    # getting the training and testing data sets
    newdf_tr = pandas.DataFrame(index=range(0,size_training[0]),columns=['NombreCliente','NombreProducto','Town','State'], dtype='str')
    newdf_te = pandas.DataFrame(index=range(0,size_testing[0]),columns=['NombreCliente','NombreProducto','Town','State'], dtype='str')
    
    target = training_data_copy['Demanda_uni_equil'].values
    training_data = training_data_copy.drop(['Demanda_uni_equil','Venta_hoy','Venta_uni_hoy','Dev_uni_proxima','Dev_proxima'],axis=1).values
    testing_data = testing_data_copy.drop(['id'],axis=1).values
    size_tr = training_data.shape
    size_te = testing_data.shape
    
    
    # first simple learning system without using client,product or city data
    from sklearn import linear_model
    ## lasso model
    #lasso_model = linear_model.Lasso(alpha=.1) 
#    #lasso_model.fit(training_data,target)
#    #output_model = lasso_model.predict(testing_data)
#    
#    # ridge model
#    ridge_model = linear_model.Ridge(alpha=1) 
#    ridge_model.fit(training_data,target)
#    output_model1 = ridge_model.predict(testing_data)
    
#    # gradient boosting regressor
    from sklearn.ensemble import GradientBoostingRegressor
    grad_boosting_model =  GradientBoostingRegressor(loss='ls',
                                    n_estimators=100, max_depth=5,
                                    learning_rate=.3,subsample=1)
    grad_boosting_model.fit(training_data,target)
    output_model1 = grad_boosting_model.predict(testing_data)
    
    
     # implementing generalized linear model
#    import statsmodels.api as sm
#     
#    gamma_model = sm.GLM(target,training_data,family=sm.families.Gamma())
#    gamma_results = gamma_model.fit()
#    output_model1 = gamma_model.predict(testing_data)

    
    
    
    
    output_model = output_model+output_model1
    count = count+1
    print(count)

output = {'id':test_ids.astype(int),
          
          'Demanda_uni_equil':output_model/count
          }
outputdf = pandas.DataFrame(output)
outputdf = outputdf[['id','Demanda_uni_equil']]
siz = outputdf.shape
outputdf.ix[outputdf.Demanda_uni_equil<=0,'Demanda_uni_equil']=0.1

# Write to a csv file
outputdf.to_csv('grupo_bimbo_submission.csv',sep = ',',index=False)