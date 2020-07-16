import Model_Functions as mf
import numpy as np
import pandas as pd

#Recovering Data from previous analyse

print('Recovery Control Output')
print('----------------------------------')

D_S_parameters =  mf.Recovery('D_S_parameters') 
ExtractedNames =  mf.Recovery('ExtractedNames') 
SelectedFeatures =  mf.Recovery('SelectedFeatures') 
ReducedFeatures =  mf.Recovery('ReducedFeatures') 
SODA_parameters, processing_parameters =  mf.Recovery ('SODA_parameters_processing_parameters') 
ClassificationPar =  mf.Recovery('ClassificationPar')

print("")

#Main Code

Output_ID = 3

D_S_parameters = mf.DataSlicer(Output_ID,20,'Main Data')

ExtractedNames = mf.TSFRESH_Extraction(D_S_parameters) #(Extração de atributos)

SelectedFeatures = mf.TSFRESH_Selection(D_S_parameters,ExtractedNames) # (Parametros e resultados da divisão de dados)

ReducedFeatures = mf.PCA_calc(SelectedFeatures,3,'Calc') # (Feautures selecionadas, numero de PC's a manter, mode ('Test','Calc','Specific', 'Analytics'))

SODA_parameters, processing_parameters = mf.SODA(ReducedFeatures,2,5,0.25) # (Features reduzidas, granularidade mínima, granularidade máxima, passo)

ClassificationPar = mf.GroupingAlgorithm(SODA_parameters,80,20, processing_parameters) # (Labels do SODA, Porcentagem de definição, numero de ID's boas, parametros de processamento)

mf.Classification (ClassificationPar, 2,5, 1, plot_matrix=False) #(Parametros do data-set,  min_grid_size, max_grid_size,plotar matriz de confusão (True or False))
"""
ModelPar = mf.Model_Train(ClassificationPar,'euclidean',"Neural Net",2.75) #(Parametros da data-set, distância, Modelo, granularidade)

mf.Model_Predict(ModelPar)
"""
