import Model_Functions as mf
import numpy as np
import pandas as pd
from time import sleep

print('.: Welcome to our Autonomous Model Users\' Interface (AMUI)! :.' )
recovery_decision = input('\nWould you like to recover the previous analyses\' variables?(y/n) ')

if recovery_decision == 'y':

    #Recovering Data from previous analyse

    print('Recovery Control Output')
    print('----------------------------------')

    D_S_parameters =  mf.Recovery('D_S_parameters') 
    ExtractedNames =  mf.Recovery('ExtractedNames') 
    SelectedFeatures =  mf.Recovery('SelectedFeatures') 
    ReducedFeatures =  mf.Recovery('ReducedFeatures') 
    SODA_parameters, processing_parameters =  mf.Recovery ('SODA_parameters_processing_parameters') 
    ClassificationPar =  mf.Recovery('ClassificationPar')
    
    Output_ID = int(D_S_parameters['ID'])

    print('The current Data ID is ', Output_ID)
    
    print('__________________________________________')
    print('')
    sleep(2)
    


elif recovery_decision == 'n':

    print('')
    print('__________________________________________')

    k = 0

    Output_ID = input('\nWhich is the ID number you want to analyse?\n\nType the ID number and press ENTER: ')

    while k ==0:
        
        try:
            
            np.genfromtxt('/home/thiago/Repositories/Lathes_Tool_Project/Model/Input/Output_' + Output_ID + '.csv', delimiter = ',')
            
            k +=1
            print('__________________________________________')
        except:
            
            print('__________________________________________')
            print('\a\nCould not find this ID number, Try again or press Ctrl + C to quit.')
            Output_ID = input('\nType the ID number and press ENTER: ')

else: 

    print('Unavalable command, try again ...')
    exit()

#Main Code

mode_decision = input('Choose the analyse mode from the list below:\n\n1. Interative Mode (Not done yet, keep up with new releases.)\n2. Direct Mode\nType a number and press ENTER: ')

if mode_decision == '1':

    print('Not done yet, keep up with new releases.')

if mode_decision == '2':

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
