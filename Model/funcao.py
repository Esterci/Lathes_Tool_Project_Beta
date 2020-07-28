import os
import numpy as np

def my_function(Output_ID):
    
    add_path1_novo = "\\Input"
    base_path_novo = str(os.path.abspath(''))
    working_path_novo = base_path_novo + '\\Model'
    Input_path_novo = working_path_novo + add_path1_novo



    data = np.genfromtxt('{}\\Output_{}.csv'.format(Input_path_novo, Output_ID), delimiter=',')

    
    print('Metodo Funcao (data.shape): {}'.format(data.shape))



    print('----- FUNCAO -----')
    print('Base Path:')
    print(base_path_novo)

    print('Working Path:')
    print(working_path_novo)

    print('Input Path:')
    print(Input_path_novo)