import Model_Functions as mf
import numpy as np
import pandas as pd

print('___________________________________________')
print('\n.:PCA preview:.')

preview_decision = 'n' 

while preview_decision == 'n':

    pc_num = int(input('\nHow many PCs you would like to use?\n\nType a number and press ENTER: '))

    #ReducedFeatures = mf.PCA_calc(SelectedFeatures,pc_num,'Test') # (Feautures selecionadas, numero de PC's a manter, mode ('Test','Calc','Specific', 'Analytics'))
    preview_decision = input('\nWould you like to proceed with this number of PCs ?(y/n) ').strip()

print('__________________________________________')
print('\n.:PCA options:.\n')

pca_mode = int(input('\nChoose the PCA mode from the list below:\n\n1. Simple calcule\n2. Partial PC Analyses\n3. Complete PC Analyses\n\nType a number and press ENTER: '))
            
loop = 0

    #while loop == 0:

if pca_mode == 1:

    #ReducedFeatures = mf.PCA_calc(SelectedFeatures,pc_num,'Calc') # (Feautures selecionadas, numero de PC's a manter, mode ('Test','Calc','Specific', 'Analytics'))
    loop = 1
    print(loop)

elif pca_mode == 2:

    #ReducedFeatures = mf.PCA_calc(SelectedFeatures,pc_num,'Analytics') # (Feautures selecionadas, numero de PC's a manter, mode ('Test','Calc','Specific', 'Analytics'))
    loop = 2
    print(loop)

elif pca_mode == 3:

    ReducedFeatures = mf.PCA_calc(SelectedFeatures,pc_num,'Specific') # (Feautures selecionadas, numero de PC's a manter, mode ('Test','Calc','Specific', 'Analytics'))
    loop = 3
    print(loop)

else:

    print('\nWrong entry, try again...')

    pca_mode = input('\nChoose the PCA mode from the list below:\n\n1. Simple calcule\n2. Partial PC Analyses\n3. Complete PC Analyses\n\nType a number and press ENTER: ')