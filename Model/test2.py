from pathlib import Path
import os
from funcao import *

Output_ID = 4

print('----- NOVO -----')
add_path1_novo = "\\Input"
base_path_novo = str(os.path.abspath('')) # Esse '1' significa que é 1 pasta acima do arquivo em execução
working_path_novo = base_path_novo + '\\Model' # Essa linha pega o caminho do arquivo em execução
Input_path_novo = working_path_novo + add_path1_novo

print('Base Path:')
print(base_path_novo)

print('Working Path:')
print(working_path_novo)

print('Input Path:')
print(Input_path_novo)

# O que mudaria de verdade seria aqui que não precisaria de ficar usando o change directory
data = np.genfromtxt('{}\\Output_{}.csv'.format(Input_path_novo, Output_ID), delimiter=',')

print('----- ANTIGO -----')
add_path1 = "/Input/"
base_path = os.getcwd() # Essa linha pega o diretorio em que estamos trabalhando no momento,
                        # pode acabar dando erro porque temos que saber exatamente em qual diretorio a gente estava anteriormente
                        # seja numa função que foi chamada antes ou algo do tipo
working_path = os.getcwd() + '/Model'
Input_path = working_path + add_path1

print('Base path:')
print(base_path)
print('Working Path:')
print(working_path)
print('Input Path:')
print(Input_path)


# Aqui que começa a dar alguns erros no codigo, já que a gente precisa de ter sempre um tracking
# de em qual diretorio estamos trabalhando, atraves do chdir, as vezes chamamos uma função dentro do de um bloco
# que esta nesse diretorio e não no base_path e acaba que perdemos controle disso ou deixa uma função muito especifica,
# por exemplo ela só poderia ser chamada se tivesse dentro de um bloco do diretorio Input_path por exemplo
os.chdir( Input_path )

data_2 = np.genfromtxt('Output_{}.csv'.format(Output_ID), delimiter=',')

os.chdir( base_path ) # E usando desse jeito esquecer isso pode acabar dificultando demais


# Só para exemplificar que os dois jeitos pegaram o mesmo arquivo coloquei um print do shape deles
print('----- Arquivo -----')
print('Metodo Novo (data.shape): {}'.format(data.shape))
print('Metodo Antigo (data.shape): {}'.format(data_2.shape))

my_function(Output_ID)