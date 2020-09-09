import os
import glob
import numpy as np
from datetime import datetime
import data_projection_classification as dpc

#### CODIGO PARA RODAR EM TEMPO REAL ####
if __name__ == '__main__':
    # Antes de rodar esse codigo deve rodar a o 'Unified_Code.ipynb' presente nessa mesma pagina para treinar e salvar o modelo

    output_id = 50 # id da base de dados utilizada como seed 

    # paths
    #raspberry_path = os.getcwd()
    raspberry_path = os.path.dirname(os.path.abspath(__file__))
    kernel_path = raspberry_path + '/.Kernel' # path onde estara salvo o modelo
    input_path = raspberry_path + '/Input' # path onde ira buscar as timeseries brutas, onde o codigo do Marcos esta salvando
    processed_path =  input_path + '/Processed_Data' # path onde estara salvando as timeseries processadas para entrar no modelo
    classification_path = raspberry_path + '/Classification' # path onde esta salvando um csv com as classificações

    N = 250 # numero de leituras para cada serie temporal
    time_id = np.arange(1,N+1) # vetor sequencial para o id de tempo da serie temporal


    old_csv = 0 # variavel de controle para saber se existe leituras novas
    header = 0 # modificar depois de ter o contato com o Marcos, por enquanto os testes estão sendo feitos com o .csv que chega sem header

    try:
        while True:
            try:
                # procura o arquivo .csv mais recente
                all_files = glob.glob(input_path + '/*.csv')
                latest_csv = max(all_files, key=os.path.getctime)
                existe_csv = True
            except:
                existe_csv = False # caso não seja encontrado nenhum arquivo .csv na pasta, não entrara no if

            # TALVEZ SEJA BOM COLOCAR UM TIME.SLEEP() AQUI, MAS ISSO PODE ACABAR DEIXANDO O MODELO LENTO

            # verifica se esse arquivo .csv mais recente ja foi verificado antes
            if latest_csv != old_csv and existe_csv:
                old_csv = latest_csv # modifica a variavel de controle old_csv para verificar nas proximas leituras
                raw_data = np.genfromtxt(latest_csv, skip_header=header, delimiter=',') # recebe em formato de np.array o .csv mais recente
                raw_data = raw_data[:,0:6] # como esta usando só 6 canais, mantem só as 6 primeiras colunas

                rows, columns = raw_data.shape 
                data = np.zeros((rows, 8)) # a partir do tamanho dos dados brutos cria um novo array para os dados processados

                # preenche as duas primeiras colunas desse novo array com o id da serie temporal e como o time_id
                # rows/N deve sempre um numero inteiro, ja que as series temporais salvas devem ter todas o mesmo tamanho N
                for i in range(int(rows/N)):
                    data[i*N:(i+1)*N,0] = N*[i+1]
                    data[i*N:(i+1)*N,1] = time_id
                        
                for i in range(6):
                    num = 2*(raw_data[:,i] - raw_data[:,i].min())
                    den = (raw_data[:,i].max() - raw_data[:,i].min())
                    data[:,i+2] = num/den - 1 # os dados brutos dos 6 primeiros canais são passados para o array com os dados normalizados

                # salva o array formatado na pasta input
                now = datetime.now()
                timestr = now.strftime("%Y-%m-%d__%H-%M-%S")  
                np.savetxt(processed_path+'/new_data_{}.csv'.format(timestr), data, delimiter=',')
                
                ### -------- final da formatação -------- ###
                
                ### MODELO EM TEMPO REAL ###
                features = dpc.dynamic_tsfresh(output_id)
                projected_data = dpc.PCA_projection(features)
                target = dpc.Model_Predict(projected_data)


                # salva os targets utilizando a mesma data para caso queira comparar depois
                np.savetxt(classification_path+'/target_{}.csv'.format(timestr), target, delimiter=',')


    except KeyboardInterrupt: # finaliza o loop infinito com ctrl+C
        print("EOF")     