

class ProgBar:

    def __init__(self, n_elements,int_str):
        
        import sys

        self.n_elements = n_elements
        self.progress = 0

        print(int_str)

        # initiallizing progress bar

        info = '{:.2f}% - {:d} of {:d}'.format(0,0,n_elements)

        formated_bar = ' '*int(50)

        sys.stdout.write("\r")

        sys.stdout.write('[%s] %s' % (formated_bar,info))

        sys.stdout.flush()


    def update(self,prog_info=None):
        
        import sys

        if prog_info == None:

            self.progress += 1

            percent = (self.progress)/self.n_elements * 100 / 2

            info = '{:.2f}% - {:d} of {:d}'.format(percent*2,self.progress,self.n_elements)

            formated_bar = '-'* int (percent) + ' '*int(50-percent)

            sys.stdout.write("\r")

            sys.stdout.write('[%s] %s' % (formated_bar,info))

            sys.stdout.flush()


        else:

            self.progress += 1

            percent = (self.progress)/self.n_elements * 100 / 2

            info = '{:.2f}% - {:d} of {:d} '.format(percent*2,self.progress,self.n_elements) + prog_info

            formated_bar = '-'* int (percent) + ' '*int(50-percent)

            sys.stdout.write("\r")

            sys.stdout.write('[%s] %s' % (formated_bar,info))

            sys.stdout.flush()


    def message(self,prog_info):
        
        import sys

        percent = (self.progress)/self.n_elements * 100 / 2

        info = '{:.2f}% - {:d} of {:d} '.format(percent*2,self.progress,self.n_elements) + prog_info

        formated_bar = '-'* int (percent) + ' '*int(50-percent)

        sys.stdout.write("\r")

        sys.stdout.write('[%s] %s' % (formated_bar,info))

        sys.stdout.flush()


class verification_tool:

    def __init__(self,folder):

        import glob
        import numpy as np
        import pickle as pkl

        # getting aquisition files

        file_list = glob.glob(folder)

        # creating data dictionary

        data_dict = {}

        ### Reading data and saving in the dictionary

        # Initializing progress bar

        n_elements = len(file_list)

        bar = ProgBar(n_elements,'\nReading data and saving in the dictionary...')

        for file in file_list:

            # getting file name

            name,_ = file.split('/')[-1].split('.')

            # creating data and text keys

            data_dict[name] = {
                'data' : 0,
                'text' : []
            }

            with open(file, "rb") as file_object:

                data = pkl.load(data, file_object)

            data_dict[name]['data'] =data

        self.data_dict = data_dict

        # ending progress bar

        print('\n')

        
    def range_check(self):
        
        import numpy as np

        for file in self.data_dict:

            data = self.data_dict[file]['data']

            # checking if the channels are within 0 and 5

            for i in range(6):

                self.data_dict[file]['text'].append(('-> Sensor %d' % i))

                col = data[:,i]

                max = np.max(col)

                min = np.min(col)

                mean = np.mean(col)

                std = np.std(col)

                if min >= 0 and max <= 5:

                    self.data_dict[file]['text'].append('   range: ok')

                else:

                    self.data_dict[file]['text'].append('   range: ERROR!')

                    print('WARNING!! ' + file + (' presents range error on sensor %d' % i))
                    
                self.data_dict[file]['text'].append(('   mean: %f' % mean))
                
                self.data_dict[file]['text'].append(('   std: %f' % std))

        # ending range check

        print('\nrange_check done')


    def plot_time_series(self):

        import numpy as np 
        import matplotlib.pyplot as plt

        # Initializing progress bar

        n_elements = len(self.data_dict)

        bar = ProgBar(n_elements,'\nPloting time series...')

        for file in self.data_dict:

            data = self.data_dict[file]['data']

            sensors = [
                'canal_0','canal_1','canal_2',
                'canal_3','canal_4','canal_5'
                    ]

            args = np.linspace(0,len(data)-1,num=len(data),dtype=int)

            for i,sen in enumerate(sensors):

                fig, ax = plt.subplots(1,1)
                
                fig.set_size_inches(320,9)

                signal = data[:,i]

                ax.plot(args, signal, 
                        ms=5, linestyle='-')

                ax.set_title(sen)
                ax.set_ylabel("V")
                ax.grid()
                ax.set_xlim(left=0, right=len(data))

                plt.locator_params(axis='x', nbins=150)
                plt.tight_layout()

                fig.savefig('figures/' + file + '__' + sen + '.jpg',format='jpg')

                plt.close('all')

            #Updating progress bar

            bar.update()
    
        # ending progress bar

        print('\n')


    def plot_fft(self):

        import numpy as np 
        import matplotlib.pyplot as plt
        import scipy.fftpack

        # Initializing progress bar

        n_elements = len(self.data_dict)

        bar = ProgBar(n_elements,'\nPloting FFT\'s...')

        for file in self.data_dict:
            
            data = self.data_dict[file]['data']

            sensors = [
                'canal_0','canal_1','canal_2',
                'canal_3','canal_4','canal_5'
                    ]


            fig = plt.figure(figsize=(13,13))
            ax = fig.subplots(3,2)

            # Iteração sobre os arquivos (cada canal do AD)

            for i, sen in enumerate(sensors):

                # Number of samplepoints

                N_ni = int(data.shape[0])

                # sample spacing

                fs = 12e3 # Frequencia de amostragem
                T = 1.0/fs 
                x = np.linspace(0.0, N_ni*T, N_ni)
                yf = scipy.fftpack.fft(data[:N_ni,i] - data[:N_ni,i].mean())
                xf = np.linspace(0.0, 1.0/(2.0*T), N_ni//2)

                ax[i%3,i//3].set_title(sen, fontsize=15)
                ax[i%3,i//3].plot(xf, 2.0/N_ni * np.abs(yf[:N_ni//2]), label='National Instruments')
                ax[i%3,i//3].legend()

                plt.tight_layout()

            fig.savefig('figures/FFT__' + file + '.jpg',format='jpg')

            plt.close('all')

            #Updating progress bar

            bar.update()
    
        # ending progress bar

        print('\n')
           

    def write_report(self):

        ### Saving results in a txt file 

        for file in self.data_dict:
            
            # appending text

            for txt in self.data_dict[file]['text']:

                # Open the file in append & read mode ('a+')

                with open('reports/' + file + '__report.txt', "a+") as file_object:

                    # Move read cursor to the start of file.
                    file_object.seek(0)

                    # If file is not empty then append '\n'
                    data = file_object.read(100)

                    if len(data) > 0 :
                        file_object.write("\n")
                        
                    # Append text at the end of file
                    file_object.write(txt)

        print('\nReports writen')


def time_stamp():
    from datetime import datetime
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H.%M.%S")
    return timestampStr


def int_to_bol(h,l):

    W=(h*256)+l

    if W>32767:

        W = W-65536      

    valor= (W/32768)*10

    return valor


def transference_function(data):

    import numpy as np

    dim = data.shape

    n = dim[0]

    valor = np.empty([n,8], dtype = float)

    a = 0

    for a in range (n):
        b = 0
        for b in range (0,15,2):

            hi = data[a,b]

            lo = data[a,(b+1)]

            c = (b/2)

            c = round(c)

            v = int_to_bol(hi,lo)

            valor[a,c] = v
        a=a+1