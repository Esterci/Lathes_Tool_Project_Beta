import numpy as np
import glob
import pandas as pd

N = 20

files = glob.glob('../Base_Prototipo/18*')
files.sort()
files = files[2:-4]

for f in files:
    print(f)
    timestamp = f.split('/')[-1].split('.c')[0]
    

    #data = np.genfromtxt(f, delimiter=',')
    data = pd.read_csv(f, header=None).values
    L, W = data.shape
    ts_LEN = int(np.max(data[:,1]))
    ts_num = int(L/ts_LEN)

    ts_re = int(ts_num/N)

    index = np.arange(ts_num)

    for i in range(N):
        print(i)

        index_fold = np.random.choice(index, size=ts_re, replace=False)
        index_fold.sort()
        print(index_fold)

        for j in index_fold:
            new_data = data[j*ts_LEN:(j+1)*ts_LEN,:].copy()

            L2, W = new_data.shape

            if f in ['../Base_Prototipo/18-Jan-2022-14.25.20.csv', '../Base_Prototipo/18-Jan-2022-15.01.43.csv']:
                new_data = np.hstack((new_data, 2*np.ones((L2,1))))
            elif f in ['../Base_Prototipo/18-Jan-2022-12.45.51.csv', '../Base_Prototipo/18-Jan-2022-12.24.50.csv']:
                new_data = np.hstack((new_data, np.ones((L2,1))))
            else:
                new_data = np.hstack((new_data, np.zeros((L2,1))))

            if j == index_fold[0]:
                new_data2 = new_data.copy()
            else:
                new_data2 = np.vstack((new_data2, new_data))
        
        res = [i for i in index if i not in index_fold]
        index = res

        np.savetxt('../Base_Prototipo_3class/Base_Prototipo__{}__split_{}.csv'.format(timestamp,i), new_data2, delimiter=',')
    

        del new_data, new_data2
    del data


for i in range(N):
    files = glob.glob('../Base_Prototipo_3class/Base_Prototipo__*__split_{}.csv'.format(i))
    files.sort()

    for j, f in enumerate(files):
        print(f)
        if j == 0:
            data = pd.read_csv(f,header=None)
        if j != 0:
            foo = pd.read_csv(f,header=None)
            data = np.vstack((data, foo))
            del foo
    
    np.savetxt('../Base_Prototipo_3class/Base_Prototipo__split_{}.csv'.format(i), data, delimiter=',')
    del data

