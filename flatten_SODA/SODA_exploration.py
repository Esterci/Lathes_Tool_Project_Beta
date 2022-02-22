
import numpy as np
import pickle
import glob
import SODA
import time


file_list = glob.glob("data_divisions/*")
file_list.sort()

gra_list = [2.5,2.6,2.7]

for ii in range(int(len(file_list)/33)):

    for gra in gra_list:

        enlapsed_time = []
        n_good_groups = np.zeros(33)
        n_bad_groups = np.zeros(33)
        good_percent_list = []
        bad_percent_list = []

        for j in range(33):

            loc = ii*33 + j

            file = file_list[int(loc)]

            data = pickle.load(open(file, "rb"))

            n_time_series = len(data)

            print("\n\nNumber of time series: {}; Granularity: {}; Iteration {};\n".format(n_time_series,
                                                                                           gra,
                                                                                           (j+1)))

            aux = (file.split('__'))

            target_name = "targets/target__" + '__'.join(aux[1:])

            target = pickle.load(open(target_name, "rb"))

            Input = {
                'StaticData': data,
                'GridSize': gra,
                'DistanceType': 'euclidean'
            }

            # initiate timer
            start = time.time()

            # run SODA
            output = SODA.SelfOrganisedDirectionAwareDataPartitioning(Input)

            # stop timer
            end = time.time()

            # append enlapse time
            enlapsed_time.append(end - start)

            SODA_label = output['IDX']

            n_groups = max(SODA_label)

            _, n_samples = np.unique(SODA_label, return_counts=True)

            percent = np.zeros((n_groups, 2))

            for i in range(len(SODA_label)):

                group = SODA_label[i] - 1

                if target[i] == 0:
                    percent[group, 0] += 1

                else:
                    percent[group, 1] += 1

            percent[:, 0] = percent[:, 0]/n_samples

            percent[:, 1] = percent[:, 1]/n_samples

            percent = percent*100

            for p in percent:
                if p[0] >= 50:
                    n_good_groups[j] += 1
                    good_percent_list.append(p[0])
                
                else:
                    n_bad_groups[j] += 1
                    bad_percent_list.append(p[1])

        mean_enlapsed_time = np.mean(enlapsed_time)
        std_enlapsed_time = np.std(enlapsed_time)

        mean_n_good_groups = np.mean(n_good_groups)
        std_n_good_groups = np.std(n_good_groups)

        mean_n_bad_groups = np.mean(n_bad_groups)
        std_n_bad_groups = np.std(n_bad_groups)

        good_mean = np.mean(good_percent_list)
        good_std = np.std(good_percent_list)

        bad_mean = np.mean(bad_percent_list)
        bad_std = np.std(bad_percent_list)

        output_text = """numero de séries temporais: {}\n
        tempo de processamento: {:.4f} +/- {:.4f} s\n
        numero de grupos com ferramentas boas: {:.4f} +/- {:.4f}\n
        numero de grupos com ferramentas ruins: {:.4f} +/- {:.4f}\n
        porcentagem média de ferramentas boas: {:.4f} +/- {:.4f}%\n
        porcentagem média de ferramentas ruins: {:.4f} +/- {:.4f}%\n""".format(n_time_series,
                                                                               mean_enlapsed_time,
                                                                               std_enlapsed_time,
                                                                               mean_n_good_groups,
                                                                               std_n_good_groups,
                                                                               mean_n_bad_groups,
                                                                               std_n_bad_groups,
                                                                               good_mean,
                                                                               good_std,
                                                                               bad_mean,
                                                                               bad_std)

        print(output_text)

        output_dict = {
            'n_time_series':n_time_series,
            'mean_enlapsed_time':mean_enlapsed_time,
            'std_enlapsed_time':std_enlapsed_time,
            'mean_n_good_groups':mean_n_good_groups,
            'std_n_good_groups':std_n_good_groups,
            'mean_n_bad_groups':mean_n_bad_groups,
            'std_n_bad_groups':std_n_bad_groups,
            'good_mean':good_mean,
            'good_std':good_std,
            'bad_mean':bad_mean,
            'bad_std':bad_std
        }

        pickle.dump(output_dict,open("outputs/output__{}__{}__.pkl".format(n_time_series,gra), "wb"))
