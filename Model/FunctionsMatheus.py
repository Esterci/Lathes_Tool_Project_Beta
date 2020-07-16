def Classification (ClassificationPar, d, n_a, g, plot_matrix=False):
    
    #Changing Work Folder
    add_path1 = "/Classification/"
    add_path2 = "/.Kernel/"
    add_path3 = "/.Recovery/"
    base_path = os.path.dirname(os.path.abspath("Model_Unified_Code.ipynb"))
    Classification_path = base_path + add_path1
    Kernel_path = base_path + add_path2
    Recovery_path = base_path + add_path3

    # Change to Kernel directory
    os.chdir(Kernel_path)
    y_original = np.genfromtxt('FinalTarget.csv', delimiter=',')


    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma='scale'),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1,max_iter=500),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
  

    # Now change to Kernel directory
    
    #os.chdir( Kernel_path )
    
    #retval = os.getcwd()
    #print ("Current working directory %s" % retval)
     
    # preprocess dataset, split into training and test part
    Accuracy = np.zeros((n_a, len(names)))
    #"Y_60_euclidean_Labels_7_1.25.csv"
    s = str (int(ClassificationPar['Percent'] )) + '_' + d + '_Labels_' + str(int(ClassificationPar['ID'])) + '_' + str("%.2f" % g) + '.csv'
    X = np.genfromtxt(('X_' + s) , delimiter=',')    
    y_soda = np.genfromtxt(('Y_' + s), delimiter=',') 
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train_soda, y_test_soda, y_train_original, y_test_original = \
    train_test_split(X, y_soda, y_original, test_size=.4, random_state=42, stratify=y_soda)

    #Loop into numbeer od samples
    for i in range(Accuracy.shape[0]):
        k = 0
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
        
            clf.fit(X_train, y_train_soda)
            score = clf.score(X_test, y_test_original)
            Accuracy[i,k] = (score*100)
            k +=1
            if plot_matrix:
                ClassifiersLabel = list(clf.predict(X_test))
                confusionmatrix(ClassificationPar['ID'], d, g, ClassifiersLabel, 'Classifiers', name, y_test_original)
       
    #Creating Matrix for Mean an Std. Derivatio
    results = np.zeros((len(names),2))

    #Calculinng Mean and Std. Derivation 
    for i in range(len(names)):
        results[i,0] = round (np.mean(Accuracy[:,i]), 2 )
        results[i,1] = round (np.std(Accuracy[:,i]), 2)
            
    # Now change to Grouping Analyses directory
    
    os.chdir( Classification_path );
    
    #retval = os.getcwd()
    #print ("Current working directory %s" % retval)

    results = pd.DataFrame(results, index = names, columns = ['Media','Desvio'])       
    results.to_csv(("Classification_result_" + s) )
        
    
    print('*** {} - {} - {:.2f}  ***'.format(ClassificationPar['ID'], d, g))
    print('-------------------------------------')
    print(results)
    print(' ')
        
    # Now change to base directory
    
    os.chdir( base_path );
    retval = os.getcwd()
    #print ("Current working directory %s" % retval)

    return;

def GroupingAlgorithm (SODA_parameters,define_percent,n_IDs_gp0, processing_parameters):
    
    #Changing Work Folder
    
    add_path1 = "/PCA_Analyses/";
    add_path2 = "/.Kernel/";
    add_path3 = "/.Recovery/";
    add_path4 = "/Grouping_Analyses/";
    base_path = os.path.dirname(os.path.abspath("Model_Unified_Code.ipynb"))
    PCA_Analyses_path = base_path + add_path1;
    Kernel_path = base_path + add_path2;
    Recovery_path = base_path + add_path3;
    Grouping_Analyses_path = base_path + add_path4;
    
    print('             ');
    print('Grouping Algorithm Control Output');
    print('----------------------------------');
    
    
    ####   imput data    ####
    DataSetID = SODA_parameters['ID'];
    min_granularity = SODA_parameters['Min_g'];
    max_granularity = SODA_parameters['Max_g'];
    pace = SODA_parameters['Pace'];
    distances = SODA_parameters['Distances'];
    # functional engines
    #n_IDs_gp1 = 0; # non-functional engines
    #n_IDs_gp2 = 3598; # eminent  fault  engines

    for d in distances:
        
        for g in np.arange(int(min_granularity), int (max_granularity + pace), pace):
            ### Start Thread
            time_cpu_thread = cpu_usage()
            time_cpu_thread.start()
    
            s = 'SODA_' + d + '_label_' + str(DataSetID) + '_' + str("%.2f" % g) + '.csv';
            

            #### Data-base Imput ####
            
            # Now change to Kernel directory
    
            os.chdir( Kernel_path );
    
            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)

            SodaOutput = np.genfromtxt( s , delimiter=',');
            
            # Now change to PCA Analyses directory
    
            os.chdir( PCA_Analyses_path );
    
            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)
            
            SelectedFeatures = np.genfromtxt('features_reduzidas_' + str(DataSetID) + '.csv' , delimiter=',');

            #### Program Matrix's and Variables ####

            n_DA_planes = np.max(SodaOutput) + 1;
            Percent = np.zeros((int(n_DA_planes),3));
            n_IDs_per_gp = np.zeros((int(n_DA_planes),2));
            n_tot_Id_per_DA = np.zeros((int(n_DA_planes),1));
            decision = np.zeros(int(n_DA_planes));
            n_DA_excluded = 0;
            n_excluded = 0;
            n_gp0 = 0;
            n_gp1 = 0;
            n_gp2 = 0;
            n_data_def = 0;
            k = 0;

            #### Definition Percentage Calculation #####

            for i in range(SodaOutput.shape[0]):
    
                if i < n_IDs_gp0:
                    n_IDs_per_gp [int(SodaOutput[i]),0] += 1 ;
                else:
                    n_IDs_per_gp [int(SodaOutput[i]),1] += 1 ;

                n_tot_Id_per_DA [int(SodaOutput[i])] += 1 ;


            for i in range(int(n_DA_planes)):
    
                Percent[i,0] = (n_IDs_per_gp[i,0] / n_tot_Id_per_DA[i]) * 100;
                Percent[i,1] = (n_IDs_per_gp[i,1] / n_tot_Id_per_DA[i]) * 100;
                Percent[i,2] = ((n_tot_Id_per_DA[i]  -  (n_IDs_per_gp[i,0] + n_IDs_per_gp[i,1])) 
                    / n_tot_Id_per_DA[i]) * 100;
    
            #### Using Definition Percentage as Decision Parameter ####

            for i in range(Percent.shape[0]):
    
                if (Percent[i,0] >= define_percent):
                    n_gp0 = n_gp0 + 1 ;        
                elif (Percent[i,1] >= define_percent):    
                    n_gp1 = n_gp1 + 1 ;
                    decision[i] = 1;
                elif (Percent[i,2] >= define_percent):
                    n_gp2 = n_gp2 + 1 ;
                    decision[i] = 2;
                else:
                    n_DA_excluded += 1;
                    decision[i] = -1;
            
            #### Using decision matrix to determine the number of excluded data
                       
            for i in range (len (decision)):

                if decision[i] == -1:
                    
                    n_excluded += np.sum(n_IDs_per_gp[i,:]);
                    
        
            #### Passing data of well defined DA planes to SelectedData and defining labels

            SelectedData = np.zeros((int(SelectedFeatures.shape[0] - n_excluded),int(SelectedFeatures.shape[1])));
            ClassifiersLabel = np.zeros((int(SelectedFeatures.shape[0] - n_excluded)));
            
            
            for i in range (SodaOutput.shape[0]):
                if decision[int (SodaOutput[i]-1)] != -1:
    
                    SelectedData[k] = SelectedFeatures[i];
                    ClassifiersLabel [k] = int(not bool(decision[int(SodaOutput[i]-1)]))
                
                    if k < int(SelectedFeatures.shape[0] - n_excluded - 1):
                        k += 1;

            #### Printing Processed Data, ID's and Percentage
            
            # Now change to Kernel directory
    
            os.chdir( Kernel_path );
    
            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)

            np.savetxt('X_' + str(define_percent) + '_' + d + '_Labels_' + str(DataSetID) + '_' + str("%.2f" % g) + '.csv', SelectedData, delimiter=',');
            np.savetxt('Y_' + str(define_percent) + '_' + d + '_Labels_' + str(DataSetID) + '_' + str("%.2f" % g) + '.csv', ClassifiersLabel, delimiter=',');
    
            ### Interrupt Thread and recalculate parameters
            time_cpu_thread.stop()
            deltatime, mean_cpu = time_cpu_thread.join()
            for pp in processing_parameters:
                if pp['DistanceType'] == d and pp['Granularity'] == g:
                    aux = pp
                    break
            totaltime = deltatime + aux['Time']
            cpu_percent = (mean_cpu + aux['CPUPercent'])/2
            
            
            ### Printig Analitics results
            
            print(s);
            print('Number of data clouds: %d' % n_DA_planes)
            print('Number of good tools groups: %d' % n_gp0)
            print('Number of worn tools groups: %d' % n_gp1)
            print('Number of excluded data clouds: %d' % n_DA_excluded)
            print('Data representation loss: %.2f' % (100-((SelectedData.shape[0] / SelectedFeatures.shape[0]) * 100)))
            print('Analyse execution time: %.6f segundos' % totaltime)
            print('Avarage CPU usage: %.2f' % cpu_percent)
            confusionmatrix(DataSetID, d, g, ClassifiersLabel, 'SODA')
            print('---------------------------------------------------');
            
            #### Saving Processed Data, ID's and Percentage
            
            # Now change to Kernel directory
    
            os.chdir( Grouping_Analyses_path );
            
            Grouping_Analyse = open("Grouping_Analyse_ID_" + str(DataSetID) + "_min_" + str(min_granularity) + "_max_" + str(max_granularity) + '_' + str(define_percent) +"%.txt","w+");
            Grouping_Analyse.write(s)
            Grouping_Analyse.write('\nNumber of data clouds: %d\n' % n_DA_planes)
            Grouping_Analyse.write('Number of good tools groups: %d\n' % n_gp0)
            Grouping_Analyse.write('Number of worn tools groups: %d\n' % n_gp1)
            Grouping_Analyse.write('Number of excluded data clouds: %d\n' % n_DA_excluded)
            Grouping_Analyse.write('Data representation loss: %.2f\n' % (100-((SelectedData.shape[0] / SelectedFeatures.shape[0]) * 100)))
            Grouping_Analyse.write('Analyse execution time: %.6f segundos\n' % totaltime)
            Grouping_Analyse.write('Avarage CPU usage: %.2f\n' % cpu_percent)
            Grouping_Analyse.write('---------------------------------------------------')
            
            Grouping_Analyse.close();
    #np.savetxt('Percent.csv',define_percent,delimiter = ',')
    
    # Now change to base directory
    
    os.chdir( Recovery_path );
    
    np.save("define_percent.npy",define_percent)
    
    Output = {'Percent': define_percent,
              'Distances': distances,
              'ID': DataSetID}
    
    # Now change to base directory
    
    os.chdir( base_path );
    
    #retval = os.getcwd()
    #print ("Current working directory %s" % retval)

    return Output

distances = ['mahalanobis', 'euclidean', 'cityblock', 'chebyshev', 'minkowski', 'canberra']

min_granularity = 5
max_granularity = 7
pace = 0.25
gra = np.arange(min_granularity,max_granularity,pace)

for d in distances:
    for g in gra:
        try:
            Classification (ClassificationPar, d, 1, g, plot_matrix=False) # (Parametros da data-set, 
                                                                           #  distância, 
                                                                           #  numero de vezes a classificar,
                                                                           #  granularidade,
                                                                           #  plotar matriz de confusão (True or False))
        except:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath("Model_Unified_Code.ipynb")))
            os.chdir( base_path )
            print('*** {} - {} - {:.2f}  ***'.format(Output_ID, d, g))
