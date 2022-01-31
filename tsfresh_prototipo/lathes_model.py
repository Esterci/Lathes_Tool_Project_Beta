import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tsfresh
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute
from datetime import date, datetime
from tsfresh.feature_extraction import EfficientFCParameters

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from SODA import SelfOrganisedDirectionAwareDataPartitioning
import pickle


class LathesModel(object):
    """Lathes Cutting Tool Model Class
    
    Parameters
    ----------
    N_PCs: int, default=3
        Number of components to keep in PCA.
    clf: classifier, default=MLPClassifier
        sklearn binary classifier
    n_jobs: int, default=4
        The number of processes to use for parallelization in tsfresh
    granularity: float, default=3
        SODA granularity, sensibility factor for data partitioning module
    percent: float, default=50
        purity percent for grouping algorithm, must be within (50, 100) interval
        percent=50 means hard voting

    Attributes
    ----------
    N_PCs_: int
        Number of components to keep in PCA.
    n_jobs_: int
        The number of processes to use for parallelization in tsfresh
    granularity_: float
        SODA granularity, sensibility factor for data partitioning module
    percent_: float
        purity percent for grouping algorithm
    eigen_matrix_: np.array
        pca transformation eigen matrix
    nan_columns_: list
        name of columns with NaN values
    valid_columns_: list
        name of columns without NaN values
    target_: np.array
        target for each time serie used to fit the model
    relevance_table_: pd.DataFrame
        relevance table after KS test
    relevant_features_: pd.Series
        name of features selected by TSFRESH hypothesis test 
    selected_columns_: pd.Index
        name of features selected by TSFRESH hypothesis test 
    kind_to_fc_parameters_: dict
        dictionary with features selected by TSFRESH
        keys = sensor names
    X_selected_: pd.DataFrame
        train data set features after TSFRESH selection
    X_projectd_: np.array
        train data set projected in Principal Components
    variation_kept: np.array
        Percentage of variance explained by each of the selected components.
    SODA_output_: dict
        Dictionary with SODA output
    SODA_IDX_: np.array
        array with labels given by SODA algorithm
    classifiers_label_: np.array
        array with labels given by Grouping Algorithm
    GA_results: dict
        'Data_Clouds': int
            number of Data Clouds
        'Good_Tools_Groups': int
            number of Adequate Condition Data Clouds 
        'Worn_Tools_Groups': int
            number of Inadequate Condition Data Clouds 
        'Samples': int
            number of Samples
    X_test_seleected: np.array
        test data set features selected by tsfresh
    X_test_projected: np.array
        test data set projected in Principal Components
    n_timeseries_ : int
        number of timeseries in train dataset
    n_measures_: int
        number of measurements in timeseries
    n_sensors_: int
        number of sensors in dataset
    already_fitted_: bool
        if model has already been fitted 
            already_fitted = True
        else 
            already_fitted = False   
    already_tested_: bool
        if model has already been tested 
            already_tested = True
        else 
            already_test = False   
    fit_time_: datetime
        time to fit model
    tsfresh_time_: datetime
        time to fit tsfresh
    predict_time_: datetime
        time of last prediction
    tsfresh_predict_time_: datetime
        time of last prediction tsfresh
    one_class_: bool
        if one_class_ == True:
            model has only one class and has to be fitted with higher granularity
        else
            model has more than one class

    clf: sklearn classifier
        sklearn binary classifier
    scaler: sklearn.preprocessing.MinMaxScaler
        scaler to normalize input data 
    pca_scaler: sklearn.preprocessing.StandardScaler
        scaler to standardize selected features
    pca: sklearn.decomposition.PCA
        pca fitted model
    """
    def __init__(self, N_PCs=3, clf='None', n_jobs=0, granularity=3, percent=50):

        self.N_PCs_ = N_PCs
        self.granularity_ = granularity
        self.n_jobs_ = n_jobs
        self.percent_ = percent
        if clf == 'None':
            self.clf = MLPClassifier(alpha=1,max_iter=500)
        else:
            self.clf = clf

        self.eigen_matrix_ = [0]
        self.already_fitted_ = False
        self.already_tested_ = False
        self.one_class_ = False

        data = pickle.load( open( "Input.pkl", "rb" ) )

        X = data[:,:-1]
        y = data[:,-1]

        self.X_train, self.X_test, self.yy_train, self.yy_test = Lathes_train_test_split(X, 
                                    y, 
                                    test_size=0.3, 
                                    random_state=12)

        print('X_train shape:', self.X_train.shape)
        print('y_train shape:', self.yy_train.shape)
        print('X_test shape:', self.X_test.shape)
        print('y_test shape:', self.yy_test.shape)
    
    def _copy(self, params):
        """ Suport function to copy model instance """
        for p in params:
            exec('self.{} = {}'.format(p, params[p]))

    def copy(self):
        """ Copy model instance """
        C = LathesModel(self.N_PCs_, self.clf, self.n_jobs_, self.granularity_, self.percent_)
        if self.already_fitted_ == True:
            param_names = ['GA_results_', 'N_PCs_', 'SODA_IDX_', 'SODA_output_', 'X_projected_', 'X_selected_',
                           'already_fitted_', 'already_tested_', 'classifiers_label_', 'clf', 'granularity_', 
                           'kind_to_fc_parameters_', 'n_jobs_', 'n_measures_', 'n_sensors', 'n_timeseries_', 'nan_columns_', 
                           'pca', 'pca_scaler', 'percent_', 'relevance_table_', 'relevant_features_', 'scaler', 
                           'selected_columns_', 'target_', 'valid_columns_', 'variation_kept_','fit_time_',
                           'tsfresh_time_', 'one_class_']
            param_dict = {}
            for p in param_names:
                exec('param_dict[p] = self.{}'.format(p))

            C._copy(param_dict)
            if self.already_tested_ == True:
                param_names = ['already_tested_''predict_time_', 'tsfresh_predict_time_', 'X_test_projected_', 'X_test_selected_']

                param_dict = {}
                for p in param_names:
                    exec('param_dict[p] = self.{}'.format(p))

                C._copy(param_dict)

        return C

    def reset(self):
        """ Reset model 
        This function will reset the model, if methods 'fit_after_tsfresh' or 'predict_after_tsfresh'
        are called it will execute the whole fit or prediction methods again"""
        self.eigen_matrix_ = [0]
        self.already_fitted_ = False
        self.already_tested_ = False
        self.one_class_ = False

    ### Fitting Methods

    def _normalization(self):
        """ Normalize input data in fit stage """

        self.X_train,self.yy_train

        self.n_timeseries_ = int(self.X_train[:,0].max())
        self.n_measures_ = int(self.X_train[:,1].max())
        self.n_sensors_ = int(self.X_train.shape[1]-2)
        self.target_ = self.yy_train[::self.n_measures_]

        del self.yy_train

        info = self.X_train[:,0:2]
        data = self.X_train[:,2:]

        del self.X_train

        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(data)

        self.df = pd.DataFrame(np.concatenate((info,data), axis=1), columns= ['id','time'] + 
                        ['Sensor_' + str(x) for x in range(1,self.n_sensors_+1)])

    def _tsfresh_extraction(self):
        """ Feature Extraction in fit stage
        After extraction columns with NaN values are dropped"""

    
        
        self.extracted_features = tsfresh.extract_features(self.df, column_id="id", column_sort="time", 
                                                        default_fc_parameters = EfficientFCParameters(),
                                                        chunksize=100,
                                                        n_jobs=self.n_jobs_,
                                                        disable_progressbar=False)

        del self.df
        
        features = self.extracted_features.columns
        self.nan_columns_ = []
        self.valid_columns_ = []
        for col in features:
            if self.extracted_features.loc[:,col].hasnans:
                self.nan_columns_.append(col)
            else:
                self.valid_columns_.append(col)
        
        self.extracted_features.drop(self.nan_columns_, axis=1)

    def _tsfresh_selection(self):
        """ Feature Selection for fit stage """
        y = pd.Series(self.target_, index=self.extracted_features.index)

        self.relevance_table_ = calculate_relevance_table(self.extracted_features, y)

        self.relevant_features_ = self.relevance_table_[self.relevance_table_.relevant].feature

        self.X_selected_ = self.extracted_features.loc[:, self.relevant_features_]
        
        self.selected_columns_ = self.X_selected_.columns

        self.kind_to_fc_parameters_ = tsfresh.feature_extraction.settings.from_columns(self.X_selected_,
                                                        n_jobs=self.n_jobs_)


    def _pca(self):
        """ PCA calculation and projection for fit stage """
        self.pca_scaler = StandardScaler()
        X_scaled = self.pca_scaler.fit_transform(self.X_selected_)

        self.pca = PCA(n_components=self.N_PCs_)
        self.pca.fit(X_scaled)

        self.X_projected_  = self.pca.transform(X_scaled)

        self.variation_kept_ = self.pca.explained_variance_ratio_*100

    def _soda(self):
        """ SODA Data Partitioning Algorithm for fit stage """
        Input = {'GridSize':self.granularity_, 'StaticData':self.X_projected_, 'DistanceType': 'euclidean'}
        self.SODA_output_ = SelfOrganisedDirectionAwareDataPartitioning(Input)

        self.SODA_IDX_ = self.SODA_output_['IDX']

    def _grouping_algorithm(self): 
        """ Grouping Algorithm for fit stage """         
         #### Program Matrix's and Variables ####
        n_DA_planes = np.max(self.SODA_IDX_)
        Percent = np.zeros((int(n_DA_planes),3))
        n_IDs_per_gp = np.zeros((int(n_DA_planes),2))
        n_tot_Id_per_DA = np.zeros((int(n_DA_planes),1))
        decision = np.zeros(int(n_DA_planes))
        n_gp0 = 0
        n_gp1 = 0

        #### Definition Percentage Calculation #####

        for i in range(self.target_.shape[0]):
            if self.target_[i] == 0:
                n_IDs_per_gp [int(self.SODA_IDX_[i]-1),0] += 1 
            else:
                n_IDs_per_gp [int(self.SODA_IDX_[i]-1),1] += 1 

            n_tot_Id_per_DA [int(self.SODA_IDX_[i]-1)] += 1 


        for i in range(int(n_DA_planes)):

            Percent[i,0] = (n_IDs_per_gp[i,0] / n_tot_Id_per_DA[i]) * 100
            Percent[i,1] = (n_IDs_per_gp[i,1] / n_tot_Id_per_DA[i]) * 100
                    
        #### Using Definition Percentage as Decision Parameter ####

        for i in range(Percent.shape[0]):

            if (Percent[i,0] > self.percent_):
                n_gp0 += 1
                decision[i] = 0
            else:
                n_gp1 += 1
                decision[i] = 1
                        
                #### Defining labels

        self.classifiers_label_ = []

        for i in range (len(self.SODA_IDX_)):
            self.classifiers_label_.append(decision[int (self.SODA_IDX_[i]-1)])
                    


        ### Printig Analitics results
                    
        self.GA_results_ = {'Data_Clouds':n_DA_planes,
                            'Good_Tools_Groups': n_gp0,
                            'Worn_Tools_Groups': n_gp1,
                            'Samples': int(len(self.SODA_IDX_))}
    
    ### Prediction Methods

    def _predict_normalization(self):
        """ Normalize input data for prediction stage
        This step is executed using 'scaler' fitted in .fit"""
        info = self.X_test[:,0:2]
        data = self.X_test[:,2:]

        del self.X_test

        data = self.scaler.transform(data)

        self.precit_df = pd.DataFrame(np.concatenate((info,data), axis=1), columns= ['id','time'] + 
                            ['Sensor_' + str(x) for x in range(1,self.n_sensors_+1)])
            
    def _predict_tsfresh_extraction(self):
        """ Feature Extraction for prediction stage 
        This step is executed using 'kind_to_fc_parameters_' constructed in .fit"""
        columns = []
        for i,x in enumerate(self.kind_to_fc_parameters_):
            aux = pd.DataFrame(np.hstack((self.precit_df.loc[:,:'time'].values,
                                self.precit_df.loc[:,x].values.reshape((-1,1)))),
                                columns=['id','time',x])

            del self.precit_df
                
            aux2 = tsfresh.extract_features(aux, column_id="id", column_sort="time",
                                            default_fc_parameters=self.kind_to_fc_parameters_[x],
                                            n_jobs=self.n_jobs_)

            for j in range(len(aux2.columns.tolist())):columns.append(aux2.columns.tolist()[j])

            if i == 0:
                extracted_features = np.array(aux2.values)
            else:
                extracted_features = np.hstack((extracted_features,aux2.values))

        final_features = pd.DataFrame(extracted_features,columns=columns)
        self.X_test_selected_ = impute(final_features[self.selected_columns_])      

    def _predict_pca(self):
        """ Project predict data using PCA fitted in .fit"""
        X_scaled = self.pca_scaler.transform(self.X_test_selected_)

        self.X_test_projected_ = self.pca.transform(X_scaled)

    ### Main Methods

    def fit(self, X, y):
        """Fit the model with X and target y

        Parameters
        ----------
        X : array-like, shape (n_timeseries_*n_measures_, n_sensors+2)
            Training data, in following format

            | ID            | Time_ID     |  Sensor 1 | ... |  Sensor n | 
            |---------------|-------------|-----------| ... |-----------|
            | 1             | 1           |  1.91     | ... | -1.03     |
            | 1             | 2           |  1.06     | ... |  1.17     |
            | ...           | ...         | ...       | ... | ...       |
            | 1             | n_measures_ |  0.48     | ... | -1.69     |
            | 2             | 1           |  0.78     | ... |  1.45     |
            | 2             | 2           | -0.21     | ... |  0.46     |
            | ...           | ...         | ...       | ... | ...       |
            | 2             | n_measures_ | -1.18     | ... | -0.09     |
            | ...           | ...         | ...       | ... | ...       |
            | n_timeseries_ | 1           | -0.85     | ... |  0.07     |
            | n_timeseries_ | 2           | -1.18     | ... |  0.64     |
            | ...           | ...         | ...       | ... | ...       |
            | n_timeseries_ | n_measures_ | -0.83     | ... | -0.97     |

        y : np.array, shape (n_timeseries_*n_measures_)
            Target for training data
        """

        start = datetime.now()

        X_norm = self._normalization(X, y)

        X_extracted = self._tsfresh_extraction(X_norm)

        self._tsfresh_selection(X_extracted)

        self.tsfresh_time_ = datetime.now() - start

        self._pca()

        self._soda()

        self._grouping_algorithm()
    
        try:
            self.clf.fit(self.X_projected_, self.classifiers_label_)
            self.one_class_ = False
        except:
            self.one_class_ = True

        self.already_fitted_ = True

        self.fit_time_ = datetime.now() - start

    def fit_predict(self, X, y):
        """Fit the model with X and target y and predict the target after that

        Parameters
        ----------
        X : array-like, shape (n_timeseries_*n_measures_, n_sensors+2)
            Training data

        y : np.array, shape (n_timeseries_*n_measures_)
            Target for training data
        
        Returns
        -------
        y_pred : np.array shape(n_timeseries,)
            Predictions for training data
        """
        self.fit(X, y)

        y_pred = self.clf.predict(self.X_projected_)

        return y_pred

    def predict(self,X):
        """Predict using the trained model

        Parameters
        ----------
        X : np.array, shape (n_timeseries_*n_measures_, n_sensors+2)
            The input data.

        Returns
        -------
        y_pred : np.array (n_timeseries,)
            The predicted class for each timeseries presented to the model.
        """

        if self.one_class_:
            return
        else:
            start = datetime.now()

            X_norm = self._predict_normalization(X)

            self._predict_tsfresh_extraction(X_norm)

            self.tsfresh_predict_time_ = datetime.now() - start

            self._predict_pca()

            y_pred = self.clf.predict(self.X_test_projected_)

            self.already_tested_ = True

            self.predict_time_ = datetime.now() - start

        return y_pred
    
    def fit_after_tsfresh(self,X,y):
        """Fit the model with X and target y after TSFRESH extraction
        and selection already had been performed.
        This method is useful for train the model after change some parameter
        as N_PCs, granularity or classifier without the need of execute the TSFRESH
        extraction module again.

        If the model wasn't fitted before the model will be fitted from the start.

        Parameters
        ----------
        X : array-like, shape (n_timeseries_*n_measures_, n_sensors+2)
            Training data

        y : np.array, shape (n_timeseries_*n_measures_)
            Target for training data
        """
        if self.already_fitted_:
            start = datetime.now()
            self._pca()

            self._soda()

            self._grouping_algorithm()

            try:
                self.clf.fit(self.X_projected_, self.classifiers_label_)
                self.one_class_ = False
            except:
                self.one_class_ = True

            self.fit_time_ = datetime.now() - start + self.tsfresh_time_
        else:
            print('Fitting from start!')
            self.fit(X,y)

    def predict_after_tsfresh(self, X):
        """ Predict using trained model same dataset that was last predicted.
        This method is useful for predict the model after change some parameter
        as N_PCs, granularity or classifier without the need of execute the TSFRESH
        extraction module again.

        If the model wasn't predicted before the model will predict from the start.

        Parameters
        ----------
        X : array-like, shape (n_timeseries_*n_measures_, n_sensors+2)
            Training data

        y : np.array, shape (n_timeseries_*n_measures_)
            Target for training data
        """
        if self.already_tested_:
            if self.one_class_:
                return
            else:
                start = datetime.now()
                y_pred = self.clf.predict(self.X_test_projected_)

                self.predict_time_ = datetime.now() - start + self.tsfresh_predict_time_

        else:
            print('Predicting from start!')
            y_pred = self.predict(X)
            
        return y_pred

    def change_hyperparams(self, params):
        """ Change model Hyperparams 
        This function is useful to change a model param without the need of reconstruct the model

        Parameters
        ----------
        params: dict 
            dictionary with some of the following keys
            'N_PCs': int
                Number of components to keep in PCA.
            'clf': classifier
                sklearn binary classifier
            'n_jobs': int
                The number of processes to use for parallelization in tsfresh
            'granularity': float
                SODA granularity, sensibility factor for data partitioning module
            'percent': float
                purity percent for grouping algorithm, must be within (50, 100) interval
        """

        for p in params:
            if p == 'clf':
                exec('self.{} = {}'.format(p, params[p]))
            else:
                exec('self.{}_ = {}'.format(p, params[p]))

    ### PCA Analysis

    def _create_eigen_matrix(self):
        """ Data manipulation for PCA Analytics plots """
        if type(self.eigen_matrix_) == np.ndarray:
            return
        else:
            try:
                self.eigen_matrix_ = abs(np.array(self.pca.components_))

                for i in range (self.eigen_matrix_.shape[0]):
                    LineSum = sum(self.eigen_matrix_[i,:])
                    for j in range (self.eigen_matrix_.shape[1]):
                        self.eigen_matrix_[i,j] = ((self.eigen_matrix_[i,j]*100)/LineSum)

                # Weighted Contribution for each feature
                weighted_contribution = (self.eigen_matrix_.T * self.variation_kept_.T).sum(1)/self.variation_kept_.sum()

                df_weighted_contribution = pd.DataFrame(weighted_contribution.reshape(1,-1), columns=self.selected_columns_)                
                df_weighted_contribution = df_weighted_contribution.sort_values(by=0, axis=1, ascending=False)

                #Creating Separated dictionaries for Sensors and Features Contribution 
                sensors_names = [None] * int(df_weighted_contribution.shape[1])
                features_names = [None] * int(df_weighted_contribution.shape[1])
                general_features = [None] * int(df_weighted_contribution.shape[1])

                c = '__'
                for i, names in zip(range (df_weighted_contribution.shape[1]), df_weighted_contribution.columns):
                    words = names.split(c)
                    sensors_names[i] = words[0]
                    general_features[i]= words[1]
                    features_names[i] = c.join(words[1:])
                    
                unique_sensors_names = np.unique(sensors_names).tolist()
                unique_general_features = np.unique(general_features).tolist()
                unique_features_names = np.unique(features_names).tolist()

                self.sensors_contribution_ = dict.fromkeys(unique_sensors_names, 0)
                self.general_features_contribution_ = dict.fromkeys(unique_general_features, 0)
                self.features_contribution_ = dict.fromkeys(unique_features_names, 0)      

                #Creating dictionaries from Data Frame orientation
                weighted_contribution = {}
                for col in df_weighted_contribution.columns:
                    parts = col.split(c)
                    
                    kind = parts[0]
                    feature = c.join(parts[1:])
                    feature_name = parts[1]
                    
                    if kind not in weighted_contribution:
                        weighted_contribution[kind] = {}
                    
                    self.sensors_contribution_[kind] += df_weighted_contribution.loc[0,col]
                    self.general_features_contribution_[feature_name] += df_weighted_contribution.loc[0,col]
                    self.features_contribution_[feature] += df_weighted_contribution.loc[0,col]
                    weighted_contribution[kind][feature] = df_weighted_contribution.loc[0,col]       

            except:
                raise Exception('Model not fitted!')

    def plot_variation_held(self, PATH=False, figsize=[16,8], title_fontsize=22, 
                            y_fontsize=27, x_fontsize=20, 
                            y_ticks_fontsize=22, x_ticks_fontsize=22, show=False):
        """ Plot Percentage of Variation Held for PCs kept
        
        Parameters
        ----------
        PATH: str or PATH
            PATH to save figure"""
        try:
            fig = plt.figure(figsize=figsize)
            fig.suptitle('Percentage of Variance Held by PCs', fontsize=title_fontsize)
            ax = fig.subplots(1,1)
            ax.bar(x=['PC' + str(x) for x in range(1,(self.N_PCs_+1))],height=self.variation_kept_[0:self.N_PCs_])
            ax.set_ylabel('Percentage of Variance Held',fontsize=y_fontsize)
            ax.set_xlabel('Principal Components',fontsize=x_fontsize)
            ax.tick_params(axis='x', labelsize=x_ticks_fontsize)
            ax.tick_params(axis='y', labelsize=y_ticks_fontsize)
            ax.grid()
            if show:
                plt.show()
            if PATH:
                fig.savefig(PATH, bbox_inches='tight')
        except:
            raise Exception('Model not fitted!')
    
    def plot_contribution_per_PC(self, PATH=False, figsize=[16,8], title_fontsize=22, show=False):
        """ Plot feature contribution per each PC
        
        Parameters
        ----------
        PATH: str or PATH
            PATH to save figure"""
        try:
            self._create_eigen_matrix()

            figsize = [figsize[0], figsize[1]*self.N_PCs_]
            fig = plt.figure(figsize=figsize)
            fig.suptitle('Contribution percentage per PC', fontsize=title_fontsize)

            ax = fig.subplots(int(self.N_PCs_),1)

            for i in range (int(self.N_PCs_)):
                s = self.eigen_matrix_[i,:]
                ax[i].bar(x=range(0,(self.eigen_matrix_.shape[1])),height=s)
                ax[i].set(xlabel='Features', ylabel='Contribution Percentage', title = 'PC ' + str(i+1))
                ax[i].grid()

                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for axs in ax.flat:
                    axs.label_outer()

            if show:
                plt.show()
            if PATH:
                fig.savefig(PATH, bbox_inches='tight')
        except:
            raise Exception('Model not fitted!')

    def plot_sensor_contribution(self, PATH=False, figsize=[16,8], title_fontsize=22, 
                            y_fontsize=27, x_fontsize=20, 
                            y_ticks_fontsize=22, x_ticks_fontsize=22, show=False, sensor_names=False):
        """ Plot sensors weighted contribution
        
        Parameters
        ----------
        PATH: str or PATH
            PATH to save figure"""
        try:
            self._create_eigen_matrix()
            #Ploting Contribution Sensors Results        
            fig = plt.figure(figsize=figsize)

            fig.suptitle('Sensors Weighted Contribution Percentage', fontsize=title_fontsize)

            ax = fig.subplots(1,1)
            if sensor_names:
                ax.bar(sensor_names, self.sensors_contribution_.values())
            else:
                s = self.sensors_contribution_
                ax.bar(*zip(*s.items()))
            plt.ylabel('Relevance Percentage',fontsize=y_fontsize)
            plt.xlabel('Sensors',fontsize=x_fontsize)
            plt.tick_params(axis='x', labelsize=x_ticks_fontsize)
            plt.tick_params(axis='y', labelsize=y_ticks_fontsize)
            ax.grid()

            if show:
                plt.show()
            if PATH:
                fig.savefig(PATH, bbox_inches='tight')
        except:
            raise Exception('Model not fitted!')

    def plot_features_contribution(self, PATH=False, figsize=[16,8], title_fontsize=16, 
                            y_fontsize=20, x_fontsize=20, 
                            y_ticks_fontsize=16, x_ticks_fontsize=18, show=False):
        """ Plot features weighted contribution
        
        Parameters
        ----------
        PATH: str or PATH
            PATH to save figure"""
        try:
            self._create_eigen_matrix()
            #Ploting Cntribution Features Results
            fig = plt.figure(figsize=figsize)

            fig.suptitle('Features Weighted Contribution Percentage', 
                        fontsize=title_fontsize)
            ax = fig.subplots(1,1)
            s = dict(sorted(self.features_contribution_.items(), 
                            key=lambda item: item[1], reverse=True))

            ax.bar(np.arange(len(s)), s.values())
                    
            plt.ylabel('Relevance Percentage',fontsize=y_fontsize)
            plt.xlabel('Features',fontsize=x_fontsize)
            plt.tick_params(axis='x', labelsize=x_ticks_fontsize)
            plt.tick_params(axis='y', labelsize=y_ticks_fontsize)
            ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
            plt.grid()
            if show:
                plt.show()
            if PATH:
                fig.savefig(PATH, bbox_inches='tight')
        except:
            raise Exception('Model not fitted!')

    def plot_best_features_contribution(self, best=20, PATH=False, 
                            figsize=[16,8], title_fontsize=16, 
                            y_fontsize=20, x_fontsize=20, 
                            y_ticks_fontsize=16, x_ticks_fontsize=18, show=False):
        """ Plot best features weighted contribution
        
        Parameters
        ----------
        best: int
            number of features to plot
        PATH: str or PATH
            PATH to save figure"""
        try:
            self._create_eigen_matrix()

            fig = plt.figure(figsize=figsize)

            fig.suptitle('Best Features Weighted Contribution Percentage', 
                        fontsize=title_fontsize)

            ax = fig.subplots(1,1)
            s = dict(sorted(self.features_contribution_.items(), 
                            key=lambda item: item[1], reverse=True))
            s_20 = list(s.values())[0:best]

            ax.bar(x=['X' + str(x) for x in range(1,(best+1))],height=s_20)
            plt.ylabel('Relevance Percentage',fontsize=y_fontsize)
            plt.xlabel('Features',fontsize=x_fontsize)
            plt.tick_params(axis='x', labelsize=x_ticks_fontsize)
            plt.tick_params(axis='y', labelsize=y_ticks_fontsize)
            ax.grid()
            ax.set_ylim([s_20[-1]-0.05,s_20[0]+0.05])
            if show:
                plt.show()
            if PATH:
                fig.savefig(PATH, bbox_inches='tight')
        except:
            raise Exception('Model not fitted!')

    def plot_scatter_features(self, PATH=False, figsize=[14,10], size=50, label_fontsize=20,
                    label_pad=18, ticks_fontsize=16, cmap='viridis', show=False):
        """ Plot dataset projected on first 2 or 3 PCs divided by SODA
        
        Parameters
        ----------
        PATH: str or PATH
            PATH to save figure"""
        try:
            best_sensor = max(self.sensors_contribution_)


            s = dict(sorted(self.features_contribution_.items(), 
                            key=lambda item: item[1], reverse=True))
            best_features = list(s.keys())
            features = []
            for f in best_features:
                features.append(best_sensor+'__'+f)

            x = self.X_selected_[features[0]].values
            y = self.X_selected_[features[1]].values
            z = self.X_selected_[features[2]].values
                                  
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            colors = pd.Series(self.target_)
            colors[self.target_==0] = 'tab:blue'
            colors[self.target_==1] = 'tab:red'
            ax.scatter(x, y, z, c=colors, s=size, cmap='seismic')
                    
            plt.ylabel('X2',fontsize=label_fontsize,labelpad=label_pad)
            plt.xlabel('X1',fontsize=label_fontsize, labelpad=label_pad)
            ax.set_zlabel('X3', fontsize=label_fontsize, labelpad=int(2/3*label_pad))
            plt.tick_params(axis='x', labelsize=ticks_fontsize)
            plt.tick_params(axis='y', labelsize=ticks_fontsize)
            plt.tick_params(axis='z', labelsize=ticks_fontsize)

            legend_elements = [Line2D([0], [0], marker='o', color='tab:blue', linestyle='None', 
                          markersize=15, label='Adequate Tools'),
                   Line2D([0], [0], marker='o', color='tab:red', linestyle='None', 
                          markersize=15, label='Inadequate Tools')]

            plt.legend(handles=legend_elements, fontsize=15)

            ax.grid()
            if show:
                plt.show()
            if PATH:
                fig.savefig(PATH, bbox_inches='tight')
        except:
            raise Exception('Model not fitted!')
    
    ### Clouds Plot

    def plot_soda(self, PATH=False, figsize=[14,10], s=50, label_fontsize=20,
                    label_pad=18, ticks_fontsize=16, cmap='viridis', show=False):
        """ Plot dataset projected on first 2 or 3 PCs divided by SODA
        
        Parameters
        ----------
        PATH: str or PATH
            PATH to save figure"""
        try:
            if self.N_PCs_ == 2:
                x = self.X_projected_[:,0]
                y = self.X_projected_[:,1]
                                    
                fig = plt.figure(figsize=figsize)
                colors = self.SODA_IDX_
                plt.scatter(x, y, c=colors, s=s, edgecolor='k', cmap=cmap)
                plt.ylabel('PC2',fontsize=label_fontsize,labelpad=label_pad)
                plt.xlabel('PC1',fontsize=label_fontsize, labelpad=label_pad)
                plt.tick_params(axis='x', labelsize=ticks_fontsize)
                plt.tick_params(axis='y', labelsize=ticks_fontsize)
                plt.grid()
                if show:
                    plt.show()
                if PATH:
                    fig.savefig(PATH, bbox_inches='tight')

            if self.N_PCs_ >= 3:
                x = self.X_projected_[:,0]
                y = self.X_projected_[:,1]
                z = self.X_projected_[:,2]
                                    
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')

                colors = self.SODA_IDX_
                ax.scatter(x, y, z, c=colors, s=s, edgecolor='k', cmap=cmap)
                    
                plt.ylabel('PC2',fontsize=label_fontsize,labelpad=label_pad)
                plt.xlabel('PC1',fontsize=label_fontsize, labelpad=label_pad)
                ax.set_zlabel('PC3', fontsize=label_fontsize, labelpad=int(2/3*label_pad))
                plt.tick_params(axis='x', labelsize=ticks_fontsize)
                plt.tick_params(axis='y', labelsize=ticks_fontsize)
                plt.tick_params(axis='z', labelsize=ticks_fontsize)
                ax.grid()
                if show:
                    plt.show()
                if PATH:
                    fig.savefig(PATH, bbox_inches='tight')
        except:
            raise Exception('Model not fitted!')

    def plot_GA(self, PATH=False, figsize=[14,10], s=50, label_fontsize=20,
                    label_pad=18, ticks_fontsize=16, cmap='viridis', show=False):
        """ Plot dataset projected on first 2 or 3 PCs divided by Grouping Algorithm
        
        Parameters
        ----------
        PATH: str or PATH
            PATH to save figure"""
        try:
            if self.N_PCs_ == 2:
                x = self.X_projected_[:,0]
                y = self.X_projected_[:,1]
                                    
                fig = plt.figure(figsize=figsize)
                colors = self.classifiers_label_
                plt.scatter(x, y, c=colors, s=s, edgecolor='k', cmap=cmap)
                plt.ylabel('PC2',fontsize=label_fontsize,labelpad=label_pad)
                plt.xlabel('PC1',fontsize=label_fontsize, labelpad=label_pad)
                plt.tick_params(axis='x', labelsize=ticks_fontsize)
                plt.tick_params(axis='y', labelsize=ticks_fontsize)
                plt.grid()
                if show:
                    plt.show()
                if PATH:
                    fig.savefig(PATH, bbox_inches='tight')

            if self.N_PCs_ >= 3:
                x = self.X_projected_[:,0]
                y = self.X_projected_[:,1]
                z = self.X_projected_[:,2]
                                    
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')

                colors = self.classifiers_label_
                ax.scatter(x, y, z, c=colors, s=s, edgecolor='k', cmap=cmap)
                    
                plt.ylabel('PC2',fontsize=label_fontsize,labelpad=label_pad)
                plt.xlabel('PC1',fontsize=label_fontsize, labelpad=label_pad)
                ax.set_zlabel('PC3', fontsize=label_fontsize, labelpad=int(2/3*label_pad))
                plt.tick_params(axis='x', labelsize=ticks_fontsize)
                plt.tick_params(axis='y', labelsize=ticks_fontsize)
                plt.tick_params(axis='z', labelsize=ticks_fontsize)
                ax.grid()
                if show:
                    plt.show()
                if PATH:
                    fig.savefig(PATH, bbox_inches='tight')
        except:
            raise Exception('Model not fitted!')

    ### Misc

from sklearn.model_selection import train_test_split

def Lathes_train_test_split(X, y, test_size, random_state):

    
    n_measures = int(X[:,1].max())
    n_timeseries = int(X[:,0].max())
    
    index = X[::n_measures,0].astype(int) - 1
    target = y[::n_measures]

    L, W = X.shape

    train_idx, test_idx, train_target, test_target = train_test_split(index, target, test_size=test_size, 
                                                                    stratify=target, random_state=random_state)
    train_idx.sort()
    test_idx.sort()

    L_train = train_idx.shape[0]
    train_index = np.zeros(L_train*n_measures, dtype=np.int32)
    for ii in range(L_train):
        train_index[ii*n_measures:(ii+1)*n_measures] = list(range(train_idx[ii]*n_measures,
                                                                    (train_idx[ii]+1)*n_measures)) 
    L_test = test_idx.shape[0]
    test_index = np.zeros(L_test*n_measures, dtype=np.int32)
    for ii in range(L_test):
        test_index[ii*n_measures:(ii+1)*n_measures] = list(range(test_idx[ii]*n_measures,
                                                                    (test_idx[ii]+1)*n_measures)) 
        
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    return X_train, X_test, y_train, y_test
