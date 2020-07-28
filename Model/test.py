import Model_Functions as mf
import numpy as np
import pandas as pd


if __name__ == '__main__':
    Output_ID = 4

    IDs_per_group = 21

    N_PCs = 3

    min_granularity = 1
    max_granularity = 3
    pace = 0.25

    Percentage = 60

    Repeticoes = 1

    D_S_parameters = mf.DataSlicer(Output_ID, IDs_per_group, 'Main Data')

    ExtractedNames = mf.TSFRESH_Extraction(D_S_parameters)

    SelectedFeatures = mf.TSFRESH_Selection(D_S_parameters, ExtractedNames)

    ReducedFeatures = mf.PCA_calc(SelectedFeatures, N_PCs, 'Calc')

    SODA_parameters, processing_parameters = mf.SODA(ReducedFeatures,min_granularity, max_granularity, pace)

    ClassificationPar = mf.GroupingAlgorithm(SODA_parameters, Percentage, processing_parameters)

    mf.Classification(ClassificationPar, min_granularity, max_granularity, pace, Repeticoes, False)
