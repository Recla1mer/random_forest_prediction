#from locale import currency
from time import time
from turtle import position

from matplotlib.style import available
import Data_into_Matrix as DiM
import numpy as np
import os
import copy
import random

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("INFO")

from matplotlib import cm

# from matplotlib.colors import LinearSegmentedColormap

# accuracy of model
from sklearn.model_selection import train_test_split
#from sklearn.metrics import explained_variance_score, r2_score
#from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle

# Random forest regressor:
from sklearn.ensemble import RandomForestRegressor

# Extra tree regressor:
#from sklearn.ensemble import ExtraTreesRegressor

# deep learning:
#from sklearn.neural_network import MLPRegressor

# Stacking with linear:
#from sklearn.linear_model import RidgeCV
#from sklearn.svm import LinearSVR
#from sklearn.ensemble import StackingRegressor

import pickle

PREDICTIONS_DIRECTORY_NAME = "Predictions/"
VISUALIZATION_PREDICTIONS_DIRECTORY_NAME = "Pred_nice/"
PICKLE_DIRECTORY_NAME = "Pickle_data/"

#check if directory exists:
if not os.path.isdir(PICKLE_DIRECTORY_NAME):
    os.mkdir(PICKLE_DIRECTORY_NAME)
if not os.path.isdir(PREDICTIONS_DIRECTORY_NAME):
    os.mkdir(PREDICTIONS_DIRECTORY_NAME)
if not os.path.isdir(VISUALIZATION_PREDICTIONS_DIRECTORY_NAME):
    os.mkdir(VISUALIZATION_PREDICTIONS_DIRECTORY_NAME)

single_no_filter = [[[]], [], [0, 200]]
single_copper_oxides_filter = [[["Cu", "O"]], [], [0, 200]]
single_non_conventional_filter = [[["Cu", "O"], ["Fe", "As"], ["Fe", "Se"]], [], [0, 200]]

RF_standard_grid = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "criterion": "squared_error",
    "max_depth": None,
    "max_features": 1.0,
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 100,
    "n_jobs": None,
    "oob_score": False,
    "random_state": None,
    "verbose": 0,
    "warm_start": False,
}

global_filter_properties_kwargs = {
    "Path": DiM.SUPERCON_PATH,
    "with_properties": False,
    "source": DiM.suggested_source,
    "features": DiM.suggested_features,
    "filtering_arguments": single_no_filter,
    "cT_in_log_values": True,
    "epsilon": 0.005,
    "only_properties": False
}

global_predicting_kwargs = {
    "set_grid": RF_standard_grid,
    "split_test_size": 0.3,
    "ml_model": RandomForestRegressor(),
    "accuracy_model": mean_absolute_error,
    "random_split": None
}


def get_directory(string):
    """
    SIDE FUNCTION

    Tries to separate the string (path) into filename and directories. Returns directories
    if existing.

    ARGUMENTS:
    - string: (string)
    """
    position = -1
    for i in range(0, len(string)):
        if string[i] == "/":
            position = i
    if position == -1:
        return ""
    else:
        return string[0:position+1]


def get_file_name(string):
    """
    SIDE FUNCTION

    Tries to extract the filename from the given string (path).

    ARGUMENTS:
    - string: (string)
    """
    position = -1
    for i in range(0, len(string)):
        if string[i] == "/":
            position = i
        if string[i] == ".":
            return string[position+1:i]


def apply_filter_and_properties(
    Path, 
    with_properties, 
    filtering_arguments, 
    source, 
    features, 
    cT_in_log_values, 
    epsilon, 
    only_properties
    ):
    """
    SIDE FUNCTION

    DESCRIPTION:
    Function that is called in almost every following MAIN FUNCTION. It will create the
    chemical composition matrix as set by the Arguments.

    RETURNS:    
    column labels of chemical composition matrix, the matrix itself, critical temperature 
    array

    ARGUMENTS   (Settings for creating chemical composition matrix. 
                In the following they will be keyword arguments.):
    - Path: Path to SuperCon database
    - with_properties:  If "True":  Elemental properties will be added to the chemical 
                                    compositions matrix
                        If "False": The matrix consists only of chemical elements
    - filtering_arguments:  sets what kind of datapoints will be filtered out
                            see 'filter_dataset' in 'Data_into_Matrix'
    - source:   Sources that are supposed to collect elemental properties
    - features:  Source features the sources are supposed to collect
    - cT_in_log_values: If "True":  transition temperatures will be transformed to: 
                                    ln(cT + epsilon)
                        If "False": transition temperatures will be given in K
    - epsilon:  float that will be added to avoid ln(0), see "cT_in_log_values"
    - only_properties:  If "True":  The data consists only of elemental properties 
                                    (chemical compositions will be removed)
                        If "False": The data consists of chemical compositions and 
                                    elemental properties
                        ATTENTION:  Only makes sense if "with_properties" is set to "True"
    """
    if with_properties:
        ccm1, cT = DiM.create_cc_with_properties(Path, source, features)
    else:
        ccm1, cT = DiM.create_cc(Path)

    ccm = copy.deepcopy(ccm1)[1:]

    # print(len(ccm))
    ccm, cT = DiM.filter_dataset(
        ccm=copy.deepcopy(ccm),
        cT=copy.deepcopy(cT),
        filter_for_temperature=filtering_arguments[2],
        filter_out_element_combinations=filtering_arguments[0],
    )

    if cT_in_log_values:
        cT = DiM.transform_to_logarithmic(copy.deepcopy(cT), epsilon=epsilon)

    if only_properties:
        copy_ccm = copy.deepcopy(ccm)
        ccm = []
        length_ccm = len(ccm1[0])
        for i in range(0, length_ccm):
            if ccm1[0][i] in DiM.all_used_feature_names:
                props_start = i
                break
        for copy_cc in copy_ccm:
            ccm.append(copy_cc[props_start:length_ccm])
    
    return [ccm1[0]], ccm, cT


def average_accuracy(
    kwargs, 
    num_for_average, 
    ccm_for_prediction, 
    cT_for_prediction, 
    validation_set = False
    ):
    """
    SIDE FUNCTION

    DESCRIPTION:
    Function that is called in every MAIN FUNCTION which is supposed to calculate the
    accuracy of the algorithm. The accuracies have to be calculated multiple times and
    afterwards be averaged.

    If 'oob_score' in the regressor is set to True, the accuracy will be calculated
    using out of bag examples. Otherwise it will be calculated with the set accuracy model

    BACKGROUND-INFORMATION:
    The random forest regression evaluation depends on up to 2 random operations. The 
    regressor itself (Argument: 'random_state') and the 'train_test_split' 
    (Argument: 'random_state') which is needed to split the dataset into training and 
    test set to calculate the accuracy. The second can be avoided if one use the 
    'oob_score' argument from the random forest regression. These random arguments also
    have an effect on the prediction power. Therefore the evaluation has to be performed
    multiple times to average accuracy.

    RETURNS:    
    array of calculated accuracies, mean, variance, standard deviation of accuracies

    ARGUMENTS:
    - kwargs: keyword arguments used in MAIN FUNCTION
    - num_for_average: integer stating how often the accuracy will be calculated
    - ccm_for_prediction: chemical composition matrix created in MAIN FUNCTION
    - cT_for_prediction: critical temperature array created in MAIN FUNCTION

    Following ARGUMENT is needed to alter the algorithm for one specific MAIN FUNCTION:
    - validation_set:   
        if 'True':  test set will be split again into validation and test set. Accuracies,
                    mean, variance and standard deviation of validation set will be 
                    returned as well (only needed in 'hyperparameter_tuning')
        if 'False': normal functioning as explained above (default)
    """
    accuracies = []
    if validation_set:
        accuracies_validation = []

    for i in range(0, num_for_average):
        reg = kwargs["ml_model"].set_params(**kwargs["set_grid"])
        if kwargs["set_grid"]["oob_score"] == True:
            reg.fit(ccm_for_prediction, cT_for_prediction)
            accuracies.append(reg.oob_score_)
        else:
            if validation_set:
                X_train, X_split, y_train, y_split = train_test_split(
                    copy.deepcopy(ccm_for_prediction),
                    copy.deepcopy(cT_for_prediction),
                    test_size=kwargs["split_test_size"],
                    random_state=kwargs["random_split"],
                )
                X_validate, X_test, y_validate, y_test = train_test_split(
                    X_split, y_split, test_size=0.5
                )

                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_validate)
                y_test_pred = reg.predict(X_test)
                accuracies_validation.append(kwargs["accuracy_model"](y_validate, y_pred))
                accuracies.append(kwargs["accuracy_model"](y_test, y_test_pred))
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    copy.deepcopy(ccm_for_prediction),
                    copy.deepcopy(cT_for_prediction),
                    test_size=kwargs["split_test_size"],
                    random_state=kwargs["random_split"],
                )
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                accuracies.append(kwargs["accuracy_model"](y_test, y_pred))
    
    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    var_acc = np.var(accuracies)
    std_error = np.std(accuracies)
    if validation_set:
        accuracies_validation = np.array(accuracies_validation)
        mean_acc_val = np.mean(accuracies_validation)
        var_acc_val = np.var(accuracies_validation)
        std_error_val = np.std(accuracies_validation)

    if validation_set:
        return accuracies, mean_acc, var_acc, std_error, accuracies_validation, mean_acc_val, var_acc_val, std_error_val
    return accuracies, mean_acc, var_acc, std_error


def compartement(ccm, cT, number_bins):
    """
    SIDE FUNCTION

    DESCRIPTION:
    This function is given 2 arrays and an integer. ccm[i] (array of strings) corresponds 
    to cT[i] (array of integers). They will be split into bins, depending on cT[i] values. 
    Step size of bin = (Max(cT)-Min(cT))/number_bins. Afterwards the nested arrays, that 
    contain the same elements as ccm and cT before just this time divided into bins, are 
    returned

    This function is used in 'vpc'. Principal component analysis is performed there. Since
    the SuperCon dataset consists of many low temperature values, you might not see the
    higher values in the pca because low temperatures are plotted on top of them 
    (temperature is shown as color). Therefore the dataset is split into bins before, the 
    ones with higher temperature will be plotted last and be therefore on top of the low
    temperature ones.

    ARGUMENTS:
    - ccm: chemical composition matrix
    - cT: critical temperature array
    - number_bins: sets in how many different bins the dataset is split into.
    """
    max_cT = cT[0]
    min_cT = cT[0]
    for i in cT:
        if i > max_cT:
            max_cT = i
        if i < min_cT:
            min_cT = i
    difference = max_cT - min_cT
    step = difference / number_bins

    comparted_ccm = []
    comparted_cT = []
    for i in range(0, number_bins):
        comparted_ccm.append([])
        comparted_cT.append([])

    for i in range(0, len(cT)):
        current_cT = cT[i]
        for j in range(number_bins - 1, -1, -1):
            if current_cT >= (j) * step:
                comparted_ccm[j].append(ccm[i])
                comparted_cT[j].append(cT[i])
                break

    return comparted_ccm, comparted_cT


def vpc(
    number_bins=10,
    print_info=False,
    pickle_name="pca/pca_last_pickle.pkl",
    **kwargs
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    Performs principal component analysis and returns plot of 2 pca features

    CORRESPONDING PLOTTING FUNCTION in 'plot_helper': 'plot_pca'

    ARGUMENTS:
    - number_bins:  sets in how many different bins the dataset is split into. (see 
                    'compartement' for the use of this)
    - print_info:   prints additional information gained from pca to console
    - pickle_name:  path to where created data will be stored

    NEW-KEYWORD-ARGUMENTS   (Settings for machine learning.
                            In the following they will be listed with keyword arguments):
    - set_grid: (dictionary) standard grid of arguments that will be applied on regressor
    - split_test_size:  (float between 0 and 1) determines size of test + validation set 
                        in comparison to training set
    - ml_model: Machine Learning Model that shall be used
    - accuracy_model:   (function) This function will be used to calculate the accuracy of 
                        actual and predicted values
    - random_split: 'random_state' for train_test_split

    KEYWORD_ARGUMENTS (kwargs):
    - Path                         
    - with_properties               
    - source                        
    - features                      
    - filtering_arguments
    - cT_in_log_values 
    - epsilon  
    - only_properties 

    """
    #check if directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name))

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])

    ccm1, ccm, cT = apply_filter_and_properties(**kwargs)
    
    min_cT = min(cT)
    max_cT = max(cT)

    cT = np.array(cT)

    #pca = PCA(n_components=2)
    pca = PCA()
    X_r = pca.fit_transform(ccm, cT)

    # get variance from features
    if kwargs["only_properties"]:
        name_column = []
        for i in range(0, len(ccm1[0])):
            if ccm1[0][i] in DiM.all_used_feature_names:
                start_properties = i
                break
        name_column = copy.deepcopy(ccm1[0][start_properties:])
    else:
        name_column = copy.deepcopy(ccm1[0])

    #print(pca.get_feature_names_out(name_column))
    #print(pca.get_params)
    print("variance ratio component 1: " + str(pca.explained_variance_ratio_[0]))
    print("variance ratio component 2: " + str(pca.explained_variance_ratio_[1]))

    components = pca.components_

    components_one = [copy.deepcopy(components[0][0])]
    corresponding_name_one = [copy.deepcopy(name_column[0])]
    components_two = [copy.deepcopy(components[1][0])]
    corresponding_name_two = [copy.deepcopy(name_column[0])]

    components_one_abs = [copy.deepcopy(components[0][0])]
    corresponding_name_one_abs = [copy.deepcopy(name_column[0])]
    components_two_abs = [copy.deepcopy(components[1][0])]
    corresponding_name_two_abs = [copy.deepcopy(name_column[0])]

    sum_amount_one = abs(copy.deepcopy(components[0][0]))
    sum_amount_two = abs(copy.deepcopy(components[1][0]))
    sum_one = copy.deepcopy(components[0][0])
    sum_two = copy.deepcopy(components[1][0])

    for i in range(1, len(components[0])):
        appended = False
        sum_amount_one += abs(components[0][i])
        sum_one += components[0][i]
        for j in range(0, len(components_one)):
            if components[0][i] < components_one[j]:
                components_one.insert(j, components[0][i])
                corresponding_name_one.insert(j, name_column[i])
                appended = True
                break
        if appended == False:
            components_one.append(components[0][i])
            corresponding_name_one.append(name_column[i])
        appended = False
        for j in range(0, len(components_one_abs)):
            if abs(components[0][i]) < abs(components_one_abs[j]):
                components_one_abs.insert(j, components[0][i])
                corresponding_name_one_abs.insert(j, name_column[i])
                appended = True
                break
        if appended == False:
            components_one_abs.append(components[0][i])
            corresponding_name_one_abs.append(name_column[i])

    for i in range(1, len(components[1])):
        appended = False
        sum_amount_two += abs(components[1][i])
        sum_two += components[1][i]
        for j in range(0, len(components_two)):
            if components[1][i] < components_two[j]:
                components_two.insert(j, components[1][i])
                corresponding_name_two.insert(j, name_column[i])
                appended = True
                break
        if appended == False:
            components_two.append(components[1][i])
            corresponding_name_two.append(name_column[i])
        appended = False
        for j in range(0, len(components_two_abs)):
            if abs(components[1][i]) < abs(components_two_abs[j]):
                components_two_abs.insert(j, components[1][i])
                corresponding_name_two_abs.insert(j, name_column[i])
                appended = True
                break
        if appended == False:
            components_two_abs.append(components[1][i])
            corresponding_name_two_abs.append(name_column[i])
    
    if print_info:
        print("Explained variance ratio: ", end="")
        print(pca.explained_variance_ratio_)
        print("")

        print("Explained variance: ", end="")
        print(pca.explained_variance_)
        print("")

        print("Singular values: ", end="")
        print(pca.singular_values_)
        print("")

        print("number of features inside fit: ", end="")
        print(pca.n_features_in_)
        print("")
        # print(pca.feature_names_in_)

        print("Components first axis?:\n")
        for i in range(0, len(components_one)):
            print(str(corresponding_name_one[i]) + ": " + str(components_one[i]))
        print("\nSum: " + str(sum_one))
        print("Amount: " + str(sum_amount_one))
        print("\nComponents second axis?:\n")
        for i in range(0, len(components_two)):
            print(str(corresponding_name_two[i]) + ": " + str(components_two[i]))
        print("\nSum: " + str(sum_two))
        print("Amount: " + str(sum_amount_two))

        print("Components first axis (absolute value)?:\n")
        for i in range(0, len(components_one_abs)):
            print(str(corresponding_name_one_abs[i]) + ": " + str(components_one_abs[i]))
        print("\nComponents second axis (absolute value)?:\n")
        for i in range(0, len(components_two_abs)):
            print(str(corresponding_name_two_abs[i]) + ": " + str(components_two_abs[i]))

    comparted_ccm, comparted_cT = compartement(X_r, cT, number_bins)

    pickle_data_save = dict()
    pickle_data_save["ccm"] = comparted_ccm
    pickle_data_save["cT"] = comparted_cT
    pickle_data_save["min"] = min_cT
    pickle_data_save["max"] = max_cT

    pickle_data_save["ordered_comp_one"] = components_one
    pickle_data_save["ordered_name_one"] = corresponding_name_one
    pickle_data_save["ordered_comp_two"] = components_two
    pickle_data_save["ordered_name_two"] = corresponding_name_two

    pickle_data_save["ordered_comp_one_abs"] = components_one_abs
    pickle_data_save["ordered_name_one_abs"] = corresponding_name_one_abs
    pickle_data_save["ordered_comp_two_abs"] = components_two_abs
    pickle_data_save["ordered_name_two_abs"] = corresponding_name_two_abs

    pickle_data_save["evr"] = pca.explained_variance_ratio_

    with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
        pickle.dump(pickle_data_save, fid)


def best_feature_candidates(
    pickle_name = "best_features/bf_last_pickle.pkl",
    collect_features = ["O", "Cu"],
    **kwargs
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    This function will collect the feature values of the given features for the settings
    on creating the chemical composition matrix. 
    The corresponding plot function will create a histogram.

    ARGUMENTS:
    - pickle_name: path to where data will be stored
    - collect_features: list of features which values are supposed to be collected
    
    KEYWORD_ARGUMENTS (kwargs):
    - Path
    - with_properties
    - source
    - features
    - cT_in_log_values
    - epsilon
    - only_properties
    """
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name))

    #retrieve default settings:
    filter_prop_args = dict()

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])
        filter_prop_args[keys] = kwargs[keys]

    ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)

    positions = copy.deepcopy(collect_features)
    for i in range(0, len(ccm1[0])):
        for j in range(0, len(collect_features)):
            if collect_features[j] == ccm1[0][i]:
                del positions[j]
                positions.insert(j, i)
    
    best_features = {}
    temp_dep_features = {}
    for key in collect_features:
        best_features[key] = []
        temp_dep_features[key] = [[],[]]

    for i in range(0, len(cT)):
        for j in range(0, len(positions)):
            temp_dep_features[collect_features[j]][0].append(ccm[i][positions[j]])
            temp_dep_features[collect_features[j]][1].append(cT[i])
            best_features[collect_features[j]].append(ccm[i][positions[j]])
    
    pickle_data_save = dict()
    pickle_data_save["collect_features"] = best_features
    pickle_data_save["temperature"] = cT
    pickle_data_save["temp_dependence"] = temp_dep_features
    with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
        pickle.dump(pickle_data_save, fid)


def cT_distribution(
    temperature_range=[0, 200],
    filter_out_comb=[["Cu", "O"], ["Fe", "As"], ["Fe", "Se"]],
    combine_filtered=False,
    remove_filtered=False,
    pickle_name="distribution/dist_last_pickle.pkl",
    **kwargs
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    Collects distribution of temperatures depending on settings on how chemical
    composition matrix is created.
    The corresponding plot function will create histogram on distribution.

    CORRESPONDING PLOTTING FUNCTION in 'plot_helper': 'plot_distribution'

    ARGUMENTS:
    - temperature_range: determines which temperature range of values will be collected
    - filter_out_comb: combinations that will be filtered out
    - combine_filtered: collect filtered datapoints combined
    - remove_filtered: collect only remaining datapoints
    - pickle_name:  path to where created data will be stored


    KEYWORD_ARGUMENTS (kwargs):
    - Path                          - set_grid
    - with_properties               - split_test_size
    - source                        - ml_model
    - features                      - accuracy_model
    - filtering_arguments           - random_split
    - cT_in_log_values 
    - epsilon  
    - only_properties 
    """
    #check if directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name))

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])
    
    if kwargs["with_properties"]:
        DiM.create_cc_with_properties(kwargs["Path"], kwargs["source"], kwargs["features"])
    else:
        DiM.create_cc(kwargs["Path"])

    seperated_data = []

    for comb in filter_out_comb:
        DiM.filter_dataset(
            filter_for_temperature=[0, 200], filter_out_element_combinations=[comb]
        )
        file = open(
            DiM.CHANGING_DATA_DIRECTORY_NAME + DiM.FILTERED_COMPOSITIONS_FILE_NAME,
            "r",
        )
        filtered_lines = file.readlines()
        seperated_data.append(filtered_lines)
        file.close()

    if remove_filtered:
        del seperated_data
        seperated_data = []
    if combine_filtered:
        copy_seperated = copy.deepcopy(seperated_data)
        del seperated_data
        seperated_data = []
        for data in copy_seperated:
            for datapoint in data:
                seperated_data.append(datapoint)
        seperated_data = [seperated_data]

    file = open(DiM.CHANGING_DATA_DIRECTORY_NAME + DiM.USED_DATA_FILE_NAME, "r")
    used_lines = file.readlines()
    seperated_data.append(used_lines)
    file.close()

    max_cT = temperature_range[1]
    min_cT = temperature_range[0]
    cT = []
    for i in range(0, len(seperated_data)):
        this_cT = []
        for j in range(0, len(seperated_data[i])):
            cc = DiM.correct_ending(seperated_data[i][j])
            current_cT = DiM.extract_cT(cc)
            if min_cT <= current_cT and current_cT <= max_cT:
                this_cT.append(current_cT)
        cT.append(this_cT)

    if kwargs["cT_in_log_values"]:
        cT_copy = copy.deepcopy(cT)
        del cT
        cT = []
        for i in range(0, len(seperated_data)):
            this_cT = DiM.transform_to_logarithmic(
                copy.deepcopy(cT_copy[i]), epsilon=kwargs["epsilon"]
            )
            cT.append(this_cT)

    pickle_data_save = dict()
    pickle_data_save["cT"] = cT
    pickle_data_save["min"] = min_cT
    with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
        pickle.dump(pickle_data_save, fid)


def ML_feature_importance(
    pickle_name="ML_feat_importance/mlfi_last_pickle.pkl",
    num_for_average=10,
    **kwargs
    ):
    """
    """
    #check if directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name))

    #retrieve default settings
    filter_prop_args = dict()

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])
        filter_prop_args[keys] = kwargs[keys]
    for keys in global_predicting_kwargs:
        kwargs.setdefault(keys, global_predicting_kwargs[keys])

    ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)
        
    ccm_for_prediction = np.array(ccm)
    cT_for_prediction = np.array(cT)

    feature_importances = []
    for i in range(0, num_for_average):
        reg = kwargs["ml_model"].set_params(**kwargs["set_grid"])
        if kwargs["set_grid"]["oob_score"] == True:
            reg.fit(ccm_for_prediction, cT_for_prediction)
            feature_importances.append(reg.feature_importances_)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                copy.deepcopy(ccm_for_prediction),
                copy.deepcopy(cT_for_prediction),
                test_size=kwargs["split_test_size"],
                random_state=kwargs["random_split"],
            )
            reg.fit(X_train, y_train)
            feature_importances.append(reg.feature_importances_)

    mean_feat_imp = np.mean(feature_importances, axis=0)
    std_feat = np.std(feature_importances, axis=0)

    ordered_imp = [mean_feat_imp[0]]
    ordered_std = [std_feat[0]]
    ordered_names = [ccm1[0][0]]

    for i in range(1, len(mean_feat_imp)):
        appended = False
        for j in range(0, len(ordered_imp)):
            if mean_feat_imp[i] < ordered_imp[j]:
                appended = True
                ordered_imp.insert(j, mean_feat_imp[i])
                ordered_std.insert(j, std_feat[i])
                ordered_names.insert(j, ccm1[0][i])
                break
        if not appended:
            ordered_imp.append(mean_feat_imp[i])
            ordered_std.append(std_feat[i])
            ordered_names.append(ccm1[0][i])

    pickle_data_save = dict()
    pickle_data_save["feat_imp"] = ordered_imp
    pickle_data_save["std_feat"] = ordered_std
    pickle_data_save["names"] = ordered_names
    with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
        pickle.dump(pickle_data_save, fid)


"""

Hyperparameter tuning

"""


def continuous_hp_evaluation(
    search_grid,
    pickle_name="hp_visualize/hpv_last_pickle.pkl",
    print_progress_to_console = False,
    num_for_average=10,
    **kwargs
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    Collects data of accuracy depending on a hyperparameter which has continuos setting
    interval, for example: (0,1.0]

    ARGUMENTS:
    - search_grid:  (dictionary) hyperparameters and values that shall be checked
    - pickle_name:  path to where created data will be stored
    - print_progress_to_console:    if 'True' print progress report to console 
                                    (since it might take a while)
    - num_for_average:  integer stating how often the accuracy will be calculated for 
                        every case

    KEYWORD_ARGUMENTS (kwargs):
    - Path                          - set_grid
    - with_properties               - split_test_size
    - source                        - ml_model
    - features                      - accuracy_model
    - filtering_arguments           - random_split
    - cT_in_log_values 
    - epsilon  
    - only_properties 
    """
    #check if directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name))

    #retrieve default settings
    filter_prop_args = dict()

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])
        filter_prop_args[keys] = kwargs[keys]
    for keys in global_predicting_kwargs:
        kwargs.setdefault(keys, global_predicting_kwargs[keys])

    ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)
        
    ccm_for_prediction = np.array(ccm)
    cT_for_prediction = np.array(cT)

    hyperparameters = []
    hyper_values = []
    acc_params = {}
    time_params = {}
    std_params = {}
    var_params = {}

    for key in search_grid:
        hyperparameters.append(key)
        hyper_values.append(search_grid[key])
        acc_params[key] = []
        time_params[key] = []
        std_params[key] = []
        var_params[key] = []

    for i in range(0, len(hyperparameters)):
        current_grid = copy.deepcopy(kwargs["set_grid"])

        if print_progress_to_console:
            print(
                "\nPROGRESS - REPORT: Calculating the accuracys for "
                + hyperparameters[i]
            )
        
        progress_counter = 0

        for value in hyper_values[i]:
            progress_counter += 1
            current_grid[hyperparameters[i]] = value

            start = time()
            accuracies, mean_acc, var_acc, std_error = average_accuracy(
                kwargs, 
                num_for_average, 
                ccm_for_prediction, 
                cT_for_prediction, 
                use_grid=current_grid
                )
            end = time()

            this_comp_time = (end - start)/num_for_average

            acc_params[hyperparameters[i]].append(mean_acc)
            time_params[hyperparameters[i]].append(this_comp_time)
            var_params[hyperparameters[i]].append(var_acc)
            std_params[hyperparameters[i]].append(std_error)

            if print_progress_to_console:
                print(
                    "PROGRESS - REPORT: "
                    + str(progress_counter)
                    + " of "
                    + str(len(hyper_values[i]))
                    + " values were checked"
                )

        pickle_data_save = dict()
        pickle_data_save["hyper_val"] = hyper_values
        pickle_data_save["hyper_para"] = hyperparameters
        pickle_data_save["acc"] = acc_params
        pickle_data_save["time"] = time_params
        pickle_data_save["standard_deviation"] = std_params
        pickle_data_save["variance"] = var_params

        with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
            pickle.dump(pickle_data_save, fid)


def hyperparameter_tuning(
    search_grid,
    print_results_to_console=False,
    pickle_name="hp_settings/last_hp_tune.pkl",
    num_for_average=10,
    **kwargs
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    This function performs hyperparameter tuning on the given 'search_grid'. Meaning it
    will check every combination of hyperparameters inside. The best set of 
    hyperparameters as well as all settings made by 'kwargs' is stored to pickle file.

    ARGUMENTS:
    - search_grid:  (dictionary) hyperparameters and values that shall be checked
    - print_results_to_console: If True: prints additional information to console, like
                                accuracy and settings of all combinations
    - pickle_name:  path to where created data will be stored
    - num_for_average: integer stating how often the accuracy will be calculated
    
    KEYWORD_ARGUMENTS (kwargs):
    - Path                          - set_grid
    - with_properties               - split_test_size
    - source                        - ml_model
    - features                      - accuracy_model
    - filtering_arguments           - random_split
    - cT_in_log_values 
    - epsilon  
    - only_properties 
    """
    #check if directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name))

    #retrieve default settings
    filter_prop_args = dict()

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])
        filter_prop_args[keys] = kwargs[keys]
    for keys in global_predicting_kwargs:
        kwargs.setdefault(keys, global_predicting_kwargs[keys])
    default_ml_model = copy.deepcopy(kwargs["ml_model"])

    ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)

    ccm_for_prediction = np.array(ccm)
    cT_for_prediction = np.array(cT)

    hyperparameters = []
    hyper_values = []
    acc_params = {}  # tracks settings of params for accuracies

    for key in search_grid:
        hyperparameters.append(key)
        hyper_values.append(search_grid[key])

    for key in hyperparameters:
        acc_params[key] = []
        if key in kwargs["set_grid"]:
            kwargs["set_grid"].pop(key)

    hyper_positions = []
    current_positions = []
    for values in hyper_values:
        hyper_positions.append(len(values) - 1)
        current_positions.append(0)

    break_loop = False
    accuracies = []
    test_set_accuracies = []

    while True:
        for i in range(0, len(hyperparameters)):
            kwargs["set_grid"][hyperparameters[i]] = hyper_values[i][current_positions[i]]

        av_acc_args = {
            "kwargs": kwargs, 
            "num_for_average": num_for_average, 
            "ccm_for_prediction": ccm_for_prediction,
            "cT_for_prediction": cT_for_prediction
            }

        if kwargs["set_grid"]["oob_score"] == True:
            mean_acc = average_accuracy(**av_acc_args)[1]
            mean_acc_val = 0
        else:
            all = average_accuracy(validation_set=True, **av_acc_args)
            mean_acc = all[1]
            mean_acc_val = all[5]

        accuracy_appended = False
        for i in range(0, len(accuracies)):
            if mean_acc < accuracies[i]:
                accuracy_appended = True
                accuracies.insert(i, mean_acc)
                test_set_accuracies.insert(i, mean_acc_val)
                for j in range(0, len(hyperparameters)):
                    acc_params[hyperparameters[j]].insert(
                        i, hyper_values[j][current_positions[j]]
                    )
                break
        if not accuracy_appended:
            accuracies.append(mean_acc)
            test_set_accuracies.append(mean_acc_val)
            for j in range(0, len(hyperparameters)):
                acc_params[hyperparameters[j]].append(
                    hyper_values[j][current_positions[j]]
                )
        position = 0
        
        while True:
            if position > len(current_positions) - 1:
                break_loop = True
                break
            if current_positions[position] < hyper_positions[position]:
                current_positions[position] += 1
                for j in range(0, position):
                    current_positions[j] = 0
                break
            else:
                position += 1
        if break_loop:
            break

    if print_results_to_console:
        print("\nAll calculated accuracies in order:")
        print(accuracies)
        if not kwargs["set_grid"]["oob_score"]:
            print("\nCorresponding accuracies of the unbiased test set:")
            print(test_set_accuracies)
        print("\nCorresponding hyperparameters and their values:")
        print(acc_params)
        print("")
        print(
            "Highest calculated value of accuracy: "
            + str(accuracies[len(accuracies) - 1])
        )
        if not kwargs["set_grid"]["oob_score"]:
            print(
                "Accuracy on unbiased test set: "
                + str(test_set_accuracies[len(accuracies) - 1])
            )
        print("Corresponding hyperparameters:")
        for i in range(0, len(hyperparameters)):
            print(hyperparameters[i], end=" = ")
            print(acc_params[hyperparameters[i]][len(accuracies) - 1])

        print("")
        print("Lowest calculated value of accuracy: " + str(accuracies[0]))
        if not kwargs["set_grid"]["oob_score"]:
            print("Accuracy on unbiased test set: " + str(test_set_accuracies[0]))
        print("Corresponding hyperparameters:")
        for i in range(0, len(hyperparameters)):
            print(hyperparameters[i], end=" = ")
            print(acc_params[hyperparameters[i]][0])

    #evaluate if high or low value of "accuracy" is good:
    if kwargs["set_grid"]["oob_score"] == True:
        return_highest_acc = True
    else:
        nice_accuracy = kwargs["accuracy_model"]([1, 1, 1], [0.99, 0.99, 0.99])
        bad_accuracy = kwargs["accuracy_model"]([1, 1, 1], [0.01, 0.01, 0.01])
        if nice_accuracy - bad_accuracy > 0:
            return_highest_acc = True
        else:
            return_highest_acc = False

    if return_highest_acc:
        for i in range(0, len(hyperparameters)):
            kwargs["set_grid"][hyperparameters[i]] = acc_params[hyperparameters[i]][
                len(accuracies) - 1
            ]
            kwargs["ml_model"]=default_ml_model

        with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
            pickle.dump(kwargs, fid)

        if kwargs["set_grid"]["oob_score"] == True:
            return kwargs["set_grid"], accuracies[len(accuracies) - 1]
        else:
            return kwargs["set_grid"], test_set_accuracies[len(accuracies) - 1]
    else:
        for i in range(0, len(hyperparameters)):
            kwargs["set_grid"][hyperparameters[i]] = acc_params[hyperparameters[i]][0]
        kwargs["ml_model"]=default_ml_model

        with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
            pickle.dump(kwargs, fid)

        return kwargs["set_grid"], test_set_accuracies[0]


def create_average_with_shuffle(
    X_train, y_train, ml_model, set_grid, accuracy_model, num_cal, size, 
    num_for_average, X_test = None, y_test = None
    ):
    """
    SIDE FUNCTION

    ARGUMENTS:
    - X_train, y_train: training data
    - size: how much of training data to be used
    - num_cal: how often to calculate accuracy for given size
    - X_test, y_test: test data: might be 'None' if 'oob_score' is supposed to be used.

    KNOWN ARGUMENTS (see documentations above):
    - ml_model
    - set_grid
    - accuracy_model
    - num_for_average

    DESCRIPTION:
    Averages the accuracy for each training size in "accuracy_training_set_size_plot".
    Since the power of the algorithm depends on datapoints one has to calculate the
    accuracy (which is also calculated multiple times, see 'average_accuracy') multiple 
    times for different datasets.
    """
    final_accuracies = []
    final_std = []
    final_var = []
    for i in range(0, num_cal):
        X_train, y_train = shuffle(X_train, y_train, random_state=None)
        accuracies = []

        for j in range(0, num_for_average):
            reg = ml_model.set_params(**set_grid)
            reg.fit(X_train[0:size], y_train[0:size])
            try:
                if X_test == None or y_test == None:
                    accuracies.append(reg.oob_score_)
            except:
                y_pred = reg.predict(X_test)
                accuracies.append(accuracy_model(y_test, y_pred))
        
        accuracies = np.array(accuracies)
        final_accuracies.append(np.mean(accuracies))
        final_var.append(np.var(accuracies))
        final_std.append(np.std(accuracies))
    
    final_accuracies = np.array(final_accuracies)
    final_std = np.array(final_std)
    final_var = np.array(final_var)

    mean_acc = np.mean(final_accuracies)
    std_err = np.std(final_accuracies) + np.mean(final_std)
    var_acc = np.var(final_accuracies) + np.mean(final_var)

    return mean_acc, std_err, var_acc


def accuracy_training_size(
    split_ratio=0.1,
    num_cal_for_averaging=10,
    pickle_name="accuracy_size/as_last_pickle.pkl",
    num_for_average=10,
    **kwargs
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    This function collects accuracy depending on size of training set for made settings
    on creating chemical composition matrix and machine learning model.

    ARGUMENTS:
    - split_ratio: (float between 0 and 1) sets stepsize depending on size of training 
                    dataset
    - num_cal_for_averaging:    determines how many different sets of the reduced training 
                                set will be used to calculate accuracy, final accuracy for 
                                each size will be the mean value them (see
                                'create_average_with_shuffle' for the use of that)
    - pickle_name:  path to where created data will be stored
    - num_for_average: integer stating how often the accuracy will be calculated
    
    KEYWORD_ARGUMENTS (kwargs):
    - Path                          - set_grid
    - with_properties               - split_test_size
    - source                        - ml_model
    - features                      - accuracy_model
    - filtering_arguments           - random_split
    - cT_in_log_values 
    - epsilon  
    - only_properties 
    """
    #check if directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name))

    #retrieve default settings
    filter_prop_args = dict()

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])
        filter_prop_args[keys] = kwargs[keys]
    for keys in global_predicting_kwargs:
        kwargs.setdefault(keys, global_predicting_kwargs[keys])

    ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)

    ccm_for_prediction = np.array(ccm)
    cT_for_prediction = np.array(cT)

    if kwargs["set_grid"]["oob_score"] == True:
        size = len(ccm_for_prediction)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            ccm_for_prediction,
            cT_for_prediction,
            test_size=kwargs["split_test_size"],
            random_state=kwargs["random_split"],
        )
        size = len(X_train)

    accuracies = []
    standard_deviations = []
    variances = []
    training_size = []

    step_size = size * split_ratio
    current_size = int(step_size)
    size_reached = False

    while current_size <= size:
        shuffle_data = {
            "ml_model": kwargs["ml_model"],
            "set_grid": kwargs["set_grid"],
            "accuracy_model": kwargs["accuracy_model"],
            "num_cal": num_cal_for_averaging,
            "size": current_size,
            "num_for_average": num_for_average
            }
        if kwargs["set_grid"]["oob_score"] == True:
            mean_acc, std_err, var_acc = create_average_with_shuffle(
                X_train = copy.deepcopy(ccm_for_prediction),
                y_train = copy.deepcopy(cT_for_prediction),
                **shuffle_data
                )
        else:
            mean_acc, std_err, var_acc = create_average_with_shuffle(
                X_train = copy.deepcopy(X_train),
                y_train = copy.deepcopy(y_train),
                X_test = copy.deepcopy(X_test),
                y_test = copy.deepcopy(y_test),
                **shuffle_data
                )
        accuracies.append(mean_acc)
        standard_deviations.append(std_err)
        variances.append(var_acc)
        training_size.append(current_size)

        current_size += step_size
        current_size = int(current_size) + 1
        if current_size > size and not size_reached:
            size_reached = True
            current_size = size

    pickle_data_save = dict()
    pickle_data_save["training"] = training_size
    pickle_data_save["acc"] = accuracies
    pickle_data_save["standard_deviation"] = standard_deviations
    pickle_data_save["variance"] = variances
    with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
        pickle.dump(pickle_data_save, fid)


def predicted_actual(
    pickle_name="predicted_actual/pa_last_pickle.pkl",
    num_for_average=10,
    **kwargs
    ):
    """ 
    MAIN FUNCTION

    DESCRIPTION:
    Creates data to visualise accuracy. Predicted and actual value will be calculated
    depending on made settings on creating the chemical composition matrix and the 
    machine learning model.

    ARGUMENTS:
    - pickle_name:  path to where created data will be stored
    - num_for_average: integer stating how often the accuracy will be calculated

    KEYWORD_ARGUMENTS (kwargs):
    - Path                          - random_split
    - with_properties               - split_test_size
    - source                        - ml_model
    - features                      - accuracy_model
    - cT_in_log_values 
    - epsilon  
    - only_properties 
    """
    #check if directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + get_directory(pickle_name))

    #retrieve default settings:
    filter_prop_args = dict()

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])
        filter_prop_args[keys] = kwargs[keys]
    for keys in global_predicting_kwargs:
        kwargs.setdefault(keys, global_predicting_kwargs[keys])

    ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)

    ccm_for_prediction = np.array(ccm)
    cT_for_prediction = np.array(cT)

    X_train, X_test, y_train, y_test = train_test_split(
        copy.deepcopy(ccm_for_prediction),
        copy.deepcopy(cT_for_prediction),
        test_size=kwargs["split_test_size"],
        random_state=kwargs["random_split"],
    )

    all_predicted_values = []
    for j in range(0, num_for_average):
        reg = kwargs["ml_model"].set_params(**kwargs["set_grid"])
        reg.fit(X_train, y_train)
        all_predicted_values.append(reg.predict(X_test))

    #print(all_predicted_values[0], all_predicted_values[1])
    all_predicted_values = np.array(all_predicted_values)
    mean_pred = np.mean(all_predicted_values, axis=0)
    var_pred = np.var(all_predicted_values, axis=0)
    std_error = np.std(all_predicted_values, axis=0)
        
    non_numpy = []
    for entry in mean_pred:
        non_numpy.append(entry)
    predicted_values = copy.deepcopy(non_numpy)
    non_numpy = []
    for entry in var_pred:
        non_numpy.append(entry)
    var_values = copy.deepcopy(non_numpy)
    non_numpy = []
    for entry in std_error:
        non_numpy.append(entry)
    std_values = copy.deepcopy(non_numpy)
    non_numpy = []
    for entry in y_test:
        non_numpy.append(entry)
    actual_values = copy.deepcopy(non_numpy)

    pickle_data_save = dict()
    pickle_data_save["pred"] = predicted_values
    pickle_data_save["actual"] = actual_values
    pickle_data_save["standard_deviation"] = std_values
    pickle_data_save["variance"] = var_values
    with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
        pickle.dump(pickle_data_save, fid)


"""

Predictions

"""


def accuracy_for_current_grid(num_for_average=10, use_cross_val=False, **kwargs):
    """ 
    MAIN FUNCTION

    DESCRIPTION:
    Quick evaluation of accuracy depending on made settings for creating chemical 
    composition matrix and machine learning model. Prints result to console.

    ARGUMENTS:
    - num_for_average: integer stating how often the accuracy will be calculated
    - use_cross_val: If 'True': Use cross validation for determing accuracy

    KEYWORD_ARGUMENTS (kwargs):
    - Path                          - set_grid
    - with_properties               - split_test_size
    - source                        - ml_model
    - features                      - accuracy_model
    - filtering_arguments           - random_split
    - cT_in_log_values 
    - epsilon  
    - only_properties 
    """
    #retrieve default settings:
    filter_prop_args = dict()

    for keys in global_filter_properties_kwargs:
        kwargs.setdefault(keys, global_filter_properties_kwargs[keys])
        filter_prop_args[keys] = kwargs[keys]
    for keys in global_predicting_kwargs:
        kwargs.setdefault(keys, global_predicting_kwargs[keys])

    ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)

    ccm_for_prediction = np.array(ccm)
    cT_for_prediction = np.array(cT)

    if use_cross_val:
        reg_model = kwargs["ml_model"](**kwargs["set_grid"])
        return cross_val_score(reg_model, ccm_for_prediction, cT_for_prediction)
        """
        return cross_val_score(
            kwargs["ml_model"], 
            ccm_for_prediction, 
            cT_for_prediction, 
            fit_params=kwargs["set_grid"]
            )
        """
    else:
        accuracies, mean_acc, var_acc, std_error = average_accuracy(
            kwargs, 
            num_for_average,
            ccm_for_prediction,
            cT_for_prediction
        )

        return accuracies, mean_acc, var_acc, std_error


def remove_mean_from_dataset(list):
    """
    SIDE FUNCTION

    DESCRIPTION:
    'make_predictions' creates text file with promising predictions in the style of
    the SuperCon database file. Additionally it adds ';' after the transition
    temperature where standard deviation and variance follows. This function removes them.

    ARGUMENTS:
    -  list: array that holds lines of text file
    """
    new_list = []
    for element in list:
        this_appended = False
        for i in range(0, len(element)):
            if element[i] == ";":
                this_appended = True
                new_list.append(element[0:i])
                break
        if not this_appended:
            new_list.append(element)
    
    return new_list
                

def remove_all_but(list, string="pkl"):
    """
    SIDE FUNCTION

    DESCRIPTION:
    This function is given a list of strings which will represent file names. All
    files that are not of type: 'string' will be removed

    ARGUMENTS:
    - list: list of strings
    - string: (string)
    """
    remove_el = []
    for element in list:
        for char in range(0, len(element)):
            if element[char] == ".":
                if element[char+1:len(element)] != string:
                    remove_el.append(element)
                break
    for el in remove_el:
        list.remove(el)
    return list


def get_real_mean_from_log(mean_arr, std_arr, epsilon):
    """
    SIDE FUNCTION

    DESCRIPTION:
    If transition temperatures where predicted in log values and you want to transform
    them back you will also want to transform standard deviation and variation back.
    This function takes care of that.

    ARGUMENTS:
    - mean_arr: array that holds mean values
    - std_arr: array that holds standard deviation values

    KNOWN ARGUMENTS:
    - epsilon:
    """
    new_mean_arr = []
    new_std_arr = []
    new_var = []
    for i in range(0, len(mean_arr)):
        low = DiM.transform_to_euler([mean_arr[i]-std_arr[i]], epsilon=epsilon)[0]
        high = DiM.transform_to_euler([mean_arr[i]+std_arr[i]], epsilon=epsilon)[0]
        this_new_std = (high-low)/2
        this_new_mean = low + this_new_std
        new_mean_arr.append(this_new_mean)
        new_std_arr.append(this_new_std)
        new_var.append(this_new_std**2)
    
    return new_mean_arr, new_std_arr, new_var


def visualisation_of_prediction(path_name, names, predictions, std_error, var):
    """
    SIDE FUNCTION

    DESCRIPTION:
    This function is supposed to put the predictions again into a file. By doing that it 
    will also order them after their transition temperature and print them in a nice way.

    ARGUMENTS:
    - path_name: (string) path to file that stores contents
    - names: (string) chemical compositions in string form
    - predictions: (list of floats) predicted mean of transition temperatures
    - std_error: (list of floats) standard deviation of transition temperatures
    - var: (list of floats) variance of transition temperatures
    """
    predictions_file = open(
        VISUALIZATION_PREDICTIONS_DIRECTORY_NAME + path_name + "_vis.txt", "w"
        )
    
    ordered_pred = [predictions[0]]
    ordered_std = [std_error[0]]
    ordered_var = [var[0]]
    ordered_names = [names[0]]

    for i in range(1, len(predictions)):
        this_appended = False
        for j in range(0, len(ordered_pred)):
            if predictions[i] < ordered_pred[j]:
                this_appended = True
                ordered_pred.insert(j, predictions[i])
                ordered_std.insert(j, std_error[i])
                ordered_var.insert(j, var[i])
                ordered_names.insert(j, names[i])
                break
        if not this_appended:
            ordered_pred.append(predictions[i])
            ordered_std.append(std_error[i])
            ordered_var.append(var[i])
            ordered_names.append(names[i])
        
    for i in range(len(ordered_pred)-1, -1, -1):
        predictions_file.write(ordered_names[i] + ", ( " + str(round(ordered_pred[i], 3)) + " +- " + str(round(ordered_std[i], 3)) + " )\n")
    
    predictions_file.close()


def make_predictions(
    data_location, 
    setting_location, 
    promising_cT=10, 
    add_documentation=True, 
    print_progression_to_console=True,
    num_for_average=10,
    create_pickle = True,
    revert_log = True,
    pickle_name = "Predictions"
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    Collects data created by:   
    - DiM.create_pickle_predictors (provides chemical composition matrix of chemical
                                    compositions which transition temperature shall be 
                                    predicted)
    - hyperparameter_tuning (provides settings on creating the chemical composition matrix
                            of training data and machine learning model)
    
    It will either predict all chemical compositions in a single file or in every
    file inside a directory FOR every setting in either a file or all files inside
    a directory. Results of transition temperatures above 'promising_cT' will be written 
    to a text file. All temperatures as well as feature values of 'collect_features'
    will be written to a pickle file as in 'best_feature_candidates'.

    ARGUMENTS:
    - data_location: path to file or directory where chemical compositions are stored
    - setting_location: path to file or directory where settings are stored
    - add_documentation: If 'True': add additional informations to text file
    - print_progression_to_console: If 'True': print progression to console
    - num_for_average: (integer > 0) how often each chemical composition will be predicted
    - revert_log: If 'True': If transition temperatures were created in log values they
                            will be transformed back.
    - collect_features: values of which features will be collected
    - pickle_name: path to directory where pickle file will be stored
    """
    #check if pickle directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + pickle_name):
        os.mkdir(PICKLE_DIRECTORY_NAME + pickle_name)

    #check save location existence:
    if not os.path.isdir(PREDICTIONS_DIRECTORY_NAME):
        os.mkdir(PREDICTIONS_DIRECTORY_NAME)
    
    #check if multiple files are suppposed to be predicted:
    if os.path.isdir(data_location):
        data_files = os.listdir(data_location)
        data_directory = data_location + "/"
    else:
        data_files = [data_location]
        data_directory = ""
    
    #check if predictions are supposed to be made with multiple settings:
    if os.path.isdir(setting_location):
        kwargs_files = os.listdir(setting_location)
        kwargs_directory = setting_location + "/"
    else:
        kwargs_files = [setting_location]
        kwargs_directory = ""

    data_files = remove_all_but(copy.deepcopy(data_files))
    kwargs_files = remove_all_but(copy.deepcopy(kwargs_files))
    
    progression_marker_data = 0
    for data_path in data_files:
        progression_marker_data += 1

        #load overfull predicting matrix and strings
        with open(data_directory + data_path, "rb") as fid:
            pickle_data_loaded = pickle.load(fid)
        
        all_strings = pickle_data_loaded["string"]
        all_matrix = pickle_data_loaded["matrix"]
        all_element_order = pickle_data_loaded["order"]
        del pickle_data_loaded

        #acces file to write predictions:
        data_path_name = get_file_name(data_path)
        predictions_file = open(
            PREDICTIONS_DIRECTORY_NAME + data_path_name + ".txt", "w"
            )

        progression_marker_setting = 0

        promising_predicted_values = []
        promising_std_errors = []
        promising_var = []
        promising_names = []

        for kwarg_path in kwargs_files:
            with open(kwargs_directory + kwarg_path, "rb") as fid:
                pickle_data_loaded = pickle.load(fid)
            
            kwargs = pickle_data_loaded
            kwargs["random_split"]=random.randint(0, 100)

            filter_prop_args = dict()
            for keys in global_filter_properties_kwargs:
                filter_prop_args[keys] = kwargs[keys]

            ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)

            ccm_for_prediction = np.array(ccm)
            cT_for_prediction = np.array(cT)
            
            #delete unneccessary columns and rows in predicting matrix
            copy_matrix = np.array(copy.deepcopy(all_matrix))
            copy_strings = copy.deepcopy(all_strings)
            copy_element_order = copy.deepcopy(all_element_order)

            not_in = []
            for i in range(0, len(copy_element_order)):
                if copy_element_order[i] not in ccm1[0]:
                    if copy_element_order[i] not in DiM.all_used_feature_names:
                        not_in.append(copy_element_order[i])

            wrong_element_pos = []
            for i in range(0, len(copy_strings)):
                current_elements = DiM.extract_elements(copy_strings[i])
                for el in current_elements:
                    if el in not_in:
                        wrong_element_pos.append(i)
                        break

            copy_matrix = np.delete(copy_matrix, wrong_element_pos, 0)
            for index in sorted(wrong_element_pos, reverse=True):
                del copy_strings[index]

            deletion_counter = 0
            for i in range(0, len(copy_element_order)):
                if copy_element_order[i-deletion_counter] not in ccm1[0]:
                    del copy_element_order[i-deletion_counter]
                    copy_matrix = np.delete(copy_matrix, i-deletion_counter, 1)
                    deletion_counter += 1

            if len(copy_matrix) != len(copy_strings):
                print("\n\nPROBLEM: matrix not as long as strings\n\n")
            if len(copy_matrix[0]) != len(ccm1[0]):
                print("\n\nPROBLEM: matrixes are not equal")
            
            for i in range(0, len(ccm1[0])):
                if copy_element_order[i] != ccm1[0][i]:
                    for j in range(i+1, len(copy_element_order)):
                        if copy_element_order[j] == ccm1[0][i]:
                            copy_matrix[:, [j,i]] = copy_matrix[:, [i,j]]
                            cop_this = copy_element_order[i]
                            copy_element_order[i] = copy_element_order[j]
                            copy_element_order[j] = cop_this
            
            #alter append condition
            if kwargs["cT_in_log_values"] == True:
                write_condition = DiM.transform_to_logarithmic([copy.deepcopy(promising_cT)], 
                    epsilon=kwargs["epsilon"]
                    )[0]
            else:
                write_condition = copy.deepcopy(promising_cT)

            #predict matrix:
            made_predictions = []

            for i in range(0, num_for_average):
                kwargs["ml_model"] = RandomForestRegressor()
                reg = kwargs["ml_model"].set_params(**kwargs["set_grid"])
                if kwargs["set_grid"]["oob_score"] == True:
                    reg.fit(ccm_for_prediction, cT_for_prediction)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        ccm_for_prediction,
                        cT_for_prediction,
                        test_size=kwargs["split_test_size"],
                        random_state=kwargs["random_split"],
                    )
                    reg.fit(X_train, y_train)

                #y_pred = reg.predict(copy_matrix)
                made_predictions.append(reg.predict(copy_matrix))

            made_predictions = np.array(made_predictions)

            y_pred = np.mean(made_predictions, axis=0)
            var_pred = np.var(made_predictions, axis=0)
            std_error = np.std(made_predictions, axis=0)

            setting_name = get_file_name(kwarg_path)

            if create_pickle:
                pickle_data_save = dict()
                pickle_data_save["strings"] = copy_strings
                pickle_data_save["pred"] = y_pred
                pickle_data_save["standard_deviation"] = std_error
                pickle_data_save["variance"] = var_pred
                with open(PICKLE_DIRECTORY_NAME + pickle_name + "/" + data_path_name + "_" + setting_name + ".pkl", "wb") as fid:
                    pickle.dump(pickle_data_save, fid)

            if add_documentation:
                predictions_file.write(setting_name + ":\n\n")
                predictions_file.write(str(kwargs["set_grid"]))
                predictions_file.write("\n\n")
            
            count_promising = 0

            if kwargs["cT_in_log_values"] == True and revert_log:
                for k in range(0, len(y_pred)):
                    if (y_pred[k] + std_error[k]) >= write_condition:
                        #print(y_pred[k] + std_error[k])
                        count_promising += 1

                        promising_predicted_values.append(y_pred[k])
                        promising_std_errors.append(std_error[k])
                        promising_var.append(var_pred[k])
                        promising_names.append(copy_strings[k])

                        predictions_file.write(copy_strings[k] + ","
                            + str(round(DiM.transform_to_euler([copy.deepcopy(y_pred[k])], 
                                epsilon=kwargs["epsilon"])[0], 4,
                                )
                            ) + ";" 
                            + str(round(DiM.transform_to_euler([copy.deepcopy(std_error[k])], 
                                epsilon=kwargs["epsilon"])[0], 4,
                                )
                            ) + ";" + 
                            str(round(DiM.transform_to_euler([copy.deepcopy(var_pred[k])], 
                                epsilon=kwargs["epsilon"])[0], 4,
                                )
                            ) + "\n"
                        )
            else:
                for k in range(0, len(y_pred)):
                    if (y_pred[k] + std_error[k]) >= write_condition:
                        #print(y_pred[k] + std_error[k])
                        count_promising += 1

                        promising_predicted_values.append(y_pred[k])
                        promising_std_errors.append(std_error[k])
                        promising_var.append(var_pred[k])
                        promising_names.append(copy_strings[k])

                        predictions_file.write(copy_strings[k] + ","
                            + str(round(y_pred[k], 4)) + ";"
                            + str(round(std_error[k], 4)) + ";"
                            + str(round(var_pred[k], 4)) + "\n"
                        )

            if add_documentation:
                predictions_file.write("\n" + str(count_promising) + " of "
                    + str(len(y_pred)) + " appear to be promising\n\n"
                )
            
            if print_progression_to_console:
                print("PROGRESSION-REPORT: " + str(progression_marker_setting) + " of " 
                    + str(len(kwargs_files)) + " predicting models did their work"
                    )
        
        predictions_file.close()

        if kwargs["cT_in_log_values"] == True and revert_log:
            promising_predicted_values, promising_std_errors, promising_var = get_real_mean_from_log(
                copy.deepcopy(promising_predicted_values),
                copy.deepcopy(promising_std_errors),
                kwargs["epsilon"]
            )

        visualisation_of_prediction(
            data_path_name, 
            promising_names, 
            promising_predicted_values, 
            promising_std_errors, 
            promising_var
        )
        
        if print_progression_to_console:
            print("PROGRESSION-REPORT: " + str(progression_marker_data) + " of " + str(len(data_files)) 
                + " files of data were predicted")


def best_predicted(Path_to_predicted):
    """ 
    SIDE FUNCTION

    DESCRIPTION:
    Does the main work for 'analyse_predicted'

    ARGUMENTS:
    - Path_to_predicted: (string) Path to where predictions are stored
    """
    for i in range(0, len(Path_to_predicted)):
        if Path_to_predicted[i] == "/":
            start = i + 1
        if Path_to_predicted[i] == ".":
            end = i
    use_directory = Path_to_predicted[0 : start - 1] + "_ordered"
    use_filename = Path_to_predicted[start:end]
    use_name = VISUALIZATION_PREDICTIONS_DIRECTORY_NAME + "/" + use_filename + "/" + use_filename

    if not os.path.isdir(VISUALIZATION_PREDICTIONS_DIRECTORY_NAME):
        os.mkdir(VISUALIZATION_PREDICTIONS_DIRECTORY_NAME)
    owd = os.getcwd()
    os.chdir(VISUALIZATION_PREDICTIONS_DIRECTORY_NAME)
    if not os.path.isdir(use_filename):
        os.mkdir(use_filename)
    os.chdir(owd)

    all_entrys = []

    file = open(Path_to_predicted, "r")
    used_lines = file.readlines()
    file.close()

    used_lines = remove_mean_from_dataset(copy.deepcopy(used_lines))

    for line in used_lines:
        try:
            cc = DiM.correct_ending(line)
            current_cT = float(DiM.extract_cT(copy.deepcopy(cc)))
            all_entrys.append(cc)
        except:
            continue

    compositions = []
    number_occured = []
    critical_temperatures = []

    for entry in all_entrys:
        cc = DiM.extract_cc(copy.deepcopy(entry))
        cT = DiM.extract_cT(copy.deepcopy(entry))

        exists_already = False
        for i in range(0, len(compositions)):
            if cc == compositions[i]:
                exists_already = True
                number_occured[i] += 1
                critical_temperatures[i].append(cT)

        if not exists_already:
            compositions.append(cc)
            number_occured.append(1)
            critical_temperatures.append([cT])

    # sort occured
    sort_compositions = [compositions[0]]
    sort_occured = [number_occured[0]]
    sort_cT = [critical_temperatures[0]]
    for i in range(1, len(number_occured)):
        appended = False
        for j in range(0, len(sort_occured)):
            if number_occured[i] < sort_occured[j]:
                sort_occured.insert(j, number_occured[i])
                sort_compositions.insert(j, compositions[i])
                sort_cT.insert(j, critical_temperatures[i])
                appended = True
                break
        if not appended:
            sort_occured.append(number_occured[i])
            sort_compositions.append(compositions[i])
            sort_cT.append(critical_temperatures[i])
    sort_sum = []
    sort_average = []
    for i in range(0, len(sort_cT)):
        sum = 0
        for tmp in sort_cT[i]:
            sum += tmp
        sort_sum.append(round(sum, 3))
        sort_average.append(round(sum / sort_occured[i], 3))

    max_length_comp = 0
    max_length_tmp = 0
    max_length_sum = 0
    max_length_average = 0
    max_length_occ = 0
    max_times_occured = max(sort_occured)

    for i in range(0, len(sort_occured)):
        if len(str(sort_occured[i])) > max_length_occ:
            max_length_occ = len(str(sort_occured[i]))
        if len(str(sort_compositions[i])) > max_length_comp:
            max_length_comp = len(str(sort_compositions[i]))
        if len(str(sort_sum[i])) > max_length_sum:
            max_length_sum = len(str(sort_sum[i]))
        if len(str(sort_average[i])) > max_length_average:
            max_length_average = len(str(sort_average[i]))
        for tmp in sort_cT[i]:
            if len(str(tmp)) > max_length_tmp:
                max_length_tmp = len(str(tmp))

    new_file = open(use_name + "_sort_occured.txt", "w")
    for i in range(0, len(number_occured)):
        add_comp_length = max_length_comp - len(sort_compositions[i])
        add_occ_length = max_length_occ - len(str(sort_occured[i]))
        add_sum_length = max_length_sum - len(str(sort_sum[i]))
        add_average_length = max_length_average - len(str(sort_average[i]))

        new_file.write(
            sort_compositions[i]
            + " " * add_comp_length
            + " | "
            + str(sort_occured[i])
            + " " * add_occ_length
            + " | "
        )
        for j in range(0, len(sort_cT[i])):
            add_tmp_length = max_length_tmp - len(str(sort_cT[i][j]))

            if j == len(sort_cT[i]) - 1:
                new_file.write(
                    str(sort_cT[i][j])
                    + " " * add_tmp_length
                    + " " * (max_times_occured - sort_occured[i]) * (max_length_tmp + 3)
                )
            else:
                new_file.write(str(sort_cT[i][j]) + " " * add_tmp_length + " ; ")

        new_file.write(
            " -> sum: "
            + str(sort_sum[i])
            + " " * add_sum_length
            + " -> average: "
            + str(sort_average[i])
            + "\n"
        )
    new_file.close()

    # sort sum
    new_sort_compositions = [sort_compositions[0]]
    new_sort_occured = [sort_occured[0]]
    new_sort_cT = [sort_cT[0]]
    new_sort_sum = [sort_sum[0]]
    new_sort_average = [sort_average[0]]

    for i in range(1, len(sort_sum)):
        appended = False
        for j in range(0, len(new_sort_sum)):
            if sort_sum[i] < new_sort_sum[j]:
                new_sort_occured.insert(j, sort_occured[i])
                new_sort_compositions.insert(j, sort_compositions[i])
                new_sort_cT.insert(j, sort_cT[i])
                new_sort_sum.insert(j, sort_sum[i])
                new_sort_average.insert(j, sort_average[i])
                appended = True
                break
        if not appended:
            new_sort_occured.append(sort_occured[i])
            new_sort_compositions.append(sort_compositions[i])
            new_sort_cT.append(sort_cT[i])
            new_sort_sum.append(sort_sum[i])
            new_sort_average.append(sort_average[i])

    new_file = open(use_name + "_sort_sum.txt", "w")
    for i in range(0, len(number_occured)):
        add_comp_length = max_length_comp - len(new_sort_compositions[i])
        add_occ_length = max_length_occ - len(str(new_sort_occured[i]))
        add_sum_length = max_length_sum - len(str(new_sort_sum[i]))
        add_average_length = max_length_average - len(str(new_sort_average[i]))

        new_file.write(
            new_sort_compositions[i]
            + " " * add_comp_length
            + " | "
            + str(new_sort_occured[i])
            + " " * add_occ_length
            + " | "
        )
        for j in range(0, len(new_sort_cT[i])):
            add_tmp_length = max_length_tmp - len(str(new_sort_cT[i][j]))

            if j == len(new_sort_cT[i]) - 1:
                new_file.write(
                    str(new_sort_cT[i][j])
                    + " " * add_tmp_length
                    + " "
                    * (max_times_occured - new_sort_occured[i])
                    * (max_length_tmp + 3)
                )
            else:
                new_file.write(str(new_sort_cT[i][j]) + " " * add_tmp_length + " ; ")

        new_file.write(
            " -> sum: "
            + str(new_sort_sum[i])
            + " " * add_sum_length
            + " -> average: "
            + str(new_sort_average[i])
            + "\n"
        )
    new_file.close()

    # sort average
    new_sort_compositions = [sort_compositions[0]]
    new_sort_occured = [sort_occured[0]]
    new_sort_cT = [sort_cT[0]]
    new_sort_sum = [sort_sum[0]]
    new_sort_average = [sort_average[0]]

    for i in range(1, len(sort_average)):
        appended = False
        for j in range(0, len(new_sort_average)):
            if sort_average[i] < new_sort_average[j]:
                new_sort_occured.insert(j, sort_occured[i])
                new_sort_compositions.insert(j, sort_compositions[i])
                new_sort_cT.insert(j, sort_cT[i])
                new_sort_sum.insert(j, sort_sum[i])
                new_sort_average.insert(j, sort_average[i])
                appended = True
                break
        if not appended:
            new_sort_occured.append(sort_occured[i])
            new_sort_compositions.append(sort_compositions[i])
            new_sort_cT.append(sort_cT[i])
            new_sort_sum.append(sort_sum[i])
            new_sort_average.append(sort_average[i])

    new_file = open(use_name + "_sort_average.txt", "w")
    for i in range(0, len(number_occured)):
        add_comp_length = max_length_comp - len(new_sort_compositions[i])
        add_occ_length = max_length_occ - len(str(new_sort_occured[i]))
        add_sum_length = max_length_sum - len(str(new_sort_sum[i]))
        add_average_length = max_length_average - len(str(new_sort_average[i]))

        new_file.write(
            new_sort_compositions[i]
            + " " * add_comp_length
            + " | "
            + str(new_sort_occured[i])
            + " " * add_occ_length
            + " | "
        )
        for j in range(0, len(new_sort_cT[i])):
            add_tmp_length = max_length_tmp - len(str(new_sort_cT[i][j]))

            if j == len(new_sort_cT[i]) - 1:
                new_file.write(
                    str(new_sort_cT[i][j])
                    + " " * add_tmp_length
                    + " "
                    * (max_times_occured - new_sort_occured[i])
                    * (max_length_tmp + 3)
                )
            else:
                new_file.write(str(new_sort_cT[i][j]) + " " * add_tmp_length + " ; ")

        new_file.write(
            " -> sum: "
            + str(new_sort_sum[i])
            + " " * add_sum_length
            + " -> average: "
            + str(new_sort_average[i])
            + "\n"
        )
    new_file.close()


def analyse_predicted(data_location="Predictions", combine=False, remove=[]):
    """
    MAIN FUNCTION

    DESCRIPTION:
    'make_predictions' created predictions for chemical compositions probably multiple 
    times with different settings each time. This function will create multiple files
    that help you evaluate which prediction might be thrustwhorty. Each file contains
    the same information about each predicted value:
    - How often it was predicted to have a temperature about the threshold by each
        different setting for the predicting model and training data.
    - The average of temperature of all predicting models combined
    - The sum of temperature of all predicting models combined
    Each different file will sort the data after one of the above metioned information

    ARGUMENTS:
    - path to file or directory of predictions
    - combine:

    """
    if os.path.isdir(data_location):
        data_files = os.listdir(data_location)
        data_directory = data_location + "/"
    else:
        data_files = [data_location]
        data_directory = ""
        combine = False

    data_files = remove_all_but(copy.deepcopy(data_files), string="txt")

    if combine:
        combined_file = open(data_directory + "combined.txt", "w")
        for file in data_files:
            file = open(data_directory + file, "r")
            used_lines = file.readlines()
            file.close()
            deletion_counter = 0
            for i in range(0, len(used_lines)):
                for el in remove:
                    if el in used_lines[i-deletion_counter]:
                        del used_lines[i-deletion_counter]
                        deletion_counter += 1
                        break
            for i in used_lines:
                combined_file.write(i)
        combined_file.close()
        best_predicted(data_directory + "combined.txt")
    else:
        for file in data_files:
            best_predicted(data_directory + file)


def predict_specific(
    predict_compositions,
    setting_location, 
    data_path_name = "last",
    promising_cT=10, 
    add_documentation=True, 
    print_progression_to_console=True,
    num_for_average=10,
    create_pickle = True,
    revert_log = True,
    test_temps = [],
    pickle_name = "Predictions",
    Path = ""
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    Collects data created by:   
    - DiM.create_pickle_predictors (provides chemical composition matrix of chemical
                                    compositions which transition temperature shall be 
                                    predicted)
    - hyperparameter_tuning (provides settings on creating the chemical composition matrix
                            of training data and machine learning model)
    
    It will either predict all chemical compositions in a single file or in every
    file inside a directory FOR every setting in either a file or all files inside
    a directory. Results of transition temperatures above 'promising_cT' will be written 
    to a text file. All temperatures as well as feature values of 'collect_features'
    will be written to a pickle file as in 'best_feature_candidates'.

    ARGUMENTS:
    - data_location: path to file or directory where chemical compositions are stored
    - setting_location: path to file or directory where settings are stored
    - add_documentation: If 'True': add additional informations to text file
    - print_progression_to_console: If 'True': print progression to console
    - num_for_average: (integer > 0) how often each chemical composition will be predicted
    - revert_log: If 'True': If transition temperatures were created in log values they
                            will be transformed back.
    - collect_features: values of which features will be collected
    - pickle_name: path to directory where pickle file will be stored
    """
    #check if pickle directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + pickle_name):
        os.mkdir(PICKLE_DIRECTORY_NAME + pickle_name)

    #check save location existence:
    if not os.path.isdir(PREDICTIONS_DIRECTORY_NAME):
        os.mkdir(PREDICTIONS_DIRECTORY_NAME)
    
    #check if predictions are supposed to be made with multiple settings:
    if os.path.isdir(setting_location):
        kwargs_files = os.listdir(setting_location)
        kwargs_directory = setting_location + "/"
    else:
        kwargs_files = [setting_location]
        kwargs_directory = ""

    kwargs_files = remove_all_but(copy.deepcopy(kwargs_files))

    predictions_file = open(PREDICTIONS_DIRECTORY_NAME + data_path_name + ".txt", "w")
    
    for kwarg_path in kwargs_files:
        with open(kwargs_directory + kwarg_path, "rb") as fid:
            pickle_data_loaded = pickle.load(fid)
            
        kwargs = pickle_data_loaded
        kwargs["random_split"]=random.randint(0, 100)
        
        if len(Path) > 0:
            kwargs["Path"] = Path

        filter_prop_args = dict()
        for keys in global_filter_properties_kwargs:
            filter_prop_args[keys] = kwargs[keys]

        ccm1, ccm, cT = apply_filter_and_properties(**filter_prop_args)

        ccm_for_prediction = np.array(ccm)
        cT_for_prediction = np.array(cT)

        done = False
        for i in range(0, len(ccm1[0])):
            if ccm1[0][i] in DiM.all_used_feature_names:
                element_order = copy.deepcopy(ccm1[0][0:i])
                done=True
                break
        if not done:
            element_order = ccm1[0]

        new_combinations = []
        count_deletions = 0
        deletion_counter = 0

        compositions_to_predict = copy.deepcopy(predict_compositions)
        test = copy.deepcopy(test_temps)
        report_marker = round(len(compositions_to_predict)/1)
        for i in range(0, len(compositions_to_predict)):
            if i%report_marker == 0:
                print("PROGRESSION-REPORT: " + str(round(i/report_marker)*10) + "% of compositions were transformed into a matrix")

            append = True
            comp_string = compositions_to_predict[i-count_deletions]
            comp_elements, comp_values = DiM.vector_from_simple_string(comp_string)
            for co_el in comp_elements:
                if co_el not in ccm1[0]:
                    append = False
            if append:
                new_simple_cc = DiM.expand_vector(element_order, comp_elements, comp_values)
                if kwargs["with_properties"]:
                    try:
                        simple_properties = DiM.add_properties(comp_string, kwargs["source"], kwargs["features"])
                        for simple_prop in simple_properties:
                            new_simple_cc.append(simple_prop)
                    except:
                        del compositions_to_predict[i-count_deletions]
                        count_deletions += 1
                        append = False
                if append:
                    new_combinations.append(new_simple_cc)
            else:
                if len(test) != 0:
                    del test[i-deletion_counter]
                    deletion_counter += 1
        
        if kwargs["with_properties"]:
            for i in range(0, len(kwargs["source"])):
                for propi in kwargs["features"][i]:
                    element_order.append(propi)

        new_combinations = np.array(new_combinations)
        for i in range(0, len(ccm1[0])):
            if element_order[i] != ccm1[0][i]:
                for j in range(i+1, len(element_order)):
                    if element_order[j] == ccm1[0][i]:
                        new_combinations[:, [j,i]] = new_combinations[:, [i,j]]
                        cop_this = element_order[i]
                        element_order[i] = element_order[j]
                        element_order[j] = cop_this

        progression_marker_setting = 0

        promising_predicted_values = []
        promising_std_errors = []
        promising_var = []
        promising_names = []
            
        #alter append condition
        if kwargs["cT_in_log_values"] == True:
            write_condition = DiM.transform_to_logarithmic([copy.deepcopy(promising_cT)], 
                epsilon=kwargs["epsilon"]
                )[0]
        else:
            write_condition = copy.deepcopy(promising_cT)

        #predict matrix:
        made_predictions = []
        accuracies = []
        for i in range(0, num_for_average):
            kwargs["ml_model"] = RandomForestRegressor()
            reg = kwargs["ml_model"].set_params(**kwargs["set_grid"])
            if kwargs["set_grid"]["oob_score"] == True:
                reg.fit(ccm_for_prediction, cT_for_prediction)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    ccm_for_prediction,
                    cT_for_prediction,
                    test_size=kwargs["split_test_size"],
                    random_state=kwargs["random_split"],
                )
                reg.fit(X_train, y_train)

            y_predict = reg.predict(new_combinations)
            if len(test) != 0:
                accuracies.append(mean_absolute_error(test, y_predict))
            made_predictions.append(y_predict)
        
        if len(test) != 0:
            accuracies = np.array(accuracies)
            mean_acc = np.mean(accuracies)
            print(mean_acc)

        made_predictions = np.array(made_predictions)
        #print(len(made_predictions))

        y_pred = np.mean(made_predictions, axis=0)
        var_pred = np.var(made_predictions, axis=0)
        std_error = np.std(made_predictions, axis=0)

        setting_name = get_file_name(kwarg_path)

        if create_pickle:
            pickle_data_save = dict()
            pickle_data_save["collect_features"] = compositions_to_predict
            pickle_data_save["pred"] = y_pred
            pickle_data_save["standard_deviation"] = std_error
            pickle_data_save["variance"] = var_pred
            with open(PICKLE_DIRECTORY_NAME + pickle_name + "/" + data_path_name + "_" + setting_name + ".pkl", "wb") as fid:
                pickle.dump(pickle_data_save, fid)

        if add_documentation:
            predictions_file.write(setting_name + ":\n\n")
            predictions_file.write(str(kwargs["set_grid"]))
            predictions_file.write("\n\n")
            
        count_promising = 0

        if kwargs["cT_in_log_values"] == True and revert_log:
            for k in range(0, len(y_pred)):
                if (y_pred[k] + std_error[k]) >= write_condition:
                    #print(y_pred[k] + std_error[k])
                    count_promising += 1

                    promising_predicted_values.append(y_pred[k])
                    promising_std_errors.append(std_error[k])
                    promising_var.append(var_pred[k])
                    promising_names.append(compositions_to_predict[k])

                    predictions_file.write(compositions_to_predict[k] + ","
                        + str(round(DiM.transform_to_euler([copy.deepcopy(y_pred[k])], 
                            epsilon=kwargs["epsilon"])[0], 4,
                            )
                        ) + ";" 
                        + str(round(DiM.transform_to_euler([copy.deepcopy(std_error[k])], 
                            epsilon=kwargs["epsilon"])[0], 4,
                            )
                        ) + ";" + 
                        str(round(DiM.transform_to_euler([copy.deepcopy(var_pred[k])], 
                            epsilon=kwargs["epsilon"])[0], 4,
                            )
                        ) + "\n"
                    )
        else:
            for k in range(0, len(y_pred)):
                if (y_pred[k] + std_error[k]) >= write_condition:
                    #print(y_pred[k] + std_error[k])
                    count_promising += 1

                    promising_predicted_values.append(y_pred[k])
                    promising_std_errors.append(std_error[k])
                    promising_var.append(var_pred[k])
                    promising_names.append(compositions_to_predict[k])

                    predictions_file.write(compositions_to_predict[k] + ","
                        + str(round(y_pred[k], 4)) + ";"
                        + str(round(std_error[k], 4)) + ";"
                        + str(round(var_pred[k], 4)) + "\n"
                    )

        if add_documentation:
            predictions_file.write("\n" + str(count_promising) + " of "
                + str(len(y_pred)) + " appear to be promising\n\n"
            )
            
        if print_progression_to_console:
            print("PROGRESSION-REPORT: " + str(progression_marker_setting) + " of " 
                + str(len(kwargs_files)) + " predicting models did their work"
                )
        
    predictions_file.close()

    if kwargs["cT_in_log_values"] == True and revert_log:
        promising_predicted_values, promising_std_errors, promising_var = get_real_mean_from_log(
            copy.deepcopy(promising_predicted_values),
            copy.deepcopy(promising_std_errors),
            kwargs["epsilon"]
        )

    visualisation_of_prediction(
        data_path_name, 
        promising_names, 
        promising_predicted_values, 
        promising_std_errors, 
        promising_var
    )


def create_composition_with_filtered(Path=DiM.SUPERCON_PATH, number_datapoints=10, filter_elements = [["Cu", "O"]], higher = 75, mix=True):
    """
    """
    DiM.create_cc(Path)
    DiM.filter_dataset(filter_out_element_combinations=filter_elements)

    used_file = open(DiM.CHANGING_DATA_DIRECTORY_NAME + DiM.USED_DATA_FILE_NAME, "r")
    used_lines = used_file.readlines()
    filtered_file = open(DiM.CHANGING_DATA_DIRECTORY_NAME + DiM.FILTERED_COMPOSITIONS_FILE_NAME, "r")
    filter_lines = filtered_file.readlines()
    used_file.close()
    filtered_file.close()
    print(len(filter_lines))
    if number_datapoints > len(filter_lines):
        number_datapoints = len(filter_lines)

    higher_ones = []
    for i in range(0, len(filter_lines)):
        cT = DiM.extract_cT(DiM.correct_ending(copy.deepcopy(filter_lines[i])))
        if cT >= higher:
            higher_ones.append(filter_lines[i])

    random_numbers = []
    counter = 0
    if not mix:
        while counter < number_datapoints:
            random_num = random.randint(0, len(higher_ones)-1)
            if random_num in random_numbers:
                continue
            else:
                random_numbers.append(random_num)
                counter += 1
        for i in random_numbers:
            used_lines.append(higher_ones[i])
    else:
        half = int(number_datapoints/2)
        while counter < half:
            random_num = random.randint(0, len(higher_ones)-1)
            if random_num in random_numbers:
                continue
            else:
                random_numbers.append(random_num)
                counter += 1
        for i in random_numbers:
            used_lines.append(higher_ones[i])
        del random_numbers
        random_numbers = []
        while counter < number_datapoints:
            random_num = random.randint(0, len(filter_lines)-1)
            if random_num in random_numbers:
                continue
            else:
                random_numbers.append(random_num)
                counter += 1
        for i in random_numbers:
            used_lines.append(filter_lines[i])
    
    used_file = open(DiM.CHANGING_DATA_DIRECTORY_NAME + DiM.USED_DATA_FILE_NAME, "w")
    filtered_file = open(DiM.CHANGING_DATA_DIRECTORY_NAME + DiM.FILTERED_COMPOSITIONS_FILE_NAME, "w")

    for line in used_lines:
        used_file.write(line)
    for line in filter_lines:
        filtered_file.write(line)
    filtered_file.close()
    used_file.close()

    filtered_cc = []
    filtered_cT = []

    for line in filter_lines:
        cc_cT = DiM.correct_ending(line)
        cT = DiM.extract_cT(copy.deepcopy(cc_cT))
        filtered_cT.append(cT)
        cc = DiM.extract_cc(cc_cT)
        filtered_cc.append(cc)
    
    return filtered_cc, filtered_cT


def just_testing():
    """
    NO RELEVANCE

    """

    filtering_arguments = [[], [], [0, 200]]
    # filtering_arguments = [[["Cu", "O"]], [], [0, 200]]
    # filtering_arguments = [[["Cu", "O"], ["Fe", "As"], ["Fe", "Se"]], [], [0, 200]]

    # ccm1, cT = DiM.create_cc_with_properties(Path, source, features)
    ccm1, cT = DiM.create_cc(DiM.SUPERCON_PATH)
    ccm = copy.deepcopy(ccm1)[1:]

    foc = filtering_arguments[0]
    # fie = filtering_arguments[1]
    ft = filtering_arguments[2]

    # print(len(ccm))
    # ccm, cT = DiM.filter_dataset(ccm = copy.deepcopy(ccm), cT = copy.deepcopy(cT), filter_for_temperature=ft, filter_out_element_combinations=foc)

    cT = DiM.transform_to_logarithmic(copy.deepcopy(cT), epsilon=1)
    ccm_for_prediction = np.array(ccm)
    cT_for_prediction = np.array(cT)

    X_train, X_split, y_train, y_split = train_test_split(
        ccm_for_prediction, cT_for_prediction, test_size=0.3, random_state=0
    )

    reg = RandomForestRegressor(
        max_features="sqrt", bootstrap=True, oob_score=True, random_state=0
    )
    reg.fit(X_train, y_train)
    print(reg.feature_importances_)
    print(reg.oob_score_)
    y_pred = reg.predict(X_split)
    current_accuracy = mean_absolute_error(y_split, y_pred)
    print(current_accuracy)
    print(reg.score(X_split, y_split))
    print(reg.feature_importances_)
    # print(reg.decision_path(X_split))
    print(reg.n_features_in_)
    print(len(ccm1[0]))
    print(ccm1[0])


#just_testing()

# predicted_distribution(["Simple_Predictions/A2B2C15.txt", "Simple_Predictions/A2B2C15_all.txt"], histogram_type="bar", facecolor=["red", "blue"], remove_text=False, cT_in_log_values=False, legend=["A2B2C15", "A2B2C15_all"], move_text=-0.5)
# predicted_distribution(["Simple_Predictions/A2B2C15_all.txt", "Simple_Predictions/A2B3C4_all.txt"], histogram_type="bar", facecolor=["red", "blue"], remove_text=False, cT_in_log_values=False, legend=["A2B2C15_all", "A2B3C4_all"], move_text=-0.5)
# predicted_distribution(["Simple_Predictions/A2B2C15_all.txt", "Simple_Predictions/A2B3C4_all.txt"], histogram_type="bar", remove_text=False, cT_in_log_values=False, move_text=-0.5, combine_paths=True)
# best_predicted("Simple_Predictions/A2B3C4.txt")


# no features: -> removes 0 -> 16273 datapoints                                                                                                                                                                                          splitting, max_features, bootstrap, acc, unbiased_acc, distance to: auto, true
# blank:                     random_split: 939       random state: ???       random_seperate: 2943   -> removes 0    -> 16273 datapoints     acc: 0.46237 / 0.47372      acc_full: 0.7156, 0.7278, 0.7279 / 'auto', 'log2', 'sqrt'       real_new: 2353, sqrt, False, 0.490, 0.485, 0.016    sqrt always followed by log2
# without copper_oxides:     random_split: 977       random state:           random_seperate: 3809   -> removes 5548 -> 10725 datapoints     acc: 0.35217 / 0.36660      acc_full: 0.7001, 0.7116, 0.7142 / 'auto', 'log2', 'sqrt'       real_new: 3973, sqrt, False, 0.379, 0.382, 0.015
# without non-conventional:  random_split: 49        random state: 405       random_seperate: 6844   -> removes 6948 -> 9325 datapoints      acc: 0.34483 / 0.34227      acc_full: 0.6861, 0.7013, 0.7040 / 'auto', 'log2', 'sqrt'       real_new: 7325, sqrt, False, 0.359, 0.341, 0.007

# minimal features: -> removes 34 -> 16239 datapoints
# -> removed elements: T, D, Cm -> 3
# blank:                     random_split: 800       random state: 166       random_seperate: 3371   -> removes 0    -> 16239 datapoints     acc: 0.48227 / 0.49327      acc_full: 0.7168, 0.7203, 0.7233 / 'log2', 'sqrt', 'auto'       real_new: 1960, sqrt, False, 0.513, 0.481, 0.01
# without copper_oxides:     random_split: 22        random state: 18        random_seperate: 3531   -> removes 5544 -> 10695 datapoints     acc: 0.40492 / 0.38291      acc_full: 0.7083, 0.7099, 0.7158 / 'log2', 'sqrt', 'auto'       real_new: 4457, sqrt, False, 0.381, 0.392, 0.01
# without non-conventional:  random_split: 40        random state: 17        random_seperate: 6826   -> removes 6941 -> 9298 datapoints      acc: 0.35916 / 0.34758      acc_full: 0.7063, 0.7068, 0.7135 / 'sqrt', 'log2', 'auto'       real_new: 6736, sqrt, False, 0.348, 0.349, 0.017

# suggested features: -> removes 1728 -> 7675 datapoints
# -> removed elements: T, D, Am, Cm, C, Np, Pa, U, Th, Pu -> 10
# blank:                     random_split: 81        random state: 94        random_seperate: 5840   -> removes 0    -> 7675 datapoints      acc: 0.36600 / 0.35431      acc_full: 0.7176, 0.7189, 0.7269 / 'sqrt', 'log2', 'auto'       real_new: 9062, sqrt, False, 0.351, 0.356, 0.015
# without copper_oxides:     random_split: 81        random state: 94        random_seperate: 9974   -> removes 0    -> 7675 datapoints      acc: 0.35091 / 0.35431      acc_full: 0.7168, 0.7208, 0.7260 / 'log2', 'sqrt', 'auto'       real_new: 7896, sqrt, False, 0.359, 0.362, 0.02
# without non-conventional:  random_split: 90        random state: 98        random_seperate: 2460   -> removes 847  -> 6828 datapoints      acc: 0.33842 / 0.33436      acc_full: 0.7215, 0.7216, 0.7265 / 'sqrt', 'log2', 'auto'       real_new: 6618, sqrt, False, 0.324, 0.342, 0.003

# all features: -> removes 14289 -> 1984 datapoints
# -> removed elements: K, As, O, Tm, Os, S, F, Ga, Sr, N, Cl, In, Zr, Se, Hg, Eu, Rb, Np, P, H, Ge, Pa, Te, T, Lu, Br, I, Sc, D, Cs, Tc, Pu
#                       Am,, Cm, Po, Ru, Nd, Re, Tl, Sm, Tb, Ce, B, C, Gd, Ho, Pr, Er, Yb, U, Th, Dy -> 52
# blank:                     random_split: 87        random state: 53        random_seperate: 8501   -> removes 0    -> 1984 datapoints      acc: 0.30176 / 0.30599      acc_full: 0.7063, 0.7099, 0.713 / 'auto', 'log2', 'sqrt'        real_new: 3387, sqrt, False, 0.366, 0.329, 0.032
# without copper_oxides:     random_split: 87        random state: 77        random_seperate: 7061   -> removes 0    -> 1984 datapoints      acc: 0.30230 / 0.30927      acc_full: 0.7079, 0.7099, 0.7142 / 'auto', 'log2', 'sqrt'       real_new: 6644, sqrt, False, 0.352, 0.311, 0.015
# without non-conventional:  random_split: 87        random state: 68        random_seperate: 612    -> removes 0    -> 1984 datapoints      acc: 0.27002 / 0.30979      acc_full: 0.7062, 0.7099, 0.7138 / 'auto', 'log2', 'sqrt'       real_new: 5271, log2, False, 0.369, 0.351, 0.027


# single_copper_oxides_filter = [[["Cu", "O"]], [], [0, 200]]
# single_non_conventional_filter = [[["Cu", "O"], ["Fe", "As"], ["Fe", "Se"]], [], [0, 200]]
# using_properties -> miss: F, O, N, H, T, D, Am, Np, Pa, U, Th, Pu