"""
PART 1: (line: 30 - 391) Chemical Composition matrix without properties
PART 2: (line 394 - 710) Chemical Composition matrix with properties
PART 3: (line 712 and beyond) Examples on how to use the functions
"""

"""
WARNING: If you plan on using these functions I advise you to use the following structure:
    - Main directory:
        - Data_into_Matrix.py
        - directory 1:
            - SuperCon file
        - directory 2 (Changing_Data - will be created if it's missing):

You only have to pass the path to the SuperCon file to the main functions, if you want to change the name of the directory where
the ever changing data will be stored, change: CHANGING_DATA_DIRECTORY_NAME

Names and paths can be set below
"""
# Important File Names:

USED_DATA_FILE_NAME = "Used_Data.txt"
USED_DATA_NO_CT_FILE_NAME = "Used_Data_no_cT.txt"
ERROR_DOCUMENTATION_FILE_NAME = "Error_Documentation.txt"
EXCLUDED_DATA_FILE_NAME = "Excluded_Data.txt"
UNWORKING_ELEMENTS_FILE_NAME = "Unworking_Elements.txt"
WORKING_ELEMENTS_FILE_NAME = "Working_Elements.txt"
CHANGING_DATA_DIRECTORY_NAME = "Changing_Data/"
DATA_DIRECTORY_PATH = "Data/"
NEW_ELEMENTS_NAME = "Predictors.txt"
ELEMENT_ORDER_FILE_NAME = "Element_order.txt"
CURRENT_COMPOSITIONS_FILE_NAME = "Current_Compositions.txt"
FILTERED_COMPOSITIONS_FILE_NAME = "Filtered_Compositions.txt"
FILTERED_COMPOSITIONS_FROM_ANY_FILE_NAME = "Filtered_Compositions_from_matrix.txt"
FIGURE_DIRECTORY_PATH = "Figures/"

SUPERCON_PATH = "Data/SuperCon.txt"

"""
PART 1

The following functions are used to transform the input of the SuperCon-database (which looks like that: Ba0.4K0.6Fe2As2,31.2) 
into a from that can be handled by the machine learning algorithms (arrays).
The main functions are: 'create_cc' and 'random_check'. 
You don't need to call any other functions of this part since they just make the main ones work.
"""

#from __future__ import all_feature_names
import math
import random
import os
import copy
import time
import pickle
#from xml.sax.handler import all_features
import matplotlib.pyplot as plt
import numpy as np

numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
alphabet = []
alphabet_up = []
for i in range(65, 91):
    alphabet.append(chr(i))
    alphabet.append(chr(i).lower())

    alphabet_up.append(chr(i))

alphabet_up.append(",")

valid_symbols = []
for i in alphabet:
    valid_symbols.append(i)
for i in numbers:
    valid_symbols.append(i)
valid_symbols.append(",")


def correct_ending(input_string):
    """
    SIDE FUNCTION

    The inputs that are read out of the SuperCon file sometimes have a weird symbol at the end 
    (something that produces some kind of newline), which leads to errors in my functions and therefore needs to be removed.
    """
    if input_string[len(input_string)-1] in valid_symbols:
        return input_string
    return input_string[:len(input_string)-1]


def weird(input_string):
    """
    SIDE Function

    Some inputs have symbols like "!" or "=" in it, I don`t understand their use, so I get rid of all entrys
    which contain these. (In total: 8 of 16.414) 
    """
    weird = False

    for i in input_string:
        if i not in valid_symbols:
            weird = True
            break
    return weird


def correct_cc(input_string):
    """
    SIDE FUNCTION

    The function that transforms an input into a row in the chemical composition matrix ('transform_cc') expects a number after each element. 
    If an element has a contribution of "1", they leave the number out in the SuperCon database and directly continue with the next element. 
    This function adds the missing "1".
    """
    last_was_letter = False
    corrected_string = ""
    position = 0

    for i in range(0, len(input_string)):
        if input_string[i] in alphabet_up and not last_was_letter:
            last_was_letter = True

        elif input_string[i] in numbers:
            last_was_letter = False

        elif input_string[i] in alphabet_up and last_was_letter:
            corrected_string += input_string[position:i]
            corrected_string += "1"
            position = i

    corrected_string += input_string[position:len(input_string)]
    if last_was_letter:
        corrected_string += "1"

    return corrected_string


def create_array_cc(matrix, array):
    """
    SIDE FUNCTION

    In the function 'transform_cc' is for all chemical compositions a matrix created, that contains the element and the number 
    associated with it in one row. This function assigns the numbers to the correct spot (column) and fills the contribution for 
    elements the chemical composition doesn't contain with "0".
    """
    cc_array = []

    for element in matrix:
        element_in_cc = False
        for i in range(0, len(array)):
            if element == array[i][0]:
                cc_array.append(float(array[i][1]))
                element_in_cc = True
        if not element_in_cc:
            cc_array.append(0.0)

    return cc_array


def multiple_elements(string):
    """
    SIDE FUNCTION

    The SuperCon Database contains chemical compositions that contain one or more elemnts multiple times. I don't know if that is a
    mistake, so I take them out (In total: 133 of 16414). The purpose of this function is to recognize them.
    """
    elements = []
    start_element = 0
    in_element = False

    for i in range(0, len(string)):
        if string[i] == ",":
            return False
        if string[i] in alphabet and not in_element:
            in_element = True
            start_element = i
        if string[i] in numbers and in_element:
            in_element = False
            if string[start_element:i] in elements:
                return True
            else:
                elements.append(string[start_element:i])


def remove_cT_from_cc(string):
    """
    SIDE FUNCTION

    As the name suggests, it removes the critical temperature from the chemical composition.
    """
    for i in range(0, len(string)):
        if string[i] == ",":
            return string[0:i]


def File_remove_bad_datapoints(Path):
    """
    SIDE FUNCTION

    This function uses 'weird' and 'multiple_elements' to identify the 'bad' datapoints from the SuperCon database.
    It creates 4 files in the Changing_Path directory (if you didn't create it youself, the function will do it): 
        - File with excluded data
        - File with some explaining of the excluded data
        - File with usable datapoints
        - File with usable datapoints without critical temperature (important for 'malfunctioning_properties')
    
    'create_cc' can afterwards easily access the used_data file (working datapoints) and create the chemical composition matrix and
    critical temperature vector with it.

    This is function always has to be called in 'create_cc', otherwise it won't work correctly. It will either create rows with
    different length (reason in description of 'multiple_elements') or won't compute at all.
    """
    if not os.path.isdir(CHANGING_DATA_DIRECTORY_NAME):
        os.mkdir(CHANGING_DATA_DIRECTORY_NAME)

    file = open(Path, "r")
    lines = file.readlines()

    owd = os.getcwd()
    os.chdir(CHANGING_DATA_DIRECTORY_NAME)
    old_files = os.listdir()
    for old_file in old_files:
        os.remove(old_file)

    used_file = open(USED_DATA_FILE_NAME, "w")
    excluded_file = open(EXCLUDED_DATA_FILE_NAME, "w")
    documentation_file = open(ERROR_DOCUMENTATION_FILE_NAME, "w")
    used_file_no_cT = open(USED_DATA_NO_CT_FILE_NAME, "w") #needed for 'malfunctioning_properties'

    weird_datapoints = []
    weird_datapoints_location = []
    multiple_datapoints = []
    multiple_datapoints_location = []

    for i in range(1, len(lines)):
        cc = lines[i]
        cc = correct_ending(cc)
        if weird(cc):
            excluded_file.write(cc)
            excluded_file.write("\n")
            weird_datapoints.append(cc)
            weird_datapoints_location.append(i+1)
        else:
            cc = correct_cc(cc)
            if multiple_elements(cc):
                excluded_file.write(cc)
                excluded_file.write("\n")
                multiple_datapoints.append(cc)
                multiple_datapoints_location.append(i+1)
            else:
                used_file.write(cc)
                used_file.write("\n")
                used_file_no_cT.write(remove_cT_from_cc(cc))
                used_file_no_cT.write("\n")
    documentation_file.write("Chemical compositions containing one or more elements multiple times and their position in the SuperCon Database:\n\n")
    for i in range(0, len(multiple_datapoints)):
        documentation_file.write("\"" + multiple_datapoints[i] + "\" at line: " + str(multiple_datapoints_location[i]) + "\n")
    documentation_file.write("\nIn total: " + str(len(multiple_datapoints)) + " of " + str(len(lines)-1) + " datapoints\n")
    documentation_file.write("\n\nChemical compositions with unusual symbols and their position in the SuperCon database:\n\n")
    for i in range(0, len(weird_datapoints)):
        documentation_file.write("\"" + weird_datapoints[i] + "\" at line: " + str(weird_datapoints_location[i]) + "\n")
    documentation_file.write("\nIn total: " + str(len(weird_datapoints)) + " of " + str(len(lines)-1) + " datapoints")

    file.close()
    used_file.close()
    used_file_no_cT.close()
    excluded_file.close()
    documentation_file.close()

    os.chdir(owd)


def contribute_to_1(matrix):
    """
    SIDE FUNCTION

    A chemical composition is always given in a way that looks like this: 'LiFePO4'. The number to the right of the element (usually 
    written in subscript) indicates the number of atoms of that particular element in the chemical composition.
    Since all the different chemical compositions contain mostly a different amount of total atoms, I have to transform the number
    of atoms into the contribution to the chemical composition (number of atoms of each element divided by total amount of atoms).
    This function does exactly that to the numbers of the 'cc_element_contribution' matrix in the 'transform_cc' function.
    """
    total_amount = 0.0
    for i in matrix:
        total_amount += float(i[1])
    for i in range(0, len(matrix)):
        matrix[i][1]=float(matrix[i][1])/total_amount
    return matrix


def transform_cc(matrix, vector, input_string):
    """
    SIDE FUNCTION

    This function is supposed to transform the chemical composition as shown in the SuperCon-database (for example: Ba0.4K0.6Fe2As2,31.2)
    into an easily processable form: A matrix where each row represents a chemical composition and each column shows the contribution of
    an element to the chemical composition.

    To achieve that, the chemical composition from the SuperCon database (input as string) is transformed into a nested array, in which 
    each element (array) holds two elements: the element (for example: "Ba") and its contribution (for example: 0.08 (= 0.4/5)). Afterwards 
    it is passed to the 'create_array_cc' function. There it is compared with an array that holds the current order of elements (represented 
    in the columns) and can so be transformed into an array, that has the contribution to each element at the correct spot. Elements that 
    are not contained in the chemical composition are of course represented with a "0".

    Finds the function an element in a chemical composition that isn`t presented in the final matrix yet, another column representing it
    will be addded to the matrix.
    """

    looping_over_element = False
    looping_over_contribution = False
    element_starts = 0
    contribution_starts = 0

    cc_element_contribution = []

    for i in range(0, len(input_string)):
        if input_string[i] == ",":
            contribution = input_string[contribution_starts:i]
            cc_element_contribution.append([element, contribution])

            vector.append(float(input_string[i+1:len(input_string)]))
            break

        if input_string[i] not in numbers and not looping_over_element:
            looping_over_element = True
            element_starts = i

        if input_string[i] in numbers and looping_over_element:
            element = input_string[element_starts:i]
            if element not in matrix[0]:
                matrix[0].append(element)
                for j in range(1, len(matrix)):
                    matrix[j].append(0.0)

            looping_over_contribution = True
            looping_over_element = False
            contribution_starts = i

        if input_string[i] not in numbers and looping_over_contribution:
            looping_over_contribution = False
            contribution = input_string[contribution_starts:i]

            cc_element_contribution.append([element, contribution])

    cc_element_contribution = contribute_to_1(cc_element_contribution)
    matrix.append(create_array_cc(matrix[0], cc_element_contribution))

    return matrix, vector


def create_cc(Path):
    """
    MAIN FUNCTION!

    This function combines all of the functions above to create the chemical composition matrix (row: chemical composition, 
    column: contribution of a specific element to the chemical composition) and the critical temperature vector.

    WARNING: You might notice that the matrix will be longer by one element than the vector. This is because the first element 
    of the matrix is the array that holds the names of the elements represented in each column. It has no use except for checking
    if the algorithm works correctly. For further computations I will delete this element (will be commented)

    Arg: - path to SuperCon database (as .txt file)
            
    Returns: chemical composition matrix and critical temperature vector
    """
    File_remove_bad_datapoints(Path)

    supercon_file = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    lines = supercon_file.readlines()

    chemical_composition = [[]]
    critical_Temperature = []

    # step by step each chemical composition is added. The final matrix will always grow in rows, but only sometimes in columns
    for i in range(0, len(lines)):
        cc = lines[i]
        cc = correct_ending(cc)
        cc = correct_cc(cc)
        chemical_composition, critical_Temperature = transform_cc(chemical_composition, critical_Temperature, cc)
    
    return chemical_composition, critical_Temperature


def random_check(Path):
    """
    MAIN FUNCTION!

    This function lets you check if 'create_cc' works properly. It will print out an entry in the chemical composition matrix and
    critical temperature vector and its corresponding datapoint in the SuperCon database.

    Args:   - path to Supercon database (as .txt file)
    """
    cc, cT = create_cc(Path)
    used_file = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    used_lines = used_file.readlines()

    rand = random.randint(1, len(cc)-1)

    print("\nDatabase entry:")
    print(correct_ending(used_lines[rand-1]))
    print("\nCritical Temperature:")
    print(cT[rand-1])
    print("\nOrder of elements in the chemical composition matrix:")
    print(cc[0])
    print("\nEntry number " + str(rand) + " in the Chemical composition matrix:")
    print(cc[rand])

    print("\nAnd now a little more obvious (:D):")
    for i in range(0, len(cc[rand])):
        if cc[rand][i] != 0.0:
            print(cc[0][i] + ": " + str(cc[rand][i]))


"""
Part 2

Following functions are supposed to create a chemical composition matrix with elemental properties. The main functions are
'malfunctioning_properties', 'create_cc_with_properties' and 'random_check_with_properties'. The functions below are supposed to 
filter the chemical compositions for which featurizing works.
"""
import matminer.featurizers.composition.composite as mfcc
from pymatgen.core.composition import Composition
                

def extract_elements(string):
    """
    SIDE FUNCTION

    This function returns a list of elements that are contained in the chemical composition that was passed as argument to it.
    """

    elements = []
    in_element = False

    for i in range(0, len(string)):
        if string[i] in alphabet and not in_element:
            in_element = True
            start_element = i
        if string[i] in numbers and in_element:
            in_element = False
            elements.append(string[start_element:i])
    
    return elements


def coarse_check_properties(source, features):
    """
    SIDE FUNCTION

    This function performs a coarse check if there are chemical compositions in the SuperCon database for which featurize can't fetch
    values (depending on the features of the source and the source itself). It creates temporary files where they are stored and 
    can be further examinated by 'fine_check_properties'.
    """

    only_cc = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_NO_CT_FILE_NAME, "r")
    only_cc_lines = only_cc.readlines()

    not_working = open(CHANGING_DATA_DIRECTORY_NAME + source + "_temporary_file.txt", "w")
    not_working_elements = open(CHANGING_DATA_DIRECTORY_NAME + source + "_elements_temporary_file.txt", "w")

    bad_elements = []
    good_elements = []

    prop = mfcc.ElementProperty(source, features, ['mean'])

    for cc in only_cc_lines:
        cc = correct_ending(cc)
        comp = Composition(cc)

        try:
            values = prop.featurize(comp)
            for value in values:
                int(value)
        except:
            not_working.write(cc + "\n")
            elements = extract_elements(cc)
            for element in elements:
                if element not in bad_elements and element not in good_elements:
                    try:
                        values = prop.featurize(Composition(element))
                        for value in values:
                            int(value)
                        good_elements.append(element)
                    except:
                        bad_elements.append(element)
    
    for bad_element in bad_elements:
        not_working_elements.write(bad_element + "\n")
    
    only_cc.close()
    not_working.close()
    not_working_elements.close()


def fine_check_properties(source, features):
    """
    SIDE FUNCTION

    This function further analyses the unworking chemical compositions and documents for which features the chemical compositions
    don't work. It seems like there is a problem with specific elements, these are documented as well.
    """

    not_working = open(CHANGING_DATA_DIRECTORY_NAME + source + "_temporary_file.txt", "r")
    not_working_elements = open(CHANGING_DATA_DIRECTORY_NAME + source + "_elements_temporary_file.txt", "r")

    nw_lines = not_working.readlines()
    nwe_lines = not_working_elements.readlines()

    documentation_file = open(CHANGING_DATA_DIRECTORY_NAME + ERROR_DOCUMENTATION_FILE_NAME, "a")

    #check elements
    bad_elements = []
    for element in nwe_lines:
        element = correct_ending(element)
        comp = Composition(element)

        bad_features = []
        for feature in features:
            prop = mfcc.ElementProperty(source, [feature], ['mean'])
            try:
                values = prop.featurize(comp)
                for value in values:
                    int(value)
            except:
                bad_features.append(feature)
        bad_elements.append([element, bad_features])
    
    #check chemical compositions
    bad_cc = []
    for cc in nw_lines:
        cc = correct_ending(cc)
        comp = Composition(cc)

        bad_features = []
        for feature in features:
            prop = mfcc.ElementProperty(source, [feature], ['mean'])
            try:
                values = prop.featurize(comp)
                for value in values:
                    int(value)
            except:
                bad_features.append(feature)
        bad_cc.append([cc, bad_features])
    
    #write element documentation to file
    documentation_file.write("\n\n\nElements not working for features in: " + source + "\n\n")
    for bad in bad_elements:
        documentation_file.write("\"" + bad[0] + "\"" + " not working for: ")
        for feat in bad[1]:
            documentation_file.write(feat + " | ")
        documentation_file.write("\n")
    documentation_file.write("\nIn total: " + str(len(nwe_lines)) + " elements")

    #write chemical composition documentation to file
    documentation_file.write("\n\n\nChemical Compositions not working for features in: " + source + "\n\n")
    for bad in bad_cc:
        documentation_file.write("\"" + bad[0] + "\"" + " not working for: ")
        for feat in bad[1]:
            documentation_file.write(feat + " | ")
        documentation_file.write("\n")
    documentation_file.write("\nIn total: " + str(len(nw_lines)) + " chemical compositions")

    not_working.close()
    not_working_elements.close()
    documentation_file.close()


def combine_bad_cc(source, excluded_property_cc):
    """
    SIDE FUNCTION

    To create the file that holds the chemical compositions that work fine with featurizers, the ones that don't work first have to be
    collected from up to 3 different files. This function writes them to an array an returns it.
    """
    not_working = open(CHANGING_DATA_DIRECTORY_NAME + source + "_temporary_file.txt", "r")
    nw_lines = not_working.readlines()
    for cc in nw_lines:
        cc = correct_ending(cc)
        if cc not in excluded_property_cc:
            excluded_property_cc.append(cc)

    not_working.close()
    return excluded_property_cc


def combine_bad_elements(source, excluded_property_element):
    """
    SIDE FUNCTION

    Does the same as 'combine_bad_cc' for not working elements.
    """

    not_working = open(CHANGING_DATA_DIRECTORY_NAME + source + "_elements_temporary_file.txt", "r")
    nw_lines = not_working.readlines()
    for cc in nw_lines:
        cc = correct_ending(cc)
        if cc not in excluded_property_element:
            excluded_property_element.append(cc)

    not_working.close()
    return excluded_property_element


def malfunctioning_properties(Path, data_source, source_features):
    """
    MAIN FUNCTION

    This function combines all functions from PART 2 to create 3 files that hold following information:
        - Chemical Compositions that work for the sources and features passed to the function
        - Chemical Compositions that were excluded
        - documentation on why certain chemical compositions were excluded
    
    Arguments:
        - Path to SuperCon database
        - Sources for data (= list, for example: ["pymatgen", "magpie"])
        - features of sources (= list of lists, for example: [['X', 'atomic_mass', 'atomic_radius'], ['AtomicWeight', 'MeltingT']])
    """

    File_remove_bad_datapoints(Path)
    excluded_data = open(CHANGING_DATA_DIRECTORY_NAME + EXCLUDED_DATA_FILE_NAME, "a")
    unworking_elements = open(CHANGING_DATA_DIRECTORY_NAME + UNWORKING_ELEMENTS_FILE_NAME, "w")
    old_used_data = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    old_used_data_no_cT = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_NO_CT_FILE_NAME, "r")
    old_used_lines = old_used_data.readlines()
    old_used_no_cT_lines = old_used_data_no_cT.readlines()
    old_used_data.close()
    old_used_data_no_cT.close()
    
    excluded_property_cc = []
    excluded_property_elements = []

    for source in range(0, len(data_source)):
        coarse_check_properties(data_source[source], source_features[source])
    for source in range(0, len(data_source)):
        fine_check_properties(data_source[source], source_features[source])
    for source in range(0, len(data_source)):
        excluded_property_cc = combine_bad_cc(data_source[source], excluded_property_cc)
    for bad_cc in excluded_property_cc:
        excluded_data.write(bad_cc + "\n")
    for source in range(0, len(data_source)):
        excluded_property_elements = combine_bad_elements(data_source[source], excluded_property_elements)
    for bad_el in excluded_property_elements:
        unworking_elements.write(bad_el + "\n")

    used_data = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "w")
    used_data_no_cT = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_NO_CT_FILE_NAME, "w")

    for i in range(0, len(old_used_no_cT_lines)):
        cc_no_cT = correct_ending(old_used_no_cT_lines[i])
        if cc_no_cT not in excluded_property_cc:
            cc = correct_ending(old_used_lines[i])
            used_data.write(cc + "\n")
            used_data_no_cT.write(cc_no_cT + "\n")
    
    used_data.close()
    used_data_no_cT.close()
    excluded_data.close()
    unworking_elements.close()

    #remove unnecessary files
    wanted_files = [EXCLUDED_DATA_FILE_NAME, USED_DATA_FILE_NAME, USED_DATA_NO_CT_FILE_NAME, ERROR_DOCUMENTATION_FILE_NAME, UNWORKING_ELEMENTS_FILE_NAME]
    owd = os.getcwd()
    os.chdir(CHANGING_DATA_DIRECTORY_NAME)
    files = os.listdir()
    for file in files:
        if file not in wanted_files:
            os.remove(file)
    os.chdir(owd)


def create_cc_with_properties(Path, data_source, source_features):
    """
    MAIN FUNCTION

    This function works similar to 'create_cc'. The datapoints that will be used are this time evaluated by 'malfunctioning_properties'.
    After it created the matrix like 'create_cc' it will then add the elemental properties to the matrix as well.

    Args:   - Path to SuperCon database
            - Sources for data (= list, for example: ["pymatgen", "magpie"])
            - features of sources (= list of lists, for example: [['X', 'atomic_mass', 'atomic_radius'], ['AtomicWeight', 'MeltingT']])
    
    Returns:    - chemical composition matrix with elemenatl properties and critical temperature vector
    """

    malfunctioning_properties(Path, data_source, source_features)

    working_elements = open(CHANGING_DATA_DIRECTORY_NAME + WORKING_ELEMENTS_FILE_NAME, "w")
    used_data = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    lines = used_data.readlines()
    used_data_no_cT = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_NO_CT_FILE_NAME, "r")
    no_cT_lines = used_data_no_cT.readlines()
    used_data.close()
    used_data_no_cT.close()

    ccm = [[]]
    cT = []

    for i in range(0, len(lines)):
        cc = lines[i]
        cc = correct_ending(cc)
        cc = correct_cc(cc)
        ccm, cT = transform_cc(ccm, cT, cc)
    
    for el in ccm[0]:
        working_elements.write(el + "\n")

    for source in range(0, len(data_source)):
        prop = mfcc.ElementProperty(data_source[source], source_features[source], ['mean'])
        for feature in source_features[source]:
                ccm[0].append(feature)
        for line in range(0, len(no_cT_lines)):
            cc = correct_ending(no_cT_lines[line])
            comp = Composition(cc)
            values = prop.featurize(comp)
            for value in values:
                if math.isnan(value):
                    print("Bad value appended for: " + cc)
                ccm[line+1].append(float(value))
    
    used_data.close()
    working_elements.close()
    used_data_no_cT.close()
    
    return ccm, cT


def random_check_with_properties(Path, data_source, source_features):
    """
    MAIN FUNCTION!

    This function lets you check if 'create_cc_with_properties' works properly. It will print out an entry in the chemical composition 
    matrix and critical temperature vector and its corresponding datapoint in the SuperCon database.

    Args:   - Path to SuperCon database
            - Sources for data (= list, for example: ["pymatgen", "magpie"])
            - features of sources (= list of lists, for example: [['X', 'atomic_mass', 'atomic_radius'], ['AtomicWeight', 'MeltingT']])
    """
    cc, cT = create_cc_with_properties(Path, data_source, source_features)
    used_file = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    used_lines = used_file.readlines()

    rand = random.randint(1, len(cc)-1)

    print("\nDatabase entry:")
    print(correct_ending(used_lines[rand-1]))
    print("\nCritical Temperature:")
    print(cT[rand-1])
    print("\nOrder of elements in the chemical composition matrix:")
    print(cc[0])
    print("\nEntry number " + str(rand) + " in the Chemical composition matrix:")
    print(cc[rand])

    print("\nAnd now a little more obvious (:D):")
    for i in range(0, len(cc[rand])):
        if cc[rand][i] != 0.0:
            print(cc[0][i] + ": " + str(cc[rand][i]))



all_pymatgen_features = ['X', 'atomic_mass', 'atomic_radius', 'mendeleev_no', 'electrical_resistivity', 
                'velocity_of_sound', 'thermal_conductivity', 'melting_point', 'bulk_modulus', 'coefficient_of_linear_thermal_expansion']

all_magpie_features = ['MendeleevNumber', 'AtomicWeight', 'MeltingT', 'Column', 'Row', 'CovalentRadius', 'Electronegativity', 'GSvolume_pa',
                'GSbandgap', 'GSmagmom', 'SpaceGroupNumber']

all_deml_features = ['atom_num', 'atom_mass', 'atom_radius', 'molar_vol', 'heat_fusion', 'boiling_point', 'heat_cap', 
                'first_ioniz', 'electronegativity', 'electric_pol', 'GGAU_Etot', 'mus_fere', 'FERE correction']

all_used_feature_names = ['X', 'atomic_mass', 'atomic_radius', 'mendeleev_no', 'electrical_resistivity', 
                'velocity_of_sound', 'thermal_conductivity', 'melting_point', 'bulk_modulus', 'coefficient_of_linear_thermal_expansion',
                'MendeleevNumber', 'AtomicWeight', 'MeltingT', 'Column', 'Row', 'CovalentRadius', 'Electronegativity', 'GSvolume_pa',
                'GSbandgap', 'GSmagmom', 'SpaceGroupNumber', 'atom_num', 'atom_mass', 'atom_radius', 'molar_vol', 'heat_fusion', 
                'boiling_point', 'heat_cap', 'first_ioniz', 'electronegativity', 'electric_pol', 'GGAU_Etot', 'mus_fere', 'FERE correction']

failing_names = [
"velocity_of_sound", 
"bulk_modulus", 
"coefficient_of_linear_thermal_expansion", 
"electrical_resistivity", 
"atomic_mass", "atomic_radius", "mendeleev_no", "thermal_conductivity", "melting_point", "X",
"heat_cap",
"electric_pol",
"GGAU_Etot",
"mus_fere",
"FERE correction",
"electronegativity",
"heat_fusion",
"boiling_point",
"first_ioniz",
"atom_num", "atom_mass", "atom_radius", "molar_vol",
"MendeleevNumber", "AtomicWeight", "MeltingT", "Column", "Row", "CovalentRadius", 
"Electronegativity", "GSvolume_pa", "GSbandgap", "GSmagmom", "SpaceGroupNumber"]

failing_elements = [
["Tm", "S", "F", "Np", "As", "Sr", "Pa", "T", "P", "Lu", "I", "Br", "Sc", "D", "Cs", "Eu", "Tc", "Am", "Po", "Cm"],
["Os", "F", "O", "Ga", "N", "In", "Zr", "Np", "Sr", "H", "Ge", "Pa", "Tc", "Pu", "Am", "Po", "Cm", "T", "D"],
["S", "F", "O", "N", "Cl", "Hg", "Rb", "Se", "Np", "As", "H", "K", "Pa", "Te", "P", "I", "Br", "Cs", "Tc", "Pu", "Am", "Po", "Cm", "T", "D"],
["F", "O", "N", "H", "T", "D", "Am", "Cm"],
["T", "D"], ["T", "D", "Cm"], ["T", "D"], ["T", "D"], ["T", "D"], [],
["Tm", "Os", "Nd", "Re", "Tl", "Tb", "Ce", "Gd", "Ho", "Pr", "Ru", "Er", "Np", "H", "Pa", "T", "U", "Th", "Lu", "Yb", "Dy", "Eu", "Tc", "Pu", "Sm", "Am", "Po", "Cm"],
["Tm", "Os", "Nd", "Re", "Tl", "Tb", "Ce", "B", "C", "Gd", "Ho", "Pr", "Ru", "Er", "Np", "H", "Pa", "T", "U", "Th", "Lu", "Yb", "Dy", "Eu", "Tc", "Pu", "Sm", "Am", "Po", "Cm"],
["Tm", "Os", "Nd", "Re", "Tl", "Tb", "Ce", "B", "C", "Gd", "Ho", "Pr", "Ru", "Er", "Np", "H", "Pa", "T", "U", "Th", "Lu", "Yb", "Dy", "Eu", "Tc", "Pu", "Sm", "Am", "Po", "Cm", "I", "Br", "Cs"],
["Tm", "Os", "Nd", "Re", "Tl", "Tb", "Ce", "B", "C", "Gd", "Ho", "Pr", "Ru", "Er", "Np", "H", "Pa", "T", "U", "Th", "Lu", "Yb", "Dy", "Eu", "Tc", "Pu", "Sm", "Am", "Po", "Cm", "I", "Br", "Cs"],
["Tm", "Os", "Nd", "Re", "Tl", "Tb", "Ce", "B", "C", "Gd", "Ho", "Pr", "Ru", "Er", "Np", "H", "Pa", "T", "U", "Th", "Lu", "Yb", "Dy", "Eu", "Tc", "Pu", "Sm", "Am", "Po", "Cm", "I", "Br", "Cs"],
["Tb", "Np", "Pa", "T", "U", "Th", "Yb", "D", "Eu", "Pu", "Am", "Cm"],
["C", "T", "D", "Pu", "Am", "Cm"],
["Np", "Pa", "T", "U", "Th", "D", "Pu", "Am", "Cm"],
["Np", "Pa", "T", "U", "Th", "D", "Pu", "Am", "Cm"],
["T", "D"], ["T", "D"], ["T", "D", "Cm"], ["T", "D"],
["T", "D"], ["T", "D"], ["T", "D"], ["T", "D"], ["T", "D"], ["T", "D"],
["T", "D"], ["T", "D"], ["T", "D"], ["T", "D"], ["T", "D"]
]

failing_elements_dict = dict()
for i in range(0, len(failing_names)):
    failing_elements_dict[failing_names[i]] = failing_elements[i]

"""
COMMENT:

Analysing all features from all sources using 'malfunctioning_properties' reveals following features as best choice:
"""

suggested_pymatgen_features = ['thermal_conductivity'] #add electrical_resistivity -> loose basic elements

suggested_magpie_features = ['MendeleevNumber', 'AtomicWeight', 'MeltingT', 'Column', 'Row', 'CovalentRadius', 'Electronegativity',
                                'GSvolume_pa', 'GSbandgap', 'GSmagmom', 'SpaceGroupNumber']

suggested_deml_features = ['atom_num', 'atom_radius', 'molar_vol', 'boiling_point', 'heat_fusion', 'first_ioniz'] #remove boiling_point for significant more data


minimum_loss_pymatgen_features = ['thermal_conductivity'] #add electrical_resistivity -> loose basic elements

minimum_loss_magpie_features = ['MendeleevNumber', 'AtomicWeight', 'MeltingT', 'Column', 'Row', 'CovalentRadius', 'Electronegativity', 'GSvolume_pa',
                                'GSbandgap', 'GSmagmom', 'SpaceGroupNumber']

minimum_loss_deml_features = ['atom_num', 'atom_radius', 'molar_vol'] #remove boiling_point for significant more data

suggested_source = ["pymatgen", "magpie", "deml"]
suggested_features = [suggested_pymatgen_features, suggested_magpie_features, suggested_deml_features]
minimum_loss_features = [minimum_loss_pymatgen_features, minimum_loss_magpie_features, minimum_loss_deml_features]
all_features = [all_pymatgen_features, all_magpie_features, all_deml_features]



"""
PART 3

Following functions are supposed to create new chemical compositions
"""


def extract_total_number(string: str):
    """
    SIDE FUNCTION

    This function takes a chemical composition as string without critical temperature (from the file that is created during the process of 
    generating the chemical composition matrix) and returns the total number of atoms
    """
    cc_numbers = []
    in_number = False

    for i in range(0, len(string)):
        if string[i] in numbers and not in_number:
            in_number = True
            start_number = i
        if string[i] in alphabet and in_number:
            in_number = False
            cc_numbers.append(string[start_number:i])
    cc_numbers.append(string[start_number:len(string)])
    total = 0
    for i in cc_numbers:
        total += float(i)
    
    return total


def same_element(string1, string2):
    """
    SIDE FUNCTION

    This function compares two chemical compositions (given as string) and returns 'True' if at least one element is contained in both of them.
    Otherwise it returns 'False'
    """
    elements1 = extract_elements(string1)
    elements2 = extract_elements(string2)

    for el in elements1:
        if el in elements2:
            return True
    return False


def get_all_elements(input_string, current_elements, max):
    """
    SIDE FUNCTION

    This function is used to collect all elements and their maximum value. It is given a chemical composition (as string), an array that holds
    elements that were already collected (by calling this function before) and another array that holds the maximum number of atoms of each element.
    If there is an element in the string, that is not contained in the array yet, it will be added. Otherwise the function will only check, if
    the elements number of atoms in the string is higher than the one associated with it in the second array and replace it, if that is the case.
    """
    elements = []
    value_elements = []

    in_element = True
    start = 0
    for i in range(0, len(input_string)):
        if input_string[i] == ",":
            value_elements.append(input_string[start:i])
            break
        if input_string[i] in alphabet and not in_element:
            value_elements.append(input_string[start:i])
            start = i
            in_element = True
        if input_string[i] in numbers and in_element:
            in_element = False
            elements.append(input_string[start:i])
            start = i
    
    for i in range(0, len(value_elements)):
        value_elements[i] = float(value_elements[i])
    
    for i in range(0, len(elements)):
        inside = False
        for j in range(0, len(current_elements)):
            if elements[i] == current_elements[j]:
                inside = True
                if max[j] < value_elements[i]:
                    max[j] = value_elements[i]
        if not inside:
            current_elements.append(elements[i])
            max.append(value_elements[i])
    
    return current_elements, max


def combine_element_and_composition(elements, max_num_atoms, min_num_atoms):
    """
    SIDE FUNCTION

    RETURNS:    array with all compositions for the passed elements

    This function helps to create all chemical compositions for the elements that were passed it. To do so it accesses a file with compositions 
    that have 'n' number of elements. It will iterate over the passed elements. First it will remove all compositions that contain the element and
    then add all of those compositions to the basic compositions of the element (for example: He1, He2, He3, ...).

    REMARK: This actually works because everything is ordered

    Arguments:  - elements: array that holds elements (as strings)
                - max_num_atoms:    array that holds the maximum value for each element
                - min_num_atoms:    array that holds the minimum value for each element
    """
    compositions_file = open(CHANGING_DATA_DIRECTORY_NAME + CURRENT_COMPOSITIONS_FILE_NAME, "r")
    lines = compositions_file.readlines()
    compositions = []
    for line in lines:
        line = correct_ending(line)
        compositions.append(line)
    compositions_file.close()
    del lines
    basic = []
    for i in range(0, len(elements)):
        el_with_at = []
        for j in range(min_num_atoms[i], max_num_atoms[i]+1):
            el_with_at.append(elements[i] + str(j))
        basic.append(el_with_at)
    ccs = []
    for i in range(0, len(basic)):
        el = elements[i]
        counter = 0
        for j in range(0, len(compositions)):
            if el in extract_elements(compositions[j-counter]):
                del compositions[j-counter]
                counter += 1
            else:
                break
        for j in range(0, len(basic[i])):
            for k in range(0, len(compositions)):
                ccs.append(basic[i][j] + compositions[k])
    
    new_file = open(CHANGING_DATA_DIRECTORY_NAME + CURRENT_COMPOSITIONS_FILE_NAME, "w")
    for comp in compositions:
        new_file.write(comp + "\n")
    
    return ccs


def create_predictors(within_atoms_range = True, around_range_ratio = 0, upper_boarder_atoms = 100, number_elements = 4, use_only_elements = [], unwanted_elements = ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Lr", "No", "Md", "Fm", "Es", "Cf", "Bk", "Cm", "Am", "Pu", "Np", "U", "Pa", "Th", "Ac", "Pm"], write_to_file = False):
    """
    MAIN FUNCTION

    This function creates, based on the input, new chemical compositions. It will create an array with the available elements and their possible 
    number of atoms (from 1 to maximum - might be element specific). For example: [Ba1, Ba2, Fe1, K1, K2, K3, ...]. For this array it will basically
    create all possible arrangements (with no respect to order and no element twice in a chemical composition) until it has created all chemical
    compositions with 1 to 'number_elements' number of elements. 

    WARNING:    Depending on your input, this function might take a lot of processing time and create a pretty large file. At the beginning it will
                tell you how many entries will approximately be created and how much file space this might consume. (This information will not be 
                exact and rather give you an inpression)
    
    ATTENTION:  The elements that will be used will not only depend on the arguments you pass to this function. This function accesses data that was
                created in the process of creating the chemical composition matrix (with 'create_cc' or 'create_cc_with_properties'), and will only
                use elements, that are contained in the chemical composition matrix
    
    Arguments:  - Path: path to the SuperCon - File, or where ever you are gaining your information from (string)
                - within_atoms_range:   * if 'True' (set by default):   The number of atoms of each element will not be bigger than the number it has
                                                                        in any of the chemical compositions in the given path
                                        * if 'False':   The maximum number of atoms for each element is only set by 'upper_boarder_atoms'
                - around_range_ratio:   If 'within_atoms_range' is set to 'True' you can adjust the maximum with this argument. As the name suggests
                                        it will increase (or decrease for negative values) the maximum by the given ratio
                                        For illustration: maximum = old_maximum(1+around_range_ratio)
                - upper_boarder_atoms:  Determines the maximum number of atoms each element can have
                - number_elements:  Determines the maximum number of elements a created chemical composition can have
                - use_only_elements:    Only uses the elements from this list to create new chemical compositions. If an element is not contained in
                                        the chemical compositions from the path, it can not be used and will be removed
                - unwanted_elements:    The elements from this list will not be used to create new chemical compositions
                - write_to_file:    * if 'True':    creates two files, one holding all compositions with 'number_elements' number of elements
                                                    the other holding all elements, as well as their maximum and mininum, that were used to create
                                                    the compositions
                                                    -> This option only makes sense inside the function 'Predicting_Model'
                                    * if 'False' (set by default):  doesn't create the above mentioned files
    """
    number_elements = int(number_elements)
    if number_elements < 1:
        number_elements = 1
        print("The number of elements (represented by 'number_elements') must be an integer an can not be smaller than 1")

    if around_range_ratio < -1:
        around_range_ratio = -1
        print("ATTENTION: \"around_range_ratio\" smaller than -1 does not make sense, because then every minimum value would be bigger than the maximum. It was changed to -1.")

    supercon_file = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    lines = supercon_file.readlines()

    elements = []
    max = []
    min = []

    for i in range(0, len(lines)):
        cc = lines[i]
        cc = correct_ending(cc)
        cc = correct_cc(cc)
        elements, max = get_all_elements(cc, elements, max)
    
    if len(use_only_elements) > 0:
        if len(unwanted_elements) > 0:
            print("ATTENTION: 'use_only_elements' and 'unwanted_elements' contain elements. Unwanted elements will be removed, even if they are also in 'use_only_elements'")
        counter = 0
        for i in range(0, len(elements)):
            if elements[i-counter] not in use_only_elements or elements[i-counter] in unwanted_elements:
                del elements[i-counter]
                del max[i-counter]
                counter += 1
    else:
        counter = 0
        for i in range(0, len(elements)):
            if elements[i-counter] in unwanted_elements:
                del elements[i-counter]
                del max[i-counter]
                counter += 1
        
    for i in range(0, len(elements)):
        min.append(1)
    
    if within_atoms_range:
        for i in range(0, len(max)):
            max[i] = int(max[i]+around_range_ratio*max[i])
            if max[i] > upper_boarder_atoms:
                max[i] = upper_boarder_atoms
                print("ATTENTION: The maximum value for \"" + elements[i] + "\" exceeded the upper boarder of atoms (=" + str(upper_boarder_atoms) + "). It was reduced to that.")
    else:
        max = []
        for i in range(0, len(elements)):
            max.append(upper_boarder_atoms)
    
    if write_to_file:
        element_order_file = open(CHANGING_DATA_DIRECTORY_NAME + ELEMENT_ORDER_FILE_NAME, "w")
        for elo in range(0, len(elements)):
            element_order_file.write(elements[elo] + "," + str(max[elo]) + "," + str(min[elo]) + "\n")
    
    #elements = ["He", "Ba", "K", "N", "H", "Ti", "Pl", "Zi", "Z", "I", "io", "U"]
    #elements = ["A", "B", "C", "D", "E"]
    #max = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    ccs = []
    for i in range(0, len(elements)):
        for j in range(min[i], max[i]+1):
            ccs.append(elements[i] + str(j))

    file = open(CHANGING_DATA_DIRECTORY_NAME + CURRENT_COMPOSITIONS_FILE_NAME, "w")
    for cc in ccs:
        file.write(cc + "\n")
    file.close()

    for iterations in range(1, number_elements):
        new_ccs = combine_element_and_composition(elements, max, min)
        file = open(CHANGING_DATA_DIRECTORY_NAME + CURRENT_COMPOSITIONS_FILE_NAME, "w")
        for new_cc in new_ccs:
            ccs.append(new_cc)
            file.write(new_cc + "\n")
        file.close()

    if not write_to_file:
        os.remove(CHANGING_DATA_DIRECTORY_NAME + CURRENT_COMPOSITIONS_FILE_NAME)
        
    return ccs


def element_order_transform(string: str):
    """
    SIDE FUNCTION

    This function helps to read out the data contained in the file, that is created when 'write_to_file' in 'create_predictors' is set to 'True'.
    For exmaple: input: "Ba,5,1" -> output: Ba,5,1 (the comma in the output is not an actual comma but stands for seperation)
    """
    start = 0
    data = []
    for i in range(0, len(string)):
        if string[i] == ",":
            data.append(string[start:i])
            start = i+1
    data.append(string[start:len(string)])
    return data[0], int(data[1]), int(data[2])


def split_cc_cT(string: str):
    """
    SIDE FUNCTION

    This function gets an entry from the SuperCon database (as string) as input and returns a list with 2 elements. The first is the chemical
    composition (string), the second the critical Temperature
    """
    for i in range(0, len(string)):
        if string[i] == ",":
            return [string[0:i], float(string[i+1:len(string)])]


def promising_elements(promising_cT = 10, return_number_ratio = 0, return_of_best_ratio = 0, return_ratio_occured_total = 0):
    """
    MAIN FUNCTION

    This function checks for elements in the SuperCon database that lead to a certain (given by user) critical temperature. To elements will be 
    ordered depending on how often they were in a chemical composition that has a transition temperature equal or higher than 'promising_cT'. 
    What elements of all collected will be returned in the end is again up to the user settings (read the Arguments for more information).

    ATTENTION:  This function will only check entries, that have been used to create the chemical composition matrix before, since it accesses
                data that was created in the process. This could also have happened in one of your last compilations.

    Arguments:  - Path: path to the SuperCon - File, or where ever you are gaining your information from (string)
                -  promising_cT:    Every element inside a chemical composition that leads to a transition temperature equal or higher than the value 
                                    of this argument will be collected (default = 10)
                - Ways to set what elements are returned:
                    * return_number_ratio:  The function will return the ratio number of elements (in reference to the total amount of collected 
                                            elements). To clarify: All_elements_collected*return_number_ratio = number_of_returned_elements.
                                            The function will not randomly choose elements to return, instead it returnes the elements, that are
                                            connected with the most cases of transition temperature equal or higher than 'promising_cT'
                    * return_of_best_ratio: The function will first determine which element is connected to the most cases of transition temperature
                                            equal or higher than 'promising_cT' (I will in the following refer with 'best' to it). Afterwards it
                                            returnes the elements, that are at least connected to 'best' * 'return_of_best_ratio' number of cases
                                            that lead to a transition temperature equal or higher than 'promising_cT'
                    * return_ratio_occured_total:   This time the function will return arguments depending on their ratio of 
                                                    'number_the_element_was_in_a_cc_that_lead_to_promising_cT' / 'number_the_element_was_in_ccs'.
                                                    It will again determine the best ratio and then return elements similar to 
                                                    'return_of_best_ratio'
    Returns:    -   list with elements
                -   list with the number of times the element was in a chemical composition that lead to a transition temperature equal or higher
                    than 'promising_cT'
                - list with the number of times the element was in a chemical composition
                - list with the ratio of returned list 2 and 3

    Remark: In the upper section when I was talking about 'element was connected to cases that lead to a transition temperature...' I of course
            meant: Element that was contained in a chemical composition with a transition temperature equal or higer than 'promising_cT'

    I assume it might be a bit difficult to understand what I meant with the different ways to set the elements which will be returned. Therefore I
    have some examples below:
    - return_number_ratio = 0.1: If the function collected 80 elements in total, the 'best' 8 will be returned
    - return_of_best_ratio = 0.8:   If 500 is the highest number an element appeared in a chemical composition that had a transition temperature 
                                    equal or higher than 'promising_cT', every element that appeared at least 400 times will be returned
    - return_ratio_occured_total = 0.5: If the highest ratio (represented by returned list 4) is 0.8, every element with a ratio of at least
                                        0.4 will be returned
    As you see, the higher way number 1, the higher the returned number of elements. The higher way number 2 and 3, the lower the returned elements.

    Suggestion: I prefer to use way number 3, because a ratio usually says more about the impact if the total number of occured times has about the 
                same size
    
    Remark: I know this is not a good way to determine what chemical composition will lead to a high transition temperature, but creating
            combinations of all elements with each element having a certain number of atoms in total, will lead to unimaginable sizes
    """

    if return_number_ratio > 0 and return_of_best_ratio > 0:
        print("It doesn't make sense to use more than 1 option that determine what elements are returned. 'return_of_best_ratio' was set to 0")
        return_of_best_ratio = 0
    if return_number_ratio > 0 and return_ratio_occured_total > 0:
        print("It doesn't make sense to use more than 1 option that determine what elements are returned. 'return_number_ratio' was set to 0")
        return_number_ratio = 0
    if return_of_best_ratio > 0 and return_ratio_occured_total > 0:
        print("It doesn't make sense to use more than 1 option that determine what elements are returned. 'return_of_best_ratio' was set to 0")
        return_of_best_ratio = 0

    supercon_file = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    lines = supercon_file.readlines()

    promising_els = []
    occured_times = []
    total_times = []

    for i in range(0, len(lines)):
        cc = lines[i]
        cc = correct_ending(cc)
        cc = correct_cc(cc)
        cc, cT = split_cc_cT(cc)
        els = extract_elements(cc)
        if cT >= promising_cT:
            for el in els:
                not_in = True
                for pe in range(0, len(promising_els)):
                    if promising_els[pe] == el:
                        occured_times[pe] += 1
                        total_times[pe] += 1
                        not_in = False
                if not_in:
                    promising_els.append(el)
                    occured_times.append(1)
                    total_times.append(1)
        else:
            for ele in els:
                for pel in range(0, len(promising_els)):
                    if promising_els[pel] == ele:
                        total_times[pel] += 1

    ordered_els = [promising_els[0]]
    ordered_times = [occured_times[0]]
    ordered_total = [total_times[0]]

    for i in range(1, len(occured_times)):
        bigger = False
        appended = False
        for j in range(0, len(ordered_times)):
            if occured_times[i] > ordered_times[j]:
                bigger = True
            if occured_times[i] < ordered_times[j]:
                appended = True
                if bigger:
                    ordered_times.insert(j, occured_times[i])
                    ordered_els.insert(j, promising_els[i])
                    ordered_total.insert(j, total_times[i])
                else:
                    ordered_times.insert(0, occured_times[i])
                    ordered_els.insert(0, promising_els[i])
                    ordered_total.insert(0, total_times[i])
                break
        if not appended:
            ordered_times.append(occured_times[i])
            ordered_els.append(promising_els[i])
            ordered_total.append(total_times[i])

    ratio = []
    for i in range(0, len(ordered_times)):
        ratio.append(ordered_times[i]/ordered_total[i])
    
    if return_number_ratio > 0:
        return_elements = int(len(ordered_els)*return_number_ratio)
        return ordered_els[len(ordered_els)-return_elements:len(ordered_els)], ordered_times[len(ordered_times)-return_elements:len(ordered_times)], ordered_total[len(ordered_total)-return_elements:len(ordered_total)], ratio[len(ratio)-return_elements:len(ratio)]
    elif return_of_best_ratio > 0:
        boarder = ordered_times[len(ordered_times)-1]*return_of_best_ratio
        print(boarder)
        for i in range(len(ordered_els)-1, 0, -1):
            if ordered_times[i] < boarder:
                return ordered_els[i+1:len(ordered_els)], ordered_times[i+1:len(ordered_times)], ordered_total[i+1:len(ordered_total)], ratio[i+1:len(ratio)]
    elif return_ratio_occured_total > 0:
        new_el = []
        new_oc = []
        new_total = []
        new_ratio = []
        best = 0
        for i in range(0, len(ordered_total)):
            if ratio[i] > best:
                best = ratio[i]
        return_ratio_occured_total*=best
        for i in range(0, len(ordered_total)):
            if ratio[i] >= return_ratio_occured_total:
                new_el.append(ordered_els[i])
                new_oc.append(ordered_times[i])
                new_total.append(ordered_total[i])
                new_ratio.append(ratio[i])
        return new_el, new_oc, new_total, new_ratio
    else:
        return ordered_els, ordered_times, ordered_total, ratio


def nice_print_promising_elements(promising_cT = 10, return_number_ratio = 0, return_of_best_ratio = 0, return_ratio_occured_total = 0):
    """
    MAIN FUNCTION

    This function is just here to visualize what 'promising_elements' returns. It takes the same arguments as 'promising_elements' and just prints
    the output to the console
    """
    elements, occured, total, ratio = promising_elements(promising_cT=promising_cT, return_number_ratio=return_number_ratio, return_of_best_ratio=return_of_best_ratio, return_ratio_occured_total=return_ratio_occured_total)

    size_el = len("Elements")
    for i in elements:
        if len(i) > size_el:
            size_el = len(i)
    size_occ = len("Hit")
    for i in occured:
        if len(str(i)) > size_occ:
            size_occ = len(str(i))
    size_total = len("Total")
    for i in total:
        if len(str(i)) > size_total:
            size_total = len(str(i))
    size_ratio = len("Ratio")
    for i in ratio:
        if len(str(i)) > size_ratio:
            size_ratio = len(str(i))
    print("")
    print("Elements" + " "*(size_el-len("Elements")) + "   Hit" + " "*(size_occ-len("Hit")) + "   Total" + " "*(size_total-len("Total")) + "   Ratio")
    print("-"*(size_el+size_occ+size_ratio+size_total+9))
    for i in range(0, len(elements)):
        ed = size_el-len(elements[i])
        od = size_occ-len(str(occured[i]))
        td = size_total-len(str(total[i]))
        rd = size_ratio-len(str(ratio[i]))
        print(elements[i], end=" "*ed + "   ")
        print(occured[i], end = " "*od + "   ")
        print(total[i], end = " "*td + "   ")
        print(ratio[i])
    print("\nIn total: " + str(len(elements)) + " elements\n")


def vector_from_simple_string(predictor_string):
    """
    SIDE FUNCTION

    This function gets a chemical composition without critical temperature (as string) and returns the elements and values of it in form of
    two arrays.

    This function might feel unnecessary because it works similar to 'transform_cc'. It is a little different because it transforms strings that
    are without critical temperature. I could have used this function above, but I didn't want to change the code that I wrote a while ago
    since it is working well.
    """
    elements = []
    values = []
    in_element = True
    start = 0
    for i in range(0, len(predictor_string)):
        if predictor_string[i] in alphabet and not in_element:
            in_element = True
            values.append(float(predictor_string[start:i]))
            start = i
        if predictor_string[i] in numbers and in_element:
            elements.append(predictor_string[start:i])
            in_element = False
            start = i
    values.append(float(predictor_string[start:len(predictor_string)]))

    return elements, values


def expand_vector(element_order, elements, values_of_elements):
    """
    SIDE FUNCTION

    This function takes the arrays that were created in 'vector_from_simple_string' and returns an array, that holds the values at the same spots, 
    as in the chemical composition matrix. To do this, the order of elements (first row in matrix) must be passed to this function as well.
    """
    total = 0
    for i in range(0, len(values_of_elements)):
        total += values_of_elements[i]
    for i in range(0, len(values_of_elements)):
        values_of_elements[i] = values_of_elements[i]/total
    expanded_vector = []
    for i in range(0, len(element_order)):
        if element_order[i] not in all_used_feature_names:
            expanded_vector.append(0.0)
        else:
            break
    for i in range(0, len(elements)):
        for j in range(0, len(element_order)):
            if elements[i] == element_order[j]:
                expanded_vector[j] = values_of_elements[i]
                break
    return expanded_vector


def add_properties(predicting_string, source, source_features):
    """
    SIDE FUNCTION

    This function returns the elemental features for the given string. It uses all passed sources and source features
    """
    feature_values = []
    comp = Composition(predicting_string)
    for s in range(0, len(source)):
        prop = mfcc.ElementProperty(source[s], source_features[s], ['mean'])
        values = prop.featurize(comp)
        for v in values:
            feature_values.append(v)
    return feature_values


def filter_dataset(ccm = None, cT = None, filter_for_temperature = [0,200], filter_for_elements = [], filter_out_element_combinations = [[]]):
    """
    SIDE FUNCTION

    This function applys the following filter on the data files. It will alter the "used data" and "error documentation" file. Additionally it will 
    create a file with the currently removed datapoints.

    PARAMETERS:
    - ccm: chemical composition matrix (nested array)
    - cT: transition temperature vector (array)
        - If "ccm" and "cT" are passed to the function, it will also remove the filtered datapoints from these arrays and return them at the end.
        ATTENTION:  - They have to be the corresponding arrays to the currently accessed "used data" file
                    - If only one is passed to the function, it will be ignored
    - filter_for_temperature: (array of 2 values) Any entry that isn't in this temperature range will be removed
    - filter_for_elements:  (array) if an entry has an element that is in this list it will not be removed 
                            if this array is empty, no elements will be removed (special case: "filter_out_element_combinations")
    - filter_out_element_combinations:  (nested array) Any ccm that contains both elements of each (single) array will be remomved, even if they are 
                                        in "filter_for_elements"
    """
    documentation_file = open(CHANGING_DATA_DIRECTORY_NAME + ERROR_DOCUMENTATION_FILE_NAME, "a")
    documentation_file.write("\n\nChemical compositions removed from \"filter\":\n\n")
    filtered_file = open(CHANGING_DATA_DIRECTORY_NAME + FILTERED_COMPOSITIONS_FILE_NAME, "w")

    for i in range(0, len(filter_for_elements)):
        if [filter_for_elements[i]] in filter_out_element_combinations:
            del filter_for_elements[i]

    supercon_file = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    lines = supercon_file.readlines()
    supercon_file.close()
    new_lines = []

    return_new = True
    if ccm == None or cT == None:
        return_new = False
        cT = []
        ccm = []
        for i in range(0, len(lines)):
            ccm.append(0)
            cT.append(0)

    counter = 0
    for i in range(0, len(lines)):
        append_cc = False
        cc = correct_ending(lines[i])
        current_cT = extract_cT(copy.deepcopy(cc))
        if current_cT >= filter_for_temperature[0] and current_cT <= filter_for_temperature[1]:
            elements_in_line = extract_elements(copy.deepcopy(cc))
            for el in elements_in_line:
                if el in filter_for_elements or len(filter_for_elements)==0:
                    append_cc = True
            if append_cc:
                for comb in filter_out_element_combinations:
                    el_not_in = False
                    for el in comb:
                        if el not in elements_in_line:
                            el_not_in = True
                    if not el_not_in and len(comb)>0:
                        append_cc = False
                        break
                    
        if append_cc:
            new_lines.append(cc)
        else:
            del ccm[i-counter]
            del cT[i-counter]
            counter += 1
            documentation_file.write(cc + "\n")
            filtered_file.write(cc + "\n")
    
    documentation_file.write("\nIn total: " + str(counter) + " datapoints")
    documentation_file.close()
    filtered_file.close()
    supercon_file = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "w")
    for ccs in new_lines:
        supercon_file.write(ccs + "\n")
    supercon_file.close()

    if return_new:
        return ccm, cT


def used_data_ordered(ccm=None, cT=None, print_first = 10, Path = CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME):
    """
    MAIN FUNCTION

    This functions orders the current "used_data" file. If both "ccm" and "cT" that correspond to the data in "used data" are passed to it,
    they will be ordered as well.

    ADDITIONAL PARAMETERS:
    - print_first: (integer) determines how many of the datapoints with the highest transition temperature will be printed to the console
    """
    supercon_file = open(Path, "r")
    lines = supercon_file.readlines()
    supercon_file.close()

    return_new = True
    if ccm == None or cT == None:
        return_new = False
        cT = []
        ccm = []
        for i in range(0, len(lines)):
            ccm.append(0)
            cT.append(extract_cT(correct_ending(lines[i])))

    new_lines = [correct_ending(lines[0])]
    new_ccm = [ccm[0]]
    new_cT = [cT[0]]

    for i in range(1, len(cT)):
        appended = False
        for j in range(0, len(new_cT)):
            if cT[i] < new_cT[j]:
                new_cT.insert(j, cT[i])
                new_ccm.insert(j, ccm[i])
                new_lines.insert(j, correct_ending(lines[i]))
                appended = True
                break
        if not appended:
            new_cT.append(cT[i])
            new_ccm.append(ccm[i])
            new_lines.append(correct_ending(lines[i]))
    
    supercon_file = open(Path, "w")
    for i in range(0, len(new_lines)):
        supercon_file.write(new_lines[i])
        if i < len(new_lines) - 1:
            supercon_file.write("\n")
        if i < print_first:
            print(new_lines[len(new_lines)-i-1])
    supercon_file.close()

    if return_new:
        return new_ccm, new_cT
    

def filter_random_np_matrix(random_ccm, random_cT, order_elements, filter_for_temperature = [0,200], filter_for_elements = [], filter_out_element_combinations = [[]]):
    """
    MAIN FUNCTION

    This function will in analogy to "filter_dataset" remove entrys from a matrix. 

    Returns: remaining ccm, remaining cT, removed ccm, removed cT

    PARAMETERS:
    - random_ccm: (nested array) chemical composition matrix that is supposed to go through the filter
    - random_cT: (array) transition temperature vector that is supposed to go through the filter
    - order_elements: (array) that tells the function to which element each column of "random_ccm" corresponds
    - filter_for_temperature: (array of 2 values) Any entry that isn't in this temperature range will be removed
    - filter_for_elements:  (array) if an entry has an element that is in this list it will not be removed 
                            if this array is empty, no elements will be removed (special case: "filter_out_element_combinations")
    - filter_out_element_combinations:  (nested array) Any ccm that contains both elements of each (single) array will be remomved, even if they are 
                                        in "filter_for_elements"
    """
    filtered_file = open(CHANGING_DATA_DIRECTORY_NAME + FILTERED_COMPOSITIONS_FROM_ANY_FILE_NAME, "a")
    filtered_file.write("\nSet to filter: ")
    filtered_file.write("temperature_range: " + str(filter_for_temperature[0]) + " to " + str(filter_for_temperature[1]) + "\n")
    filtered_file.write("\ninclude elements: ")
    for el in filter_for_elements:
        filtered_file.write(el + "  ")
    filtered_file.write("\nremove compositions with combination: ")
    for combi in filter_out_element_combinations:
        for el in range(0, len(combi)):
            if el < len(combi)-1:
                filtered_file.write(combi[el] + ";")
            else:
                filtered_file.write(combi[el])
        filtered_file.write("  ")
    filtered_file.write("\n\n")

    for i in range(0, len(filter_for_elements)):
        if [filter_for_elements[i]] in filter_out_element_combinations:
            del filter_for_elements[i]

    removed_matrix = []
    removed_cT = []
    counter = 0
    for i in range(0, len(random_ccm)):
        append_cc = False
        current_cT = random_cT[i-counter]
        cc = ""
        elements_in_line = []
        relative_atom_numbers = []
        for j in range(0, len(random_ccm[i-counter])):
            if random_ccm[i-counter][j] != 0 and order_elements[j] not in all_used_feature_names:
                elements_in_line.append(order_elements[j])
                relative_atom_numbers.append(random_ccm[i-counter][j])
        for j in range(0, len(elements_in_line)):
            cc += elements_in_line[j]
            cc += str(round(relative_atom_numbers[j], 2))
        cc += ","
        cc += str(current_cT)
        if current_cT >= filter_for_temperature[0] and current_cT <= filter_for_temperature[1]:
            for el in elements_in_line:
                if el in filter_for_elements or len(filter_for_elements)==0:
                    append_cc = True
            if append_cc:
                for comb in filter_out_element_combinations:
                    el_not_in = False
                    for el in comb:
                        if el not in elements_in_line:
                            el_not_in = True
                    if not el_not_in and len(comb)>0:
                        append_cc = False
                        break

        if not append_cc:
            removed_matrix.append(random_ccm[i-counter])
            removed_cT.append(random_cT[i-counter])
            random_ccm = np.delete(random_ccm, i-counter, axis=0)
            random_cT = np.delete(random_cT, i-counter)
            counter += 1
            filtered_file.write(cc + "\n")

    filtered_file.close()
    return random_ccm, random_cT, removed_matrix, removed_cT
    

     
def transform_to_logarithmic(vector, epsilon = 0.5):
    """
    SIDE FUNCTION

    Takes vector of numbers and returns vector of corresponding logarithmic (values+epsilon) -> to avoid ln(0)
    """
    for i in range(0, len(vector)):
        vector[i] = math.log(vector[i]+epsilon)
    return vector


def transform_to_euler(vector, epsilon = 0.5):
    """
    SIDE FUNCTION

    Takes vector of numbers and returns vector with corresponding exponential values + epsilon -> to fix epsilon in "transform_to_logarithmic"
    """
    for i in range(0, len(vector)):
        vector[i] = math.exp(vector[i]) - epsilon
    return vector


def extract_cT(string):
    """
    SIDE FUNCTION

    extracts critical temperature from entry in supercon database (removes chemical composition)
    """
    for i in range(0, len(string)):
        if string[i] == ',':
            start = i
    return float(string[start+1:len(string)])


def extract_cc(string):
    """
    SIDE FUNCTION

    extracts chemical composition from entry in supercon database (removes critical temperature)
    """
    for i in range(0, len(string)):
        if string[i] == ',':
            end = i
            break
    return string[0:end]


def extract_pattern_numbers(string: str):
    """
    SIDE FUNCTION

    This function is given a pattern (for example: "A1A2A2A3" or "A1B1C1G4H6") and returns an array with the numbers (for the previous example:
    ["1", "2", "2", "3"], ["1", "1", "1", "4", "6"])
    """
    pattern_numbers = []
    in_number = False
    start = 0
    for i in range(0, len(string)):
        if string[i] in numbers and not in_number:
            in_number = True
            start = i
        if string[i] not in numbers and in_number:
            in_number = False
            pattern_numbers.append(string[start:i])
    if in_number:
        pattern_numbers.append(string[start:len(string)])
    return pattern_numbers
        

def create_composition_from_pattern(string: str):
    """
    SIDE FUNCTION

    This function is given a pattern (for example: "A1A2A2A3") and returns all possible combinations for this pattern, with all existing elements
    in the previous created chemical composition matrix for training the data.

    ATTENTION: Expects real pattern (with sorted number of atoms)
    """
    pattern = correct_cc(string)
    pattern_numbers = extract_pattern_numbers(pattern)

    supercon_file = open(CHANGING_DATA_DIRECTORY_NAME + USED_DATA_FILE_NAME, "r")
    lines = supercon_file.readlines()

    elements = []
    maxx = [] #no need for us, but the function that collects the elements needs it to work

    for i in range(0, len(lines)):
        cc = lines[i]
        cc = correct_ending(cc)
        cc = correct_cc(cc)
        elements, maxx = get_all_elements(cc, elements, maxx)
    del maxx
    #elements = ["A", "B", "C", "D"]

    times_same_number = [1]
    last_pattern_num = pattern_numbers[0]
    counter = 0
    for i in range(1, len(pattern_numbers)):
        if pattern_numbers[i-counter] != last_pattern_num:
            last_pattern_num = pattern_numbers[i-counter]
            times_same_number.append(1)
        else:
            times_same_number[len(times_same_number)-1] += 1
            del pattern_numbers[i-counter]
            counter += 1

    basic_combinations = []
    for pat_num in pattern_numbers:
        current_basic = []
        for el in elements:
            current_basic.append(el+pat_num)
        basic_combinations.append(current_basic)
    del current_basic

    first_ccs = []
    for i in range(0, len(basic_combinations)):
        repeat = times_same_number[i]
        first_ccs.append(basic_combinations[i])
        for j in range(1, repeat):
            new_comp = []
            current_comps = copy.deepcopy(first_ccs[len(first_ccs)-1])
            for k in range(0, len(basic_combinations[i])):
                counter = 0
                for l in range(0, len(current_comps)):
                    if same_element(current_comps[l-counter], basic_combinations[i][k]):
                        del current_comps[l-counter]
                        counter += 1
                for l in range(0, len(current_comps)):
                    new_comp.append(basic_combinations[i][k]+current_comps[l])
            del first_ccs[len(first_ccs)-1]
            first_ccs.append(new_comp)

    ccs = first_ccs[0]
    for i in range(1, len(first_ccs)):
        new_comp = []
        for comp in ccs:
            for j in range(0, len(first_ccs[i])):
                if not same_element(comp, first_ccs[i][j]):
                    new_comp.append(comp + first_ccs[i][j])
        del ccs
        ccs = new_comp

    return ccs


def create_composition_matrix_from_pattern(string: str, element_order, with_properties, source = suggested_source, source_features = suggested_features, return_strings = True, remove_failing_entry = True):
    """
    MAIN FUNCTION

    This function creates new chemical compositions with the pattern that was passed to it. Those can be used for prediction later on. All elements 
    that were used in the chemical composition matrix for training the machine learning model, will be used to create these new compositions.

    ATTENTION: Expects real pattern (with sorted number of atoms)

    RETURNS: new chemical compositions as matrix (optional: and their corresponding strings)

    PARAMETERS:
    - string: pattern for creation (for example: "A1A2A2A3" or "A1B1C1G4H6", letters don't matter but numbers have to be sepaerated by at least one)
    - element_order: (array) - first array in the created chemical composition matrix from SuperCon database data
    - with_properties:  If True: creates matrix with elemental properties
                        If False: creates matrix without elemental properties
    - source: (array) holds all sources used for retrieving elemental properties
    - source_features: (nested array) holds array of elemental properties (for example ["atomic_weight", "electronegativity"]) for each source
    - return strings: If "True": also returns the chemical compositions as strings
    """
    compositions_to_predict = create_composition_from_pattern(string)

    new_combinations = []
    count_deletions = 0

    report_marker = round(len(compositions_to_predict)/100)
    for i in range(0, len(compositions_to_predict)):
        if i%report_marker == 0:
            print("PROGRESSION-REPORT: " + str(round(i/report_marker)) + "% of compositions were transformed into a matrix")

        append = True
        comp_string = compositions_to_predict[i-count_deletions]
        comp_elements, comp_values = vector_from_simple_string(comp_string)
        new_simple_cc = expand_vector(element_order, comp_elements, comp_values)
        if with_properties:
            try:
                simple_properties = add_properties(comp_string, source, source_features)
                for simple_prop in simple_properties:
                    new_simple_cc.append(simple_prop)
            except:
                del compositions_to_predict[i-count_deletions]
                count_deletions += 1
                append = False
        if append:
            new_combinations.append(new_simple_cc)
    
    if return_strings:
        return new_combinations, compositions_to_predict
    else:
        return new_combinations


def create_pickle_predictors(patterns, Path=SUPERCON_PATH):
    """
    creates compositions to predict in a matrix (with material properties) and writes
    it to a pickle file.

    Arguments:
    - patterns: list of chemical patterns to create chemical compositions from
    """
    original_ccm1, original_cT = create_cc(Path)

    name_order = original_ccm1[0]
    for i in range(0, len(suggested_source)):
        for j in all_features[i]:
            name_order.append(j)

    for pat in patterns:
        matrix = []
        strings = create_composition_from_pattern(pat)
        counter = 0
        report_marker = round(len(strings)/10)

        for string in strings:
            counter += 1
            if counter%report_marker == 0:
                print("PROGRESSION-REPORT: " + str(round(counter/report_marker)) + "% done")
            comp_elements, comp_values = vector_from_simple_string(string)
            new_simple_cc = expand_vector(original_ccm1[0], copy.deepcopy(comp_elements), copy.deepcopy(comp_values))
            comp = Composition(string)
            start_time = time.time()
            for i in range(0, len(suggested_source)):
                copy_features = copy.deepcopy(all_features[i])
                removed_position = []
                deletion_counter = 0
                for o in range(0, len(copy_features)):
                    this_features_failing_elements = failing_elements_dict[copy_features[o-deletion_counter]]
                    for p in range(0, len(comp_elements)):
                        if comp_elements[p] in this_features_failing_elements:
                            removed_position.append(o)
                            del copy_features[o-deletion_counter]
                            deletion_counter += 1
                            break
                try:
                    prop = mfcc.ElementProperty(suggested_source[i], copy_features, ['mean'])
                    values = prop.featurize(comp)
                    deletion_counter = 0
                    for v in range(0, len(values)+len(removed_position)):
                        if v in removed_position:
                            new_simple_cc.append(None)
                            deletion_counter += 1
                        else:
                            new_simple_cc.append(values[v-deletion_counter])
                except:
                    for q in range(0, len(suggested_source)):
                        for m in range(0, len(all_features[q])):
                            new_simple_cc.append(None)
            end_time = time.time()
            print(round(end_time-start_time,3))
            matrix.append(new_simple_cc)

        #print(len(name_order), len(matrix[0]))
        
        pickle_save = dict()
        pickle_save["matrix"] = matrix
        pickle_save["string"] = strings
        pickle_save["order"] = name_order

        with open("Pickle_data/predictors/" + pat + ".pkl", "wb") as fid:
            pickle.dump(pickle_save, fid)


all_patterns = ["A2B2C15", "A2B3C4", "A2B3C9", "A3B3C5", "A3B4C8", "AB2C7", "AB3C7", "AB5C7", "A2B2C3", "A2B3C5", "A2B4C5", "A3B3C8", 
                "A4B6C7", "AB2C8", "AB3C8", "ABC", "A2B2C5", "A2B3C6", "A2B5C12", "A3B4C12", "AB2C10", "AB2C9", "AB4C12", "A2B2C7",   
                "A2B3C7", "A2B5C5", "A3B4C4", "AB2C2", "AB3C4", "AB4C4", "A2B3C3", "A2B3C8", "A3B3C4", "A3B4C5", "AB2C3", "AB3C5",  
                "AB5C5", "AB2C4", "AB2C6", "AB3C3", "AB4C5", "AB4C7", "AB5C6", "AB6C12", "AB6C8", "ABC2", "ABC4", "AB2C5", "AB2C8",  
                "AB3C6", "AB4C6", "AB4C8", "AB5C8", "AB6C6", "AB7C12", "ABC3"]