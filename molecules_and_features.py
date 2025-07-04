import math
import os

import pandas as pd
from numpy import inf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors


def generate_molecules(dataset_name, data_dir=None, endpoint=None):
    """
    Takes in the path to a QSAR-ready .sdf file and generates a list of rdkit molecule objects
    Returns training and external validation sets without molecules that could not be created from SMILES strings or
    molecules with no information about the desired endpoint
    Returns prediction sets without molecules that could not be created from SMILES strings

    :param dataset_name: String representing dataset name in .sdf file
    :param data_dir: The project directory containing the dataset
    :param endpoint: Desired binary or continuous endpoint with threshold to be modeled (for training sets and external
    validation sets, defaults to None); Enter None for prediction sets

    :return molecules: List of rdkit molecule objects
    """

    # Checks for appropriate input
    assert data_dir is not None, \
        f'Please create an environment variable called {data_dir} pointing to the project directory containing the ' \
        f'dataset.'

    # Instantiates data_dir and sdf_file variables
    sdf_file = os.path.join(data_dir, f'{dataset_name}.sdf')

    # Checks for appropriate input
    assert os.path.exists(
        sdf_file), f'The dataset entered {dataset_name} is not present in data_directory as a  .sdf file'

    if endpoint is None:
        # Returns rdkit Mol objects for molecules in prediction set .sdf files that were able to be generated
        molecules = [mol for mol in Chem.SDMolSupplier(sdf_file) if mol is not None]

        # Checks for appropriate output
        assert molecules != [], f'No molecules could be generated from the dataset provided ({dataset_name}).'

        return molecules

    else:
        # Returns rdkit Mol objects for molecules in training and evaluation set .sdf files that were able to be
        # generated and have information pertaining to the desired endpoint
        molecules = [mol for mol in Chem.SDMolSupplier(sdf_file) if mol is not None and mol.HasProp(endpoint)]

        # Checks for appropriate output
        assert molecules != [], f'No molecules could be generated with the given endpoint ({endpoint})'

        return molecules


def calc_rdkit(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules, calculates molecular descriptors for each molecule, and returns a machine
    learning-ready pandas DataFrame.

    :param molecules: List of rdkit molecule objects with no None values
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert None not in molecules, 'The list of molecules entered contains None values.'

    # Generates molecular descriptor calculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])

    # Calculates descriptors and stores in pandas DataFrame
    X = pd.DataFrame([list(calculator.CalcDescriptors(mol)) for mol in molecules],
                     index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules],
                     columns=list(calculator.GetDescriptorNames()))

    # Imputes the data and replaces NaN values with mean from the column
    desc_matrix = X.fillna(X.mean())

    # Removes descriptors with infinity values
    desc_matrix = desc_matrix.loc[:, ~desc_matrix.isin([inf, -inf]).any(axis=0)]

    # Checks for appropriate output
    assert len(desc_matrix.columns) != 0, 'All features contained at least one null value. No descriptor matrix ' \
                                          'could be generated.'

    return desc_matrix


def calc_ecfp6(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules and returns ECFP6 fingerprints for a list of rdkit molecules

    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecule objects with no None values

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert None not in molecules, 'The list of molecules entered contains None values.'

    data = []

    for mol in molecules:
        ecfp6 = [float(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024)]
        data.append(ecfp6)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def calc_fcfp6(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules and returns FCFP6 fingerprints for a list of rdkit molecules

    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecules with no None values

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert None not in molecules, 'The list of molecules entered contains None values.'

    data = []

    for mol in molecules:
        fcfp6 = [float(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024, useFeatures=True)]
        data.append(fcfp6)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def calc_maccs(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules and returns MACCS fingerprints for a list of rdkit molecules

    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecules with no None values

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert None not in molecules, 'The list of molecules entered contains None values.'

    data = []

    for mol in molecules:
        maccs = [float(x) for x in MACCSkeys.GenMACCSKeys(mol)]
        data.append(maccs)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def get_activities(molecules, name_col='CASRN', endpoint=None, threshold=None):
    """
    Takes in a list of rdkit molecules and returns a vector with a value of 1 if the indexed molecule fits the desired
    binary endpoint and 0 if it does not

    :param molecules: List of rdkit molecule objects with no None values
    :param name_col: Name of the field to index the resulting Series.  Needs to be a valid property of all molecules
    :param endpoint: Desired property to be modeled (defaults to None). Needs to be a valid property of all molecules
    :param threshold: Toxicity threshold value for binary endpoints based on continuous data where values falling
    below the threshold will constitute an active response and vice versa(i.e. LD50 in mg/kg, defaults to None)

    :return y: Activity vector as pandas Series
    """

    # Checks for appropriate input
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert all(mol.HasProp(endpoint) for mol in molecules), 'The desired endpoint is not valid for all molecules in ' \
                                                            'the input list.'
    assert all(mol.GetProp(endpoint) != '' and mol.GetProp(endpoint) != 'NA' for mol in molecules), \
        'The desired endpoint is not valid for all molecules in the input list.'
    assert all(mol.HasProp(name_col) for mol in molecules), \
        f'The input parameter name_col {name_col} must be a valid property of all molecules to be modeled.'
    assert endpoint is not None, 'Please enter a binary endpoint or continuous endpoint and threshold value.'

    # Instantiates activity vector
    y = []

    # Checks for binary endpoints
    if molecules[0].GetProp(endpoint).title() in ['True', 'False']:
        for mol in molecules:
            # Populates the activity vector with a 1 for molecules with a 'True' endpoint and vice versa
            y.append(int(eval(mol.GetProp(endpoint).title())))

        return pd.Series(y, index=[mol.GetProp(name_col) for mol in molecules])

    elif molecules[0].GetProp(endpoint) in ['0.0', '0', '1.0', '1']:

        for mol in molecules:
            y.append(int(float(mol.GetProp(endpoint))))

        return pd.Series(y, index=[mol.GetProp(name_col) for mol in molecules])

    # Populates the activity vector with a 1 for molecules with a continuous endpoint below the threshold and vice versa

    else:
        y_continuous = []
        
        for mol in molecules:
            continuous_value = float(mol.GetProp(endpoint))
            y_continuous.append(math.log10(continuous_value))
            
        return pd.Series(y_continuous, index=[mol.GetProp(name_col) for mol in molecules])


def load_external_desc(dataset_name, features, data_dir=None, pred_set=False, training_set=None, endpoint=None,
                       threshold=None):
    """
    Loads externally generated descriptor values stored in .csv files saved to a sub-folder of data_dir titled
    'external_descriptors' with the format '(sdf_file)_descriptors.sdf'

    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    assert data_dir is not None, \
        'Please set an environmental variable pointing to your project directory to use this function.'
    assert os.path.exists(data_dir), \
        f'Please create an environment variable called {data_dir} pointing to the project directory containing the ' \
        f'dataset.'

    file = os.path.join(data_dir, 'external_descriptors', f'{dataset_name}_{features}.csv')

    assert os.path.exists(
        file), f'Externally generated descriptor values must be stored in .csv files saved to a sub-folder of ' \
               f'data_dir titled \'external_descriptors\' with the format \'{dataset_name}_{features}.csv\''

    X = pd.read_csv(file, index_col=0)
    df = X.copy()
    df.loc[:, df.isnull().all()] = -999
    df.fillna(X.mean(), inplace=True)
    df = df.loc[:, ~df.isin([inf, -inf]).any(axis=0)]

    if pred_set:
        X_train, y_train = make_dataset('{}.sdf'.format(training_set), data_dir=os.getenv('NICEATM_ACUTE_ORAL_DATA'),
                                        pred_set=False, features=features, endpoint=endpoint, threshold=threshold)
        df = df[X_train.columns]

    return df


def get_classes(molecules, name_col='CASRN', class_col='Class'):
    return pd.Series([float(mol.GetProp(class_col)) for mol in molecules], index=[mol.GetProp(name_col) if mol.HasProp(
        name_col) else '' for mol in molecules])


def make_dataset(sdf_file, data_dir=None, pred_set=False, features='MACCS', name_col='CASRN', endpoint=None,
                 threshold=None, cache=True):
    """
    :param sdf_file: Name of the .sdf file from which to make a dataset
    :param data_dir: Environmental variable pointing to the project directory
    :param pred_set: True if the dataset is a prediction set, False otherwise (defaults to False)
    :param features: Molecular descriptor set to be used for modeling (defaults to MACCS keys); If externally
    generated descriptors are used, they must be saved to a sub-folder of data_dir titled 'external_descriptors' with
    the format 'datasetname_features.sdf'
    :param name_col: Name of the field to index the resulting Series.  Needs to be a valid property of all molecules
    :param endpoint: Desired property to be modeled (defaults to None). Needs to be a valid property of all
    molecules
    :param threshold: Toxicity threshold value for binary endpoints based on continuous data where values falling
    below the threshold will constitute an active response and vice versa(i.e. LD50 in mg/kg, defaults to None)

    If it is the first time the descriptors (and activities, when applicable) are being loaded, it will generate a
    descriptor matrix and activity vector for training and external validation sets and then cache the results to a
    .csv file for easier future loading. Descriptor matrices will be generated and cached for prediction sets.

    :return (X, y): (Feature matrix as a pandas DataFrame, Class labels as a pandas Series (Only returned for
    training and external validation sets))
    """

    # Checks for appropriate input
    assert sdf_file[-4:] == '.sdf', 'The input parameter sdf_file must be in .sdf file format.'
    assert data_dir is not None, 'Please set an environmental variable pointing to your project directory to use this' \
                                 'function.'

    if endpoint is None and not pred_set:
        raise Exception('Please enter a binary or continuous endpoint for modeling.')

    if endpoint == 'Class':
        threshold = 'X'

    descriptor_fxs = {
        'rdkit': lambda molecules: calc_rdkit(molecules, name_col=name_col),
        'ECFP6': lambda molecules: calc_ecfp6(molecules, name_col=name_col),
        'FCFP6': lambda molecules: calc_fcfp6(molecules, name_col=name_col),
        'MACCS': lambda molecules: calc_maccs(molecules, name_col=name_col)
    }

    dataset_name = sdf_file.split('.')[0]
    data_dir = os.getenv(data_dir)

    if not os.path.exists(os.path.join(data_dir, 'caches')):
        os.makedirs(os.path.join(data_dir, 'caches'))

    # Creates X and y for external features
    if os.path.exists(os.path.join(data_dir, 'external_descriptors', f'{dataset_name}_{features}.csv')):
        molecules = generate_molecules(dataset_name, data_dir, endpoint)

        if threshold is not None:
            X, y = load_external_desc(dataset_name, features, data_dir), get_activities(molecules, name_col,
                                                                                        endpoint, threshold)
            X = X.loc[y.index]
            df = X.copy()
            df['Class'] = y

            if cache:
                df.to_csv(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_{endpoint}_{threshold}.csv'))

            return X, y

        if pred_set:
            if os.path.exists(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_prediction_set.csv')):
                return pd.read_csv(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_prediction_set.csv'),
                                   index_col=0)

            else:
                X = load_external_desc(dataset_name, features, data_dir)

                if cache:
                    X.to_csv(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_{endpoint}.csv'))

                return X

        else:
            # changing the line of code here from get_act to get_classes 
            # because if im following the logic correctly anytime
            # threshold is none, it would be a categorical endpoint and should call get_classes
            X, y = load_external_desc(dataset_name, features, data_dir), get_classes(molecules, name_col, endpoint)
            X = X.loc[y.index]
            df = X.copy()
            df['Class'] = y

            if cache:
                df.to_csv(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_{endpoint}.csv'))

            return X, y

    if pred_set:
        if os.path.exists(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_prediction_set.csv')):
            return pd.read_csv(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_prediction_set.csv'),
                               index_col=0)

        else:
            molecules = generate_molecules(dataset_name, data_dir, endpoint=None)
            X = descriptor_fxs[features](molecules)
            df = X.copy()

            if cache:
                df.to_csv(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_prediction_set.csv'))

            return X

    else:
        molecules = generate_molecules(dataset_name, data_dir, endpoint=endpoint)
        X = descriptor_fxs[features](molecules)
        
        if threshold is not None:
            y = get_activities(molecules, name_col, endpoint, threshold)
        
        else:
            y = get_classes(molecules, name_col, endpoint)
        
        df = X.copy()
        df['Class'] = y

        if cache:
            if threshold is not None:
                df.to_csv(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_{endpoint}_{threshold}.csv'))

            else:
                df.to_csv(os.path.join(data_dir, 'caches', f'{dataset_name}_{features}_{endpoint}.csv'))

        return X, y
