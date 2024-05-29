import pandas as pd
import numpy as np
import json
from rdkit.Chem.PandasTools import LoadSDF
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.inchi import MolToInchi

def can_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles),True)
    except:
        return np.nan

def can_smiles_finchi(inchi):
    try:
        return Chem.MolToSmiles(Chem.MolFromInchi(inchi),True)
    except:
        return np.nan

def extract_children(data):
    """
    Recursively extract children from the hierarchical JSON data.
    """
    children = []
    
    def recurse(node):
        if isinstance(node, dict):
            if 'children' in node and isinstance(node['children'], list):
                for child in node['children']:
                    recurse(child)
            else:
                children.append(node)
        elif isinstance(node, list):
            for item in node:
                recurse(item)

    recurse(data)
    return children

def parse_KEGG_json_to_df(json_file_path):
    """
    Read a hierarchical JSON file and convert its children into a pandas DataFrame.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    children = extract_children(data)
    
    # Convert the list of children to a DataFrame
    df = pd.DataFrame(children)
    df[['cpd', 'cpd_name']] = df['name'].str.split('  ', expand=True)
    df = df[['cpd','cpd_name']]
    
    return df

def preprocess_db(df, smiles_col=None, inchi_col=None, drop_cols=None, rename_cols=None):
    if smiles_col is None and inchi_col is None:
        raise ValueError("Either 'smiles_col' or 'inchi_col' must be provided")
    
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if rename_cols:
        df = df.rename(columns=rename_cols)
    if inchi_col:
        df['can_smiles'] = df[inchi_col].apply(can_smiles_finchi)
    else:
        df['can_smiles'] = df[smiles_col].apply(can_smiles)
    return df.dropna(subset=['can_smiles'])

def intersect_smiles(df1, df2, smiles_col1='Smiles', smiles_col2='can_smiles'):
    return list(set(df1[smiles_col1]).intersection(df2[smiles_col2]))

def neutral(test):
    def process_suffix(suffix, increment):
        if not suffix:
            return 'H2' if increment > 0 else 'H'

        if suffix[0].isdigit():
            i = 1
            while i < len(suffix) and suffix[i].isdigit():
                i += 1
            n = int(suffix[:i]) + increment
            return f'H{n}' + suffix[i:]
        return 'H2' + suffix if increment > 0 else 'H' + suffix

    if '-' in test:
        prefix, rest = test.split('-')[0], test.split('-')[0]
        if 'H' in rest:
            prefix, suffix = rest.split('H')
            return prefix + process_suffix(suffix, 1)
        return prefix + 'H'
    
    if '+' in test and 'H' in test:
        prefix, suffix = test.split('+')[0].split('H')
        return prefix + process_suffix(suffix, -1)
    
    return test