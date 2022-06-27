import pickle
import os
import sys
import gc
import multiprocessing
import functools
import numpy as np
import pandas as pd
from pathlib import Path

from minedatabase.pickaxe import Pickaxe


# Get location of this file
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Setup logging before starting script
import logging
logging.getLogger().setLevel(logging.INFO)
logfile = f"{__location__}/dataframe.log"
if(os.path.isfile(logfile)):
        os.remove(logfile)
file_handler = logging.FileHandler(logfile)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger = logging.getLogger("create_dataframe")
logger.addHandler(file_handler)


# Create ModelSEED dictionaries
with open("biochem.pik", "rb") as f:
    biochem = pickle.load(f)

modelseed_inchikeys_full = {
    biochem.get_seed_compound(cpd_id).inchikey: cpd_id
    for cpd_id in biochem.compounds if
    biochem.get_seed_compound(cpd_id).inchikey is not None
}

# Only keep the first 14 charcters (before first hyphen)
modelseed_inchikeys_partial = {
    key.split("-")[0]: val for key, val in modelseed_inchikeys_full.items()
}


def make_default_dataframe():
    """Make the default dataframe from ModelSEED data. Includes compound IDs, peak 
    numbers, and SMILES columns.
    """
    inchikey_connectivity = {
        i: biochem.get_seed_compound(cpd_id).inchikey.split("-")[0]
        for i, cpd_id in enumerate(biochem.compounds.keys())
        if biochem.get_seed_compound(cpd_id).inchikey is not None
    }
    df = pd.DataFrame(
        {
            "InChI_Key": list(inchikey_connectivity.values()),
            "ModelSEED id": [
                cpd.get("id", None) for i, cpd in enumerate(biochem.compounds.values())
                if i in inchikey_connectivity
            ],
            "Name":[
                cpd.get("name", None) for i, cpd in enumerate(biochem.compounds.values())
                if i in inchikey_connectivity
            ],
            "SMILES": [
                cpd.get("smiles", None) for i, cpd in enumerate(biochem.compounds.values())
                if i in inchikey_connectivity
            ],
        }    
    )
    return df.set_index("InChI_Key", drop=True)


def load_df_from_pickle(file_name: str):
    """Load a previous dataframe from a pcikle object"""
    with open(file_name, "rb") as file:
        return pickle.load(file)


def load_pickaxe_from_pickle(file_path: str) -> Pickaxe:
    """Loads a pickeled Pickaxe object from the specified file path and returns the 
    Pickaxe object.
    # TODO: Load from remote file?

    Arguments:
        (str) file_path: Path to pickle object

    Returns:
        Pickaxe object
    """
    with open(file_path, "rb") as f:
        pk = pickle.load(f)

    return pk


def _get_inchikeys_and_smiles(index, cpd):
    key = cpd["InChI_key"].split("-")[0]
    sm = cpd["SMILES"]
    return key, sm if key not in index and cpd["Type"] != "Coreactant" else None


def _get_inchikeys_and_generations(cpd):
    key = cpd["InChI_key"].split("-")[0]
    gen = cpd["Generation"]
    return key, gen if cpd["Type"] != "Coreactant" else None


def _get_item_from_dict(d, key):
    """Function def to allow multiprocessing indexing of dictionary"""
    return d[key] if key in d else None


def add_pickaxe_run_to_df(df, pk, col_name, processes):
    """Loads the generation number a compound is found into the dataframe"""
    # Get all inchikeys, smiles, and generations in pickaxe object
    # stored as list of tuple (inchikey, smiles)
    get_inchikeys_and_smiles_partial = functools.partial(
        _get_inchikeys_and_smiles,
        df.index
    )
    with multiprocessing.Pool(processes=processes) as pool:
        new_inchikey_smiles = pool.map(
            get_inchikeys_and_smiles_partial,
            pk.compounds.values()
        )
        inchikeys_generations = pool.map(
            _get_inchikeys_and_generations,
            pk.compounds.values()
        )

    # Remove all None from list
    while None in new_inchikey_smiles:
        new_inchikey_smiles.pop(new_inchikey_smiles.index(None))
    while None in inchikeys_generations:
        inchikeys_generations.pop(inchikeys_generations.index(None))

    # Add new indexes to list
    new_indexes_df = pd.DataFrame(
        {
            "InChI_Key": [tup[0] for tup in new_inchikey_smiles],
            "SMILES": [tup[1] for tup in new_inchikey_smiles],
        }
    )
    new_indexes_df = new_indexes_df.set_index("InChI_Key", drop=True)
    df = pd.concat([df, new_indexes_df], axis=0)

    # Convert incikeys and generations tuples into dict
    generations_by_inchikey = dict(inchikeys_generations)

    # Make sure generations list is ordered by getting items in order of df index
    get_item_partial = functools.partial(_get_item_from_dict, generations_by_inchikey)
    with multiprocessing.Pool(processes=processes) as pool:
        generations_list = pool.map(get_item_partial, df.index)

    df[col_name] = generations_list
    return df


def load_col_names_and_files_from_csv(file_name: str):
    """Loads a list of column names and Pickaxe run pickle files from a csv file"""
    df = pd.read_csv(file_name)
    return list(df["column_name"]), list(df["pickle_path"])


def validate_files(files):
    for file in files:
        if not Path(file).is_file():
            logger.error(f"File {file} is not valid")
            raise ValueError(f"File {file} is not valid")

if __name__ == "__main__":
    # Get command line arguments and input into variables
    args = sys.argv[1:]
    cols, files = load_col_names_and_files_from_csv(args[0])

    validate_files(files)

    logger.info(f"Loaded column names: {cols}")
    logger.info(f"Loaded the following Pickaxe files:")
    for pik_file in files:
        logger.info(f"    {pik_file}")
    logger.info("-" * 50)

    df = make_default_dataframe()

    logger.info(f"Loaded default dataframe")

    processes = multiprocessing.cpu_count() // 2

    logger.info(f"Using {processes} processes to open datasets")
    
    for col_name, pik_file in zip(cols, files):
        logger.info(f"Adding {col_name} column")
        with open(pik_file, "rb") as f:
            pk = pickle.load(f)
            df = add_pickaxe_run_to_df(df, pk, col_name, processes)
        
            # Finally delete the pickaxe object and call for garbage collection
            del pk
            gc.collect()
            logger.info(f"Done adding {col_name} to dataframe")

    logger.info(f"Saving dataframe as pickle to {__location__}/{args[1]}")
    with open(f"{__location__}/{args[1]}", "wb") as outfile:
        pickle.dump(df, outfile)

    print("Done computing the pandas dataframe")
