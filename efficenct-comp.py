import datetime
import multiprocessing
import pickle
import time

import pymongo

from rdkit import Chem

import minedatabase

from minedatabase.filters import MCSFilter, MetabolomicsFilter, MWFilter
from minedatabase.pickaxe import Pickaxe
from minedatabase.rules import metacyc_generalized, BNICE

from minedatabase.reactions import transform_all_compounds_with_full


def make_pickaxe_object(
    rules: str,
    coreactants: str,
    starting_compounds: str,
    target_compounds: str = None,
    filter_after_final_gen: bool = False,
    prune_between_gens: bool = False,
) -> minedatabase.pickaxe.Pickaxe:
    """Instantiate and returns a pickaxe object with the given set of rules,
    coreactants, and starting compounds. Additional info?
    
    Arguments:
        (str) rules: Path to tsv file of rules
        (str) coreactants: Path to tsv file of coreactants
        (str) starting_compounds: Path to csv file of starting compounds smiles
        (str) target_compounds: Path to csv file with target compound smiles
        (bool): filter_after_final_gen: Wether to apply filters to the final generation
        (bool): prune_between_gens: Wether to prune the network between generations

    Returns:
        A Pickaxe object with the given arguments loaded
    """
    pk = Pickaxe(
        coreactant_list=coreactants,
        rule_list=rules,
        react_targets=target_compounds,
        filter_after_final_gen=filter_after_final_gen,
        prune_between_gens=prune_between_gens,
        neutralise=True,
        kekulize=True,
        # Prevent mongo database. Do not change
        database=None,
        mongo_uri=None,
        image_dir=None,
        inchikey_blocks_for_cid=1,
        quiet=False,
        errors=False  # Print RDKit warnings and errors
    )
    
    pk.load_compound_set(starting_compounds)

    return pk


old_pk = make_pickaxe_object(
    # "minedatabase/data/metacyc_rules/metacyc_generalized_rules.tsv",
    # "minedatabase/data/metacyc_rules/metacyc_intermediate_rules.tsv",
    # "minedatabase/data/metacyc_rules/metacyc_coreactants.tsv",
    "minedatabase/data/original_rules/EnzymaticReactionRules.tsv",  # These rules do not work
    "minedatabase/data/original_rules/EnzymaticCoreactants.tsv",
    # "jvci_syn3A.csv"
    "example_data/starting_cpds_ten.csv"
    # "jvci_random_7.csv"
)
old_pk.transform_all(1, 1)