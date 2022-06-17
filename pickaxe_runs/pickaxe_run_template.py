import datetime
import multiprocessing
import pickle
import time
import os
import traceback

from rdkit import Chem

import minedatabase

from minedatabase.filters import MCSFilter, MetabolomicsFilter, MWFilter
# from minedatabase.filters.thermodynamics import ThermoFilter
from minedatabase.pickaxe import Pickaxe

import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# NOTE: Not all imports are necessary for the default script
# TODO: Making a logger object instead of writing to a file seems like a good idea, but
# also low priority


# ================================== HELPER FUNCTIONS ==================================
def make_pickaxe_object(
    rules: str,
    coreactants: str,
    starting_compounds: str,
    target_compounds: str = None,
    explicit_h: bool = True,
    filter_after_final_gen: bool = True,
    prune_between_gens: bool = False,
) -> minedatabase.pickaxe.Pickaxe:
    """Instantiate and returns a pickaxe object with the given set of rules,
    coreactants, and starting compounds. Additional info?
    
    Arguments:
        (str) rules: Path to tsv file of rules
        (str) coreactants: Path to tsv file of coreactants
        (str) starting_compounds: Path to csv file of starting compounds smiles
        (str) target_compounds: Path to csv file with target compound smiles
        (bool): explicit_h: 
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
        explicit_h=explicit_h,
        # Prevent mongo database. Do not change
        database=None,
        mongo_uri=None,
        image_dir=None,
        inchikey_blocks_for_cid=1,
        quiet=True,
        errors=False  # Print RDKit warnings and errors
    )
    
    pk.load_compound_set(starting_compounds)

    return pk


def load_pickaxe_object_from_pickle(file_name: str) -> minedatabase.pickaxe.Pickaxe:
    """Loads a saved Pickaxe object from a pickle file.
    
    Arguments:
        (str) file_name: Path to pickle file
        
    Returns:
        A Pickaxe object
    """
    with open(file_path, "rb") as f:
        pk = pickle.load(f)

    return pk


def _write_info_to_file(info_text: str, file_name: str = f"{__location__}/info.txt"):
    """Helper function to write some info to a text file. Use the triple quote string
    to automatically include new line characters

    Arguments:
        (str) info_text: Text to write to the file
        (str) file_name: Optional file name/path to write to. Default is info.txt

    Returns:
        None
    """
    with open(file_name, "a") as f:
        f.write(info_text)


def transform_pickaxe_compounds(
    pk: Pickaxe,
    generations: int,
    processes: int = 6,  # Computer has 8 cores
    write_info: bool = True,  # Currently ignored
    delete_old_files: bool = False  # Currently ignored
):
    """Helper function to expand for a number of generations one at a time saving the 
    pickaxe object using pickle after each generation. Also writes information to 
    info.txt about the progress of the expansion.

    Arguments:
        (Pickaxe) pk: Pickaxe object
        (int) generations: Number of generations to expand
        (int) processes: Optional number of processes to run concurrently. Default is 6
        (bool) write_info: Optional bool to specify wether to write information about
            the run to a file. CURRENTLY NOT IMPLEMENTED
        (bool) delete_old_files: CURRENTLY NOT IMPLEMENTED

    Returns:
        None
    """
    # TODO: Include more info in general info header
    # NOTE: This method could easily be broken into smaller methods

    # Get last generation of pickaxe object so old runs can be loaded and run further
    # TODO: test this feature
    previous_gen = pk.generation
    _write_info_to_file(
        f"""
        ------- General Info for Pickaxe Run -------
        starting gen: {previous_gen}
        generations:  {generations}
        processes:    {processes}
        filters:      {[fil.filter_name for fil in pk.filters]}

        ----------------------------------------------------------------
        """
    )

    total_time = 0  # in seconds
    for gen in range(0, generations + 1):
        start_time = time.time()

        # Run the expansion
        # try / except block for catching errors during a long run without losing data  
        pk.generation = gen - 1  # Hacky way to get pickaxe to expand on next generation
        try:
            pk.transform_all(processes=processes, generations=gen)
        except KeyboardInterrupt:
            with open(
                f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik", "wb"
            ) as pik_file:
                pickle.dump(pk, pik_file)

            _write_info_to_file(
                f"""
                ------- Pickaxe Run Interrupted -------
                The run was stopped during generation {gen} of {generations} due to a 
                keyboard interrupt.
                The current Pickaxe object has been saved to

                {f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik"}

                Elapsed time:     {str(datetime.timedelta(seconds=total_time))}
                """
            )
        except Exception as e:
            with open(
                f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik", "wb"
            ) as pik_file:
                pickle.dump(pk, pik_file)

            _write_info_to_file(
                f"""
                ------- Pickaxe Run Errored Out -------
                The run was stopped during generation {gen} of {generations} due to an
                unexpected error.
                The current Pickaxe object has been saved to

                {f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik"}

                The stack trace of the error:

                {str(e)}
                {traceback.format_exc()}

                Elapsed time:     {str(datetime.timedelta(seconds=total_time))}
                """
            )
            raise e
        
        # Update time variables and write to info 
        end_time = time.time()
        gen_time = end_time - start_time
        total_time += gen_time
        _write_info_to_file(
            f"""
            ------- Generation {gen} of {generations} -------
            Computation time: {str(datetime.timedelta(seconds=gen_time))}
            Elapsed time:     {str(datetime.timedelta(seconds=total_time))}
            New Compounds:    {pk.num_new_compounds}
            New Reactions:    {pk.num_new_reactions}
            
            {pk.filters[0].info_string}
            """
            # NOTE: Info string from filter currently hardcoded
        )

        # Export the pickaxe object to a pickle file
        with open(
            f"{__location__}/pickaxe_run gen {gen} of {generations}.pik", "wb"
        ) as pik_file:
            pickle.dump(pk, pik_file)

    # Write final info after completing pickaxe run
    _write_info_to_file(
        f"""
        ------- Pickaxe Run Complete -------
        Elapsed time:     {str(datetime.timedelta(seconds=total_time))}

        """
    )



# ======================================== MAIN ========================================
if __name__ == "__main__":
    # Make a new pickaxe object
    pk = make_pickaxe_object(
        rules="/Users/mgiammar/Documents/MINE-Database/minedatabase/data/original_rules/EnzymaticReactionRules.tsv",
        coreactants="/Users/mgiammar/Documents/MINE-Database/minedatabase/data/original_rules/EnzymaticCoreactants.tsv",
        # starting_compounds="/Users/mgiammar/Documents/MINE-Database/jvci_syn3A.csv"
        starting_compounds="/Users/mgiammar/Documents/MINE-Database/jvci_random_7.csv"
    )

    # Add filters to the pickaxe object
    pk.filters = []

    # NOTE: REMEMBER TO NOT CALL transform_all USE CUSTOM transform_pickaxe_compounds
    transform_pickaxe_compounds(pk, 2, processes=1)

    print("PICKAXE RUN COMPLETED")