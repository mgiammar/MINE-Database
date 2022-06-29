import datetime
import multiprocessing
import pickle
import time
import os
import traceback
import logging

from rdkit import Chem

import minedatabase

from minedatabase.filters import MCSFilter, MetabolomicsFilter, MWFilter
from minedatabase.filters import MultiRoundSimilarityClusteringFilter
from minedatabase.filters import SimilarityClusteringFilter
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
    # _write_info_to_file(
    #     f"""
    #     ------- General Info for Pickaxe Run -------
    #     starting gen: {previous_gen}
    #     generations:  {generations}
    #     processes:    {processes}
    #     filters:      {[fil.filter_name for fil in pk.filters]}

    #     ----------------------------------------------------------------
    #     """
    # )

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

                logger.error(f"------- Pickaxe Run Interrupted -------")
                logger.error(f"The run was stopped during generation {gen} of {generations} due to a ")
                logger.error(f"keyboard interrupt.")
                logger.error(f"The current Pickaxe object has been saved to")
                logger.error(f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik")
                logger.error(f"Elapsed time:     {str(datetime.timedelta(seconds=total_time))}")

            # _write_info_to_file(
            #     f"""
            #     ------- Pickaxe Run Interrupted -------
            #     The run was stopped during generation {gen} of {generations} due to a 
            #     keyboard interrupt.
            #     The current Pickaxe object has been saved to

            #     {f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik"}

            #     Elapsed time:     {str(datetime.timedelta(seconds=total_time))}
            #     """
            # )
        except Exception as e:
            with open(
                f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik", "wb"
            ) as pik_file:
                pickle.dump(pk, pik_file)

                logger.error(f"------- Pickaxe Run Errored Out -------")
                logger.error(f"The run was stopped during generation {gen} of {generations} due to an")
                logger.error(f"unexpected error.")
                logger.error(f"The current Pickaxe object has been saved to")
                logger.error(f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik")
                logger.error(f"The stack trace of the error:")
                logger.error(f"{str(e)}")
                logger.error(f"{traceback.format_exc()}")
                logger.error(f"Elapsed time:     {str(datetime.timedelta(seconds=total_time))}")

            # _write_info_to_file(
            #     f"""
            #     ------- Pickaxe Run Errored Out -------
            #     The run was stopped during generation {gen} of {generations} due to an
            #     unexpected error.
            #     The current Pickaxe object has been saved to

            #     {f"{__location__}/pickaxe_run interrupt {gen} of {generations}.pik"}

            #     The stack trace of the error:

            #     {str(e)}
            #     {traceback.format_exc()}

            #     Elapsed time:     {str(datetime.timedelta(seconds=total_time))}
            #     """
            # )
            raise e
        
        # Update time variables and write to info 
        end_time = time.time()
        gen_time = end_time - start_time
        total_time += gen_time
        
        logger.info(f"------- Generation {gen} of {generations} -------")
        logger.info(f"Computation time: {str(datetime.timedelta(seconds=gen_time))}")
        logger.info(f"Elapsed time:     {str(datetime.timedelta(seconds=total_time))}")
        logger.info(f"New Compounds:    {pk.num_new_compounds}")
        logger.info(f"New Reactions:    {pk.num_new_reactions}")

        # _write_info_to_file(
        #     f"""
        #     ------- Generation {gen} of {generations} -------
        #     Computation time: {str(datetime.timedelta(seconds=gen_time))}
        #     Elapsed time:     {str(datetime.timedelta(seconds=total_time))}
        #     New Compounds:    {pk.num_new_compounds}
        #     New Reactions:    {pk.num_new_reactions}
        #     """
        #     # NOTE: Info string from filter currently hardcoded
        # )

        # Export the pickaxe object to a pickle file
        with open(
            f"{__location__}/pickaxe_run gen {gen} of {generations}.pik", "wb"
        ) as pik_file:
            pickle.dump(pk, pik_file)

    logger.info(f"------- Pickaxe Run Complete -------")
    logger.info(f"Elapsed time:     {str(datetime.timedelta(seconds=total_time))}")
    # Write final info after completing pickaxe run
    # _write_info_to_file(
    #     f"""
    #     ------- Pickaxe Run Complete -------
    #     Elapsed time:     {str(datetime.timedelta(seconds=total_time))}

    #     """
    # )



# ======================================== MAIN ========================================
if __name__ == "__main__":
    # Setup logging object
    logging.getLogger().setLevel(logging.INFO)
    logfile = f"{__location__}/pk_run.log"
    if(os.path.isfile(logfile)):
            os.remove(logfile)
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger("run_pickaxe")
    logger.addHandler(file_handler)

    rule_file = "/homes/mgiammar/Documents/MINE-Database/minedatabase/data/original_rules/EnzymaticReactionRules.tsv"
    coreactant_file = "/homes/mgiammar/Documents/MINE-Database/minedatabase/data/original_rules/EnzymaticCoreactants.tsv"
    # starting_compounds_file = "/homes/mgiammar/Documents/MINE-Database/jvci_random_7.csv"
    starting_compounds_file = "/homes/mgiammar/Documents/MINE-Database/jvci_syn3A.csv"

    logger.info(f"-------- Start of Pickaxe Run --------")
    logger.info(f"Creating Pickaxe object with")
    logger.info(f"Rules:       {rule_file}")
    logger.info(f"Coreactants: {coreactant_file}")
    logger.info(f"Compounds:   {starting_compounds_file}")

    # Make a new pickaxe object
    pk = make_pickaxe_object(
        rules=rule_file,
        coreactants=coreactant_file,
        starting_compounds=starting_compounds_file
    )

    logger.info("Pickaxe object instantiated")

    # Add filters to the pickaxe object
    # cluster_filter = SimilarityClusteringFilter(
    #     cutoff=0.3,
    #     compounds_per_cluster=2,
    #     fingerprint_bits=512
    # )
    # pk.filters = [cluster_filter]

    # random_filter = RandomSubselectionFilter(
    #     max_compounds=10000, generation_list=[1, 2, 3, 4, 5, 6, 7]
    # )
    multi_cluster = MultiRoundSimilarityClusteringFilter(
        cutoff=[0.1, 0.2, 0.3, 0.4],
        cluster_size_cutoff=[80, 40, 20, 10],
        compounds_selected_per_cluster=[2, 2, 2, 2],
        generation_list=None,
        max_compounds=75000,
    )
    pk.filters=[multi_cluster]

    # TODO: Add more info about filter objects, possibly within the filter file
    logger.info(f"Added {len(pk.filters)} to the Pickaxe object")

    # NOTE: REMEMBER TO NOT CALL transform_all USE CUSTOM transform_pickaxe_compounds
    processes = os.cpu_count() // 8
    generations = 3

    logger.info(f"-------- Start of Transformation --------")
    logger.info(f"Generations: {generations}")
    logger.info(f"Processes:   {processes}")

    transform_pickaxe_compounds(pk, generations=generations, processes=processes)

    print("PICKAXE RUN COMPLETED")
