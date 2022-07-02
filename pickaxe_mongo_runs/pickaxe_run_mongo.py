import argparse
import datetime
import functools
import itertools
import logging
import multiprocessing
import numpy as np
import os
import pathlib
import pickle
import sys
import time
import traceback

from rdkit import Chem
from rdkit.Chem import AllChem

import minedatabase
from minedatabase.pickaxe import Pickaxe
from minedatabase.filters import MWFilter
from minedatabase.filters import RandomSubselectionFilter
from minedatabase.filters import SimilarityClusteringFilter
from minedatabase.filters import MultiRoundSimilarityClusteringFilter

import pymongo
from pymongo import UpdateOne


# Setup known compounds in ModelSEED
import modelseedpy
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


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


FILTER_NAME_MAP = {
    "Molecular Weight": MWFilter,
    "Random Subselection": RandomSubselectionFilter,
    "Similarity Clustering Filter": SimilarityClusteringFilter,
    "Multi Round Similarity Clustering Filter": MultiRoundSimilarityClusteringFilter,
}


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


def transform_pickaxe_compounds(
    pk: Pickaxe,
    generations: int,
    processes: int,
    logger: logging.Logger,
    mongo_host: str,
    mongo_port: int,
    collection: str,
    run_info_id: str,
):
    """Helper function to expand for a number of generations one at a time logging info
    in-between each generation. At end of expansion, writes pickaxe compounds to a mongo
    database from the given mongo arguments.

    Arguments:
        (Pickaxe) pk: Pickaxe object
        (int) generations: Number of generations to expand
        (int) processes: Number of processes to run concurrently.
        (logging.Logger): Logging object
        (str) mongo_host:
        (int) mongo_port:
        (str) collection:
        (str) run_info_id:
    Returns:
        None
    """
    # TODO: Write info to mongo database metadata collection

    total_time = 0  # in seconds
    for gen in range(0, generations + 1):
        start_time = time.time()

        # Run the expansion
        # try / except block for catching errors during a long run without losing data  
        pk.generation = gen - 1  # Hacky way to get pickaxe to expand on next generation
        try:
            pk.transform_all(processes=processes, generations=gen)

        except KeyboardInterrupt:
            logger.error(f"------- Pickaxe Run Interrupted -------")
            logger.error(f"The run was stopped during generation {gen} of {generations} due to a ")
            logger.error(f"keyboard interrupt.")
            
            if write_cpds_to_database:
                logger.error(f"Attempting to save to save current progress to a mongo database")

                save_to_mongo(
                    pk=pk,
                    processes=processes,
                    logger=logger,
                    mongo_host=mongo_host,
                    mongo_port=mongo_port,
                    collection=collection,
                    run_info_id=run_info_id,
                )

                run_col = pymongo.MongoClient(
                    host=mongo_host, port=mongo_port
                ).pickaxe_filtering.pickaxe_run_info
                run_col.update_one(
                    {"_id": run_info_id}, {"$set": {"errored_out": True}}
                )
                
                logger.error(f"Successfully saved to mongo db at {mongo_url}")
                logger.error(f"Elapsed time:     {str(datetime.timedelta(seconds=total_time))}")

        except Exception as e:
            logger.error(f"------- Pickaxe Run Errored Out -------")
            logger.error(f"The run was stopped during generation {gen} of {generations} due to an")
            logger.error(f"unexpected error.")
            logger.error(f"The stack trace of the error:")
            logger.error(f"{str(e)}")
            logger.error(f"{traceback.format_exc()}")

            if write_cpds_to_database:
                logger.error(f"Attempting to save to save current progress to a mongo database")

                save_to_mongo(
                    pk=pk,
                    processes=processes,
                    logger=logger,
                    mongo_host=mongo_host,
                    mongo_port=mongo_port,
                    collection=collection,
                    run_info_id=run_info_id,
                )

                run_col = pymongo.MongoClient(
                    host=mongo_host, port=mongo_port
                ).pickaxe_filtering.pickaxe_run_info
                run_col.update_one(
                    {"_id": run_info_id}, {"$set": {"errored_out": True}}
                )

                logger.error(f"Successfully saved to mongo db at {mongo_url}")
                logger.error(f"Elapsed time:     {str(datetime.timedelta(seconds=total_time))}")
            raise e
        
        # Update time variables
        end_time = time.time()
        gen_time = end_time - start_time
        total_time += gen_time
        
        logger.info(f"------- Generation {gen} of {generations} -------")
        logger.info(f"Computation time: {str(datetime.timedelta(seconds=gen_time))}")
        logger.info(f"Elapsed time:     {str(datetime.timedelta(seconds=total_time))}")
        logger.info(f"New Compounds:    {pk.num_new_compounds}")
        logger.info(f"New Reactions:    {pk.num_new_reactions}")

        # NOTE: Run is not saved to database every generation. This reduces number
        # of times the compounds dict needs to be iterated over

    write_start_time = time.time()
    logger.info("Writing Pickaxe run info to mongo database")

    save_to_mongo(
        pk=pk,
        processes=processes,
        logger=logger,
        mongo_host=mongo_host,
        mongo_port=mongo_port,
        collection=collection,
        run_info_id=run_info_id,
    )

    write_time = time.time() - write_start_time
    logger.info(f"Finished writing to database in {str(datetime.timedelta(seconds=write_time))} seconds")

    run_col = pymongo.MongoClient(
        host=mongo_host, port=mongo_port
    ).pickaxe_filtering.pickaxe_run_info
    run_col.update_one(
        {"_id": run_info_id}, {"$set": {"completed": True}}
    )

    logger.info(f"------- Pickaxe Run Complete -------")
    logger.info(f"Elapsed time:     {str(datetime.timedelta(seconds=total_time))}")


def setup_logger(logpath: str) -> logging.Logger: 
    """Function to take the logging setup out of the main function. Instantiates a new
    logger object with name `run_pickaxe` and sets appropriate levels
    """
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
    
    return logger


def get_args_from_argparse():
    """Argument parser setup function. Adds arguments and returns parsed parsed args"""
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument(
        "name",
        help="Unique name to give to the run. Additionally, a logfile with <name>.log will be created for the run",
        type=str,
    )
    # Optional arguments
    parser.add_argument(
        "-r",
        "--rule",
        help="Path to rule .tsv file",
        default="/homes/mgiammar/Documents/MINE-Database/minedatabase/data/original_rules/EnzymaticReactionRules.tsv"
    )
    parser.add_argument(
        "-c",
        "--coreactants",
        help="Path to coreactants .tsv file",
        default="/homes/mgiammar/Documents/MINE-Database/minedatabase/data/original_rules/EnzymaticCoreactants.tsv"
    )
    parser.add_argument(
        "-s",
        "--start",
        help="Path to starting compounds .csv file",
        default="/homes/mgiammar/Documents/MINE-Database/jvci_syn3A.csv"
    )
    parser.add_argument(
        "-f",
        "--filter",
        help="Path to filter JSON file to parse filters from",
        type=str
    )
    parser.add_argument(
        "-g",
        "--generations",
        help="Target number of generations to expand",
        type=int,
        default=3
    )
    parser.add_argument(
        "-p",
        "--processes",
        help="Number of processes to use during expansion",
        type=int,
        default=max(multiprocessing.cpu_count() // 15, 1)  # Arbitrary number
    )
    parser.add_argument(
        "--mongo_host",
        help="Mongo DB hostname",
        type=str,
        # default="localhost"  # TODO: make sequoia host when transfer
        default="sequoia.mcs.anl.gov"
    )
    parser.add_argument(
        "--mongo_port",
        help="Mongo DB hostname",
        type=int,
        default=27017
    )
    parser.add_argument(
        "--collection",
        help="Mongo DB collection to add documents to",
        type=str,
        default="JVCI_pickaxe_runs"
    )
    # keyword arguments to pass to pickaxe object
    parser.add_argument(
        "--explicit_h",
        help="Use explicit hydrogens in expansion",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--prune_between_generations",
        help="Wether to prune the compounds in-between each generation",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--filter_after_final_gen",
        help="Wether to filter the Pickaxe object after the final generation",
        type=bool,
        default=True
    )
    return parser.parse_args()


def parse_filter_json(filter_file: str) -> list:
    """Takes a json file with a list of dict representations of filters and instantiates
    filter objects from there
    
    Arguments:
        (str) filter_file: Filepath to the filter json

    Returns:
        List of minedatabase Filter objects
    """
    if not pathlib.Path(filter_path).is_file():
        msg = f"File {filter_file} is not a valid path."
        logger.error(msg)
        raise ValueError(msg)
    
    with open(filter_file, "r") as f:
        json_data = json.load(f)

    filters = []
    json_data = json_data["filters"]
    for filter_dict in json_data:
        name = filter_dict.pop("filter_name")
        filters.append(FILTER_NAME_MAP[name](**filter_dict))

    return filters


def add_mongo_metadata_header(db, args, pk):
    """Adds information about the pickaxe run into the mongo DB `pickaxe_run_info`
    collection.

    Arguments:
        TODO

    Returns:
        Post ID of entry; will be used to label which runs a compound is found in
    """
    pickaxe_run_info = db.pickaxe_run_info

    post = vars(args)
    post["filters"] = [fil.get_filter_fields_as_dict for fil in pk.filters]
    post["start_time"] = datetime.datetime.utcnow()
    post["completed"] = False
    post["errored_out"] = False

    return str(pickaxe_run_info.insert_one(post).inserted_id)


def chunk_dict(_dict: dict, chunk_size: int):
    """Generator function to chunk dictionary into smaller dicts of chunk_size"""
    it = iter(_dict)
    for i in range(0, len(_dict), chunk_size):
        yield {key: _dict[key] for key in itertools.islice(it, chunk_size)}


# def chunk_list(it, it_length, chunk_size: int):
#     """Lazily split up a generator object. Length must be supplied """
#     for i in range(0, it_length, chunk_size):
#         yield itertools.islice(it, chunk_size)


def add_compounds_to_mongo_db(
    mongo_host,
    mongo_port,
    database,
    collection,
    logger,  # TODO: add to docstring
    cpd_list
):
    """Adds formatted compound document to the provided mongo database collection. Each
    document (compound) id is the compound's connectivity InChI_key and the values are:
     - SMILES: str
     - mol_wt: float (in g/mol)
     - in_ModelSEED: bool
     - found_in: dict

     The found_in dict has the run_info_id string as key and the generation number as
     value; at insertion, however, this dictionary is empty and populated in a later
     function

    Arguments:
        (str) mongo_host: Hostname for the mongo database
        (int) mongo_port: Port number for the mongo database
        (str) database: Name of the database to use  #  Currently unused
        (str) collection: Collection name in the database to use
        (list) cpd_list: Dictionary of compound attributes

    """
    logger.info("In function add_compounds_to_mongo_db")
    client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
    db = client.pickaxe_filtering

    document_list = [
        {
            "_id": cpd["InChI_key"].split("-")[0],
            "SMILES": cpd["SMILES"],
            "mol_wt": AllChem.CalcExactMolWt(Chem.MolFromSmiles(cpd["SMILES"])),
            "in_ModelSEED": cpd["InChI_key"].split("-")[0] in modelseed_inchikeys_partial,
            "found_in": {}
        }
        for cpd in cpd_list
    ]
    
    db[collection].insert_many(document_list)

    # NOTE Should make try except block when adding, maybe some logic to handle errors
    # too
    # client.close()

    return None


def add_found_in_data_to_document(
    mongo_host,
    mongo_port,
    database,
    collection,
    run_info_id,
    logger,
    cpd_list,
):
    """Updates the found_in dict field in each compound with the run_info_id and the
    compounds generation number. 
    
    Arguments:
        (str) mongo_host: Hostname for the mongo database
        (int) mongo_port: Port number for the mongo database
        (str) database: Name of the database to use  #  Currently unused
        (str) collection: Collection name in the database to use
        (str) run_info_id: String of run info document id
        (list) cpd_list: Dictionary of compound attributes

    Returns:
        TODO
    """
    logger.info("In function add_found_in_data_to_document")
    client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
    db = client.pickaxe_filtering

    # Tuple of args to pass to pymongo.UpdateOne method
    update_tuples = [
        (
            {"_id": cpd["InChI_key"].split("-")[0]},
            {"$set": {f"found_in.{run_info_id}": cpd["Generation"]}}
        )
        for cpd in cpd_list
    ]
    res = db[collection].bulk_write([UpdateOne(*tup) for tup in update_tuples])
    
    # client.close()

    return None


def save_to_mongo(
    pk: Pickaxe,
    processes: int,
    logger: logging.Logger,
    mongo_host: str,
    mongo_port: int,
    collection: str,
    run_info_id: str,
):
    """Method which writes all compounds in the supplied pickaxe object to a mongo db
    collection
    
    Arguments:
        (minedatabase.pickaxe.Pickaxe) pk: Pickaxe object to get compounds from
        (str) collection: Collection name to write compounds to
        (str) run_info_id: ID of run info document. Used when recording what runs and
            which generation a compound is found in.
        (str) mongo_host: Hostname for mongo database.
        (int) mongo_port: Port number for mongo database.
    
        # (TODO) db:  Cant pass DB object itself, need to pass stuff to make client
    Returns:
        None
    """
    logger.info("in function add_to_mongo")
    # TODO: Add logger outputs throughout method

    # Get set of all new compound IDs to add to collection
    compound_collection = db[collection]
    db_compound_ids = {str(_id) for _id in compound_collection.find().distinct('_id')}
    pickaxe_compound_ids = {cpd["InChI_key"].split("-")[0] for cpd in pk.compounds.values()}
    new_compounds = pickaxe_compound_ids - db_compound_ids

    logger.info(f"len new compounds {len(new_compounds)}")

    # Skip adding new compounds to mongo if no new compounds. Otherwise numpy array
    # split will throw an error
    if len(new_compounds) > 0:

        # Create processes number of Process objects each acting upon a chunk of the new
        # compounds. Compounds is chunked into processes number of chunks
        new_cpds_list = [
            cpd for cpd in pk.compounds.values()
            if cpd["InChI_key"].split("-")[0] in new_compounds
        ]
        use_processes = min(len(new_cpds_list), processes)  # No empty lists
        new_cpds_list = np.array_split(new_cpds_list, use_processes)

        # Setup and run the Process objects
        processors = [
            multiprocessing.Process(
                target=add_compounds_to_mongo_db,
                args=(
                    mongo_host,
                    mongo_port,
                    "pickaxe_filtering",
                    collection,
                    logger,
                    new_cpds_list[i]
                )
            )
            for i in range(use_processes)
        ]
        _ = [p.start() for p in processors]
        _ = [p.join() for p in processors]

        logger.info("Finished adding compounds to mongo")

    # Ensure there will be no empty processes (will throw out of bounds error)
    cpd_list = [cpd for cpd in pk.compounds.values()]
    use_processes = min(len(cpd_list), processes)
    cpd_list = np.array_split(cpd_list, use_processes)

    # Setup and run Process objects
    processors = [
        multiprocessing.Process(
            target=add_found_in_data_to_document,
            args=(
                mongo_host,
                mongo_port,
                "pickaxe_filtering",
                collection,
                run_info_id,
                logger,
                cpd_list[i]
            )
        )
        for i in range(use_processes)
    ]
    _ = [p.start() for p in processors]
    _ = [p.join() for p in processors]
    
    logger.info("Finished updating found_in attribute in mongo ")


# ======================================== MAIN ========================================
if __name__ == "__main__":
    # Parse arguments function and get values
    args = get_args_from_argparse()

    logfile = f"{__location__}/{args.name}.log"
    logger = setup_logger(logfile)

    logger.info(f"-------- Start of Pickaxe Run --------")
    logger.info(f"Starting run {args.name} with the following arguments")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    logger.info(f"Instantiating Pickaxe object...")

    # Make a new pickaxe object
    pk = make_pickaxe_object(
        rules=args.rule,
        coreactants=args.coreactants,
        starting_compounds=args.start,
        explicit_h=args.explicit_h,
        filter_after_final_gen=args.filter_after_final_gen,
        prune_between_gens=args.filter_after_final_gen
    )

    logger.info("Pickaxe object instantiated")

    filter_file = args.filter
    if filter_file is not None:
        logger.info(f"Parsing filter file...")
        filters = parse_filter_json()
        pk.filter = filters
        logger.info(f"Added {len(filters)} to the Pickaxe object")
    else:
        logger.info(f"No filters to add")

    # Mongo DB stuff
    logger.info(f"Attempting to connect to Mongo DataBase with")
    logger.info(f"host = {args.mongo_host}")
    logger.info(f"port = {args.mongo_port}")
    try:
         client = pymongo.MongoClient(host=args.mongo_host, port=args.mongo_port)

    except Exception as e:
        logger.error(f"Unable to connect to Mongo DB")
        logger.error(f"The following error occurred:")
        logger.error(f"{str(e)}")
        logger.error(f"{traceback.format_exc()}")
        
        raise e

    db = client.pickaxe_filtering  # Database name currently hardcoded

    # method to add run header to mongo db
    logger.info(f"Adding pickaxe run header to Mongo DB")
    run_info_id = add_mongo_metadata_header(db, args, pk)
    logger.info(run_info_id)

    logger.info(f"-------- Start of Transformation --------")
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Processes:   {args.processes}")

    transform_pickaxe_compounds(
        pk=pk,
        generations=args.generations,
        processes=args.processes,
        logger=logger,
        mongo_host=args.mongo_host,
        mongo_port=args.mongo_port,
        collection=args.collection,
        run_info_id=run_info_id
    )

    print("PICKAXE RUN COMPLETED")
