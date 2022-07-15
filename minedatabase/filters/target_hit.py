from minedatabase.filters.base_filter import Filter
from minedatabase.pickaxe import Pickaxe

import pandas as pd

import logging
logger = logging.getLogger("pickaxe_run")


class TargetCompoundHitFilter(Filter):
    """This filter takes in a csv file with target compound inchikeys and whitelists
    all compounds which match the target inckikeys. NOTE: This file only whitelists
    compounds, that is, it will always return an empty set for compounds to remove.
    
    """
    def __init__(
        self,
        target_compounds_path: str,
        generation_list=None,
        priority: int=10,
        all_compounds: bool=True,
    ):
        # Attributes for all filters
        self._filter_name = "Target Compound Whitelist"
        self.generation_list=generation_list
        self.priority = priority
        self.all_compounds = all_compounds

        self.target_compounds_path = target_compounds_path
        self.target_compounds = self._load_target_compounds(target_compounds_path)

    @property
    def filter_name(self) -> str:
        return self._filter_name

    def get_filter_fields_as_dict(self) -> dict:
        """Returns property info about filter as a dict"""
        return {
            "filter_name": self._filter_name,
            "generation_list": self.generation_list,
            "priority": self.priority,
            "all_compounds": self.all_compounds,
            "target_compounds_path": self.target_compounds_path,
        }

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        logger.info("filter info not implemented")
    
    def _post_print_footer(self, pickaxe: Pickaxe) -> None:
        """Post filtering info"""
        logger.info(f"Done filtering Generation {pickaxe.generation}")
        logger.info("-----------------------------------------------")

    def _choose_items_to_filter(self, pickaxe: Pickaxe, *args, **kwargs):
        """Get whitelisted compounds (those that match known inchikeys). All other 
        arguments are ignored and only the standard sets are returned.
        """
        whitelist_compounds = {
            cpd_id for cpd_id, cpd in pickaxe.compounds.items()
            if cpd["Generation"] == pickaxe.generation and "InChI_key" in cpd
            and cpd["InChI_key"].split("-")[0] in self.target_compounds
        }

        return set(), set(), whitelist_compounds, set()

    def _load_target_compounds(self, file_path):
        """Loads the inchikeys of target compounds in the csv file pointed to by 
        file_path
        """
        cpds_csv = pd.read_csv(file_path)
        known_cpds = [row[1]["InChI_key"] for row in cpds_csv.iterrows()]
        return known_cpds