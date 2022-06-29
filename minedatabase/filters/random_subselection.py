"""Filter for choosing a random sub-selection up to a maximum number of compounds of the
compounds generated. after a generation
"""

import time
import numpy as np

from minedatabase.filters.base_filter import Filter
from minedatabase.pickaxe import Pickaxe

import logging
logger = logging.getLogger("run_pickaxe")


class RandomSubselectionFilter(Filter):
    """Filter which chooses up to max_compounds for expansion in the next generation.
    If the total number of compounds for a generation are less than max_compounds, then
    all compounds are chosen for expansion.

    This filter should be fast since no computations about compounds need to be made.

    Parameters
    ----------
    (int) max_compounds:
        Maximum number of compounds to keep from the generation
    (list) generation_list:
        Optional list of integers corresponding to which generations to apply filter to

    Attributes
    ----------
        (int) max_compounds:
            Maximum number of compounds to keep from the generation
        (list) generation_list:
            Optional list of integers corresponding to which generations to apply filter to
        # (bool) strict_filter: If True, the filter will return a compound ID set for
        #     compounds to remove. Otherwise an empty set will be returned and only the 
        #     compounds selected will be marked for expansion. Default is True
    """

    def __init__(self, max_compounds, generation_list=None) -> None:
        self._filter_name = "Random Subselection"
        self.max_compounds = max_compounds
        self.generation_list = generation_list

    @property
    def filter_name(self) -> str:
        return self._filter_name

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        logger.info(f"Sampling {self.max_compounds} randomly selected compounds.")
    
    def _post_print_footer(self, pickaxe: Pickaxe) -> None:
        """Post filtering info"""
        logger.info(f"Done filtering Generation {pickaxe.generation}")
        logger.info("-----------------------------------------------")

    # def _should_filter_this_generation(self):
    #     """Returns True if this filter should be applied for this generation, False
    #     otherwise.
    #     """
    #     # NOTE: Could have made one bool expression, but more comprehensible to split up
    #     if self.generation == 0:
    #         return False
        
    #     if self.generation_list is None:
    #         return True

    #     return (self.generation - 1) in self.generation_list

    def _choose_items_to_filter(self, pickaxe, processes):
        """Randomly chooses max_compounds from the Pickaxe object to expand in the
        next generation
        """
        cpds_remove_set = set()
        rxn_remove_set = set()

        generation = pickaxe.generation
        # if not self._should_filter_this_generation():
        #     return cpds_remove_set, rxn_remove_set

        cpd_ids = [
            cpd_id for cpd_id, cpd in pickaxe.compounds.items()
            if cpd["Generation"] == generation and
            cpd["Type"] not in ["Coreactant"]
        ]
        keep_ids = np.random.choice(
            cpd_ids,
            replace=False,
            size=min(self.max_compounds, len(cpd_ids))
        )
        for cpd_id in pickaxe.compounds.keys():
            pickaxe.compounds[cpd_id]["Expand"] = cpd_id in keep_ids

        cpds_remove_set =  set(cpd_ids).difference(set(keep_ids))
        return cpds_remove_set, rxn_remove_set
        
