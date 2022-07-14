import abc
import time
from copy import copy
from typing import List, Set

import rdkit.rdBase as rkrb
# import rdkit.RDLogger as rkl

from minedatabase.pickaxe import Pickaxe


# logger = rkl.logger()
# logger.setLevel(rkl.ERROR)
# rkrb.DisableLog("rdApp.error")


import logging
logger = logging.getLogger("run_pickaxe")


class Filter(metaclass=abc.ABCMeta):
    """Abstract base class used to generate filters.

    The Filter class provides the framework for interaction with pickaxe expansions.
    Each filter subclass must inherit properties from the Filter class.
    All subclasses must implement properties and methods decorated with
    @abc.abstractmethod. Feel free to override other non-private methods as
    well, such as _pre_print() and _post_print().
    """

    @property
    @abc.abstractmethod
    def filter_name(self) -> str:
        """Obtain name of filter."""
        pass

    @abc.abstractmethod
    def _choose_items_to_filter(
        self,
        pickaxe: Pickaxe,
        processes: int,
        compound_ids_to_check: set,
        reaction_ids_to_check: set,
        whitelist_compound_ids: set,
        whitelist_reaction_ids: set
    ) -> Set[str]:
        """Return list of compounds to remove from pickaxe object.

        Parameters
        ----------
        pickaxe : Pickaxe
            Instance of Pickaxe being used to expand and filter the network.
        processes : int
            The number of processes to use, by default 1.
        generation : int
            Which generation the expansion is in.
        compound_ids_to_check : set
            Set of compound ids previously selected for removal by a filter
        reaction_ids_to_check : set
            Set of reaction ids previously selected for removal by a filter
        whitelist_compound_ids : set
            Set of compound ids previously selected for whitelist by a filter
        whitelist_reaction_ids : set
            Set of reaction ids previously selected for whitelist by a filter
        """
        pass

    def apply_filter(
        self,
        pickaxe: Pickaxe,
        previously_removed: set,
        processes: int = 1,
        generation: int = 0,
        print_on: bool = True,
    ) -> None:
        """Apply filter from Pickaxe object.

        Parameters
        ----------
        pickaxe : Pickaxe
            The Pickaxe object to filter.
        previously_removed : set
            Set of pickaxe compound ids which were previously selected for removal by
            filters
        processes : int
            The number of processes to use, by default 1.
        print_on : bool
            Whether or not to print filtering results.
        previous_cpds: set
            Set of compounds selected from previous filters. Used when applying a filter
            to compounds only selected by previous filter
        """
        time_start = time.time()

        self.generation = generation
        print("FILTER GEN", self.generation)

        if not self._should_filter_this_generation():
            logger.info(f"Not applying {self.filter_name} this generation")
            return set(), set()
            

        if print_on:
            n_total = self._get_n(pickaxe, "total")
            self._pre_print_header(pickaxe)
            self._pre_print()

        (
            compound_ids_to_check,
            reaction_ids_to_check,
            whitelist_compound_ids,
            whitelist_reaction_ids
        ) = self._choose_items_to_filter(
            pickaxe, processes, previously_removed
        )

        # Do not apply filter results in until all filters have been applied
        # self._apply_filter_results(
        #     pickaxe, compound_ids_to_check, reaction_ids_to_check
        # )

        if print_on:
            n_filtered = self._get_n(pickaxe, "filtered")
            self._post_print(pickaxe, n_total, n_filtered, time.time() - time_start)
            self._post_print_footer(pickaxe)

        return (
            compound_ids_to_check,
            reaction_ids_to_check,
            whitelist_compound_ids,
            whitelist_reaction_ids
        )

    @abc.abstractmethod
    def get_filter_fields_as_dict(self) -> dict:
        """Returns information about the filter type and associated parameter values
        as a dictionary with each key describing the attribute type and value being the
        value of that attribute. Implemented so that filter data can be added to a
        mongo database.
        """
        raise NotImplementedError

    def _should_filter_this_generation(self):
        """Method used to check if a filter should be applied this generation"""
        if not hasattr(self, "seen_generations"):
            self.seen_generations = []

        # Do not filter on zeroth generation
        if self.generation <= 0:
            logger.info("Not filtering zeroth generation")
            return False

        # Do not filter if the generation has already been seen
        if self.generation in self.seen_generations:
            logger.info("Generation has already been filtered")
            self.seen_generations.append(self.generation)
            return False

        self.seen_generations.append(self.generation)

        # Filter if no generation_list attribute or if attribute is None
        if not hasattr(self, "generation_list") or self.generation_list is None:
            return True

        return (self.generation - 1) in self.generation_list

    def _pre_print_header(self, pickaxe: Pickaxe) -> None:
        """Print header before filtering.

        Parameters
        ----------
        pickaxe : Pickaxe
            Instance of Pickaxe being used to expand and filter the network.
        """
        print("----------------------------------------")
        print(f"Filtering Generation {pickaxe.generation}\n")

        logger.info("----------------------------------------")
        logger.info(f"Filtering Generation {pickaxe.generation}")

    def _pre_print(self) -> None:
        """Print filter being applied."""
        print(f"Applying filter: {self.filter_name}")

    def _post_print(
        self, pickaxe: Pickaxe, n_total: int, n_filtered: int, time_sample: float
    ) -> None:
        """Print results of filtering.

        Parameters
        ----------
        pickaxe : Pickaxe
            Instance of Pickaxe being used to expand and filter the network.
            Unused here, but may be useful in your implementation.
        n_total : int
            Total number of compounds.
        n_filtered : int
            Number of compounds remaining after filtering.
        times_sample : float
            Time in seconds from time.time().
        """
        # print(
        #     f"{n_filtered} of {n_total} compounds remain after applying "
        #     f"filter: {self.filter_name}"
        #     f"--took {round(time_sample, 2)}s.\n"
        # )

        logger.info(f"{n_filtered} of {n_total} compounds remain after applying")
        logger.info(f"filter: {self.filter_name}")
        logger.info(f"--took {round(time_sample, 2)}s.")

    def _post_print_footer(self, pickaxe: Pickaxe) -> None:
        """Print end of filtering.

        Parameters
        ----------
        pickaxe : Pickaxe
            Instance of Pickaxe being used to expand and filter the network.
        """
        print(f"Done filtering Generation {pickaxe.generation}")
        print("----------------------------------------\n")

    def _get_n(self, pickaxe: Pickaxe, n_type: str) -> int:
        """Get current number of compounds to be filtered.

        Parameters
        ----------
        pickaxe : Pickaxe
            Instance of Pickaxe being used to expand and filter the network.
        n_type : str
            Whether to return "total" number of "filtered" number of compounds.

        Returns
        -------
        n : int
            Either the total or filtered number of compounds.
        """
        n = 0
        for cpd_dict in pickaxe.compounds.values():
            is_in_current_gen = cpd_dict["Generation"] == pickaxe.generation
            is_predicted_compound = cpd_dict["_id"].startswith("C")
            if is_in_current_gen and is_predicted_compound:
                if n_type == "total":
                    n += 1
                elif n_type == "filtered" and cpd_dict["Expand"]:
                    n += 1
        return n

    def _apply_filter_results(
        self,
        pickaxe: Pickaxe,
        # NOTE: The arguments passed are sets, not lists. Also, having default values be
        # empty list can be dangerous. Better to be None and then check
        compound_ids_to_check: List[str] = [],
        reaction_ids_to_delete: List[str] = [],
        whitelist_cpd_ids: List[str] = [],
        whitelist_rxn_ids: List[str] = [],
    ) -> None:
        """Apply filter results to Pickaxe object.

        Remove compounds and reactions that can be removed.
        For a compound to be removed it must:
            1. Not be flagged for expansion
            2. Not have a coproduct in a reaction marked for expansion
            3. Start with "C"

        Parameters
        ----------
        pickaxe : Pickaxe
            Instance of Pickaxe being used to expand and filter the network,
            this method modifies the Pickaxe object's compound documents.
        compound_ids_to_check : List[str]
            List of compound IDs to try to remove, if possible.
        """
        # print("check_cpds ", compound_ids_to_check)
        # print("remove_rxns", reaction_ids_to_delete)
        # print("len cpds 1 ", len(pickaxe.compounds))


        # TODO: These functions to filter off compounds and reactions are confusing and
        # seem inefficient. Might be better to re-write them with more understandable
        # logic and actually understand the behavior
        def should_delete_reaction(rxn_id: str) -> bool:
            """Returns whether or not a reaction can safely be deleted."""
            if rxn_id in whitelist_rxn_ids:
                return False

            products = pickaxe.reactions[rxn_id]["Products"]
            for _, c_id in products:
                if (
                    c_id in whitelist_cpd_ids or
                    (c_id.startswith("C") and c_id not in cpds_to_remove)
                ):
                    return False
            # Every compound isn't in cpds_to_remove
            return True

        def remove_reaction(rxn_id):
            """Removes reaction and any resulting orphan compounds"""
            cpds_to_return = set()
            # Remove affiliations of reaction and check for orphans
            product_ids = [cpd[1] for cpd in pickaxe.reactions[rxn_id]["Products"]]
            for prod_id in product_ids:
                if prod_id.startswith("C"):
                    pickaxe.compounds[prod_id]["Product_of"].remove(rxn_id)
                    cpds_to_return.add(prod_id)
            compound_ids = [cpd[1] for cpd in pickaxe.reactions[rxn_id]["Reactants"]]
            for cpd_id in compound_ids:
                if cpd_id.startswith("C"):
                    pickaxe.compounds[cpd_id]["Reactant_in"].remove(rxn_id)
                    # cpds_to_return.add(cpd_id)  # We are not adding reactants
            # Delete reaction itself
            del pickaxe.reactions[rxn_id]

            return cpds_to_return

        # Process reactions to delete
        # Loop through reactions to add compounds to check and to delete reactions
        if reaction_ids_to_delete:
            cpd_check_from_rxn = set()
            for rxn_id in reaction_ids_to_delete:
                cpd_check_from_rxn = cpd_check_from_rxn.union(remove_reaction(rxn_id))

            # Check for orphaned compounds due to reaction deletion
            while len(cpd_check_from_rxn) != 0:
                cpd_id = cpd_check_from_rxn.pop()
                # Orphan compound is one that has no reaction connecting it
                if cpd_id in pickaxe.compounds:
                    product_of = copy(pickaxe.compounds[cpd_id].get("Product_of", []))
                    cpd_type = pickaxe.compounds[cpd_id]["Type"]
                    # Delete if no reactions
                    if product_of == [] and cpd_type not in ("Starting Compound", "Coreactant"):
                        # Delete out reactions
                        reactant_in = copy(
                            pickaxe.compounds[cpd_id].get("Reactant_in", [])
                        )
                        for rxn_id in reactant_in:
                            cpd_check_from_rxn = cpd_check_from_rxn.union(
                                remove_reaction(rxn_id)
                            )
                        # Now delete compound
                        del pickaxe.compounds[cpd_id]

        # print("len cpds 2 ", len(pickaxe.compounds))
        # Go through compounds_ids_to_check and delete cpds/rxns as needed
        if compound_ids_to_check:
            cpds_to_remove = set()
            rxns_to_check = []

            compound_ids_to_check = set(compound_ids_to_check)
            for cpd_id in compound_ids_to_check:
                cpd_dict = pickaxe.compounds.get(cpd_id)
                if not cpd_dict:
                    continue

                if (
                    cpd_id not in whitelist_cpd_ids and
                    not cpd_dict["Expand"] and
                    cpd_id.startswith("C")
                ):
                    cpds_to_remove.add(cpd_id)

                    rxns_to_check.extend(pickaxe.compounds[cpd_id]["Product_of"])
                    rxns_to_check.extend(pickaxe.compounds[cpd_id]["Reactant_in"])

            rxns_to_check = set(rxns_to_check)
            # Function to check to see if should delete reaction
            # If reaction has compound that won't be deleted keep it
            # Check reactions for deletion
            for rxn_id in rxns_to_check:
                if should_delete_reaction(rxn_id):
                    for _, c_id in pickaxe.reactions[rxn_id]["Products"]:
                        if c_id.startswith("C"):
                            if rxn_id in pickaxe.compounds[c_id]["Product_of"]:
                                pickaxe.compounds[c_id]["Product_of"].remove(rxn_id)

                    for _, c_id in pickaxe.reactions[rxn_id]["Reactants"]:
                        if c_id.startswith("C"):
                            if rxn_id in pickaxe.compounds[c_id]["Reactant_in"]:
                                pickaxe.compounds[c_id]["Reactant_in"].remove(rxn_id)

                    del pickaxe.reactions[rxn_id]
                else:
                    # Reaction is dependent on compound that is flagged to be
                    # removed. Don't remove compound
                    products = pickaxe.reactions[rxn_id]["Products"]
                    cpds_to_remove -= set(i[1] for i in products)

                    # for _, c_id in products:
                    #     if c_id in cpds_to_remove:
                    #         cpds_to_remove -= {c_id}

            # Remove compounds and reactions if any found
            for cpd_id in cpds_to_remove:
                del pickaxe.compounds[cpd_id]
        # print("len cpds 3 ", len(pickaxe.compounds))
            


if __name__ == "__main__":
    pass
