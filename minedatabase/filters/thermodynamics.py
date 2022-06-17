import multiprocessing
import functools
import datetime
from typing import Set

import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from equilibrator_api import Q_

from minedatabase.filters.base_filter import Filter
from minedatabase.pickaxe import Pickaxe
from minedatabase.thermodynamics import Thermodynamics


logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")


class ThermoFilter(Filter):
    """A filter that removes reactions and compounds with bad ∆Gr

    This filter allows for the specification of a pH, Ionic strength, pMg and using
    these to calculate ∆Gr. Reeactions are then filtered out based on ∆Gr.

    Parameters
    ----------
    eq_uri : str
        URI of ccache. Can be an sqlite or postgres URI. If no uri is given
        the default is the system default eQuilibrator uri.
    dg_max : float
        Maximum ∆Gr in kJ/mol, by default 0.
    pH : float
        pH of the expansion, by default 7
    ionic_strength : float
        ionic strength of the expansion, by default 0
    pMg : float
        pMg of the expansion, by default 3
    generation_list : list
        Generations to apply filter -- empty list filters all, by default empty list
    last_generation_only : bool
        Whether or not to only filter last generation, by default False

    Attributes
    ----------
    dg_max : Q_
    pH : Q_
        pH of the expansion
    ionic_strength : Q_
        ionic strength of the expansion
    pMg : Q_
        pMg of the expansion
    generation_list : list
        Generations to apply filter -- empty list filters all, by default empty list
    last_generation_only : bool
        Whether or not to only filter last generation, by default False
    """

    def __init__(
        self,
        eq_uri=None,
        dg_max=0,
        p_h=7,
        ionic_strength=0,
        p_mg=3,
        physiological=False,
        generation_list=[],
        last_generation_only=False,
    ) -> None:
        self._filter_name = "Thermodynamic Filter"
        self.dg_max = Q_(f"{dg_max}kJ/mol")
        self.p_h = Q_(f"{p_h}")
        self.ionic_strength = Q_(f"{ionic_strength}M")
        self.p_mg = Q_(f"{p_mg}")
        self.physiological = physiological
        self.generation_list = generation_list
        self.last_generation_only = last_generation_only
        self.info_string = None

        self.thermo = Thermodynamics()
        if not eq_uri:
            eq_uri = ""
        if "post" in eq_uri:
            self.thermo.load_thermo_from_postgres(eq_uri)
        elif "sql" in eq_uri:
            self.thermo.load_thermo_from_sqlite(eq_uri)
        else:
            self.thermo.load_thermo_from_sqlite()

    @property
    def filter_name(self) -> str:
        return self._filter_name

    def _pre_print(self) -> None:
        """Print before filtering."""
        print(f"Filter out reactions with ∆Gr < {self.dg_max}")

    def _post_print(
        self, pickaxe: Pickaxe, n_total: int, n_filtered: int, time_sample: float
    ) -> None:
        """Print after filtering."""
        info_string = (
            f"{n_filtered} of {n_total} "
            f"compounds selected after thermodynamic filtering in "
            f"{str(datetime.timedelta(seconds=time_sample))}."
        )
        self.info_string = info_string
        print(
            (
                f"{n_filtered} of {n_total} "
                f"compounds selected after thermodynamic filtering in "
                f"{str(datetime.timedelta(seconds=time_sample))}."
            )
        )


    # Flag all compounds with Expand = False if created from an unfavorable reaction
    # If reaction does not have any expandable compounds, remove that reaction and
    # its compounds.

    def _calc_compound_delta_g_formation(smiles_str: str) -> tuple:
        """Calculates the delta G of formation for the given compound with smiles string
        using equilibrator_assets. Return tuple which is then used to update the delta
        G of formation for each compound in the pickaxe compound dict
        """
        pass


    def _choose_items_to_filter(self, pickaxe: Pickaxe, processes: int = 1) -> Set[str]:
        """
        Check the compounds against the MW constraints and return
        compounds to filter.
        """
        cpds_remove_set = set()
        rxns_remove_set = set()

        # TODO put these statements together
        # No reactions to filter for
        if len(pickaxe.reactions) == 0:
            print("No reactions to calculate ∆Gr for.")
            return cpds_remove_set, rxns_remove_set

        if self.last_generation_only and pickaxe.generation != self.generation:
            print("Not filtering for this generation using thermodynamics.")
            return cpds_remove_set, rxns_remove_set

        if self.generation_list and (self.generation - 1) not in self.generation_list:
            print("Not filtering for this generation using thermodynamics.")
            return cpds_remove_set, rxns_remove_set

        print(
            f"Filtering Generation {pickaxe.generation} "
            f"with ∆G <= {self.dg_max} at pH={self.p_h}, "
            f"I={self.ionic_strength}, pMg={self.p_mg}"
        )

        reactions_to_check = []
        for cpd in pickaxe.compounds.values():
            # Compounds are in generation and correct type
            if cpd["Generation"] == pickaxe.generation and cpd["Type"] not in [
                "Coreactant",
                "Target Compound",
            ]:
                reactions_to_check.extend(cpd["Product_of"])

        # reactions_to_check = set(reactions_to_check)

        # NOTE: Delta G is computed from the reaction itself, not individual compounds
        # Multithreading idea here probably wont work.
        # # First create a set of all compounds ids which are compounds or reactions
        # # NOTE: This could probably be reduced to one list comprehension per reaction
        # c_ids_to_calculate = set(
        #     [react[1] for r_id in reactions_to_check for react in pk.reactions[r_id]["Reactants"]] +
        #     [prod[1] for r_id in reactions_to_check for prod in pk.reactions[r_id]["Products"]]
        # )

        # # Next calculate delta G of formation for each compound and add that value to
        # # the pickaxe compounds dictionary
        # with multiprocessing.Pool(processes=processes) as pool:
        #     pool.map_async(_calc_compound_delta_g_formation, c_ids_to_calculate)

        partial_parallel_physiological = functools.partial(
            parallel_check_reaction_physiological,
            self.thermo,
            self.dg_max,
            pickaxe
        )
        def parallel_check_reaction_physiological(
            thermo,
            dg_max,
            pickaxe,
            rxn_id,  # rxn_id must be last
        ):
            """DOCSTRING"""
            rxn_dg = thermo.physiological_dg_prime_from_rid(
                r_id=rxn_id, pickaxe=pickaxe
            )
            if rxn_dg > dg_max:
                return rxn_id
            return None

        partial_parallel_custom = functools.partial(
            parallel_check_reaction_custom,
            self.thermo,
            pickaxe,
            Q_(f"{self.p_h}"),
            Q_(f"{self.ionic_strength}"),
            Q_(f"{self.p_mg}")
        )
        def parallel_check_reaction_custom(
            thermo,
            dg_max,
            pickaxe,
            p_h,
            ionic_strength,
            p_mg,
            rxn_id,  # rxn_id must be last
        ):
            """DOCSTRING"""
            rxn_dg = self.thermo.dg_prime_from_rid(
                    r_id=rxn_id,
                    pickaxe=pickaxe,
                    p_h=p_h,
                    ionic_strength=ionic_strength,
                    p_mg=p_mg,
                )
            if rxn_dg > dg_max:
                return rxn_id
            return None

        if processes > 1:
            if self.physiological:
                with multiprocessing.Pool(processes=processes) as pool:
                    rxns_remove_set = set(
                        pool.map_async(partial_parallel_physiological, reactions_to_check)
                    )
            else:
                with multiprocessing.Pool(processes=processes) as pool:
                    rxns_remove_set = set(
                        pool.map_async(partial_parallel_custom, reactions_to_check)
                    )
        else:
            for rxn_id in reactions_to_check:
                if self.physiological:
                    rxn_dg = self.thermo.physiological_dg_prime_from_rid(
                        r_id=rxn_id, pickaxe=pickaxe
                    )
                else:
                    rxn_dg = self.thermo.dg_prime_from_rid(
                        r_id=rxn_id,
                        pickaxe=pickaxe,
                        p_h=Q_(f"{self.p_h}"),
                        ionic_strength=Q_(f"{self.ionic_strength}"),
                        p_mg=Q_(f"{self.p_mg}"),
                    )
                if rxn_dg >= self.dg_max:
                    rxns_remove_set.add(rxn_id)

        # Ensure None is removed from rxns_remove_set by first adding and then removing
        rxns_remove_set.add(None)
        rxns_remove_set.remove(None)

        return cpds_remove_set, rxns_remove_set


# # ======================================================================================
# # ========================== Overwriting base filter methods ===========================
# # ======================================================================================

#     def apply_filter(
#         self,
#         pickaxe: Pickaxe,
#         processes: int = 1,
#         generation: int = 0,
#         print_on: bool = True,
#     ) -> None:
#         """Apply filter from Pickaxe object.

#         Parameters
#         ----------
#         pickaxe : Pickaxe
#             The Pickaxe object to filter.
#         processes : int
#             The number of processes to use, by default 1.
#         print_on : bool
#             Whether or not to print filtering results.
#         """
#         time_sample = time.time()

#         self.generation = generation

#         if print_on:
#             n_total = self._get_n(pickaxe, "total")
#             self._pre_print_header(pickaxe)
#             self._pre_print()

#         compound_ids_to_check, reaction_ids_to_check = self._choose_items_to_filter(
#             pickaxe, processes
#         )

#         self._apply_filter_results(
#             pickaxe, compound_ids_to_check, reaction_ids_to_check
#         )

#         if print_on:
#             n_filtered = self._get_n(pickaxe, "filtered")
#             self._post_print(pickaxe, n_total, n_filtered, time_sample)
#             self._post_print_footer(pickaxe)

#     def _pre_print_header(self, pickaxe: Pickaxe) -> None:
#         """Print header before filtering.

#         Parameters
#         ----------
#         pickaxe : Pickaxe
#             Instance of Pickaxe being used to expand and filter the network.
#         """
#         print("----------------------------------------")
#         print(f"Filtering Generation {pickaxe.generation}\n")

#     def _pre_print(self) -> None:
#         """Print filter being applied."""
#         print(f"Applying filter: {self.filter_name}")

#     def _post_print(
#         self, pickaxe: Pickaxe, n_total: int, n_filtered: int, time_sample: float
#     ) -> None:
#         """Print results of filtering.

#         Parameters
#         ----------
#         pickaxe : Pickaxe
#             Instance of Pickaxe being used to expand and filter the network.
#             Unused here, but may be useful in your implementation.
#         n_total : int
#             Total number of compounds.
#         n_filtered : int
#             Number of compounds remaining after filtering.
#         times_sample : float
#             Time in seconds from time.time().
#         """
#         print(
#             f"{n_filtered} of {n_total} compounds remain after applying "
#             f"filter: {self.filter_name}"
#             f"--took {round(time.time() - time_sample, 2)}s.\n"
#         )

#     def _post_print_footer(self, pickaxe: Pickaxe) -> None:
#         """Print end of filtering.

#         Parameters
#         ----------
#         pickaxe : Pickaxe
#             Instance of Pickaxe being used to expand and filter the network.
#         """
#         print(f"Done filtering Generation {pickaxe.generation}")
#         print("----------------------------------------\n")

#     def _get_n(self, pickaxe: Pickaxe, n_type: str) -> int:
#         """Get current number of compounds to be filtered.

#         Parameters
#         ----------
#         pickaxe : Pickaxe
#             Instance of Pickaxe being used to expand and filter the network.
#         n_type : str
#             Whether to return "total" number of "filtered" number of compounds.

#         Returns
#         -------
#         n : int
#             Either the total or filtered number of compounds.
#         """
#         n = 0
#         for cpd_dict in pickaxe.compounds.values():
#             is_in_current_gen = cpd_dict["Generation"] == pickaxe.generation
#             is_predicted_compound = cpd_dict["_id"].startswith("C")
#             if is_in_current_gen and is_predicted_compound:
#                 if n_type == "total":
#                     n += 1
#                 elif n_type == "filtered" and cpd_dict["Expand"]:
#                     n += 1
#         return n

#     def _apply_filter_results(
#         self,
#         pickaxe: Pickaxe,
#         compound_ids_to_check: List[str] = [],
#         reaction_ids_to_delete: List[str] = [],
#     ) -> None:
#         """OVERWRITTEN FROM BASE FILTER OBJECT
        
#         Apply filter results to Pickaxe object.

#         Remove compounds and reactions that can be removed.
#         For a compound to be removed it must:
#             1. Not be flagged for expansion
#             2. Not have a coproduct in a reaction marked for expansion
#             3. Start with "C"

#         Parameters
#         ----------
#         pickaxe : Pickaxe
#             Instance of Pickaxe being used to expand and filter the network,
#             this method modifies the Pickaxe object's compound documents.
#         compound_ids_to_check : List[str]
#             List of compound IDs to try to remove, if possible.
#         """
#         # TODO: These functions to filter off compounds and reactions are confusing and
#         # seem inefficient. Might be better to re-write them with more understandable
#         # logic and actually understand the behavior
#         def should_delete_reaction(rxn_id: str) -> bool:
#             """Returns whether or not a reaction can safely be deleted."""
#             products = pickaxe.reactions[rxn_id]["Products"]
#             for _, c_id in products:
#                 if c_id.startswith("C") and c_id not in cpds_to_remove:
#                     return False
#             # Every compound isn't in cpds_to_remove
#             return True

#         def remove_reaction(rxn_id):
#             """Removes reaction and any resulting orphan compounds"""
#             cpds_to_return = set()
#             # Remove affiliations of reaction and check for orphans
#             product_ids = [cpd[1] for cpd in pickaxe.reactions[rxn_id]["Products"]]
#             for prod_id in product_ids:
#                 if prod_id.startswith("C"):
#                     pickaxe.compounds[prod_id]["Product_of"].remove(rxn_id)
#                     cpds_to_return.add(prod_id)
#             compound_ids = [cpd[1] for cpd in pickaxe.reactions[rxn_id]["Reactants"]]
#             for cpd_id in compound_ids:
#                 if cpd_id.startswith("C"):
#                     pickaxe.compounds[cpd_id]["Reactant_in"].remove(rxn_id)
#                     cpds_to_return.add(cpd_id)
#             # Delete reaction itself
#             del pickaxe.reactions[rxn_id]

#             return cpds_to_return

#         # Process reactions to delete
#         # Loop through reactions to add compounds to check and to delete reactions
#         if reaction_ids_to_delete:
#             cpd_check_from_rxn = set()
#             for rxn_id in reaction_ids_to_delete:
#                 cpd_check_from_rxn = cpd_check_from_rxn.union(remove_reaction(rxn_id))

#             # Check for orphaned compounds due to reaction deletion
#             while len(cpd_check_from_rxn) != 0:
#                 cpd_id = cpd_check_from_rxn.pop()
#                 # Orphan compound is one that has no reaction connecting it
#                 if cpd_id in pickaxe.compounds:
#                     product_of = copy(pickaxe.compounds[cpd_id].get("Product_of", []))
#                     # Delete if no reactions
#                     if not product_of:
#                         # Delete out reactions
#                         reactant_in = copy(
#                             pickaxe.compounds[cpd_id].get("Reactant_in", [])
#                         )
#                         for rxn_id in reactant_in:
#                             cpd_check_from_rxn = cpd_check_from_rxn.union(
#                                 remove_reaction(rxn_id)
#                             )
#                         # Now delete compound
#                         del pickaxe.compounds[cpd_id]

#         # Go through compounds_ids_to_check and delete cpds/rxns as needed
#         if compound_ids_to_check:
#             cpds_to_remove = set()
#             rxns_to_check = []

#             compound_ids_to_check = set(compound_ids_to_check)
#             for cpd_id in compound_ids_to_check:
#                 cpd_dict = pickaxe.compounds.get(cpd_id)
#                 if not cpd_dict:
#                     continue

#                 if not cpd_dict["Expand"] and cpd_id.startswith("C"):
#                     cpds_to_remove.add(cpd_id)

#                     rxns_to_check.extend(pickaxe.compounds[cpd_id]["Product_of"])
#                     rxns_to_check.extend(pickaxe.compounds[cpd_id]["Reactant_in"])

#             rxns_to_check = set(rxns_to_check)
#             # Function to check to see if should delete reaction
#             # If reaction has compound that won't be deleted keep it
#             # Check reactions for deletion
#             for rxn_id in rxns_to_check:
#                 if should_delete_reaction(rxn_id):
#                     for _, c_id in pickaxe.reactions[rxn_id]["Products"]:
#                         if c_id.startswith("C"):
#                             if rxn_id in pickaxe.compounds[c_id]["Product_of"]:
#                                 pickaxe.compounds[c_id]["Product_of"].remove(rxn_id)

#                     for _, c_id in pickaxe.reactions[rxn_id]["Reactants"]:
#                         if c_id.startswith("C"):
#                             if rxn_id in pickaxe.compounds[c_id]["Reactant_in"]:
#                                 pickaxe.compounds[c_id]["Reactant_in"].remove(rxn_id)

#                     del pickaxe.reactions[rxn_id]
#                 else:
#                     # Reaction is dependent on compound that is flagged to be
#                     # removed. Don't remove compound
#                     products = pickaxe.reactions[rxn_id]["Products"]
#                     cpds_to_remove -= set(i[1] for i in products)

#                     # for _, c_id in products:
#                     #     if c_id in cpds_to_remove:
#                     #         cpds_to_remove -= {c_id}

#             # Remove compounds and reactions if any found
#             for cpd_id in cpds_to_remove:
#                 del pickaxe.compounds[cpd_id]
