"""Use fingerprint similarity to cluster created compounds and select only a certain
number of compounds from each cluster to expand during the next generation. Motivation
for this filter is maintaining a diverse set of compounds while limiting redundant 
molecules from the same reactions
"""

import time
import json
import itertools
import multiprocessing
from functools import partial
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from minedatabase.filters.base_filter import Filter
from minedatabase.filters.clusterer import parse_dict_to_clusterer
from minedatabase.pickaxe import Pickaxe
from minedatabase.utils import Chunks

import logging
logger = logging.getLogger("run_pickaxe")


class SimilarityClusteringFilter(Filter):
    """Filter which clusters new compounds based on their fingerprint similarity. 
    Compounds within the cutoff will be put into the same cluster and 
    selected_per_cluster compounds will be selected randomly from each cluster.

    Attributes
    ----------
        (float) cutoff: Value between 0.0 and 1.0 denoting the similarity cutoff for
            each cluster. Lower values mean more selective (fewer compounds per cluster)
        (int) compounds_selected_per_cluster: Number of compounds to randomly select
            from each cluster
        (list) generation_list: List of generations to apply filter to. The default is
            None and will be applied to all generations after the first generation (ie
            do not cluster initial metabolomics data).
        (BaseClusterer) clusterer: An instance of the clustering class, or its dict
            representation for creating a BaseClusterer class
        (int) priority: The priority number for this filter. Lower has more priority
        (bool): all_compounds: If true, apply to all pickaxe compounds in generation,
            otherwise only select from compounds previously selected / not knocked out
            by a filter
    """
    def __init__(
        self,
        cutoff,
        compounds_selected_per_cluster,
        clusterer,  # class instance or dict
        generation_list=None,
        priority: int=10,
        all_compounds: bool=True,
    ):
        # Attributes for all filters
        self._filter_name = "Similarity Clustering Filter"
        self.generation_list=generation_list
        self.priority = priority
        self.all_compounds = all_compounds

        # Attributes for clustering filter
        self.cutoff = cutoff
        self.compounds_selected_per_cluster = compounds_selected_per_cluster
        if isinstance(clusterer, dict):
            self.clusterer = parse_dict_to_clusterer(clusterer)
        else:    
            self.clusterer = clusterer

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
            "cutoff": self.cutoff,
            "compounds_selected_per_cluster": self.compounds_selected_per_cluster,
            "clusterer": self.clusterer.get_fields_as_dict()
        }

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        logger.info(f"Creating clusters for {n_compounds} using")
        logger.info(f"Tanimoto cutoff:        {self.cutoff}")
        logger.info(f"Compounds per cluster:  {self.compounds_selected_per_cluster}")
    
    def _post_print_footer(self, pickaxe: Pickaxe) -> None:
        """Post filtering info"""
        logger.info(f"Done filtering Generation {pickaxe.generation}")
        logger.info("-----------------------------------------------")

    def _choose_items_to_filter(self, pickaxe, processes, previously_removed):
        """Creates clusters based on fingerprint similarity using RDKit and samples 
        compounds_per_cluster from each cluster to keep expanding
        
        TODO: Complete docstring
        """
        cpds_remove_set = set()
        rxns_remove_set = set()
        self.clusterer.matrix = None

        # Dictionary with compound id as key and smiles for that compound as value
        # TODO: Take into account all compounds or previously selected
        smiles_by_cid = {
             cpd_id: cpd["SMILES"] for cpd_id, cpd in pickaxe.compounds.items()
            if cpd["Generation"] == pickaxe.generation and 
            (self.all_compounds or cpd_id not in previously_removed)
        }

        clusters = self.clusterer.generate_clusters(smiles_by_cid, self.cutoff)

        cpd_ids_to_keep = set()  # TODO This become whitelist compounds
        for cluster in clusters:
            keep_ids = np.random.choice(
                cluster,
                replace=False,
                size=min(self.compounds_selected_per_cluster, len(cluster))
            )
            cpd_ids_to_keep = cpd_ids_to_keep.union(keep_ids)

        # Return IDs of all compounds in this generation to not keep
        cpds_remove_set = set(smiles_by_cid.keys()).difference(cpd_ids_to_keep)
        return cpds_remove_set, rxns_remove_set, set(), set()


class MultiRoundSimilarityClusteringFilter(Filter):
    """Filter which clusters new compounds based on their fingerprint similarity over
    multiple clustering rounds. During each round, compounds are clustered based on that
    round's similarity cutoff. 
    
    All clusters above a certain cutoff (cluster_cutoff_size) for that round
    have a number of compounds (selected_per_cluster) randomly selected for expansion.
    Any compounds in clusters below the cutoff are kept for the next round and 
    re-clustered with the next cutoff value and a number are chosen.

    Lists passed when instantiating this class must be of the same length.

    Attributes
    ----------
        (list) cutoff: List of floats between 0.0 and 1.0 denoting the similarity
            cutoff for that round.
        (list) compounds_selected_per_cluster: Number of compounds to randomly select
            from each cluster that round
        (list) cluster_size_cutoff: Only clusters above this size will have compounds
            selected from. Any compounds in clusters below this size will be
            re-clustered in the next round.
        (list) generation_list: List of generations to apply filter to. The default is
            None and will be applied to all generations after the first generation (ie
            do not cluster initial metabolomics data).
        (BaseClusterer) clusterer: An instance of the clustering class, or its dict
            representation for creating a BaseClusterer class
        (int) priority: The priority number for this filter. Lower has more priority
        (bool): all_compounds: If true, apply to all pickaxe compounds in generation,
            otherwise only select from compounds previously selected / not knocked out
            by a filter
    """
    def __init__(
        self,
        cutoff,
        compounds_selected_per_cluster,
        cluster_size_cutoff,
        clusterer,  # class instance or dict
        generation_list=None,
        priority: int=10,
        all_compounds: bool=True,
    ):
        # Attributes for all filters
        self._filter_name = "Multi Round Similarity Clustering Filter"
        self.generation_list=generation_list
        self.priority = priority
        self.all_compounds = all_compounds

        # Attributes for clustering filter
        self.cutoff = cutoff
        self.compounds_selected_per_cluster = compounds_selected_per_cluster
        self.cluster_size_cutoff = cluster_size_cutoff
        if isinstance(clusterer, dict):
            self.clusterer = parse_dict_to_clusterer(clusterer)
        else:    
            self.clusterer = clusterer

        # Check list lengths
        if not (
            len(self.cutoff) ==
            len(self.compounds_selected_per_cluster) ==
            len(self.cluster_size_cutoff)
        ):
            raise ValueError("Lists passed do not have the same lengths")

    def get_filter_fields_as_dict(self) -> dict:
        """Returns property info about filter as a dict"""
        return {
            "filter_name": self._filter_name,
            "generation_list": self.generation_list,
            "priority": self.priority,
            "all_compounds": self.all_compounds,
            "cutoff": self.cutoff,
            "compounds_selected_per_cluster": self.compounds_selected_per_cluster,
            "cluster_size_cutoff": self.cluster_size_cutoff,
            "clusterer": self.clusterer.get_fields_as_dict()
        }

    @property
    def filter_name(self) -> str:
        return self._filter_name

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        logger.info(f"Creating clusters for {n_compounds} using")
        logger.info(f"Tanimoto cutoff:        {self.cutoff}")
        logger.info(f"Compounds per cluster:  {self.compounds_selected_per_cluster}")
        logger.info(f"Cluster size cutoff:    {self.cluster_size_cutoff}")

    def _choose_items_to_filter(self, pickaxe, processes, previously_removed):
        """Creates clusters based on fingerprint similarity using RDKit and samples 
        compounds_per_cluster from each cluster to keep expanding
        
        TODO: Complete docstring
        """
        cpds_remove_set = set()
        rxns_remove_set = set()
        self.clusterer.matrix = None

        # Dictionary with compound id as key and smiles for that compound as value
        smiles_by_cid = {
             cpd_id: cpd["SMILES"] for cpd_id, cpd in pickaxe.compounds.items()
            if cpd["Generation"] == pickaxe.generation and 
            (self.all_compounds or cpd_id not in previously_removed)
        }
        all_cids = set(smiles_by_cid.keys())

        # TODO: Wrap this loop logic in its own function
        cpd_ids_to_keep = set()
        for _cutoff, _cpds_per_cluster, _cluster_size in zip(
            self.cutoff, self.compounds_selected_per_cluster, self.cluster_size_cutoff
        ):
            # Generate clusters
            clusters = self.clusterer.generate_clusters(smiles_by_cid, _cutoff)

            cluster_lengths = [len(cs) for cs in clusters]
            logger.info(f"Cluster info for {len(smiles_by_cid)} compounds")
            logger.info(f"Cutoff:              {_cutoff}")
            logger.info(f"Cluster:             {_cpds_per_cluster}")
            logger.info(f"Size:                {_cluster_size}")
            logger.info(f"Number of Clusters:  {len(clusters)}")
            logger.info(f"Cluster size max     {np.max(cluster_lengths)}")
            logger.info(f"Cluster size min     {np.min(cluster_lengths)}")
            logger.info(f"Cluster size average {np.average(cluster_lengths)}")
            logger.info(f"Cluster size std     {np.std(cluster_lengths)}")

            # Partition clusters into keep and dont keep
            if len(clusters) == 0:
                logger.warn("No clusters created. Skipping")

            # Partition into recluster and drop while selection compounds to keep
            drop_cids = []
            recluster_cids = []
            round_compound_ids = []
            for cluster in clusters:
                if len(cluster) < _cluster_size:
                    recluster_cids.extend(cluster)
                    continue
                
                # Select from compounds from cluster and do not keep cluster
                drop_cids.extend(cluster)
                round_compound_ids.extend(
                    np.random.choice(
                        cluster,
                        replace=False,
                        size=min(_cpds_per_cluster, len(cluster))
                    )
                )

            cpd_ids_to_keep = cpd_ids_to_keep.union(set(round_compound_ids))
            # Drop selected compounds from matrix and reindex smiles_by_cid
            self.clusterer._drop_compounds_from_matrix(drop_cids)
            _ = [smiles_by_cid.pop(cid) for cid in drop_cids]

        cpds_remove_set = all_cids.difference(cpd_ids_to_keep)
        return cpds_remove_set, rxns_remove_set, set(), set()  
        # Filter doesn't do whitelisting
 

class TargetCompoundsClusteringFilter(Filter):
    """Filter which clusters new compounds based on their fingerprint similarity. 
    Also include ModelSEED compounds in the clustering and prefer clusters which include
    ModelSEED compounds. The workflow of this filter is as follows

    0. (during init) Pre-compute ModelSEED similarity matrix given fingerprint
        parameters and save to variable
    1. Compute similarity full similarity matrix including the ModelSEED matrix and the
        new compounds.
    2. Cluster all molecules based full similarity matrix using cutoff_1
    3. Split all clusters into clusters which include at least 1 ModelSEED compound
        and those clusters which do not include any ModelSEED compounds

        NOTE: Check if ModelSEED compound is in cluster by integer? Indexing might be
        funky and also need to make sure finding is fast.
        IDEA: Sets and intersection per cluster

    4. Select selected_per_cluster_1 compounds in each ModelSEED cluster to continue for
        expansion. Do NOT select ModelSEED compounds
    5. Re-compute similarity matrix / Extract from full matrix for all compounds in the
        not ModelSEED clusters
    6. Cluster not ModelSEED compounds again using a less strict cutoff_2
    7. Select selected_per_cluster_2 from each cluster for expansion

    Attributes
    ----------

        (list) generation_list: List of generations to apply filter to. The default is
            None and will be applied to all generations after the first generation (ie
            do not cluster initial metabolomics data).
        (list) cutoff: List of floats between 0.0 and 1.0 dictating the similarity for
            compounds to be clustered into the same group
        (list) target_cpds_selected_per_cluster: List of 
        (list) other_cpds_selected_per_cluster: List of 
        (list) target_cpds_cluster_size_cutoff: List of 
        (list) other_cpds_cluster_size_cutoff: List of 
    """

    def __init__(
        self,
        cutoff,
        target_cpds_selected_per_cluster,  # For clusters which contain target compounds
        other_cpds_selected_per_cluster,  # For clusters without any target compounds
        target_cpds_cluster_size_cutoff,
        other_cpds_cluster_size_cutoff,
        clusterer,  # class instance or dict
        generation_list=None,
        priority: int=10,
        all_compounds: bool=True,
    ):
        # Attributes for all filters
        self._filter_name = "Target Compound Similarity Clustering Filter"
        self.generation_list=generation_list
        self.priority = priority
        self.all_compounds = all_compounds

        # Attributes for clustering filter
        self.cutoff = cutoff
        self.target_cpds_selected_per_cluster = target_cpds_selected_per_cluster
        self.other_cpds_selected_per_cluster = other_cpds_selected_per_cluster
        self.target_cpds_cluster_size_cutoff = target_cpds_cluster_size_cutoff
        self.other_cpds_cluster_size_cutoff = other_cpds_cluster_size_cutoff

        if isinstance(clusterer, dict):
            self.clusterer = parse_dict_to_clusterer(clusterer)
        else:    
            self.clusterer = clusterer

        # Check list lengths
        if not (
            len(self.cutoff) ==
            len(self.target_cpds_selected_per_cluster) ==
            len(self.other_cpds_selected_per_cluster) ==
            len(self.target_cpds_cluster_size_cutoff) ==
            len(self.other_cpds_cluster_size_cutoff)
        ):
            raise ValueError("Lists passed do not have the same lengths")


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
            "cutoff": self.cutoff,
            "target_cpds_selected_per_cluster": self.target_cpds_selected_per_cluster,
            "other_cpds_selected_per_cluster": self.other_cpds_selected_per_cluster,
            "target_cpds_cluster_size_cutoff": self.target_cpds_cluster_size_cutoff,
            "other_cpds_cluster_size_cutoff": self.other_cpds_cluster_size_cutoff,
            "clusterer": self.clusterer.get_fields_as_dict()
        }

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        # logger.info(f"Creating clusters for {n_compounds} using")
        # logger.info(f"Tanimoto cutoff 1:      {self.cutoff_1}")
        # logger.info(f"Compounds per cluster 1:  {self.selected_per_cluster}")
        # logger.info(f"Tanimoto cutoff 1:      {self.cutoff_1}")
        # logger.info(f"Compounds per cluster 1:  {self.selected_per_cluster}")
        # logger.info(f"Fingerprint kwargs:     {self.fprint_kwargs}")
        # logger.info(f"Similarity metric:      {self.similarity_metric}")
        logger.info("filter info not implemented")
    
    def _post_print_footer(self, pickaxe: Pickaxe) -> None:
        """Post filtering info"""
        logger.info(f"Done filtering Generation {pickaxe.generation}")
        logger.info("-----------------------------------------------")

    def _choose_items_to_filter(self, pickaxe, processes, previously_removed):
        """Creates clusters based on fingerprint similarity using RDKit and samples 
        compounds_per_cluster from each cluster to keep expanding
        
        TODO: Complete docstring
        """
        cpds_remove_set = set()
        rxns_remove_set = set()
        self.clusterer.matrix = None

        smiles_by_cid = {
             cpd_id: cpd["SMILES"] for cpd_id, cpd in pickaxe.compounds.items()
            if cpd["Generation"] == pickaxe.generation and 
            (self.all_compounds or cpd_id not in previously_removed)
        }
        all_cids = set(smiles_by_cid.keys())

        cpd_ids_to_keep = set()
        for (
            _cutoff,
            tgt_cpds_per_cluster,
            oth_cpds_per_cluster,
            tgt_cpds_cluster_size,
            oth_cpds_cluster_size
        ) in zip(
            self.cutoff,
            self.target_cpds_selected_per_cluster,
            self.other_cpds_selected_per_cluster,
            self.target_cpds_cluster_size_cutoff,
            self.other_cpds_cluster_size_cutoff,
        ):
            clusters = self.clusterer.generate_clusters(smiles_by_cid, _cutoff)

            # Partition clusters into those with target compounds and those without
            # by using the fact InChI_key strings are 14 characters long and pickaxe
            # compound ids are longer. Might be a more robust way of doing this, but
            # oh well...
            target_clusters = []
            other_clusters = []
            for cs in clusters:
                id_lengths = {len(_id) for _id in cs}
                if 14 in id_lengths:
                    target_clusters.append(cs)
                else:
                    other_clusters.append(cs)

            # Choose keep compounds from target compound clusters

            # Choose keep compounds from other compound clusters

            drop_cids = []
            recluster_cids = []
            round_compound_ids = []
            for cluster in target_clusters:
                # Remove any target compound ids from cluster. That way no target cids
                # are selected or dropped 
                cluster = [_id for _id in cluster if len(_id) != 14]
                if len(cluster) < tgt_cpds_cluster_size:
                    recluster_cids.extend(cluster)
                    continue

                drop_cids.extend(cluster)
                round_compound_ids.extend(
                    np.random.choice(
                        cluster,
                        replace=False,
                        size=min(tgt_cpds_per_cluster, len(cluster))
                    )
                )

            for cluster in other_clusters:
                if len(cluster) < tgt_cpds_cluster_size:
                    recluster_cids.extend(cluster)
                    continue

                drop_cids.extend(cluster)
                round_compound_ids.extend(
                    np.random.choice(
                        cluster,
                        replace=False,
                        size=min(oth_cpds_per_cluster, len(cluster))
                    )
                )
            
            cpd_ids_to_keep = cpd_ids_to_keep.union(set(round_compound_ids))
            # Drop selected compounds from matrix and reindex smiles_by_cid
            self.clusterer._drop_compounds_from_matrix(drop_cids)
            _ = [smiles_by_cid.pop(cid) for cid in drop_cids]
                
        cpds_remove_set = all_cids.difference(cpd_ids_to_keep)
        return cpds_remove_set, rxns_remove_set, set(), set()


if __name__ == "__main__":
    pass
