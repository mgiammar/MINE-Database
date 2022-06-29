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
from minedatabase.pickaxe import Pickaxe
from minedatabase.utils import Chunks

import logging
logger = logging.getLogger("run_pickaxe")


def _bulk_similarity_async(fps: list, i: int):
    """Given a list of fingerprints and an index, compute the similarity between each
    fingerprint up to that index. Return 1-similarity
    
    Arguments:
        (list) fps: List of RDKit fingerprint objects
        (int) i: Index of fingerprint to compare to

    Returns:
        Similarity list where each element is 1-similarity
    """
    res =  DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    return [1-x for x in res]


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
        (dict) fprint_kwargs: Optional Keyword arguments to pass to the RDKFingerprint
            function. The default is None.
        (str) similarity_metric: Optional string defining the similarity metric used
            when comparing fingerprints. The allowed strings are [TODO]. The default is
            TODO
        (list) generation_list: List of generations to apply filter to. The default is
            None and will be applied to all generations after the first generation (ie
            do not cluster initial metabolomics data).
        (int) max_compounds: Maximum number of compounds to try and apply filter to.
            If the number of new compounds exceeds this value, the filter will not be
            applied  TODO: Make None have no restriction
        # (bool) strict_filter: If True, the filter will return a compound ID set for
        #     compounds to remove. Otherwise an empty set will be returned and only the 
        #     compounds selected will be marked for expansion. Default is True
        #     TODO implement this.
    """

    def __init__(
        self,
        cutoff,
        compounds_selected_per_cluster,
        fprint_kwargs=None,
        similarity_metric="default",
        generation_list=None,
        max_compounds=10000,
    ):
        fprint_kwargs = {} if fprint_kwargs is None else fprint_kwargs
        # TODO: Setup similarity metric

        self._filter_name = "Similarity Clustering Filter"
        self.generation_list=generation_list

        self.cutoff = cutoff
        self.compounds_selected_per_cluster = compounds_selected_per_cluster
        self.max_compounds = max_compounds

        self.fprint_kwargs = fprint_kwargs
        self.similarity_metric = similarity_metric  # String
        similarity_metric_func = None
        self.similarity_metric_func = similarity_metric_func  # Function

    @property
    def filter_name(self) -> str:
        return self._filter_name

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        logger.info(f"Creating clusters for {n_compounds} using")
        logger.info(f"Tanimoto cutoff:        {self.cutoff}")
        logger.info(f"Compounds per cluster:  {self.compounds_selected_per_cluster}")
        logger.info(f"Fingerprint kwargs:     {self.fprint_kwargs}")
        logger.info(f"Similarity metric:      {self.similarity_metric}")
    
    def _post_print_footer(self, pickaxe: Pickaxe) -> None:
        """Post filtering info"""
        logger.info(f"Done filtering Generation {pickaxe.generation}")
        logger.info("-----------------------------------------------")

    # def _should_filter_this_generation(self):
    #     """Returns True if this filter should be applied for this generation, False
    #     otherwise.
    #     """
    #     # NOTE: Could have made one bool expression, but more comprehensible to split up
    #     if self.generation <= 0:
    #         return False
        
    #     if self.generation_list is None:
    #         return True

    #     return (self.generation - 1) in self.generation_list

    def _choose_items_to_filter(self, pickaxe, processes):
        """Creates clusters based on fingerprint similarity using RDKit and samples 
        compounds_per_cluster from each cluster to keep expanding
        
        TODO: Complete docstring
        """
        cpds_remove_set = set()
        rxn_remove_set = set()

        # Do not filter on zeroth generation
        # if not self._should_filter_this_generation():
        #     return cpds_remove_set, rxn_remove_set

        # Dictionary with compound id as key and smiles for that compound as value
        smiles_by_cid = {
             cpd_id: cpd["SMILES"] for cpd_id, cpd in pickaxe.compounds.items()
            if cpd["Generation"] == pickaxe.generation
        }

        if self.max_compounds is not None and len(smiles_by_cid) > self.max_compounds:
            logger.warn("Number of compounds too long. Skipping clustering")
            return cpds_remove_set, rxn_remove_set

        # Clusters converted from index to compound id
        similarly_matrix = self._generate_similarity_matrix(smiles_by_cid, processes)
        clusters = self._generate_clusters(smiles_by_cid, similarly_matrix, self.cutoff)

        cpd_ids_to_keep = set()
        for cluster in clusters:
            keep_ids = np.random.choice(
                cluster,
                replace=False,
                size=min(self.compounds_selected_per_cluster, len(cluster))
            )
            cpd_ids_to_keep = cpd_ids_to_keep.union(keep_ids)
        
        for cpd_id in pickaxe.compounds.keys():
            pickaxe.compounds[cpd_id]["Expand"] = cpd_id in cpd_ids_to_keep

        # Return IDs of all compounds in this generation to not keep
        cpds_remove_set = set(smiles_by_cid.keys()).difference(cpd_ids_to_keep)
        return cpds_remove_set, rxn_remove_set

    def _generate_similarity_matrix(self, smiles_by_cid, processes):
        """Generates the lower triangle of the symmetric similarity matrix
        
        Arguments:
            (dict) smiles_by_cid: Dictionary where keys are compound ids and values are
                smiles
            (int) processes: Number of processes to use when running in parallel

        Returns:
            List of lists representing the lower triangle of the fingerprint similarity
            matrix between each compound
        """
        fps = [
            Chem.RDKFingerprint(Chem.MolFromSmiles(sm), **self.fprint_kwargs) 
            for sm in smiles_by_cid.values()
        ]
        nfps = len(fps)
        matrix = []

        if processes > 1:  # Use multiprocessing
            # Define partial function for multiprocessing
            bulk_similarity_partial = partial(
                _bulk_similarity_async,
                fps,
            )
            chunk_size = max([round(nfps / (processes * 6)), 1])  # NOTE: Arbitrary
            with multiprocessing.Pool(processes=processes) as pool:
                # Returned type is list
                # Do not need async since need to pass the whole list to function
                matrix = pool.map(
                    bulk_similarity_partial,
                    range(1, nfps),
                    chunk_size
                )
                # matrix = list(itertools.chain.from_iterable(res))

        else:  # No multiprocessing for computation
            for i in range(1, len(fps)):
                res = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
                matrix.append([1-x for x in res])

        return matrix

    def _generate_clusters(self, smiles_by_cid, matrix, cutoff):
        """Uses an adjacency matrix based on fingerprint similarity to generate clusters
        using the RDKit package. 
        
        Arguments:
            (dict) smiles_by_cid: Dictionary where keys are compound ids and values are
                smiles
            (list) matrix: Lower triangle of symmetric similarity matrix
            (float) cutoff: Cutoff similarity value for clustering compounds.

        Returns:
            (list) clusters: Clusters with compound ids representing compounds
        """
        nfps = len(smiles_by_cid)

        # Convert lower triangle matrix into one continuos list (dists)
        dists = list(itertools.chain.from_iterable(matrix))

        clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
        # NOTE: Could do multiprocessing on the cluster indexes to compound ids
        clusters = [
            [list(smiles_by_cid.keys())[cpd_idx] for cpd_idx in cluster]
            for cluster in clusters
        ]
        return clusters


class MultiRoundSimilarityClusteringFilter(SimilarityClusteringFilter):
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
        (dict) fprint_kwargs: Optional Keyword arguments to pass to the RDKFingerprint
            function. The default is None.
        (str) similarity_metric: Optional string defining the similarity metric used
            when comparing fingerprints. The allowed strings are [TODO]. The default is
            TODO
        (list) generation_list: List of generations to apply filter to. The default is
            None and will be applied to all generations after the first generation (ie
            do not cluster initial metabolomics data).
    """

    def __init__(self, cluster_size_cutoff: list, **kwargs):
        """Pass other arguments to SimilarityClusteringFilter by keyword"""
        super().__init__(**kwargs)
        self.cluster_size_cutoff = cluster_size_cutoff
        self._filter_name = "Multi Round Similarity Clustering Filter"

        # Check list lengths
        if not (
            len(self.cutoff) ==
            len(self.compounds_selected_per_cluster) ==
            len(self.cluster_size_cutoff)
        ):
            raise ValueError("Lists passed do not have the same lengths")

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        logger.info(f"Creating clusters for {n_compounds} using")
        logger.info(f"Tanimoto cutoff:       {self.cutoff}")
        logger.info(f"Compounds per cluster:  {self.compounds_selected_per_cluster}")
        logger.info(f"Cluster size cutoff:    {self.cluster_size_cutoff}")
        logger.info(f"Fingerprint kwargs:     {self.fprint_kwargs}")
        logger.info(f"Similarity metric:      {self.similarity_metric}")

    def _choose_items_to_filter(self, pickaxe, processes):
        """Creates clusters based on fingerprint similarity using RDKit and samples 
        compounds_per_cluster from each cluster to keep expanding
        
        TODO: Complete docstring
        """
        cpds_remove_set = set()
        rxn_remove_set = set()

        # Do not filter on zeroth generation
        # if not self._should_filter_this_generation():
        #     return cpds_remove_set, rxn_remove_set

        # Dictionary with compound id as key and smiles for that compound as value
        smiles_by_cid = {
             cpd_id: cpd["SMILES"] for cpd_id, cpd in pickaxe.compounds.items()
            if cpd["Generation"] == pickaxe.generation
        }
        all_cids = set(smiles_by_cid.keys())

        # Set expand to false for all compounds. Selected compounds will be set to True
        for cpd_id in smiles_by_cid.keys():
            pickaxe.compounds[cpd_id]["Expand"] = False

        if self.max_compounds is not None and len(smiles_by_cid) > self.max_compounds:
            logger.warn("Number of compounds too long. Skipping clustering")
            return cpds_remove_set, rxn_remove_set

        # Clusters converted from index to compound id
        similarly_matrix = self._generate_similarity_matrix(smiles_by_cid, processes)
        self._last_similarity_matrix = similarly_matrix

        # TODO: Wrap this loop logic in its own function
        cpd_ids_to_keep = set()
        for _cutoff, _cpds_per_cluster, _cluster_size in zip(
            self.cutoff, self.compounds_selected_per_cluster, self.cluster_size_cutoff
        ):
            # Generate clusters
            clusters = self._generate_clusters(
                smiles_by_cid, self._last_similarity_matrix, _cutoff
            )
            # clusters.sort(key=len, reverse=True)
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
            selected_compound_idxs = [
                list(smiles_by_cid.keys()).index(cid) for cid in drop_cids
            ]
            _ = [smiles_by_cid.pop(cid) for cid in drop_cids]
            self._last_similarity_matrix = self.remove_indexes_from_lower_tri_matrix(
                self._last_similarity_matrix, selected_compound_idxs
            )

            if len(self._last_similarity_matrix) == 0:
                logger.warn("Sequential clustering finished early. Exiting loop")
                break
#
        # Set Expand to true for compounds ids in the cpd_ids_to_keep set
        for cp_id in cpd_ids_to_keep:
            pickaxe.compounds[cp_id]["Expand"] = True

        cpds_remove_set = all_cids.difference(cpd_ids_to_keep)
        return cpds_remove_set, rxn_remove_set

    def remove_indexes_from_lower_tri_matrix(self, matrix, indexes):
        """Removes indexes at rows and 'columns' of lower triangular matrix"""
        # NOTE: This method has become messy. Probably a cleaner way of writing this
        # but oh well...
        mat = matrix.copy()
        num_rows = len(mat) + 1  # +1 since [0, 0] of full mat does not have entry
        if len(indexes) == 0:
            return mat

        indexes.sort()  # Need to be sorted so can iterate in reverse order

        # Remove rows
        _ = [mat.pop(i - 1) for i in indexes[::-1] if i - 1 >= 0]

        # Now remove the last index if it is the length of the matrix, otherwise might
        # get an out of bounds error
        if indexes[-1] + 1 == num_rows:
            indexes.pop(-1)

        if len(indexes) == 0:
            return mat

        # Remove 'columns'
        for i, row in enumerate(mat):
            # Get indexes within range of row lengths
            delete_indexes = np.asarray(indexes)[
                np.where(np.asarray(indexes) < len(row))
            ]
            mat[i] = list(np.delete(row, delete_indexes))

        # Edge case for removing index zero
        if 0 in indexes:
            mat.pop(0)

        return mat
 


class ModelSEEDSimilarityClusteringFilter(Filter):
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
        (float) cutoff_1: Value between 0.0 and 1.0 denoting the similarity cutoff for
            the first clustering with ModelSEED compounds. Lower values mean more 
            selective (fewer compounds per cluster)
        (int) selected_per_cluster_1: Number of compounds to randomly select during the
            first clustering with ModelSEED compounds from clusters which include a
            ModelSEED compound
        (float) cutoff_2: Value between 0.0 and 1.0 denoting the similarity cutoff for
            the second clustering without ModelSEED compounds. Lower values mean more 
            selective (fewer compounds per cluster)
        (int) selected_per_cluster_2: Number of compounds to randomly select during the
            second clustering without ModelSEED compounds from clusters.
        (dict) fprint_kwargs: Optional Keyword arguments to pass to the RDKFingerprint
            function. The default is None. TODO
        (str) similarity_metric: Optional string defining the similarity metric used
            when comparing fingerprints. The allowed strings are [TODO]. The default is
            TODO
        (list) generation_list: List of generations to apply filter to. The default is
            None and will be applied to all generations after the first generation (ie
            do not cluster initial metabolomics data).
        (str) ModelSEED_json_path: Optional path to JSON file which contains a (list/object)
            of ModelSEED smiles to include. Default is TODO
    """

    def __init__(
        self,
        cutoff_1,
        selected_per_cluster_1,
        cutoff_2,
        selected_per_cluster_2,
        selected_per_cluster,
        fprint_kwargs=None,
        generation_list=None,
        ModelSEED_json_path=""  # TODO
    ):
        fprint_kwargs = {} if fprint_kwargs is None else fprint_kwargs
        # TODO: Setup similarity metric

        self._filter_name = "Similarity Clustering Filter"
        self.generation_list=generation_list

        self.cutoff_1 = cutoff_1
        self.selected_per_cluster_1 = selected_per_cluster_1
        self.cutoff_2 = cutoff_2
        self.selected_per_cluster_2 = selected_per_cluster_2

        self.fprint_kwargs = fprint_kwargs
        self.similarity_metric = similarity_metric  # String
        self.similarity_metric_func = similarity_metric_func  # Function

        # Pre-compute ModelSEED similarity matrix
        self.mseed_matrix, self.mseed_fps = self._pre_compute_modelseed_similarity(
            ModelSEED_json_path
        )
        self.num_mseed_cpds = len(self.mseed_matrix)  # NOTE could be +1

    def _pre_compute_modelseed_similarity(self, json_path: str):
        """Computes the fingerprint similarity matrix of the ModelSEED compounds from
        the compound smiles in the json_path file.
        """
        # 1. Load JSON file data into memory

        # 2. Compute similarity matrix (lower half)

        # 3. Return matrix, fingerprints (and any additional data?)

        raise NotImplementedError

    @property
    def filter_name(self) -> str:
        return self._filter_name

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        logger.info(f"Creating clusters for {n_compounds} using")
        logger.info(f"Tanimoto cutoff 1:      {self.cutoff_1}")
        logger.info(f"Compounds per cluster 1:  {self.selected_per_cluster}")
        logger.info(f"Tanimoto cutoff 1:      {self.cutoff_1}")
        logger.info(f"Compounds per cluster 1:  {self.selected_per_cluster}")
        logger.info(f"Fingerprint kwargs:     {self.fprint_kwargs}")
        logger.info(f"Similarity metric:      {self.similarity_metric}")
    
    def _post_print_footer(self, pickaxe: Pickaxe) -> None:
        """Post filtering info"""
        logger.info(f"Done filtering Generation {pickaxe.generation}")
        logger.info("-----------------------------------------------")

    def _choose_items_to_filter(self, pickaxe, processes):
        """Creates clusters based on fingerprint similarity using RDKit and samples 
        compounds_per_cluster from each cluster to keep expanding
        
        TODO: Complete docstring
        """
        # TODO: Log ModelSEED cluster and num selected compounds
        # TODO: Log non ModelSEED clusters (2) and num selected compounds
        cpds_remove_set = set()
        rxn_remove_set = set()

        # Do not filter on zeroth generation
        if (
            pickaxe.generation == 0 or
            self.generation_list and 
            (self.generation - 1) not in self.generation_list
        ):
            return cpds_remove_set, rxn_remove_set

        # Clusters converted from index to compound id
        clusters = self._generate_clusters(pickaxe, processes)

        if clusters is None:
            return cpds_remove_set, rxn_remove_set

        for cluster in clusters:
            keep_ids = np.random.choice(
                cluster,
                replace=False,
                size=min(self.compounds_per_cluster, len(cluster))
            )
            for cpd_id in cluster:
                pickaxe.compounds[cpd_id]["Expand"] = cpd_id in keep_ids

        return cpds_remove_set, rxn_remove_set

    def _generate_mseed_clusters(self, pickaxe, processes):
        """Uses an adjacency matrix based on fingerprint similarity to generate clusters
        using the RDKit package
        
        TODO: Complete docstring
        """
        # 1. Compute fingerprints of molecules
        smiles_by_cpd_id = {
            cpd_id: cpd["SMILES"] for cpd_id, cpd in pickaxe.compounds.items()
            if cpd["Generation"] == pickaxe.generation
        }
        cpd_ids = list(smiles_by_cpd_id.keys())
        smiles = list(smiles_by_cpd_id.values())

        # Limit number of compounds to 25000
        if len(smiles) > 25000:
            logger.warn("Number of compounds too long. Skipping clustering")
            return None

        fps = [
            Chem.RDKFingerprint(Chem.MolFromSmiles(sm), **self.fprint_kwargs) 
            for sm in smiles
        ]
        mseed_fps_len = len(mseed_fps)
        all_fps = self.mseed_fps + fps
        all_fps_len = len(all_fps)
        
        for i in range(1, len(fps)):
            dists.append(DataStructs.BulkTanimotoSimilarity(
                fps[i], all_fps[:mseed_fps_len+i])
            )

        # TODO: rewrite distance matrix method for multiprocessing AND similarity metric
        #
        # 2. Using DataStructs.BulkTanimotoSimilarity and multiprocessing chunks,
        #    generate the adjacency matrix between all molecules


if __name__ == "__main__":
    pass
