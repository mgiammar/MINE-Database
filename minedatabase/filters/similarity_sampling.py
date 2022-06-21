"""Use fingerprint similarity to cluster created compounds and select only a certain
number of compounds from each cluster to expand during the next generation. Motivation
for this filter is maintaining a diverse set of compounds while limiting redundant 
molecules from the same reactions
"""

import time
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


def _bulk_similarity(
    fps,  # Fingerprints
    i,  # Index
):
    return DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])


class SimilarityClusteringFilter(Filter):
    """DOCSTRING"""

    def __init__(
        self,
        cutoff,
        compounds_per_cluster,
        fingerprint_bits=2048,
        generation_list=None
    ):
        self._filter_name = "Similarity Clustering Filter"
        self.generation_list=generation_list

        # NOTE: There will be more hyper-parameters to add and use later
        self.cutoff = cutoff
        self.compounds_per_cluster = compounds_per_cluster
        self.fingerprint_bits = fingerprint_bits

    @property
    def filter_name(self) -> str:
        return self._filter_name

    def _pre_print(self) -> None:
        """Print and log before filtering"""
        n_compounds = None
        logger.info(f"Creating clusters for {n_compounds} using")
        logger.info(f"Fingerprint bits:       {self.fingerprint_bits}")
        logger.info(f"Tanimoto cutoff:        {self.cutoff}")
        logger.info(f"Compounds per  cluster: {self.compounds_per_cluster}")
    
    def _post_print_footer(self, pickaxe: Pickaxe) -> None:
        """Post filtering info"""
        logger.info(f"Done filtering Generation {pickaxe.generation}")
        logger.info("-----------------------------------------------")

    def _choose_items_to_filter(self, pickaxe, processes):
        """Creates clusters based on fingerprint similarity using RDKit and samples 
        compounds_per_cluster from each cluster to keep expanding
        
        TODO: Complete docstring
        """
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

    def _generate_clusters(self, pickaxe, processes):
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
            Chem.RDKFingerprint(
                Chem.MolFromSmiles(sm), fpSize=self.fingerprint_bits
            ) for sm in smiles
        ]
        
        # 2. Using DataStructs.BulkTanimotoSimilarity and multiprocessing chunks,
        #    generate the adjacency matrix between all molecules
        nfps = len(fps)
        dists = []
        chunk_size = max([round(len(smiles) / (processes * 10)), 1])
        pool = multiprocessing.Pool(processes=processes)

        bulk_similarity_partial = partial(
            _bulk_similarity,
            fps,
        )

        for res in pool.map(
            bulk_similarity_partial,
            range(1, nfps),
            chunk_size
        ):
            dists.extend([1-x for x in res])

        # 3. Cluster with Butina
        clusters = Butina.ClusterData(dists, nfps, self.cutoff, isDistData=True)
        clusters = [[cpd_ids[cpd_idx] for cpd_idx in cluster] for cluster in clusters]
        return clusters


if __name__ == "__main__":
    pass
