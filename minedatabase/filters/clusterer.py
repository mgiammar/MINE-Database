"""Base Clustering class to use in cluster filters. Use this class to create other
custom clustering types (i.e. one using a ML approach)
"""

import time
import json
import itertools
import multiprocessing
import pathlib
from functools import partial
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from minedatabase.filters.base_filter import Filter
from minedatabase.pickaxe import Pickaxe
from minedatabase.utils import Chunks

import logging
logger = logging.getLogger("run_pickaxe")


def _bulk_similarity_async(fps: list, known_fprints: list, i: int):
    """Given a list of fingerprints and an index, compute the dissimilarity between each
    fingerprint up to that index. Return 1-similarity
    
    Arguments:
        (list) fps: List of RDKit fingerprint objects
        (list) known_fprints: List of fingerprints from target compounds
        (int) i: Index of fingerprint to compare to

    Returns:
        Similarity list where each element is 1-similarity
    """
    res = DataStructs.BulkTanimotoSimilarity(fps[i], known_fprints + fps[:i])
    return [1-x for x in res]


class BaseClusterer():
    """Base Clusterer class with unimplemented methods. Your custom classes should
    inherit from this class and override the unimplemented methods

    Attributes
    ----------
        Custom clustering classes will have their own custom attributes, but the
        following attributes are shared among all clustering classes

        (str) name: Name associated with the clusterer
        (int) max_compounds: Maximum number of compounds to try and apply the filter to.
            If max_compounds is None, then no limit will be applied. If the number of
            compounds given exceeds max_compounds, then the clustering will be skipped.
            The default is 50,000 compounds.
        
    Methods
    -------
        get_fields_as_dict: Return the attribute fields as a dictionary. Able to
        instantiate a new Clusterer class from this dict
        _should_cluster: Returns True if the cluster should be applied to the compounds,
            otherwise False.
        generate_clusters: Takes a dictionary / list of compounds
        ??? get_cluster_info: Returns some information about the clusters created.
    """
    def __init__(self, name: str, max_compounds: int=50000):
        self.name = name
        self.max_compounds = max_compounds

    def get_fields_as_dict(self):
        """Return attribute values as dict. Child classes will override this method
        so their attributes can be included
        """
        return {
            "name": self.name,
            "max_compounds": self.max_compounds
        }

    def _should_cluster(self, num_compounds: int):
        """Returns True if num_compounds is less than max_compounds, otherwise False"""
        # NOTE: Not compact one-line for future logic
        if num_compounds > self.max_compounds:
            return False
        
        return True

    def generate_clusters(self, compounds):
        """Takes in a list / dict of compounds and returns a list of clusters where
        each cluster is a tuple in that list.
        """
        raise NotImplementedError


class TanimotoSimilarityClusterer(BaseClusterer):
    """Cluster compounds based on Tanimoto fingerprint (dis)similarity scores. Generates
    a dissimilarity matrix based on the fingerprint metrics and then uses the
    RDKit.ML.Cluster.Butina module to cluster compounds based on dissimilarity.

    Attributes
    ----------
        (str) name: Name associated with the clusterer
        (int) max_compounds: Maximum number of compounds to try and apply the filter to.
            If max_compounds is None, then no limit will be applied. If the number of
            compounds given exceeds max_compounds, then the clustering will be skipped.
            The default is 50,000 compounds.
        (int) processes: Number of processes to run when doing parallel computations
        (float) cutoff: The similarity cutoff for compounds to be in the same cluster
        (dict) fingerprint_kwargs: Optional keyword arguments to pass to the
            RDKFingerprint function. The default is None TODO
        (dict) known_compounds: Dict of known compounds with InChI_key as key and
            fingerprint of compound as value, such as those in
            ModelSEED, whose fingerprint scores are added to the upper left corner of
            the similarity matrix. These compounds are grabbed from a provided CSV file
            during instantiation and precomputed. If no CSV file is provided, then the
            attribute is None.
        (list) dissimilarity_matrix: Lower triangle portion of the dissimilarity matrix
            stored as a list of lists. The diagonal is not included in the matrix, i.e.
            the 0th row corresponds to the 1st compound but the 0th column corresponds
            to the 0th compound.
        (list) ordered_ids: List of identification keys (InChI_key or Pickaxe cid) of
            all compounds in the matrix. When compound is dropped from the matrix, id
            is also dropped from this list

    Methods
    -------
        get_fields_as_dict: Return the attribute fields as a dictionary. Able to
            instantiate a new Clusterer class from this dict
        generate_clusters: Returns a list of clusters where each cluster is a list of
            compound ids based on the given clustering cutoff.
        _generate_dissimilarity_matrix: Given a set of compounds, compute the
            dissimilarity matrix. This can use the pre-computed upper corner 
            fingerprints of known compounds
        _pre_compute_known_compounds: Pre-computes the upper corner of the
            dissimilarity matrix and the fingerprints from provided known compounds.
        _drop_compounds_from_matrix: Given a list of compound ids, drop those compounds
            from the dissimilarity matrix without recomputing
    """
    # NOTE: The fact that InChI_keys are different than the Pickaxe compound IDs is used
    # to choose clusters with known compounds in them.
    def __init__(
        self,
        processes: int,
        # cutoff: float,  # Cutoff passed as argument to cluster 
        fingerprint_kwargs: dict = None,
        known_compounds_path: str = None,
        **kwargs,
    ):
        super().__init__(name="TanimotoSimilarityClusterer", **kwargs)
        self.processes = processes
        # self.cutoff = cutoff
        self.fingerprint_kwargs = {} if fingerprint_kwargs is None else fingerprint_kwargs
        self.known_compounds_path = known_compounds_path
        # Following method creates useful "private" attributes like _dissimilarity_matrix
        self._pre_compute_known_compounds()
        self.matrix = None

    def get_fields_as_dict(self):
        parent_dict = super().get_fields_as_dict()
        self_dict = {
            "processes": self.processes,
            # "cutoff": self.cutoff,  # Cutoff passed to generate_clusters argument
            "fingerprint_kwargs": self.fingerprint_kwargs,
            "known_compounds_path": self.known_compounds_path,
        }
        return {**parent_dict, **self_dict}

    def generate_clusters(self, smiles_by_cid: list, cutoff: float):
        """Clusters the list of compounds given by a dictionary of cid as key and smiles
        as value with the given cutoff.

        Arguments:
            (dict) smiles_by_cid: Dictionary with compound ids as keys and SMILES as
                values
            (float) cutoff: Cutoff similarity for compounds. Lower value means compounds
                must be more similar to appear in the same cluster
        """
        if not self._should_cluster(len(smiles_by_cid)):
            logger.warning("Too many compounds, not clustering")
            return []

        fps = [
            Chem.RDKFingerprint(Chem.MolFromSmiles(sm), **self.fingerprint_kwargs) 
            for sm in smiles_by_cid.values()
        ]
        num_known = len(self.known_compounds)
        num_new = len(fps)
        nfps = num_known + num_new

        if self.matrix is None or self.matrix == []:
            self.matrix = self._generate_dissimilarity_matrix(fps, self.processes)
        dists = list(itertools.chain.from_iterable(self.matrix))
        clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
        
        # Convert cluster indexes into InChI_key strings or compound id strings
        self.ordered_ids = list(self.known_compounds.keys()) + list(smiles_by_cid.keys())
        return [
            [self.ordered_ids[cpd_idx] for cpd_idx in cluster] for cluster in clusters
        ]

    def _pre_compute_known_compounds(self):
        """Pre computes the upper corner of dissimilarity matrix and the fingerprints
        of known compounds. If known_compounds_path is None, then simply set the
        attributes to None.
        """
        if self.known_compounds_path is None:
            # Set pre_computation attributes to None
            self.known_compounds = {}
            self._pre_computed_upper_corner = []
            return
        
        if not pathlib.Path(self.known_compounds_path).is_file():
            raise ValueError(f"Got invalid path: {self.known_compounds_path}")

        cpds_csv = pd.read_csv(self.known_compounds_path)
        known_cpds_dict = {
            row[1]["InChI_key"]: Chem.RDKFingerprint(
                Chem.MolFromSmiles(row[1]["SMILES"]), **self.fingerprint_kwargs
            ) for row in cpds_csv.iterrows()
        }
        self._pre_computed_upper_corner = []
        self.known_compounds = {}
        self._pre_computed_upper_corner = self._generate_dissimilarity_matrix(
            list(known_cpds_dict.values()), self.processes
        )

        self.known_compounds = known_cpds_dict

    def _generate_dissimilarity_matrix(self, fingerprints, processes):
        """Generates the whole dissimilarity matrix for all fingerprints. If
        _pre_computed_upper_corner is not None or empty, then the additional compound
        dissimilarities are added to the matrix. Note: The matrix created is the lower
        diagonal of the whole
        matrix.

        Arguments:
            (list) fingerprints: List of all new compound fingerprints. The fingerprints
                of any known compounds are grabbed from the saved values of the
                known_compounds dictionary.
            (int) processes: Number of processes to run in parallel
        """
        nfps = len(fingerprints)
        matrix = []
        upper_diag = self._pre_computed_upper_corner
        known_fprints = list(self.known_compounds.values())
        range_start = 1 if known_fprints == [] else 0

        # Run in multiprocessing
        if processes > 1:
            # Define partial function for multiprocessing
            bulk_similarity_partial = partial(
                _bulk_similarity_async,
                fingerprints,
                known_fprints
            )
            chunk_size = max([round(nfps / (processes * 6)), 1])  # NOTE: Arbitrary
            with multiprocessing.Pool(processes=processes) as pool:
                # Returned type is list
                # Do not need async since need to pass the whole list to function
                matrix = pool.map(
                    bulk_similarity_partial,
                    range(range_start, nfps),
                    chunk_size
                )

        else:  # No multiprocessing for computation
            for i in range(range_start, len(fps)):
                res = DataStructs.BulkTanimotoSimilarity(
                    fingerprints[i],
                    known_fprints + fingerprints[:i]
                )
                matrix.append([1-x for x in res])

        matrix = upper_diag + matrix
        return matrix

    def _drop_compounds_from_matrix(self, compounds: list):
        """Removes compound ids in the passed compounds list from the dissimilarity
        matrix without recomputing the matrix.

        Arguments:
            (list) compounds: List of string IDs for compounds to drop. If compound id
                not found in the self.ordered_ids attribute, an error will be raised.
        """
        drop_idxs = [self.ordered_ids.index(cid) for cid in compounds]
        drop_idxs.sort()
        _ = [self.ordered_ids.pop(idx) for idx in drop_idxs[::-1]]
        
        if self.matrix is None:
            return
            
        num_rows = len(self.matrix) + 1  # +1 since [0, 0] of full mat does not have entry

        if len(drop_idxs) == 0:
            return
        
        # Remove rows
        _ = [self.matrix.pop(i - 1) for i in drop_idxs[::-1] if i - 1 >= 0]

        # Now remove the last index if it is the length of the matrix, otherwise might
        # get an out of bounds error
        if drop_idxs[-1] + 1 == num_rows:
            drop_idxs.pop(-1)

        if len(drop_idxs) == 0 or len(self.matrix) == 0:
            return

        # Remove 'columns'
        for i, row in enumerate(self.matrix):
            # Get indexes within range of row lengths
            delete_indexes = np.asarray(drop_idxs)[
                np.where(np.asarray(drop_idxs) < len(row))
            ]
            self.matrix[i] = list(np.delete(row, delete_indexes))

        # Edge case for removing index zero
        if 0 in drop_idxs:
            self.matrix.pop(0)


CLUSTERER_NAME_MAP = {
    "TanimotoSimilarityClusterer": TanimotoSimilarityClusterer
}


def parse_dict_to_clusterer(_dict):
    """Takes in a dictionary representation of a Clusterer instance and returns a
    Clusterer class
    """
    name = _dict.pop("name")
    if name not in CLUSTERER_NAME_MAP:
        raise ValueError(f"Unrecognized Clusterer class name {name}.")

    return CLUSTERER_NAME_MAP[name](**_dict)




if __name__ == "__main__":
    pass

