from breizhcrops import BreizhCrops
import os
import pandas as pd

from .urls import INDEX_FILE_URLs
from ..utils import download_file
import numpy as np

class OODBreizhCrops(BreizhCrops):
    """
    An out-of-distribution wrapper around BreizhCrops that returns all samples that are NOT in the class mapping.
    This Dataset can be used to create an "other" class or to estimate the out-of-distribution performance
    """

    def __init__(self, **kwargs):
        # initialize Breizhcrops
        if "preload_ram" in kwargs.keys():
            preload_ram = kwargs["preload_ram"] # keep setting in memory
            kwargs["preload_ram"] = False # set to false for the super call
        else:
            preload_ram = False # if not specified set to False

        # initialize regular BreizhCrops Class
        super().__init__(**kwargs)

        raw_indexfile = os.path.join(self.root, "raw_index.csv")
        if not os.path.exists(raw_indexfile):
            download_file(INDEX_FILE_URLs[self.year][self.level][self.region], raw_indexfile)

        allindex = pd.read_csv(raw_indexfile, index_col=None)
        allindex = allindex.loc[allindex.sequencelength > 0]

        # overwrite index
        self.index = allindex.loc[~allindex.path.isin(self.index.path)]

        # overwrite mapping
        codes = self.index["CODE_CULTU"].unique()
        self.mapping = pd.DataFrame(np.stack([codes, np.arange(len(codes)), codes]).T,
                     columns=["code", "id", "classname"]).set_index("code")

        if preload_ram:
            self.X_list = self.preload_ram()
        else:
            self.X_list = None
