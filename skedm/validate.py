from warnings import warn
from copy import copy
from datetime import datetime

from pandas import DataFrame, Series
from numpy import any, array, isnan, ndarray

from .aux_func import IsIterable


# EDM method
def RemoveNan(self):
    """KDTree in Neighbors does not accept nan
    If ignoreNan remove Embedding rows with nan from lib_i, pred_i"""
    if self._ignoreNan:
        # type : <class 'pandas.core.series.Series'>
        # Series of bool for all Embedding columns (axis = 1) of lib_i...
        na_lib = self.Embedding_.iloc[self.lib_i_, :].isna().any(axis=1)
        na_pred = self.Embedding_.iloc[self.pred_i_, :].isna().any(axis=1)

        if na_lib.any():
            if self._name == "SMap":
                original_knn = self.knn
                original_lib_i_len = len(self.lib_i_)

            # Redefine lib_i excluding nan
            self.lib_i_ = self.lib_i_[~na_lib.to_numpy()]

            # lib_i resized, update SMap self.knn if not user provided
            if self._name == "SMap":
                if original_knn == original_lib_i_len - 1:
                    self.knn = len(self.lib_i_) - 1

        # Redefine pred_i excluding nan
        if any(na_pred):
            self.pred_i_ = self.pred_i_[~na_pred]

        # If targetVec has nan, set flag for SMap internals
        if self._name == "SMap":
            if any(isnan(self._targetVec)):
                self._targetVecNan = True

    self.PredictionValid()


# EDM method
def PredictionValid(self):
    """Validate there are pred_i to make a prediction"""
    if len(self.pred_i_) == 0:
        msg = (
            f"{self._name}: PredictionValid() No valid prediction "
            + "indices. Examine pred, E, tau, Tp parameters and/or nan."
        )
        warn(msg)


# EDM method
def Validate(self):
    """Validate DataFrame columns/target in .fit()"""

    # Ensure columns are provided, convert to iterable if bare string
    if self._columns is None:
        raise RuntimeError(f"Validate() {self._name}: columns required")

    if not IsIterable(self._columns):
        self._columns = [self._columns]

    for column in self._columns:
        if column not in self._Data.columns:
            raise RuntimeError(
                f"Validate() {self._name}: column "
                + f"{column} not found in dataFrame."
            )

    if self._target is None:
        raise RuntimeError(f"Validate() {self._name}: target required")

    if self._target not in self._Data.columns:
        raise RuntimeError(
            f"Validate() {self._name}: target "
            + f"{self._target} not found in dataFrame."
        )
