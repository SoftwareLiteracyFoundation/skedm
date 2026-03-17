"""
Class Simplex : projection of target variable from embedding library

To Do:
Cross-validation splitters: sklearn's KFold, TimeSeriesSplit etc. operate on row
indices of X and assume lib/pred semantics are not embedded in the estimator.
Provide a custom SimplexSplitter that wraps lib/pred parameters as a
BaseCrossValidator to make the estimator composable with GridSearchCV
However, under default conditions lib = pred = [1,N] KFold etc are fine.
"""

# Author: Joseph Park
# License: BSD 3 clause

from copy import copy
from warnings import warn

from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import Tags, InputTags, TargetTags, RegressorTags
from numpy import array, divide, exp, isnan, fmax, ndarray, power, subtract, sum, zeros
from numpy import isinf, isnan, ndarray
from scipy.sparse import issparse
from pandas import DataFrame

from .embed import Embed


class Simplex(RegressorMixin, BaseEstimator):
    """Simplex projection of target variable from embedding library

    Simplex is a nearest neighbors projection from a target (query) point in
    a state space (library) to time Tp. The simplex is an E+1 dimensional
    geometric object in the state space with the n=E+1 nearest neighbors
    as vertices. Each neighbor represents an E-dimensional vector.

    Parameters
    ----------
    columns : [str]
        Vector of column names used to create embedding library

    target : str
        DataFrame column name of target feature to predict

    E : int
        Embedding dimension applied in time delay embedding

    tau : int
        Embedding time delay offset. Negative are delays, positive future values.

    Tp : int
        Prediction horizon in units of time series row indices

    lib : [int]
        Vector of pairs of 1-offset integer indices defining the embedding library
        The first index of a pair is the start index, the second the stop index.
        Default lib=None will assign lib=[1,N_obs] where N_obs is the number of
        observations. lib is _not_ 0-offset but ranges from 1 to N_obs.

    pred : [int]
        Vector of pairs of 1-offset integer indices defining target prediction times
        The first index of a pair is the start index, the second the stop index.
        Default pred=None will assign pred=[1,N_obs] where N_obs is the number of
        observations. pred is _not_ 0-offset but ranges from 1 to N_obs.

    knn : int
        Number of k-nearest neighbors for the simplex. If default knn=0 it is
        set to E+1 for Simplex, and to N_obs for SMap.

    exclusionRadius : int
        Temporal exclusion radius for nearest neighbors. Neighbors closer than
        exclusionRadius indices from the target are ignored. Only applicable if
        lib and pred overlap.

    embedded : bool
        Is the input an embedding? If False (default) all columns will be time
        delay embedded with E and tau.

    noTime : bool
        X is expected to be DataFrame with time index/values/strings in first
        column. If no time vector is provided set noTime=True.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Embedding_ : DataFrame
        Embedding of X from which predictions are made

    Projection_ : DataFrame
        DataFrame with columns 'Time', 'Observations', 'Predictions', 'Pred_Variance'
        of the Simplex projection

    lib_i_ : ndarray
        List of library indices identifying time points from which embedding
        vectors are used for the projection
    
    pred_i_ : ndarray
        List of indices identifying time points at which embedding
        vectors are used to make the projection
    
    knn_neighbors_ : ndarray
        Matrix of k-nearest neighbors at each prediction time

    knn_distances_ : ndarray
        Matrix of k-nearest neighbors distances at each prediction time

    Returns
    -------
    Vector of predictions of target (y) at forecast horizon Tp.

    Examples
    --------
    >>> from skedm import Simplex
    >>> from pandas import DataFrame
    >>> df = DataFrame({'time':[t for t in range(1,21)],
                        'x':[1,1,3,4,5,5,6,8,3,3,2,6,5,5,9,3,5,1,8,2],
                        'y':[5,5,7,8,6,6,7,8,2,2,2,8,3,3,7,5,3,1,1,1]})
    >>> smplx = Simplex(columns='x',target='y',E=2)
    >>> smplx.fit(df)
    Simplex(E=2, columns='x', target='y')
    >>> smplx.predict(df)

    Notes
    -----

    See Also
    --------
    https://en.wikipedia.org/wiki/Empirical_dynamic_modeling#Simplex

    Reference
    ---------
    Sugihara, George; May, Robert M. (April 1990).
    "Nonlinear forecasting as a way of distinguishing chaos from measurement
    error in time series". Nature. 344 (6268): 734–741. doi:10.1038/344734a0.
    """

    # Used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {"no_validation"}

    def __init__(
        self,
        columns=None,
        target=None,
        E=1,
        tau=-1,
        Tp=1,
        lib=None,
        pred=None,
        knn=0,
        exclusionRadius=0,
        embedded=False,
        noTime=False
    ):
        """Parameters are considered static;
        If mutable they should be copied before being modified.
        """
        self.columns = columns
        self.target = target
        self.lib = lib
        self.pred = pred
        self.E = E
        self.Tp = Tp
        self.knn = knn
        self.tau = tau
        self.exclusionRadius = exclusionRadius
        self.embedded = embedded
        self.noTime = noTime

    # Class Methods
    from .edm_indices import CreateIndices
    from .neighbors import FindNeighbors
    from .formatting import FormatProjection, ConvertTime, AddTime
    from .validate import Validate, PredictionValid, RemoveNan
    from .edm_params import EDM_params

    def EmbedData(self):
        """Embed data : If not embedded call Embed()"""
        if not self.embedded:
            self.Embedding_ = Embed(
                dataFrame=self._Data, E=self._E, tau=self.tau, columns=self._columns
            )
        else:
            self.Embedding_ = self._Data[self.columns]  # Already an embedding

    def __sklearn_tags__(self):
        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=False),
            regressor_tags=RegressorTags(poor_score=True),
            input_tags=InputTags(allow_nan=True),
        )

    def get_feature_names_out(self, input_features=None):
        """set_output for downstream pipeline compatibility
        This function is a method of class ColumnTransformer"""
        check_is_fitted(self)
        return np.array([self._target + "_pred"], dtype=object)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Initialize lib & pred indices, embed data, find neighbors

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Observed data to be embedded, or used as embedding.

        y : accepted but silently ignored — embedding is target-independent

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        """

        # sklearn requires static __init__ parameters
        # instantiate copies that can be updated based on EDM or test criteria
        self._columns = copy(self.columns)
        self._target = copy(self.target)
        self._lib = copy(self.lib)
        self._pred = copy(self.pred)
        self._E = copy(self.E)
        self._knn = copy(self.knn)
        self._noTime = copy(self.noTime)
        self._name = "Simplex"

        self._validate_params()  # Additional parameter validation in self.Validate()

        # validate_data() validates input data and sets or checks feature
        # names and counts of the input. This mutates the estimator setting
        # n_features_in_ and feature_names_in_ attributes if reset=True.
        # Enable skip_check_array to allow pandas instead of ndarray coercion
        # Returns {ndarray, sparse matrix} or tuple of these
        Z = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
            dtype="numeric",
            validate_separately=False,
            skip_check_array=True,
            accept_sparse=False,
            ensure_all_finite="allow-nan",
            y_numeric=True,
        )

        if isinstance(Z, tuple):
            _X, _y = X, y
        else:
            _X, _y = X, None

        # A pandas DataFrame in X per pyEDM requirements is expected, set to self._Data
        # If X is not DataFrame create self._Data with _columns= ['x1','x2',..]
        # _target='y' and issue warning.
        if isinstance(_X, DataFrame):
            self._Data = _X.copy()  # DataFrame passed in : presume user API call

        elif issparse(_X):
            msg = (
                f"{self._name} sparse input not supported; got {type(_X).__name__}"
            )
            raise TypeError(msg)

        elif isinstance(_X, list) or isinstance(_X, ndarray):
            warn(
                "Simplex received ndarray but column names are load-bearing "
                "metadata for embedding and target selection. "
                "Provide a pandas DataFrame with column names and a time vector "
                "in the first column for application.",
                UserWarning,
                stacklevel=2,
            )
            if isinstance(_X, list):
                _X = array(_X)
                if isinstance(_y, list):
                    _y = array(_y)

            # Create columns names for X as ['x1','x2',..]
            if _X.ndim == 1:
                num_cols = _X.shape[0]
            elif _X.ndim == 2:
                num_cols = _X.shape[1]
            self._columns = getattr(
                self, "feature_names_in_", array([f"x{i+1}" for i in range(num_cols)])
            )
            self._Data = DataFrame(_X, columns=self._columns)
            if _y is not None and len(_y):
                self._target = "y"
                self._Data.insert(self._Data.shape[1], self._target, _y)

            # Set self._noTime True to add time vector in EDM_params()
            self._noTime = True

        else:
            msg = (
                f"{self._name} requires DataFrame or ndarray; "
                + f"got {type(X).__name__}"
            )
            raise ValueError(msg)

        if self._Data.empty:
            msg = f"{self._name} DataFrame X is empty."
            raise ValueError(msg)

        if self._Data.shape[0] < 11:
            msg = f"{self._name} skedm requires at least 10 observations."
            raise ValueError(msg)

        if not len(self._columns):
            msg = (
                f"{self._name} 0 feature(s) (shape={_X.shape}) "
                + "while a minimum of 1 is required."
            )
            raise ValueError(msg)

        if isinstance(_y, ndarray) and (any(isinf(_y)) or all(isnan(_y))):
            msg = f"{self._name} inf not allowed in target"
            raise ValueError(msg)

        self.Embedding_ = None  # DataFrame, includes nan
        self.Projection_ = None  # DataFrame Simplex & SMap output
        self.lib_i_ = None  # ndarray library indices
        self.pred_i_ = None  # ndarray prediction indices : nan removed
        self.knn_neighbors_ = None  # ndarray (N_pred, knn) sorted
        self.knn_distances_ = None  # ndarray (N_pred, knn) sorted

        self._pred_i_all = None  # ndarray prediction indices : nan included
        self._predList = []  # list of disjoint pred_i_all
        self._validLib = []  # list of valid lib indices
        self._disjointLib = False  # True if disjoint library
        self._libOverlap = False  # True if lib & pred overlap
        self._ignoreNan = True  # Remove nan from embedding
        self._noTime = False  # First column expected to be time index, set True if not
        self._xRadKnnFactor = 5  # exclusionRadius knn factor
        self._kdTree = None  # SciPy KDTree (k-dimensional tree)
        self._projection = None  # ndarray Simplex & SMap output
        self._variance = None  # ndarray Simplex & SMap output
        self._targetVec = None  # ndarray entire record
        self._targetVecNan = False  # True if targetVec has nan : SMap only
        self._time = None  # ndarray entire record numerically operable

        if y is None:  # In application y=None, in check_estimator() skip Validate()
            self.Validate()  # EDM method: set knn, E, Data(Frame) if needed
        self.EDM_params()  # Set default EDM params, add time if noTime=True
        self.CreateIndices()  # Generate lib_i & pred_i, validLib : EDM Method
        self.EmbedData()
        self.RemoveNan()
        self.FindNeighbors()

        self.is_fitted_ = True
        return self

    # -------------------------------------------------------------------
    def predict(self, X):
        """Simplex projection of target variable from embedding library

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Not used but accepted for sklearn convention.

        Returns
        -------
        y : ndarray, shape (n_samples,)
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Set reset=False to not overwrite `n_features_in_` and
        # `feature_names_in_` but check the shape is consistent.
        X = validate_data(
            self, X=X, y=None, accept_sparse=False, skip_check_array=True, reset=False
        )

        # target parameter might have changed for new prediction
        __target = self.get_params()["target"]
        if __target is None:
            __target = "y"  # Terrible. How do we fix this?
        self._targetVec = self._Data[__target]

        # If targetVec has nan, set flag for SMap internals
        if self._name == "SMap":
            if any(isnan(self._targetVec)):
                self._targetVecNan = True

        self.Project()
        self.FormatProjection()

        return self._projection

    # -------------------------------------------------------------------
    def Project(self):
        """Simplex Projection
        Sugihara & May (1990) doi.org/10.1038/344734a0"""
        # First column of knn_distances is minimum distance of all N pred rows
        minDistances = self.knn_distances_[:, 0]
        # In case there is 0 in minDistances: minWeight = 1E-6
        minDistances = fmax(minDistances, 1e-6)

        # Divide each column of the N x k knn_distances matrix by N row
        # column vector minDistances
        scaledDistances = divide(self.knn_distances_, minDistances[:, None])

        weights = exp(-scaledDistances)  # N x k
        weightRowSum = sum(weights, axis=1)  # N x 1

        # Matrix of knn_neighbors + Tp defines library target values
        knn_neighbors_Tp = self.knn_neighbors_ + self.Tp  # N x k
        libTargetValues = zeros(knn_neighbors_Tp.shape)  # N x k

        # for j in range( knn_neighbors_Tp.shape[1] ) : # for each column j of k
        #    libTargetValues[ :, j ][ :, None ] = \
        #        self._targetVec[ knn_neighbors_Tp[ :, j ] ]

        for j in range(knn_neighbors_Tp.shape[1]):  # for each column j of k
            libTargetValues[:, j] = self._targetVec[knn_neighbors_Tp[:, j]]

        # Projection is average of weighted knn library target values
        self._projection = sum(weights * libTargetValues, axis=1) / weightRowSum

        # "Variance" estimate assuming weights are probabilities
        libTargetPredDiff = subtract(libTargetValues, self._projection[:, None])
        deltaSqr = power(libTargetPredDiff, 2)
        self._variance = sum(weights * deltaSqr, axis=1) / weightRowSum

    # -------------------------------------------------------------------
    def score(self, X, y, sample_weight=None):
        """Return : coefficient of determination <r2_score> on test data.

        Override RegessorMixin score() method since predict(X) does not
        actually depend on X (the data passed to Simplex) but on the
        embedding created in fit(). Once the embedding is created it is
        used to project a target y. If a new model/embedding is desired
        based on X, a new fit() is required to build the embedding. 

        Both the embedding and the target projection are dictated by the
        lib and pred parameters defining the time series observation rows
        of the library and prediction set. By default lib = pred = [1,N]
        (all data) but in general lib and pred are disjoint (out-of-sample).

        The embedding is also governed by E & tau, and the prediction by Tp. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for estimate of y based on embedding of X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Not used.

        Returns
        -------
        score : float
            :math:R^2 of self.fit(X).predict(X) w.r.t. y.

        Notes
        -----
        """

        from sklearn.metrics import r2_score

        # Set reset=False to not overwrite `n_features_in_` and
        # `feature_names_in_` but check the shape is consistent.
        Z = validate_data(
            self, X=X, y=y, accept_sparse=False, skip_check_array=True, reset=False
        )
        if isinstance(Z, tuple):
            _X, _y = X, y
        else:
            _X, _y = X, None

        _ = self.fit(_X,_y) # Depends on columns, lib, pred, E, Tp, tau
        y_pred = self.predict(_X) # Depends on target

        return r2_score(
            y[self.pred_i_], # Limit to pred
            y_pred,
            sample_weight=sample_weight
        )
