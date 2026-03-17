"""scikit-learn common tests for skedm"""

# Authors: Joseph Park
# License: BSD 3 clause

import pytest

from sklearn.utils.estimator_checks import parametrize_with_checks
from skedm.utils.discovery import all_estimators

expected_failed = {
    "check_complex_data":"Complex data not supported in KDTree",
    "check_regressor_data_not_an_array":"skedm requires DataFrame or ndarray",
    "check_supervised_y_2d":"skedm is not supervised, target is 1d",
    "check_dtype_object":"Unknown label type raw object dtype not allowed",
    "check_methods_sample_order_invariance":"skedm is dynamical",
    "check_methods_subset_invariance":"skedm is state dependent",
    "check_fit2d_1sample":"skedm is dynamical, multiple observations required",
    "check_dict_unchanged":"skedm requires numeric data",
    "check_fit2d_predict1d":"skedm correctly creates DataFrame for this case",
    "check_fit2d_1feature":"skedm requires at least 10 observations, 1 feature is fine",
    "check_regressors_no_decision_function":"skedm requires at least 10 observations",
}

@parametrize_with_checks([est() for _, est in all_estimators()])
def test_estimators(estimator, check, request):
    """Check the compatibility with scikit-learn API"""
    
    # Extract the name of the check function
    # Sometimes 'check' is a functools.partial, so we get __name__ carefully
    check_name = check.func.__name__ if hasattr(check, 'func') else check.__name__

    if check_name in expected_failed:
        pytest.xfail(reason=expected_failed[check_name])

    check(estimator)
