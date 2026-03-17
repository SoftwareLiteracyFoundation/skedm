"""Simplex tests against pyEDM."""
import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal

import skedm
import pyEDM


# Authors: Joseph Park
# License: BSD 3 clause


@pytest.fixture
def data():
    return pyEDM.sampleData['Lorenz5D']


def test_Simplex_1(data):
    """Simplex time delay embedding cross mapping: compare to pyEDM"""
    est = skedm.Simplex(columns='V1', target='V2', E=5)
    est.fit(data)
    assert hasattr(est, "is_fitted_")

    y_pred = est.predict(data)

    pyEDM_S = pyEDM.Simplex(data, columns='V1', target='V2', E=5,
                            lib=[1,1000], pred=[1,1000])

    skedm_pred = est.Projection_['Predictions']
    pyEDM_pred = pyEDM_S['Predictions']

    assert_array_equal(skedm_pred, pyEDM_pred)


def test_Simplex_2(data):
    """Simplex multivariate embedding cross mapping: compare to pyEDM"""
    est = skedm.Simplex(columns=['V1','V3','V4','V5'], target='V2', embedded=True)
    est.fit(data)
    assert hasattr(est, "is_fitted_")

    y_pred = est.predict(data)

    pyEDM_S = pyEDM.Simplex(data, columns=['V1','V3','V4','V5'], target='V2',
                            lib=[1,1000],pred=[1,1000], embedded=True)

    skedm_pred = est.Projection_['Predictions']
    pyEDM_pred = pyEDM_S['Predictions']

    assert_array_equal(skedm_pred, pyEDM_pred)
