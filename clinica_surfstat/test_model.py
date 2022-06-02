
import os
import pytest
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal
from brainstat.stats.terms import FixedEffect
from _model import _build_model
from _inputs import _read_and_check_tsv_file


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def df():
    return _read_and_check_tsv_file(
        Path(CURRENT_DIR).parent / "data/subjects.tsv"
    )


@pytest.mark.parametrize(
        "design",
        ["1 + age", "1+age", "age +1", "age"])
def test_build_model_intercept(design, df):
    """Test that we get the same results with equivalent
    designs. Especially, the fact that adding explicitely
    the intercept doesn't change the results.
    Test also that spaces in the design expression have
    no effect.
    """
    model = _build_model(design, df)
    assert isinstance(model, FixedEffect)
    assert len(model.m.columns) == 2
    assert_array_equal(
        model.intercept,
        np.array([1, 1, 1, 1, 1, 1, 1])
    )
    assert_array_equal(
        model.age,
        np.array([78. , 73.4, 70.8, 82.3, 60.6, 72.1, 74.2])
    )


def test_build_model(df):
    model = _build_model("1 + age + sex", df)
    assert isinstance(model, FixedEffect)
    assert len(model.m.columns) == 4
    assert_array_equal(
        model.intercept,
        np.array([1, 1, 1, 1, 1, 1, 1])
    )
    assert_array_equal(
        model.age,
        np.array([78. , 73.4, 70.8, 82.3, 60.6, 72.1, 74.2])
    )
    assert_array_equal(
        model.sex_Female,
        np.array([1, 0, 1, 1, 1, 0, 1])
    )
    assert_array_equal(
        model.sex_Male,
        np.array([0, 1, 0, 0, 0, 1, 0])
    )
    model = _build_model("1 + age + sex + age * sex", df)
    assert isinstance(model, FixedEffect)
    assert len(model.m.columns) == 6
    assert_array_equal(
        model.intercept,
        np.array([1, 1, 1, 1, 1, 1, 1])
    )
    assert_array_equal(
        model.age,
        np.array([78. , 73.4, 70.8, 82.3, 60.6, 72.1, 74.2])
    )
    assert_array_equal(
        model.sex_Female,
        np.array([1, 0, 1, 1, 1, 0, 1])
    )
    assert_array_equal(
        model.sex_Male,
        np.array([0, 1, 0, 0, 0, 1, 0])
    )
    assert_array_equal(
        getattr(model, "age*sex_Female"),
        np.array([78. ,  0. , 70.8, 82.3, 60.6,  0. , 74.2])
    )
    assert_array_equal(
        getattr(model, "age*sex_Male"),
        np.array([ 0. , 73.4,  0. ,  0. ,  0. , 72.1,  0. ])
    )
