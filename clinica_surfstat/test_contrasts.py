
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def df():
    return pd.read_csv(
        Path(CURRENT_DIR).parent / "data/subjects.tsv", sep="\t"
    )


@pytest.mark.parametrize("contrast_sign", ["positive", "negative"])
def test_get_correlation_contrast(contrast_sign, df):
    from _contrasts import _get_correlation_contrast
    contrast, filename = _get_correlation_contrast("age", df, contrast_sign)
    assert isinstance(contrast, dict)
    assert isinstance(filename, dict)
    assert list(contrast.keys()) == ["age"]
    assert list(filename.keys()) == ["age"]
    assert (filename["age"].safe_substitute(
        group_label="group_label",
        contrast_name="contrast_name",
        feature_label="feature_label",
        fwhm="fwhm"
    ) == (
        "group-group_label_correlation-contrast_name"
        f"-{contrast_sign}_measure-feature_label_fwhm-fwhm"
    ))
    mult = 1 if contrast_sign == "positive" else -1
    assert_array_equal(
        contrast["age"].values,
        mult * np.array([78. , 73.4, 70.8, 82.3, 60.6, 72.1, 74.2])
    )


def test_get_group_contrast_without_interaction(df):
    from _contrasts import _get_group_contrast_without_interaction
    with pytest.raises(
        ValueError,
        match="Contrast should refer to a categorical variable for group comparison."
    ):
        _get_group_contrast_without_interaction("age", df)
    contrast, filename = _get_group_contrast_without_interaction("sex", df)
    assert isinstance(contrast, dict)
    assert isinstance(filename, dict)
    assert set(contrast.keys()) == set(['Female-lt-Male', 'Male-lt-Female'])
    assert set(filename.keys()) == set(['Female-lt-Male', 'Male-lt-Female'])
    for contrast_name in ['Female-lt-Male', 'Male-lt-Female']:
        assert (filename[contrast_name].safe_substitute(
            group_label="group_label",
            contrast_name=contrast_name,
            feature_label="feature_label",
            fwhm="fwhm"
        ) == (
            f"group-group_label_{contrast_name}_measure-feature_label_fwhm-fwhm"
        ))
        mult = 1 if contrast_name == "Female-lt-Male" else -1
        assert_array_equal(
            contrast[contrast_name].values,
            mult * np.array([ 1, -1,  1,  1,  1, -1,  1])
        )


@pytest.mark.parametrize(
        "contrast",
        ["age*sex*group", "group*sex", "group * sex "])
def test_get_group_contrast_with_interaction_error(contrast, df):
    from _contrasts import _get_group_contrast_with_interaction
    with pytest.raises(
        ValueError,
        match="The contrast must be an interaction between one continuous"
    ):
        _get_group_contrast_with_interaction(contrast, df)


@pytest.mark.parametrize(
        "contrast_name",
        ["age*sex", "sex*age"])
def test_get_group_contrast_with_interaction(contrast_name, df):
    from _contrasts import _get_group_contrast_with_interaction
    contrast, filename = _get_group_contrast_with_interaction(contrast_name, df)
    assert isinstance(contrast, dict)
    assert isinstance(filename, dict)
    assert set(contrast.keys()) == set([contrast_name])
    assert set(filename.keys()) == set([contrast_name])
    assert (filename[contrast_name].safe_substitute(
        contrast_name=contrast_name,
        feature_label="feature_label",
        fwhm="fwhm"
    ) == (
        f"interaction-{contrast_name}_measure-feature_label_fwhm-fwhm"
    ))
    assert_array_equal(
        contrast[contrast_name].values,
        np.array([ 78. , -73.4,  70.8,  82.3,  60.6, -72.1,  74.2])
    )


def test_get_contrasts_and_filenames_error(df):
    from _contrasts import _get_contrasts_and_filenames
    with pytest.raises(
        ValueError,
        match="Check out if you define the glmtype flag correctly"
    ):
        _get_contrasts_and_filenames("foo", "age", df)
