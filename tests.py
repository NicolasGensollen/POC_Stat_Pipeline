import os
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def test_extract_parameters():
    from clinica_surfstat import _extract_parameters
    input_params = {
        "sizeoffwhm": 30,
        "thresholduncorrectedpvalue": 0.2,
        "thresholdcorrectedpvalue": 0.01,
        "clusterthreshold": 0.033,
    }
    (
        fwhm, threshold_uncorrected_pvalue,
        threshold_corrected_pvalue, cluster_threshold,
    ) = _extract_parameters(input_params)
    assert fwhm == 30
    assert threshold_uncorrected_pvalue == 0.2
    assert threshold_corrected_pvalue == 0.01
    assert cluster_threshold == 0.033
    (
        fwhm, threshold_uncorrected_pvalue,
        threshold_corrected_pvalue, cluster_threshold,
    ) = _extract_parameters({})
    assert fwhm == 20
    assert threshold_uncorrected_pvalue == 0.001
    assert threshold_corrected_pvalue == 0.05
    assert cluster_threshold == 0.001


def test_read_and_check_tsv_file(tmpdir):
    from clinica_surfstat import _read_and_check_tsv_file
    with pytest.raises(
        FileNotFoundError,
        match="File foo.tsv does not exist"
    ):
        _read_and_check_tsv_file(Path("foo.tsv"))
    df = pd.DataFrame(columns=["foo"])
    df.to_csv(tmpdir / "foo.tsv", sep="\t", index=False)
    with pytest.raises(
        ValueError,
        match=r"The TSV data in .*foo.tsv should have at least 2 columns."
    ):
        _read_and_check_tsv_file(tmpdir / "foo.tsv")
    df = pd.DataFrame(columns=["foo", "bar"])
    df.to_csv(tmpdir / "foo.tsv", sep="\t", index=False)
    with pytest.raises(
        ValueError,
        match=r"The first column in .*foo.tsv should always be participant_id."
    ):
        _read_and_check_tsv_file(tmpdir / "foo.tsv")
    df = pd.DataFrame(columns=["participant_id", "bar"])
    df.to_csv(tmpdir / "foo.tsv", sep="\t", index=False)
    with pytest.raises(
        ValueError,
        match=r"The second column in .*foo.tsv should always be session_id."
    ):
        _read_and_check_tsv_file(tmpdir / "foo.tsv")
    df = _read_and_check_tsv_file(Path(CURRENT_DIR) / "data/subjects.tsv")
    assert len(df) == 7
    assert set(df.columns) == set(['participant_id', 'session_id', 'group', 'age', 'sex'])


def test_get_t1_freesurfer_custom_file_template():
    from clinica_surfstat import _get_t1_freesurfer_custom_file_template
    templ = _get_t1_freesurfer_custom_file_template(".")
    path = templ.safe_substitute(
        subject="sub-01",
        session="ses-M00",
        hemi="lh",
        fwhm=20,
    )
    assert path == "./sub-01/ses-M00/t1/freesurfer_cross_sectional/sub-01_ses-M00/surf/lh.thickness.fwhm20.fsaverage.mgh"


def test_check_contrast():
    from clinica_surfstat import _check_contrast, _read_and_check_tsv_file
    df_subjects = _read_and_check_tsv_file(Path(CURRENT_DIR) / "data/subjects.tsv")
    with pytest.raises(
        ValueError,
        match="Column foo does not exist in provided TSV file."
    ):
        _check_contrast("foo", df_subjects, "group_comparison")
    with pytest.warns(
        UserWarning,
        match="You included interaction as covariate in your model"
    ):
        absolute_contrast, contrast_sign, with_interaction = _check_contrast(
            "age*sex", df_subjects, "group_comparison"
        )
    assert with_interaction
    assert absolute_contrast == "age*sex"
    assert contrast_sign == "positive"
    bad_df = pd.DataFrame({
        "participant_id": [f"sub-0{i}" for i in range(1, 4)],
        "session_id": ["ses-M00"] * 3,
        "group": ["AD1", "AD2", "AD3"],  # 3 labels for group
    })
    with pytest.raises(
        ValueError,
        match="For group comparison, there should be just 2 different groups!"
    ):
        _check_contrast("group", bad_df, "group_comparison")
    absolute_contrast, contrast_sign, with_interaction = _check_contrast(
        "group", df_subjects, "group_comparison"
    )
    assert absolute_contrast == "group"
    assert contrast_sign == "positive"
    assert not with_interaction
    absolute_contrast, contrast_sign, with_interaction = _check_contrast(
        "-age", df_subjects, "correlation"
    )
    assert absolute_contrast == "age"
    assert contrast_sign == "negative"
    assert not with_interaction


def test_build_model():
    from brainstat.stats.terms import FixedEffect
    from clinica_surfstat import _build_model, _read_and_check_tsv_file
    df_subjects = _read_and_check_tsv_file(Path(CURRENT_DIR) / "data/subjects.tsv")
    for design in ["1 + age", "1+age", "age +1", "age"]:
        model = _build_model(design, df_subjects)
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
    model = _build_model("1 + age + sex", df_subjects)
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

    model = _build_model("1 + age + sex + age * sex", df_subjects)
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


@pytest.mark.parametrize("contrast_sign", ["positive", "negative"])
def test_get_correlation_contrast(contrast_sign):
    from clinica_surfstat import _get_correlation_contrast
    df = pd.read_csv("./data/subjects.tsv", sep="\t")
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


def test_get_group_contrast_without_interaction():
    from clinica_surfstat import _get_group_contrast_without_interaction
    df = pd.read_csv("./data/subjects.tsv", sep="\t")
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
def test_get_group_contrast_with_interaction_error(contrast):
    from clinica_surfstat import _get_group_contrast_with_interaction
    df = pd.read_csv("./data/subjects.tsv", sep="\t")
    with pytest.raises(
        ValueError,
        match="The contrast must be an interaction between one continuous"
    ):
        _get_group_contrast_with_interaction(contrast, df)


@pytest.mark.parametrize(
        "contrast_name",
        ["age*sex", "sex*age"])
def test_get_group_contrast_with_interaction(contrast_name):
    from clinica_surfstat import _get_group_contrast_with_interaction
    df = pd.read_csv("./data/subjects.tsv", sep="\t")
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


def test_get_contrasts_and_filenames_error():
    from clinica_surfstat import _get_contrasts_and_filenames
    df = pd.read_csv("./data/subjects.tsv", sep="\t")
    with pytest.raises(
        ValueError,
        match="Check out if you define the glmtype flag correctly"
    ):
        _get_contrasts_and_filenames("foo", "age", df)
