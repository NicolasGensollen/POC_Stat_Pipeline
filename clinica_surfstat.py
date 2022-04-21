
import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_FWHM = 20
DEFAULT_THRESHOLD_UNCORRECTED_P_VALUE = 0
DEFAULT_THRESHOLD_CORRECTED_P_VALUE = 0.05
DEFAULT_CLUSTER_THRESHOLD = 0.001
TSV_FIRST_COLUMN = "participant_id"
TSV_SECOND_COLUMN = "session_id"


def _extract_parameters(parameters: dict) -> Tuple[float, float, float, float]:
    fwhm = DEFAULT_FWHM
    if "sizeoffwhm" in parameters:
        fwhm = parameters["sizeoffwhm "]
    threshold_uncorrected_pvalue = DEFAULT_THRESHOLD_UNCORRECTED_P_VALUE
    if "thresholduncorrectedpvalue" in parameters:
        threshold_uncorrected_pvalue = parameters["thresholduncorrectedpvalue"]
    threshold_corrected_pvalue = DEFAULT_THRESHOLD_CORRECTED_P_VALUE
    if "thresholdcorrectedpvalue" in parameters:
        threshold_corrected_pvalue = parameters["thresholdcorrectedpvalue"]
    cluster_threshold = DEFAULT_CLUSTER_THRESHOLD
    if "clusterthreshold" in parameters:
        cluster_threshold = parameters["cluster_threshold"]
    return fwhm, threshold_uncorrected_pvalue, threshold_corrected_pvalue, cluster_threshold


def _read_and_check_tsv_file(tsv_file: PathLike, strformat: str) -> pd.DataFrame:
    tsv_data = pd.read_csv(tsv_file, sep=strformat)
    if len(tsv_data.columns) < 2:
        raise ValueError(
            f"The TSV data in {tsv_file} should have at least 2 columns."
        )
    if tsv_data.columns[0] != TSV_FIRST_COLUMN:
        raise ValueError(
            f"The first column in {tsv_file} should always be {TSV_FIRST_COLUMN}."
        )
    if tsv_data.columns[1] != TSV_SECOND_COLUMN:
        raise ValueError(
            f"The second column in {tsv_file} should always be {TSV_SECOND_COLUMN}."
        )
    return tsv_data


def _get_t1_freesurfer_custom_file_template(base_dir):
    from string import Template
    return Template(
        str(base_dir) +
        "/${subject}/${session}/t1/freesurfer_cross_sectional/${@subject}_${session}/surf/${hemi}.thickness.fwhm${fwhm}.fsaverage.mgh"
    )


def _build_thickness_array(input_dir: PathLike, surface_file, df_subjects: pd.DataFrame, fwhm) -> np.ndarray:
    from nibabel.freesurfer.mghformat import load
    thickness = []
    for idx, row in df_subjects.iterrows():
        subject = row[TSV_FIRST_COLUMN]
        session = row[TSV_SECOND_COLUMN]
        parts = (
            load(
                input_dir / surface_file.safe_substitute(
                    subject=subject, session=session, fwhm=fwhm, hemi=hemi
                )
            ).get_fdata() for hemi in ['lh', 'rh']
        )
        Y = np.vstack(parts)
        thickness.append(Y.flatten())
    thickness = np.vstack(thickness)
    assert thickness.shape[0] == len(df_subjects)
    return thickness


def _check_contrast(contrast: str, df_subjects: pd.DataFrame) -> Tuple[str, bool]:
    absolute_contrast = contrast
    with_interaction = False
    if contrast.startswith("-"):
        absolute_contrast = contrast[1:]
    if "*" in contrast:
        with_interaction = True
        print(
            "You include interaction as covariate in you model, "
            "please carefully check the format of your tsv files."
        )
    else:
        if absolute_contrast not in df_subjects.columns:
            raise ValueError(
                f"Column {absolute_contrast} does not exist in provided TSV file."
            )
        unique_labels = np.unique(df_subjects[absolute_contrast])
        if len(unique_labels) != 2:
            raise ValueError(
                "For group comparison, there should be just 2 different groups!"
            )
    return absolute_contrast, with_interaction


def clinica_surfstat(
    inputdir: PathLike,
    outputdir: PathLike,
    tsv_file: PathLike,
    design_matrix: str,
    contrast: str,
    strformat: str,
    glm_type: str,
    group_label: str,
    freesurfer_home: PathLike,
    surface_file: PathLike,
    feature_label: str,
    parameters: dict,
):
    """TODO

    Parameters
    ----------
    glm_type : {"group_comparison", "correlation"}
        Type of GLM to run.
    """
    fwhm, threshold_uncorrected_pvalue, threshold_corrected_pvalue, cluster_threshold = _extract_parameters(parameters)
    fsaveragepath = freesurferhome / Path("/subjects/fsaverage/surf")
    df_subjects = _read_and_check_tsv_file(tsv_file, strformat)
    n_subjects = len(df_subjects)
    absolute_contrast, with_interaction = _check_contrast(contrast, df_subjects)
    thickness = _build_thickness_array(input_dir, surface_file, df_subjects, fwhm)
    mask = thickness[0, :] > 0
    from nilearn.surface import load_surf_mesh
    meshes = [
        load_surf_mesh(fsaveragepath / Path(f"{hemi}.pial"))
        for hemi in ['lh', 'rh']
    ]
    coordinates = np.vstack([mesh.coordinates for mesh in meshes])
    faces = np.vstack([mesh.faces for mesh in meshes])
    average_surface = {
        "coord": coordinates,
        "tri": faces,
    }
    if glm_type == "group_comparison":
        print(f"The GLM linear model is: {designmatrix}")
        if not with_interaction:
            from brainstat.stats.terms import FixedEffect
            from brainstat.stats.SLM import SLM
            model = FixedEffect(df_subjects[absolute_contrast])
            slm_model = SLM(
                model,
                contrast=df_subjects[absolute_contrast],
                surf=average_surface,
                mask=mask,
                correction=["fdr", "rft"],
                cluster_threshold=cluster_threshold,
            )
            slm_model.fit(thickness)
            print("T-values :")
            print(slm.t)
        else:
            print(
                "The contrast here is the interaction between one "
                "continuous variable and one categorical "
                f"variable: {contrast}"
            )

    elif glm_type == "correlation":
        pass
    else:
        raise ValueError(
            "Check out if you define the glmtype flag correctly, "
            "or define your own general linear model, e,g MGLM."
        )

if __name__ == "__main__":
    current_dir = Path(
        os.path.dirname(os.path.realpath(__file__))
    )
    caps_dir = Path(
        "/network/lustre/iss02/aramis/project/clinica/data_ci/StatisticsSurface"
    )
    inputdir = caps_dir / Path("in/caps/subjects")
    surface_file = _get_t1_freesurfer_custom_file_template(input_dir)
    outputdir = None
    tsv_file = caps_dir / Path("in/subjects.tsv")
    design_matrix = "1+group"
    contrast: "group"
    strformat: "\t"
    glm_type: "group_comparison"
    group_label: "Test"
    freesurfer_home: PathLike,
    surface_file: PathLike,
    feature_label: str,
    parameters: dict,
    clinica_surfstat()
