
import os
from os import PathLike
import numpy as np
import pandas as pd
from pathlib import Path
from string import Template
from typing import Tuple, Dict
import matplotlib.pyplot as plt


DEFAULT_FWHM = 20
DEFAULT_THRESHOLD_UNCORRECTED_P_VALUE = 0
DEFAULT_THRESHOLD_CORRECTED_P_VALUE = 0.05
DEFAULT_CLUSTER_THRESHOLD = 0.001
TSV_FIRST_COLUMN = "participant_id"
TSV_SECOND_COLUMN = "session_id"


def _extract_parameters(parameters: Dict) -> Tuple[float, float, float, float]:
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
    return Template(
        str(base_dir) +
        "/${subject}/${session}/t1/freesurfer_cross_sectional/${subject}_${session}/surf/${hemi}.thickness.fwhm${fwhm}.fsaverage.mgh"
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
    input_dir: PathLike,
    output_dir: PathLike,
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
    from nilearn.surface import Mesh, load_surf_mesh
    fwhm, threshold_uncorrected_pvalue, threshold_corrected_pvalue, cluster_threshold = _extract_parameters(parameters)
    fsaverage_path = (freesurfer_home / Path("subjects/fsaverage/surf"))
    print(f"fsaverage path : {fsaverage_path}")
    df_subjects = _read_and_check_tsv_file(tsv_file, strformat)
    n_subjects = len(df_subjects)
    absolute_contrast, with_interaction = _check_contrast(contrast, df_subjects)
    thickness = _build_thickness_array(input_dir, surface_file, df_subjects, fwhm)
    mask = thickness[0, :] > 0
    meshes = [
        load_surf_mesh(str(fsaverage_path / Path(f"{hemi}.pial")))
        for hemi in ['lh', 'rh']
    ]
    coordinates = np.vstack([mesh.coordinates for mesh in meshes])
    faces = np.vstack([mesh.faces for mesh in meshes])
    ##################
    ## UGLY HACK !!! Need investigation
    ##################
    faces += 1
    #################
    average_surface = {
        "coord": coordinates,
        "tri": faces,
    }
    print(f"Absolute contrast = {absolute_contrast}")
    print(f"Mask shape = {mask.shape}")
    group_values = np.unique(df_subjects[absolute_contrast])
    contrast_g = (
        (df_subjects[absolute_contrast] == group_values[0]).astype(int) -
        (df_subjects[absolute_contrast] == group_values[1]).astype(int)
    )
    if glm_type == "group_comparison":
        print(f"The GLM linear model is: {design_matrix}")
        if not with_interaction:
            from brainstat.stats.terms import FixedEffect
            from brainstat.stats.SLM import SLM
            model = FixedEffect(df_subjects[absolute_contrast])
            slm_model = SLM(
                model,
                contrast=contrast_g,
                surf=average_surface,
                mask=mask,
                correction=["fdr", "rft"],
                cluster_threshold=cluster_threshold,
            )
            slm_model.fit(thickness)
            print("T-values :")
            print(slm_model.t)
            from nilearn.plotting import plot_surf_stat_map
            output_filename = f"group-{group_label}_{group_values[0]}-lt-{group_values[1]}_measure-{feature_label}_fwhm-{fwhm}_TStatistics"
            tstat_plot_filename = output_filename + ".png"
            print(f"Saving plot of T-map to {tstat_plot_filename}")
            plot_surf_stat_map(
                Mesh(coordinates=coordinates, faces=faces),
                slm_model.t,
                output_file=tstat_plot_filename,
            )
            from scipy.io import savemat
            t_value_with_mask = slm_model.t * mask  # Note: Is this really necessary??
            tstat_mat_filename = output_filename + ".mat"
            print(f"Saving T-map to {tstat_mat_filename}")
            savemat(
                tstat_mat_filename,
                {"tvaluewithmask": t_value_with_mask},
            )
            output_filename = f"group-{group_label}_{group_values[0]}-lt-{group_values[1]}_measure-{feature_label}_fwhm-{fwhm}_correctedPValue"
            pval_plot_filename = output_filename + ".png"
            plot_surf_stat_map(
                Mesh(coordinates=coordinates, faces=faces),
                slm_model.P["pval"]["C"],
                output_file=pval_plot_filename,
            )
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
        #"/network/lustre/iss02/aramis/project/clinica/data_ci/StatisticsSurface"
        "/Users/nicolas.gensollen/GitRepos/clinica_data_ci/data_ci/StatisticsSurface"
    )
    input_dir = caps_dir / Path("in/caps/subjects")
    output_dir = Path("./out")
    tsv_file = caps_dir / Path("in/subjects.tsv")
    design_matrix = "1+group"
    contrast = "group"
    strformat = "\t"
    glm_type = "group_comparison"
    group_label = "UnitTest"
    freesurfer_home = Path("/Applications/freesurfer/7.2.0/")
    print(f"FreeSurfer home : {freesurfer_home}")
    surface_file = _get_t1_freesurfer_custom_file_template(input_dir)
    print(f"Surface file : {surface_file}")
    feature_label = "cortical-thickness"
    parameters = dict()
    clinica_surfstat(
        input_dir,
        output_dir,
        tsv_file,
        design_matrix,
        contrast,
        strformat,
        glm_type,
        group_label,
        freesurfer_home,
        surface_file,
        feature_label,
        parameters,
    )

