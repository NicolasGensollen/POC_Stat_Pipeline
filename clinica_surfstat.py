
import os
from time import time
import warnings
from os import PathLike
import numpy as np
import pandas as pd
from pathlib import Path
from string import Template
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from nilearn.surface import Mesh, load_surf_mesh
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM
from scipy.stats import t
from functools import reduce


DEFAULT_FWHM = 20
DEFAULT_THRESHOLD_UNCORRECTED_P_VALUE = 0.001
DEFAULT_THRESHOLD_CORRECTED_P_VALUE = 0.05
DEFAULT_CLUSTER_THRESHOLD = 0.001
TSV_FIRST_COLUMN = "participant_id"
TSV_SECOND_COLUMN = "session_id"


def _extract_parameters(parameters: Dict) -> Tuple[float, float, float, float]:
    fwhm = DEFAULT_FWHM
    if "sizeoffwhm" in parameters:
        fwhm = parameters["sizeoffwhm"]
    threshold_uncorrected_pvalue = DEFAULT_THRESHOLD_UNCORRECTED_P_VALUE
    if "thresholduncorrectedpvalue" in parameters:
        threshold_uncorrected_pvalue = parameters["thresholduncorrectedpvalue"]
    threshold_corrected_pvalue = DEFAULT_THRESHOLD_CORRECTED_P_VALUE
    if "thresholdcorrectedpvalue" in parameters:
        threshold_corrected_pvalue = parameters["thresholdcorrectedpvalue"]
    cluster_threshold = DEFAULT_CLUSTER_THRESHOLD
    if "clusterthreshold" in parameters:
        cluster_threshold = parameters["clusterthreshold"]
    return fwhm, threshold_uncorrected_pvalue, threshold_corrected_pvalue, cluster_threshold


def _read_and_check_tsv_file(tsv_file: PathLike) -> pd.DataFrame:
    if not tsv_file.exists():
        raise FileNotFoundError(f"File {tsv_file} does not exist.")
    tsv_data = pd.read_csv(tsv_file, sep="\t")
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


def _build_thickness_array(
        input_dir: PathLike,
        surface_file: Template,
        df_subjects: pd.DataFrame,
        fwhm: float,
) -> np.ndarray:
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
        combined = np.vstack(parts)
        thickness.append(combined.flatten())
    thickness = np.vstack(thickness)
    if thickness.shape[0] != len(df_subjects):
        raise ValueError(
            f"Unexpected shape for thickness array : {thickness.shape}. "
            f"Expected {len(df_subjects)} rows."
        )
    return thickness


def _check_contrast(
        contrast: str,
        df_subjects: pd.DataFrame,
        glm_type: str,
) -> Tuple[str, str, bool]:
    absolute_contrast = contrast
    with_interaction = False
    contrast_sign = "positive"
    if contrast.startswith("-"):
        absolute_contrast = contrast[1:].lstrip()
        contrast_sign = "negative"
    if "*" in contrast:
        with_interaction = True
        warnings.warn(
            "You included interaction as covariate in your model, "
            "please carefully check the format of your tsv files."
        )
    else:
        if absolute_contrast not in df_subjects.columns:
            raise ValueError(
                f"Column {absolute_contrast} does not exist in provided TSV file."
            )
        if glm_type == "group_comparison":
            unique_labels = np.unique(df_subjects[absolute_contrast])
            if len(unique_labels) != 2:
                raise ValueError(
                    "For group comparison, there should be just 2 different groups!"
                )
    return absolute_contrast, contrast_sign, with_interaction


def _print_clusters(slm_model, threshold):
    """Print results related to total number of clusters
    and significative clusters.
    """
    print("#" * 40)
    print("After correction (Clusterwise Correction for Multiple Comparisons): ")
    df = slm_model.P['clus'][1]
    print(df)
    print(f"Clusters found: {len(df)}")
    print(f"Significative clusters (after correction): {len(df[df['P'] <= threshold])}")


def _plot_stat_map(mesh, texture, filename, threshold=None, verbose=True):
    from nilearn.plotting import plot_surf_stat_map
    plot_filename = filename + ".png"
    if verbose:
        print(f"--> Saving plot to {plot_filename}")
    plot_surf_stat_map(
        mesh, texture, threshold=threshold, output_file=plot_filename
    )


def _save_to_mat(struct, filename, key, verbose=True):
    from scipy.io import savemat
    #masked_texture = texture
    #mask_ = np.ones_like(texture)
    #if mask is not None:
    #    mask_ = mask
    #if threshold is None:
    #    threshold = 0.0
    #masked_texture = texture * mask_
    mat_filename = filename + ".mat"
    if verbose:
        print(f"--> Saving matrix to {mat_filename}")
    savemat(
        mat_filename,
        {key: struct},
    )


def _build_model(design_matrix: str, df: pd.DataFrame):
    """Build a brainstat model from the design matrix in
    string format.
    This function assumes that the design matrix is formatted
    in the following way:

        1 + factor_1 + factor_2 + ...

    Or:

        factor_1 + factor_2 + ... (in this case the intercept will
        be added automatically).
    """
    if len(design_matrix) == 0:
        raise ValueError("Design matrix cannot be empty.")
    if "+" in design_matrix:
        terms = [_.strip() for _ in design_matrix.split("+")]
    else:
        terms = [design_matrix.strip()]
    model = []
    for term in terms:
        # Intercept is automatically included in brainstat
        if term == "1":
            continue
        # Handles the interaction effects
        if "*" in term:
            sub_terms = [_.strip() for _ in term.split("*")]
            model_term = reduce(
                lambda x, y: x * y,
                [_build_model_term(_, df) for _ in sub_terms]
            )
        else:
            model_term = _build_model_term(term, df)
        model.append(model_term)
    if len(model) == 1:
        return model[0]
    return reduce(lambda x, y: x + y, model)


def _build_model_term(term: str, df: pd.DataFrame) -> FixedEffect:
    if term not in df.columns:
        raise ValueError(
            f"Term {term} from the design matrix is not "
            "in the columns of the provided TSV file. "
            "Please make sure that there is no typo."
        )
    return FixedEffect(df[term])


def clinica_surfstat(
    input_dir: PathLike,
    output_dir: PathLike,
    tsv_file: PathLike,
    design_matrix: str,
    contrast: str,
    glm_type: str,
    group_label: str,
    freesurfer_home: PathLike,
    surface_file: PathLike,
    feature_label: str,
    parameters: dict,
    verbose=True,
):
    """This function mimics the previous function `clinica_surfstat`
    written in MATLAB and relying on the MATLAB package SurfStat.
    This implementation is written in pure Python and rely on the
    package brainstat for GLM modeling.

    Parameters
    ----------
    input_dir : PathLike
        Input folder.

    output_dir : PathLike
        Output folder for storing results.

    tsv_file : PathLike
        Path to the TSV file `subjects.tsv` which contains the
        necessary metadata to run the statistical analysis.

        .. warning::
            The column names need to be accurate because they
            are used to defined contrast and model terms.
            Please double check for typos.

    design_matrix : str
        Design matrix in string format. For example "1+Label"

    contrast : str
        The contrast to be used in the GLM.

        .. warning::
            The contrast needs to be in the design matrix.

    glm_type : {"group_comparison", "correlation"}
        Type of GLM to run:
            - "group_comparison": Performs group comparison.
              For example "AD - ND".
            - "correlation": Performs correlation analysis.

    group_label : str

    freesurfer_home : PathLike
        Path to the home folder of Freesurfer.
        This is required to get the fsaverage templates.

    surface_file : PathLike
    """
    (
        fwhm, threshold_uncorrected_pvalue,
        threshold_corrected_pvalue, cluster_threshold,
    ) = _extract_parameters(parameters)
    fsaverage_path = (freesurfer_home / Path("subjects/fsaverage/surf"))
    if verbose:
        print(f"--> fsaverage path : {fsaverage_path}")
    df_subjects = _read_and_check_tsv_file(tsv_file)
    n_subjects = len(df_subjects)
    (
        absolute_contrast,
        contrast_sign,
        with_interaction
    ) = _check_contrast(
        contrast, df_subjects, glm_type
    )
    start = time()
    thickness = _build_thickness_array(
        input_dir, surface_file, df_subjects, fwhm
    )
    print(f"--> Building thickness array took {time() - start} seconds.")
    mask = thickness[0, :] > 0
    meshes = [
        load_surf_mesh(str(fsaverage_path / Path(f"{hemi}.pial")))
        for hemi in ['lh', 'rh']
    ]
    coordinates = np.vstack([mesh.coordinates for mesh in meshes])
    faces = np.vstack([
        meshes[0].faces,
        meshes[1].faces + meshes[0].coordinates.shape[0]
    ])
    average_mesh = Mesh(
        coordinates=coordinates,
        faces=faces,
    )
    ##################
    ## UGLY HACK !!! Need investigation
    ##################
    # Uncomment the following line if getting an error
    # with negative values in bincount in Brainstat.
    # Not sure, but might be a bug in BrainStat...
    #
    #faces += 1
    #################
    average_surface = {
        "coord": coordinates,
        "tri": faces,
    }
    if glm_type == "group_comparison":
        if verbose:
            print(f"--> The GLM linear model is: {design_matrix}")
        group_values = np.unique(df_subjects[absolute_contrast])
        contrast_g = (
            (df_subjects[absolute_contrast] == group_values[0]).astype(int) -
            (df_subjects[absolute_contrast] == group_values[1]).astype(int)
        )
        if not with_interaction:
            start = time()
            model = _build_model(design_matrix, df_subjects)
            slm_model = SLM(
                model,
                contrast=contrast_g,
                surf=average_surface,
                mask=mask,
                correction=["fdr", "rft"],
                cluster_threshold=cluster_threshold,
            )
            if verbose:
                print("--> Fitting the SLM model...")
            slm_model.fit(thickness)
            if verbose:
                print(f"--> Building and fitting the SLM model took {time() - start} seconds.")
            t_values = np.nan_to_num(slm_model.t)
            uncorrected_pvalues = 1 - t.cdf(t_values, slm_model.df)

            # The ugly code after this point is to enforce backward compatibility
            # with outputs from the previous MATLAB code...
            #
            filename_root = (
                f"group-{group_label}_{group_values[0]}-lt-{group_values[1]}_"
                f"measure-{feature_label}_fwhm-{fwhm}"
            )
            structs = [
                t_values,
                {
                    'P': uncorrected_pvalues,
                    'mask': mask,
                    'thresh': threshold_uncorrected_pvalue,
                },
                {
                    'P': slm_model.P['pval']['P'],
                    'C': slm_model.P['pval']['C'],
                    'mask': mask,
                    'thresh': threshold_corrected_pvalue,
                }
            ]
            for name, key, struct in zip(
                    ["_TStatistics", "_uncorrectedPValue", "_correctedPValue"],
                    ["tvaluewithmask", "uncorrectedpvaluesstruct", "correctedpvaluesstruct"],
                    structs,
            ):
                if isinstance(struct, dict):
                    texture = struct['P']
                else:
                    texture = struct
                _plot_stat_map(
                    average_mesh,
                    texture,
                    filename_root + name,
                    threshold=None,
                    verbose=verbose,
                )
                _save_to_mat(
                    struct,
                    filename_root + name,
                    key,
                    verbose=verbose,
                )
            _print_clusters(slm_model, threshold_corrected_pvalue)
        ################
        # The rest of the function is only copy-pasted code for now.
        # I think it can be factorized a lot once we have meaningful results.
        ################
        else:
            if verbose:
                print(
                    "--> The contrast here is the interaction between one "
                    "continuous variable and one categorical "
                    f"variable: {contrast}"
                )
            model = _build_model(design_matrix, df_subjects)
            contrast_interaction = contrast_g # TODO: CHANGE THIS
            slm_model = SLM(
                model,
                contrast=contrast_interaction,
                surf=average_surface,
                mask=mask,
                correction=["fdr", "rft"],
                cluster_threshold=cluster_threshold,
            )
            if verbose:
                print("--> Fitting the SLM model...")
            slm_model.fit(thickness)
            filename_root = (
                f"interaction-{contrast}_measure-{feature_label}_fwhm-{fwhm}"
            )
            for name, key, texture, _mask, threshold in zip(
                    ["_TStatistics", "_correctedPValue"],
                    ["tvaluewithmask", "correctedpvaluesstruct"],
                    [slm_model.t, slm_model.P["pval"]['C']],
                    [mask, None],
                    [None, cluster_threshold],
            ):
                _plot_stat_map(
                    average_mesh,
                    texture,
                    filename_root + name,
                    threshold=threshold,
                    verbose=verbose,
                )
                _save_to_mat(
                    texture,
                    _mask,
                    threshold,
                    filename_root + name,
                    key,
                    verbose=verbose,
                )
            _print_clusters(slm_model, threshold_corrected_pvalue)

    elif glm_type == "correlation":
        model = _build_model(design_matrix, df_subjects)
        contrast_ = df_subjects[absolute_contrast]
        if contrast_sign == "negative":
            contrast_ *= -1
        slm_model = SLM(
            model,
            contrast_,
            surf=average_surface,
            mask=mask,
            correction=["fdr", "rft"],
            cluster_threshold=cluster_threshold,
        )
        slm_model.fit(thickness)
        uncorrected_pvalues = 1 - t.cdf(slm_model.t, slm_model.df)
        filename_root = (
            f"group-{group_label}_correlation-{contrast}-{contrastsign}_"
            f"measure-{feature_label}_fwhm-{fwhm}"
        )
        for name, key, texture, _mask, threshold in zip(
                ["_TStatistics", "_uncorrectedPValue", "_correctedPValue"],
                ["tvaluewithmask", "uncorrectedpvaluesstruct", "correctedpvaluesstruct"],
                [slm_model.t, uncorrected_pvalues, slm_model.P["pval"]['C']],
                [mask, mask, None],
                [None, threshold_uncorrected_pvalue, cluster_threshold],
        ):
            _plot_stat_map(
                average_mesh,
                texture,
                filename_root + name,
                threshold=threshold,
                verbose=verbose,
            )
            _save_to_mat(
                texture,
                _mask,
                threshold,
                filename_root + name,
                key,
                verbose=verbose,
            )
        _print_clusters(slm_model, threshold_corrected_pvalue)
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
    design_matrix = "1 + age + sex + group"
    contrast = "group"
    glm_type = "group_comparison"
    group_label = "UnitTest"
    freesurfer_home = Path("/Applications/freesurfer/7.2.0/")
    print(f"FreeSurfer home : {freesurfer_home}")
    surface_file = _get_t1_freesurfer_custom_file_template(input_dir)
    print(f"Surface file : {surface_file}")
    feature_label = "ct"
    parameters = dict()
    clinica_surfstat(
        input_dir,
        output_dir,
        tsv_file,
        design_matrix,
        contrast,
        glm_type,
        group_label,
        freesurfer_home,
        surface_file,
        feature_label,
        parameters,
    )

