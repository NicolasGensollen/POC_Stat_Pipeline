"""Script to test the implementation of clinica_surfstat."""

import os
from pathlib import Path
from clinica_surfstat import clinica_surfstat
from _inputs import _get_t1_freesurfer_custom_file_template

current_dir = Path(
    os.path.dirname(os.path.realpath(__file__))
)
caps_dir = Path(
    #"/network/lustre/iss02/aramis/project/clinica/data_ci/StatisticsSurface"
    "/Users/nicolas.gensollen/GitRepos/clinica_data_ci/data_ci/StatisticsSurface"
)
freesurfer_home = Path("/Applications/freesurfer/7.2.0/")
input_dir = caps_dir / Path("in/caps/subjects")
output_dir = Path("./out")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tsv_file = caps_dir / Path("in/subjects.tsv")
design_matrix = "1 + group + age + sex"
contrast = "group"
glm_type = "group_comparison"
group_label = "UnitTest"
surface_file = _get_t1_freesurfer_custom_file_template(input_dir)
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
    verbose=True,
)
