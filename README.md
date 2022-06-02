# POC Statistics Surface Pipeline in BrainStat

## Install

```
$ conda create -n brainstat python=3.9
$ conda activate brainstat
$ pip install brainstat
```

## Usage

This repo only contains a proof of concept for conversion of the `StatisticsSurface` Pipeline of Clinica from MATLAB to pure Python thanks to `BrainStats`. It is not meant to be used as is and therefore, **the code isn't packaged**.

All python files are within the  `clinica_surfstat` folder. You can either open a Jupyter server and run the `POC.ipynb` notebook, or run the script `main.py` after having changed the path to the data and freesurfer home folders.

```
$ cd clinica_surfstat
$ python main.py
```



