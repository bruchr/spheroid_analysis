# Tool for spheroid data evaluation
**Roman Bruch, Mario Vitacolonna, Rüdiger Rudolf and Markus Reischl**

## Installation

Clone this repository using [`git`](https://git-scm.com/downloads) or download it as `.zip`file and extract it.

Install a conda distribution like [Anaconda](https://www.anaconda.com/products/individual).

Create the environment with conda:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate spheroid_analysis
```
Once the environment is activated, the pipeline can be run as described below.


### Data structure
The GUI supports two data structures. The first `Cell_ACDC` is based on the folder structure and file names of [Cell-ACDC](https://github.com/SchmollerLab/Cell_ACDC). The second `Raw_Data` expects the following folder structure:
```
├── Dataset
│   ├── Raw_Data
│   │   ├── ConditionA
│   │   │   ...
│   │   ├── ConditionB
│   │   │   ...
│   ├── Segmentation
│   │   ├── ConditionA
│   │   │   ...
│   │   ├── ConditionB
│   │   │   ...
│   ...
...
```
The folder structure inside `Raw_Data` and `Segmentation` can be chosen arbitrarily, e.g., multiple subfolders, as long as they are consistent between the `Raw_Data` and `Segmentation` folder.

### Using the GUI
Start the GUI with:
```
python gui.py
```

In the GUI, the user is required to first select the data structure and the input folder. The input folder is the main folder of the data structure.
Afterwards, the software reads the first image of the dataset to extract information about the image properties and automatically sets the default parameters in the GUI.

**Note**: All parameter names feature tooltips which are shown by hovering over the text.


### Using the python script
The pipeline can also be started using the `analysis.py` file in the code folder:
```
python analysis.py path/to/settings/file.json
```
The script requires a JSON configuration file. If none is specified, the configuration file in the code folder is used, if available.
A template configuration file is provided: `settings_template.json`.