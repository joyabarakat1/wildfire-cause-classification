Can Location and Season Help Identify Wildfire Causes?

This project investigates whether the location and discovery calendar of a recorded California wildfire can help distinguish its reported cause as Human or Natural. It tests whether adding latitude and longitude improves classification compared with using discovery month and weekday alone, and whether that improvement holds in later years.
The project studies causes of recorded fires. It does not predict whether a new wildfire will occur.

Dataset

The project uses the sixth edition of the US Forest Service’s Fire Program Analysis Fire-Occurrence Database (FPA FOD), compiled by Karen C. Short. It contains over 2.3 million US wildfire records from 1992 to 2020.

Original dataset DOI: https://doi.org/10.2737/RDS-2013-0009.6

Download mirror: https://www.datalumos.org/datalumos/project/249023/version/V2/view

Required file: FPA_FOD_20221014.sqlite, extracted from the SQLite-format archive. Download the raw database separately; it is not included in the repository.

The California subset contains 251,881 records. Removing 38,418 records with missing, unspecified, or undetermined broad cause leaves 213,463 records: 183,173 Human and 30,290 Natural. Explicit Human labels are retained even when the specific activity is unspecified. Natural does not mean every record is confirmed lightning.
The source permits use without additional permission or fees and requests citation.

Features and preprocessing
The models use discovery month, weekday, latitude, and longitude. Calendar features are one-hot encoded for logistic regression, and coordinates are standardized using training statistics. Discovery year defines the split rather than serving as an input. Names, identifiers, cause descriptions, final fire size, and containment information are excluded from predictors. Loading includes checks for valid dates, missing values, duplicate fire identifiers, and global coordinate ranges.

Models

Completed models include a majority-class baseline and logistic regression implemented with NumPy, comparing calendar-only features with calendar plus location. A random forest is planned.

Training uses 1992–2014, validation uses 2015–2017, and testing is reserved for 2018–2020. Adding location increased validation macro-F1 from 0.453 to 0.672. The test set has not yet been scored.

Repository contents

code/: data-loading, figure-generation, baseline scripts, and dependencies.

results/: baseline metrics, optimization histories, and data checks.

figures/: data-exploration plots and supporting summaries.

Keep the scripts together: the figure and NumPy training scripts reuse load_data from baseline.py. The separate scikit-learn entry point in baseline.py is not the source of the reported NumPy results.

Setup and running

Use Python 3.11 or newer. From the repository root folder, install the dependencies:

python -m pip install -r code/requirements.txt

Download and extract the dataset. Replace the example database path below with its actual location:

python code/make_figures.py --database "C:/path/to/FPA_FOD_20221014.sqlite" --output figures

python code/train_numpy_baselines.py --database "C:/path/to/FPA_FOD_20221014.sqlite" --output results

These commands regenerate the exploration figures and numerical baselines. Use different output folder names to preserve supplied results. No GPU is required.

Limitations

Reported causes may contain errors, coordinates may be approximate, and recording practices differ across agencies, locations, and years. Excluding unknown causes can introduce selection bias. Results apply to retained recorded fires and should not be used to assign responsibility for an individual wildfire.

Dataset reference

Short, K. C. (2022). Spatial wildfire occurrence data for the United States, 1992–2020 [FPA_FOD_20221014] (6th edition). Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.6
