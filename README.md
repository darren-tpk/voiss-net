VOISS-Net
============

This repository stores the codes and datasets related to **VOlcano Infrasound & Seismic Spectrogram Network (VOISS-Net)**, as described in Tan et al. (2024): 

*Tan, D., Fee, D., Girona, T., Haney, M. M., Witsil, A., Wech, A., Waythomas, C. & Lopez, T. (2024). Detection and characterization of seismic and acoustic signals at Pavlof Volcano, Alaska using deep learning. (submitted for review in the Journal of Geophysical Research: Solid Earth)*

VOISS-Net is a pair of Convolutional Neural Networks (one for seismic, one for acoustic) that can detect tremor in near real-time and classify them according to their spectral signature. We train the models using labeled seismic and infrasonic data from the 2021-2022 eruption of Pavlof Volcano, Alaska. Although we demonstrate its applicability to the Pavlof seismoacoustic network across different Pavlof eruptions within the manuscript, we have applied the models successfully to other volcanoes and encourage others to as well.

The codes (and labels) included within the repository can be used to re-create the labeled spectrogram dataset from the 2021-2022 eruption of Pavlof Volcano. Using the labeled spectrogram dataset, users are able to split and augment the dataset according to their preferences, and re-train separate iterations of the VOISS-Net models for each data type. The models used in the paper are also included in the ``models" subdirectory for reproducibility. Once users have selected model, they can explore model implementations in both short and long timescales using the different functions detailed within the repository. 

Documentation for this package can be found **here**. (Link to be updated)

Quickstart
----------

1. Obtain

```
git clone https://github.com/darren-tpk/VOISS_Net.git
cd VOISS_Net
```

2. Create environment and activate

```
conda env create
conda activate voiss_net
```

3. Run example to check 3 hour seismic and infrasound timeline (Fig. 6 & 7 in Tan et al. (2024))

```
python check_timeline.py
```

Dependencies
------------

Other repositories:
* [label-studio](https://github.com/HumanSignal/label-studio)
* [waveform_collection](https://github.com/uafgeotools/waveform_collection)
