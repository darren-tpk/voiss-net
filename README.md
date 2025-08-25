VOISS-Net
============

The **VOlcano Infrasound & Seismic Spectrogram Network (VOISS-Net)** is a pair of Convolutional Neural Networks (one for seismic, one for acoustic) that can detect volcanic tremor and other relevant signals in near real-time and classify them according to their spectral signature. The models and applications are described in: 

*Tan, D., Fee, D., Witsil, A., Girona, T., Haney, M., Wech, A., Waythomas, C., & Lopez, T. (2024). Detection and Characterization of Seismic and Acoustic Signals at Pavlof Volcano, Alaska, Using Deep Learning. Journal of Geophysical Research: Solid Earth, 129(6), e2024JB029194. https://doi.org/10.1029/2024JB029194.*

and

*Fee, D., Tan, D., Lyons, J., Sciotto, M., Cannata, A., Hotovec-Ellis, A., Girona, T., Wech, A., Roman, D., Haney, M. and De Angelis, S. (2025) “A generalized deep learning model to detect and classify volcano seismicity”, Volcanica, 8(1), pp. 305–323. https://doi.org/10.30909/vol/rjss1878.*

and we ask that you cite those manuscripts when using the tool.

We train the models using labeled seismic and infrasonic data from various volcanoes. Although we demonstrate its applicability to the Pavlof seismoacoustic network across different Pavlof eruptions within Tan et al. (2024), we have generalized the seismic model to other volcanoes and eruptions in the Fee et al. (2025) manuscript and model. We envision most users will want to use the generalized model (voissnet_seismic_generalized_model.keras) for detecting and characterizing volcano seismicity.

The codes included within the repository can also be used to re-create the labeled spectrogram datasets for both models. Using the labeled spectrogram dataset, users are able to split and augment the dataset according to their preferences, and re-train separate iterations of the VOISS-Net models for each data type. Once users have selected a model, they can explore model implementations in both short and long timescales using the different functions detailed within the repository. 


Quickstart
----------

1. Obtain

```
git clone --depth 1 https://github.com/darren-tpk/voiss-net.git
cd voiss-net
```

2. Create environment and activate

```
conda env create
conda activate voiss-net
```

3. Run example to check 3 hour seismic timeline for Semisopochnoi Volcano (Fig. 6 in Fee et al. (2025))

```
# Pull data from Earthscope
python timeline_checker.py

# Local miniseed directory example (pulls data from Earthscope and saves it)
python timeline_checker_local.py
```

(OPTIONAL) Train alternate models (skip to step 3 if using labels by Tan et al. (2024))
----------

1. Plot stacked spectrograms in bulk.

```
python plot_spectrograms_bulk.py
```

2. Open the Label Studio (Tkachenko et al., 2020) web browser interface, upload stacked spectrograms, and label spectrograms using the "Bounding Box" option.

```
conda activate voiss-net
label-studio
```

3. Use python to build labeled spectrograms (saved as .npy files), augment the labeled dataset, and train the CNNs. 

```
# Architecture and hyperparameters from Tan et al. (2024)
python train_voiss_net_seismic.py
python train_voiss_net_infrasound.py

# Architecture and hyperparameters from Fee et al. (2025)
python train_voiss_net_seismic_generalized.py
```

Note that the spectrogram labels for the generalized model are included as a text file: models/labels_generalized.txt. Retraining the generalized model requires creating the npy files from the times and stations in the labels_generalized.txt file.

Dependencies
------------

Other repositories:
* [waveform_collection](https://github.com/uafgeotools/waveform_collection)
* [label-studio](https://github.com/HumanSignal/label-studio)
* [PhysicsML-Combine Features](https://github.com/qingkaikong/PhysicsML-CombineFeatures/) (Home repository for GradCAM reference code, Kong et al. (2022))
