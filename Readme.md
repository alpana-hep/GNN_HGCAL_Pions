# Instructions on how to use different scripts
## folder "notebooks"
"Loss.ipynb" : In this we get the 8 discrete energy points and corresponding predicted energy distributions for simulation (validation & training), TB-data, QGSP dataset, FTFP dataset. The output root file will contain all the 1-d histograms.

"ModelPerformance_based_onSS.ipynb" : in this we divide the events in two different categories "SSin EE" "MipsInEE" and corresponding histograms.

"Equal_binned.ipynb" : The flat energy simualtion data used to train the model is binned into equal bins of 4 GeV each and we get the corresponding histograms and output root file.


## plotting scripts and getting resolution/repsonse
"pionResolution.C" : command to run , root -b 'pionResolution.C(root_filename)'
get the one d distributions and resolution/response and output root file contains

"DataOverlay.C" : command to run ,   root -b 'DataOverlay.C(rootfile)'

Get the overlay resolution and response plots

"overlayPlots.C": command to run root -b 'overlayPlots.C(root file)'
get the overlay one-d distributions




