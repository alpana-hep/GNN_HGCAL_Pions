## Get one-d distributions & calculate energy response & resolution:
# "plotResolution.C" : command to run , root -b 'pionResolution.C("root_histogram_filename for beamEn")'
get the one d distributions and resolution/response for individual datasets ("valid, train, data, QGSP, FTFP") and output root file contains the Tgraph for resolution & response for each datasets

** plot_reoslution.C_full_binned : this script is to be run over root file which has 85 histogram for training & validation (4 GeV bins = 85 energy bins [10:350]
** plot_resolution.C_beamEn: this script is to be run over root file which has only 8 histograms for eahc of datasets, 

Note: for simplicity I save these two file separately and copy it to my plotResolution.C depending upon the root file I am dealing with.

## To get the overlay of response & resolution
# "DataOverlay.C" : command to run ,   root -b 'DataOverlay.C(out_rootfile from plotResolution.C)'
get the overlay plots for response & resolution
** DataOverlay.C_full_binned : this script is to be run over root file which has 85 histogram for training & validation (4 GeV bins = 85 energy bins [10:350]
** DataOverlay.C_beamEn: this script is to be run over root file which has only 8 histograms for eahc of datasets, 

## Get the overlay energy ditributions
# "overlayPlots.C": command to run root -b 'overlayPlots.C(hist root file)'
get the overlay one-d distributions
** overlayPlots.C_full_binned : this script is to be run over root file which has 85 histogram for training & validation (4 GeV bins = 85 energy bins [10:350]
** overlayPlots.C_beamEn: this script is to be run over root file which has only 8 histograms for eahc of datasets, 

You can ignore other scripts for now
