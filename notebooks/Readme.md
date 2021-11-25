1. beamEn_modelperfo_tbdata.ipynb : In this I fill histograms for all 8 beam enegry points (20,50,80,100,120,200,250,300) for all available datasets,
 (a): for valid & training datasets  which is from flat energy samples and taking 4GeV bins around these 8 beam points e.g: for 20 GeV, I am considering [18:22] GeV.
 (b): TB data available for all these beam points
 (c): QGSP & FTFP simulation for these discrete energies (optional)
 
 2. modelPerfor_4GeVBins_sim.ipynb: In this 85 histograms are made which corresponds to 85 enegry bins in the range [10:350] GeV and this is only done for flat energy sample (validation & training)
 
 3.ModelPerformance_basedOnSS.ipynb: Model's performance is evaluated in two categories (SSinEE && MIPsInEE) . Histograms are filled for all the 5 datasets (see -1)
 
 4.  onD_hist_features.ipynb: to make one d histograms: predicted beam energy or true beam energy
