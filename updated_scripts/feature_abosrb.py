import pandas as pd #dataframes etc                                                                                                       
import matplotlib.pyplot as plt #plotting                                                                                                 
import pickle
import numpy as np
import os, sys
import seaborn as sns
import pickle
import numpy as np
import ROOT
import awkward as ak
pred_v2 ="./valid_flat/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
#print(predPickle[)
preds_trueEn = np.asarray(pickle.load(predPickle))
print(preds_trueEn[preds_trueEn>3])
preds_trueEn[preds_trueEn>3] = 3
print(preds_trueEn[preds_trueEn>3])
print(len(preds_trueEn))
predic=preds_trueEn[0:836658]
print(len(predic))

# f_energyLostEE= "/home/rusack/shared/pickles/HGCAL_TestBeam/0to1M_Energyabsorbed/energyLostEE.pickle"
# energyLostEE_Pickle = open(f_energyLostEE, "rb")
# energyLostEE_= np.asarray(pickle.load(energyLostEE_Pickle))
# print(len(energyLostEE_))

# f_energyLostFH= "/home/rusack/shared/pickles/HGCAL_TestBeam/0to1M_Energyabsorbed/energyLostFH.pickle"
# energyLostFH_Pickle = open(f_energyLostFH, "rb")
# energyLostFH_= np.asarray(pickle.load(energyLostFH_Pickle))
# print(len(energyLostFH_))


# f_energyLostBH= "/home/rusack/shared/pickles/HGCAL_TestBeam/0to1M_Energyabsorbed/energyLostBH.pickle"
# energyLostBH_Pickle = open(f_energyLostBH, "rb")
# energyLostBH_= np.asarray(pickle.load(energyLostBH_Pickle))
# print(len(energyLostBH_))
path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/ntuple_chi2_reco_5M_firstFlatEn.root"
treeName ="pion_variables_v1"
outfolder ="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/ratio_updated/correct_inputs/epoch58_onwards/DownScale_Ahcal"
tree = uproot.open("%s:%s"%(path, treeName))
energyLostEE_ = tree['energyLostEE'].array()
print(energyLostEE_)
energyLostFH_ = tree['energyLostFH'].array()
print(energyLostFH_)
energyLostBH_ = tree['energyLostBH'].array()
print(energyLostBH_)

#sys.exit()
trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_0to5M_fix_raw_ahcalTrim_up/trueE.pickle"

trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
print(trueEn_pkl[0:836658])
trueEn_pkl=trueEn_pkl[0:836658]
#sys.exit()
ratio_true ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_0to5M_fix_raw_ahcalTrim_up/ratio_target.pickle"

ratio_truePickle = open(ratio_true,"rb")
ratio_true_pkl = np.asarray(pickle.load(ratio_truePickle))
print(ratio_true_pkl[0:836658])
ratio_true_pkl=ratio_true_pkl[0:8336658]
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_0to5M_fix_raw_ahcalTrim_up/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
rawE= rawE[0:836658]

#print(rawE[0]*trueEn_pkl[0],'input')
Pred_ = rawE * predic
#Pred_= Pred[0:836658]
SSLocation_file_sim = "/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M/SsLocation.pickle"
SSLocation_sim =  open(SSLocation_file_sim,"rb")
SSLocation_arr_sim = np.asarray(pickle.load(SSLocation_sim))
SSLocation_arr_sim = SSLocation_arr_sim[0:8336658]
fout= ROOT.TFile("hist_predicVsEnrgyinAbsorber_trimAhcal_86epochs.root", 'RECREATE')
fout.cd()

hist1d_trueEn_ratio= ROOT.TH1F('hist1d_trueEn_ratio','hist1d_trueEn_ratio',1000,0,50);
hist1d_predEn_ratio= ROOT.TH1F('hist1d_predEn_ratio','hist1d_predEn_ratio',1000,0,50);
hist1d_trueEn_= ROOT.TH1F('hist1d_trueEn_','hist1d_trueEn_',600,0,600);
hist1d_predEn_= ROOT.TH1F('hist1d_predEn_','hist1d_predEn_',600,0,600);

hist2d_truEnRatio_EnergyLossEE= ROOT.TH2F("hist2d_truEnRatio_EnergyLossEE","hist2d_truEnRatio_EnergyLossEE",1000,0,50,600,0,600);
hist2d_truEnRatio_EnergyLossFH=ROOT.TH2F("hist2d_truEnRatio_EnergyLossFH","hist2d_truEnRatio_EnergyLossFH",1000,0,50,600,0,600);
hist2d_truEnRatio_EnergyLossBH=ROOT.TH2F("hist2d_truEnRatio_EnergyLossBH","hist2d_truEnRatio_EnergyLossBH",1000,0,50,600,0,600);

hist2d_predEnRatio_EnergyLossEE=ROOT.TH2F("hist2d_predEnRatio_EnergyLossEE","hist2d_predEnRatio_EnergyLossEE",1000,0,50,600,0,600);
hist2d_predEnRatio_EnergyLossFH=ROOT.TH2F("hist2d_predEnRatio_EnergyLossFH","hist2d_predEnRatio_EnergyLossFH",1000,0,50,600,0,600);
hist2d_predEnRatio_EnergyLossBH=ROOT.TH2F("hist2d_predEnRatio_EnergyLossBH","hist2d_predEnRatio_EnergyLossBH",1000,0,50,600,0,600);


hist2d_truEn_EnergyLossEE=ROOT.TH2F("hist2d_truEn_EnergyLossEE","hist2d_truEn_EnergyLossEE",600,0,600,600,0,600);
hist2d_truEn_EnergyLossFH=ROOT.TH2F("hist2d_truEn_EnergyLossFH","hist2d_truEn_EnergyLossFH",600,0,600,600,0,600);
hist2d_truEn_EnergyLossBH=ROOT.TH2F("hist2d_truEn_EnergyLossBH","hist2d_truEn_EnergyLossBH",600,0,600,600,0,600);

hist2d_predEn_EnergyLossEE=ROOT.TH2F("hist2d_predEn_EnergyLossEE","hist2d_predEn_EnergyLossEE",600,0,600,600,0,600);
hist2d_predEn_EnergyLossFH=ROOT.TH2F("hist2d_predEn_EnergyLossFH","hist2d_predEn_EnergyLossFH",600,0,600,600,0,600);
hist2d_predEn_EnergyLossBH=ROOT.TH2F("hist2d_predEn_EnergyLossBH","hist2d_predEn_EnergyLossBH",600,0,600,600,0,600);

hist2d_truEn_EnergyLost =ROOT.TH2F("hist2d_truEn_EnergyLost","hist2d_truEn_EnergyLost",600,0,600,600,0,600);
hist2d_truEn_fracEnergyLost =ROOT.TH2F("hist2d_truEn_fracEnergyLost","hist2d_truEn_fracEnergyLost",600,0,600,600,0,2);

hist2d_predEn_EnergyLost =ROOT.TH2F("hist2d_predEn_EnergyLost","hist2d_predEn_EnergyLost",600,0,600,600,0,600);
hist2d_predEn_fracEnergyLost =ROOT.TH2F("hist2d_predEn_fracEnergyLost","hist2d_predEn_fracEnergyLost",600,0,600,600,0,2);

hist2d_Sslocat_EnergyLostEE=ROOT.TH2F("hist2d_Sslocat_EnergyLostEE","hist2d_Sslocat_EnergyLostEE",41,0,41,600,0,600);
hist2d_Sslocat_EnergyLostFH=ROOT.TH2F("hist2d_Sslocat_EnergyLostFH","hist2d_Sslocat_EnergyLostFH",41,0,41,600,0,600);
hist2d_Sslocat_EnergyLostBH=ROOT.TH2F("hist2d_Sslocat_EnergyLostBH","hist2d_Sslocat_EnergyLostBH",41,0,41,600,0,600);
hist2d_Sslocat_EnergyLost=ROOT.TH2F("hist2d_Sslocat_EnergyLost","hist2d_Sslocat_EnergyLost",41,0,41,600,0,600);
hist2d_Sslocat_fracEnergyLost=ROOT.TH2F("hist2d_Sslocat_fracEnergyLost","hist2d_fracSslocat_EnergyLost",41,0,41,600,0,1);
hist2d_truevspred_ratio= ROOT.TH2F("hist2d_truevspred_ratio","hist2d_truevspred_ratio",1000,0,50,1000,0,50);
hist2d_truevspred= ROOT.TH2F("hist2d_truevspred","hist2d_truevspred",600,0,600,600,0,600);

hist2d_predVstrue_Energlossfrac1 = ROOT.TH2F("hist2d_predVstrue_Energlossfrac1","trueEn vs pred (frac_En in absorber <0.6",600,0,600,600,0,600)
hist2d_predVstrue_Energlossfrac2 = ROOT.TH2F("hist2d_predVstrue_Energlossfra2","trueEn vs pred (frac_En in absorber >0.6 & <0.8",600,0,600,600,0,600)
hist2d_predVstrue_Energlossfrac3 = ROOT.TH2F("hist2d_predVstrue_Energlossfrac3","trueEn vs pred (frac_En in absorber >0.8",600,0,600,600,0,600)


for i in range(len(trueEn_pkl)):
    hist1d_trueEn_ratio.Fill(ratio_true_pkl[i])
    hist1d_predEn_ratio.Fill(predic[i])
    hist1d_trueEn_.Fill(trueEn_pkl[i])
    hist1d_predEn_.Fill(Pred_[i])
    hist2d_truEnRatio_EnergyLossEE.Fill(ratio_true_pkl[i],energyLostEE_[i])
    hist2d_truEnRatio_EnergyLossFH.Fill(ratio_true_pkl[i],energyLostFH_[i])
    hist2d_truEnRatio_EnergyLossBH.Fill(ratio_true_pkl[i],energyLostBH_[i])
    
    hist2d_predEnRatio_EnergyLossEE.Fill(predic[i],energyLostEE_[i])
    hist2d_predEnRatio_EnergyLossFH.Fill(predic[i],energyLostFH_[i])
    hist2d_predEnRatio_EnergyLossBH.Fill(predic[i],energyLostBH_[i])

    hist2d_truEn_EnergyLossEE.Fill(trueEn_pkl[i],energyLostEE_[i])
    hist2d_truEn_EnergyLossFH.Fill(trueEn_pkl[i],energyLostFH_[i])
    hist2d_truEn_EnergyLossBH.Fill(trueEn_pkl[i],energyLostBH_[i])

    hist2d_predEn_EnergyLossEE.Fill(Pred_[i],energyLostEE_[i])
    hist2d_predEn_EnergyLossFH.Fill(Pred_[i],energyLostFH_[i])
    hist2d_predEn_EnergyLossBH.Fill(Pred_[i],energyLostBH_[i])
    total_lost = energyLostEE_[i]+energyLostFH_[i]+energyLostBH_[i]
    ratio = total_lost/trueEn_pkl[i]
    
    hist2d_predEn_EnergyLost.Fill(Pred_[i],total_lost)
    hist2d_predEn_fracEnergyLost.Fill(Pred_[i],ratio)
    hist2d_truEn_EnergyLost.Fill(trueEn_pkl[i],total_lost)
    hist2d_truEn_fracEnergyLost.Fill(trueEn_pkl[i],ratio)
    hist2d_Sslocat_EnergyLostEE.Fill(SSLocation_arr_sim[i],energyLostEE_[i])
    hist2d_Sslocat_EnergyLostFH.Fill(SSLocation_arr_sim[i],energyLostFH_[i])
    hist2d_Sslocat_EnergyLostBH.Fill(SSLocation_arr_sim[i],energyLostBH_[i])
    hist2d_Sslocat_EnergyLost.Fill(SSLocation_arr_sim[i],total_lost)
    hist2d_Sslocat_fracEnergyLost.Fill(SSLocation_arr_sim[i],ratio)
    hist2d_truevspred_ratio.Fill(ratio_true_pkl[i],predic[i])
    hist2d_truevspred.Fill(trueEn_pkl[i],Pred_[i])
    if(ratio<=0.6):
        hist2d_predVstrue_Energlossfrac1.Fill(trueEn_pkl[i],Pred_[i])
    elif(ratio>0.6 and ratio<0.8):
        hist2d_predVstrue_Energlossfrac2.Fill(trueEn_pkl[i],Pred_[i])
    elif(ratio>=0.8):
        hist2d_predVstrue_Energlossfrac3.Fill(trueEn_pkl[i],Pred_[i])
                

fout.cd()
hist2d_Sslocat_fracEnergyLost.Write()
hist2d_Sslocat_EnergyLost.Write()
hist2d_Sslocat_EnergyLostBH.Write()
hist2d_Sslocat_EnergyLostFH.Write()
hist2d_Sslocat_EnergyLostEE.Write()
hist2d_truEn_fracEnergyLost.Write()
hist2d_truEn_EnergyLost.Write()
hist2d_predEn_fracEnergyLost.Write()
hist2d_predEn_EnergyLost.Write()
hist2d_predEn_EnergyLossBH.Write()
hist2d_predEn_EnergyLossFH.Write()
hist2d_predEn_EnergyLossEE.Write()
hist2d_truEn_EnergyLossBH.Write()
hist2d_truEn_EnergyLossFH.Write()
hist2d_truevspred_ratio.Write()
hist2d_truEn_EnergyLossEE.Write()
hist2d_predEnRatio_EnergyLossBH.Write()
hist2d_predEnRatio_EnergyLossFH.Write()
hist2d_predEnRatio_EnergyLossEE.Write()
hist2d_truEnRatio_EnergyLossBH.Write()
hist2d_truEnRatio_EnergyLossFH.Write()
hist2d_truEnRatio_EnergyLossEE.Write()
hist1d_predEn_.Write()
hist1d_trueEn_.Write()
hist1d_predEn_ratio.Write()
hist1d_trueEn_ratio.Write()
hist2d_truevspred.Write()
hist2d_predVstrue_Energlossfrac1.Write()
hist2d_predVstrue_Energlossfrac2.Write()
hist2d_predVstrue_Energlossfrac3.Write()

fout.Close()
