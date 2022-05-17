import uproot
from numba import jit
import numpy as np
import awkward as ak
from time import time
import pickle
import ROOT
path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/skimmed_ntuple_data_chi2method01_0000.root"
treeName ="pion_variables_v1"
outfolder ="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/ratio_updated/correct_inputs/epoch58_onwards/DownScale_Ahcal"
tree = uproot.open("%s:%s"%(path, treeName))
energyLostEE_ = tree['energyLostEE'].array()
print(energyLostEE_)
energyLostFH_ = tree['energyLostFH'].array()
print(energyLostFH_)
energyLostBH_ = tree['energyLostBH'].array()
print(energyLostBH_)
beamEnergy = tree['beamEnergy'].array()
trueBeamEnergy = tree['trueBeamEnergy'].array()
rechit_shower_start_layer=tree['rechit_shower_start_layer'].array()
trimAhcal_chi2Reco = np.asarray(pickle.load(open("./aggre2_maxlr6e04_75ep/constlr_1feat/tb_data/pred_tb.pickle","rb"))) #tree['trimAhcal_chi2Reco'].array()
trimAhcal_chi2Reco[trimAhcal_chi2Reco>3]=3
trueBeamEnergy = beamEnergy
pred_v2 ="./aggre2_maxlr6e04_75ep/constlr_4feat/tb_data/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])
predic=preds_ratio#[0:836658]
print(len(predic))
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_updat_relwt_ahcaltrim_TB/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
rawE= rawE#[0:836658]
Pred_ = rawE * predic
trimAhcal_chi2Reco_v1 = rawE*trimAhcal_chi2Reco
trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/beamEn.pickle"

trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
#print(trueEn_pkl[0:836658])
trueEn_pkl=trueEn_pkl#[0:836658]


fout= ROOT.TFile("hist_predicVsEnrgyinAbsorber_TB_1vs4input_2aggre_constlr_trimAhcal.root", 'RECREATE')
fout.cd()

h2d_trueEnvstrueEn = ROOT.TH2F("h2d_trueEnvstrueEn","trueEn vs trueEn picklin vs root",500,0,500,500,0,500)
h2d_chi2vs_model = ROOT.TH2F("h2d_chi2vs_model","chi2 vs GNN",600,0,600,600,0,600)
h2d_chi2vs_true = ROOT.TH2F("h2d_chi2vs_true","chi2 vs true",600,0,600,600,0,600)
h2d_modelvstrue=ROOT.TH2F("h2d_modelvstrue","model pred vs true",600,0,600,600,0,600)
h2d_chi2vs_absorb = ROOT.TH2F("h2d_chi2vs_absorb","chi2 vs absorber",600,0,600,600,0,600)
h2d_modelvs_absorb = ROOT.TH2F("h2d_modelvs_absorb","GNN vs absorber",600,0,600,600,0,600)
for i in range(len(trueEn_pkl)):
    h2d_trueEnvstrueEn.Fill(trueEn_pkl[i],trueBeamEnergy[i])
    h2d_chi2vs_model.Fill(Pred_[i],trimAhcal_chi2Reco_v1[i])
    h2d_chi2vs_true.Fill(trueBeamEnergy[i],trimAhcal_chi2Reco[i])
    h2d_modelvstrue.Fill(trueBeamEnergy[i],Pred_[i])
    total_lost = energyLostEE_[i]+energyLostFH_[i]+energyLostBH_[i]
    h2d_chi2vs_absorb.Fill(trimAhcal_chi2Reco[i],total_lost)
    h2d_modelvs_absorb.Fill(Pred_[i],total_lost)


fout.cd()
h2d_trueEnvstrueEn.Write()
h2d_chi2vs_model.Write()
h2d_chi2vs_true.Write()
h2d_modelvstrue.Write()
h2d_chi2vs_absorb.Write()
h2d_modelvs_absorb.Write()
fout.Close()


