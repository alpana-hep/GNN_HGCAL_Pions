import uproot
from numba import jit
import numpy as np
import awkward as ak
from time import time
import pickle
import ROOT
path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/skimmed_ntuple_data_chi2method01_0300.root"
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
trimAhcal_chi2Reco = tree['trimAhcal_chi2Reco'].array()

trueBeamEnergy = beamEnergy
pred_v2 ="./tb_data_upscaled/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])
predic=preds_ratio#[0:836658]
#print(len(predic))test_0to5M_fix_raw_ahcalTrim_up
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_tb_fix_raw_ahcalTrim_upscaled/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
#rawE= rawE[0:836658]
Pred_ = rawE * predic

trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/beamEn.pickle"

trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
# print(trueEn_pkl[0:836658])
# trueEn_pkl=trueEn_pkl[0:836658]

#print(numpy.array_equal(trueEn_pkl,trueBeamEnergy))
print(Pred_)
fout= ROOT.TFile("hist_predicVsEnrgyinAbsorber_trimAhcal_86epochs_8endata_upscaled.root", 'RECREATE')
fout.cd()
h2d_trueEnvstrueEn =[]
h2d_chi2vs_model =[]
h2d_chi2vs_true = []
h2d_modelvstrue=[]
h2d_chi2vs_absorb=[]
h2d_modelvs_absorb=[]
h2d_ratioModel_vsAbsorb=[]
h2d_ratiochi2_vsAbsorb=[]
h2d_normModel_vsAbsorb=[]
h2d_normchi2_vsAbsorb=[]
h1d_normchi2=[]
h1d_normModel=[]
h1d_predchi2=[]
h1d_predModel=[]

Energy=[20,50,80,100,120,200,250,300]

for i_hist in range(8):
    xhigh_true= 2.0*Energy[i_hist]
    xhigh_pred = 3.0*Energy[i_hist]

    if(i_hist==0):
        xhigh_pred =4.0*Energy[i_hist]
        #xhigh_pred = 3.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(Energy[i_hist])
    h2d_trueEnvstrueEn.append(ROOT.TH2F("h2d_trueEnvstrueEn_%s"%name1,"trueEn vs trueEn picklin vs root", 500, 0, xhigh_pred, 500, 0, xhigh_pred))
    h2d_chi2vs_model.append(ROOT.TH2F("h2d_chi2vs_model_%s"%name1,"chi2 vs GNN",500, 0, xhigh_pred, 500, 0, xhigh_pred))
    h2d_chi2vs_true.append(ROOT.TH2F("h2d_chi2vs_true_%s"%name1,"chi2 vs true",500, 0, xhigh_pred, 500, 0, xhigh_pred))
    h2d_modelvstrue.append(ROOT.TH2F("h2d_modelvstrue_%s"%name1,"model pred vs true",500, 0, xhigh_pred, 500, 0, xhigh_pred))
    h2d_chi2vs_absorb.append(ROOT.TH2F("h2d_chi2vs_absorb_%s"%name1,"chi2 vs absorber" ,500, 0, xhigh_pred, 500, 0, xhigh_pred))
    h2d_modelvs_absorb.append(ROOT.TH2F("h2d_modelvs_absorb_%s" %name1,"GNN vs absorber" ,500, 0, xhigh_pred, 500, 0, xhigh_pred))
    h2d_ratioModel_vsAbsorb.append(ROOT.TH2F("h2d_ratioModel_vsAbsorb_%s"%name1,"h2d_ratioModel_vsAbsorb",500,0,10,500,0,1))
    h2d_ratiochi2_vsAbsorb.append(ROOT.TH2F("h2d_ratiochi2_vsAbsorb_%s"%name1,"h2d_ratiochi2_vsAbsorb",500,0,10,500,0,1))
    h2d_normModel_vsAbsorb.append(ROOT.TH2F("h2d_normModel_vsAbsorb_%s"%name1,"h2d_normModel_vsAbsorb",500,-5,5,500,0,1))
    h2d_normchi2_vsAbsorb.append(ROOT.TH2F("h2d_normchi2_vsAbsorb_%s"%name1,"h2d_normchi2_vsAbsorb",500,-5,5,500,0,1))
    h1d_normchi2.append(ROOT.TH1F("h1d_normchi2_%s"%name1,"h1d_normchi2",500,-5,5))
    h1d_normModel.append(ROOT.TH1F("h1d_normModel_%s"%name1,"h1d_normModel2",500,-5,5))
    h1d_predchi2.append(ROOT.TH1F("h1d_predchi2_%s"%name1,"h1d_predchi2",500,0,xhigh_pred))
    h1d_predModel.append(ROOT.TH1F("h1d_predModel_%s"%name1,"h1d_predModel2",500,0,xhigh_pred))



for i_en in range(8):
    for i in range(len(trueEn_pkl)):
        if(trueEn_pkl[i]>=Energy[i_en]-2 and trueEn_pkl[i]<=Energy[i_en]+2 ):
            h2d_trueEnvstrueEn[i_en].Fill(trueEn_pkl[i],trueBeamEnergy[i])
            h2d_chi2vs_model[i_en].Fill(Pred_[i],trimAhcal_chi2Reco[i])
            h2d_chi2vs_true[i_en].Fill(trueBeamEnergy[i],trimAhcal_chi2Reco[i])
            h2d_modelvstrue[i_en].Fill(trueBeamEnergy[i],Pred_[i])
            total_lost = energyLostEE_[i]+energyLostFH_[i]+energyLostBH_[i]
            h2d_chi2vs_absorb[i_en].Fill(trimAhcal_chi2Reco[i],total_lost)
            h2d_modelvs_absorb[i_en].Fill(Pred_[i],total_lost)
            ratio_gnn= Pred_[i]/trueEn_pkl[i]
            ratio_chi2 = trimAhcal_chi2Reco[i]/trueEn_pkl[i]
            ratio_abs = total_lost/trueEn_pkl[i]
            h2d_ratioModel_vsAbsorb[i_en].Fill(ratio_gnn,ratio_abs)
            h2d_ratiochi2_vsAbsorb[i_en].Fill(ratio_chi2,ratio_abs)
            norm_gnn = (Pred_[i] - trimAhcal_chi2Reco[i])/trueEn_pkl[i]            
            h2d_normchi2_vsAbsorb[i_en].Fill(norm_gnn,ratio_abs);
            norm_gnn = (Pred_[i] -trueEn_pkl[i])/trueEn_pkl[i]
            norm_chi2 = (trimAhcal_chi2Reco[i]-trueEn_pkl[i])/trueEn_pkl[i]
            h1d_normchi2[i_en].Fill(norm_chi2)            
            h1d_normModel[i_en].Fill(norm_gnn)
            h1d_predchi2[i_en].Fill(trimAhcal_chi2Reco[i])
            h1d_predModel[i_en].Fill(Pred_[i])

            
fout.cd()
for i_en in range(8):
    h2d_trueEnvstrueEn[i_en].Write()
    h2d_chi2vs_model[i_en].Write()
    h2d_chi2vs_true[i_en].Write()
    h2d_modelvstrue[i_en].Write()
    h2d_chi2vs_absorb[i_en].Write()
    h2d_modelvs_absorb[i_en].Write()
    h2d_ratioModel_vsAbsorb[i_en].Write()
    h2d_ratiochi2_vsAbsorb[i_en].Write()
    h2d_normchi2_vsAbsorb[i_en].Write()
    h1d_normchi2[i_en].Write()
    h1d_normModel[i_en].Write()
    h1d_predchi2[i_en].Write()
    h1d_predModel[i_en].Write()
fout.Close()


