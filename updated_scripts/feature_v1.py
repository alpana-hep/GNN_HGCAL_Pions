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
pred_v2 ="./aggre2_maxlr6e04_75ep/constlr_4feat/ep135/valid_flat/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
#print(predPickle[)
preds_trueEn = np.asarray(pickle.load(predPickle))
print(preds_trueEn[preds_trueEn>3])
preds_trueEn[preds_trueEn>3] = 3
print(preds_trueEn[preds_trueEn>3])
#sys.exit()
print(preds_trueEn)
print('pred')
trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_update_relwt_ahcaltrim_fixwt/ratio_target.pickle"

trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
print(trueEn_pkl)
#sys.exit()
print('true')
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_update_relwt_ahcaltrim_fixwt/recHitEn.pickle"


RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)
# ratio ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_update_relwt_ahcaltrim_fixwt/ratio_target.pickle"
# ratioPickle = open(ratio,"rb")
# ratio_pkl =pickle.load(ratioPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
#print(rawE[0]*trueEn_pkl[0],'input')
print(rawE)
Pred = rawE * preds_trueEn
print('final')
print(Pred)
SSLocation_file_sim = "/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_0to5M/SsLocation.pickle"
SSLocation_sim =  open(SSLocation_file_sim,"rb")
SSLocation_arr_sim = np.asarray(pickle.load(SSLocation_sim))

fout= ROOT.TFile("./aggre2_maxlr6e04_75ep/constlr_4feat/ep135/hist_1d_ratio_constlr_maxlr1e04_5M_updatedrelWt_2aggre_NSM_4infeat_135ep.root", 'RECREATE')
fout.cd()
hist_trueEN_wScaMC = ROOT.TH1F('trueEn_nosemi_wScaMC' ,'true beam energy in GeV', 500, 0, 500)
hist_trueEN_ScaMC = ROOT.TH1F('trueEn_nosemi_ScaMC' ,'true beam energy in GeV', 500, 0, 500)
hist_pred1d_wScaMC_25=ROOT.TH1F('Predi_nosemi_withoutScaling_25' ,'Predicted energy in GeV', 500, 0, 500)
hist_pred2d_wScaMC_25=ROOT.TH2F('PrediVsTrue_nosemi_withoutScaling_25' ,"Predicted energy in GeV", 500, 0, 500,500, 0, 500)
hist_pred1d_wScaMC_52=ROOT.TH1F('Predi_nosemi_withoutScaling_52', "Predicted energy in GeV", 500, 0, 500)
hist_pred2d_wScaMC_52=ROOT.TH2F('PrediVsTrue_nosemi_withoutScaling_52', "Predicted energy in GeV", 500, 0, 500,500, 0, 500)
hist_pred1d_wScaMC_75=ROOT.TH1F('true_ratio', "Predicted energy in GeV",1000, 0, 10)
hist_pred1d_wScaMC_75_v1=ROOT.TH1F('Predi_ratio', "Predicted energy in GeV",1000, 0, 10)

hist_pred2d_wScaMC_75=ROOT.TH2F('PrediVsTrue_nosemi_withoutScaling_75', "Predicted energy in GeV", 500, 0, 500,500, 0, 500)


hist_pred1d_ScaMC_25=ROOT.TH1F('Predi_nosemi_withScaling_25', "Predicted energy in GeV", 500, 0, 500)
hist_pred2d_ScaMC_25=ROOT.TH2F('PrediVsTrue_nosemi_withScaling_25', "Predicted energy in GeV", 500, 0, 500,500, 0, 500)
hist_pred1d_ScaMC_52=ROOT.TH1F('Predi_nosemi_withScaling_52' ,"Predicted energy in GeV", 500, 0, 500)
hist_pred2d_ScaMC_52=ROOT.TH2F('PrediVsTrue_nosemi_withScaling_52', "Predicted energy in GeV", 500, 0, 500,500, 0, 500)
hist_pred1d_ScaMC_75=ROOT.TH1F('Predi_nosemi_withScaling_75' ,"Predicted energy in GeV", 500, 0, 500)
hist_pred2d_ScaMC_75=ROOT.TH2F('PrediVsTrue_nosemi_withScaling_75', "Predicted energy in GeV", 1000, 0, 10,1000, 0, 10)
hist_predvsShowStart = ROOT.TH2F('Predi_vs_SS' ,"Shower Start location (x) vs predicted energy (y)",50,0,50,500, 0, 500)
#hist_ratio= ROOT.TH1F(
for i in range(len(trueEn_pkl)):
    hist_pred1d_wScaMC_25.Fill(Pred[i])
    hist_pred2d_wScaMC_25.Fill(trueEn_pkl[i]*rawE[i],Pred[i])
    hist_trueEN_wScaMC.Fill(trueEn_pkl[i]*rawE[i])
    hist_pred2d_ScaMC_75.Fill(trueEn_pkl[i],preds_trueEn[i])
    hist_pred1d_wScaMC_75.Fill(trueEn_pkl[i])
    hist_pred1d_wScaMC_75_v1.Fill(preds_trueEn[i])
    hist_predvsShowStart.Fill(SSLocation_arr_sim[i],Pred[i])
fout.cd()
hist_predvsShowStart.Write()
hist_pred1d_ScaMC_75.Write()
hist_pred1d_ScaMC_52.Write()
hist_pred1d_ScaMC_25.Write()

hist_pred1d_wScaMC_75.Write()
hist_pred1d_wScaMC_52.Write()
hist_pred1d_wScaMC_25.Write()

hist_pred2d_ScaMC_75.Write()
hist_pred2d_ScaMC_52.Write()
hist_pred2d_ScaMC_25.Write()

hist_pred2d_wScaMC_75.Write()
hist_pred2d_wScaMC_52.Write()
hist_pred2d_wScaMC_25.Write()
hist_trueEN_ScaMC.Write()
hist_trueEN_wScaMC.Write()
hist_pred1d_wScaMC_75_v1.Write()
fout.Close()
