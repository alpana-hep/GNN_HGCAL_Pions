import pandas as pd #dataframes etc
import matplotlib.pyplot as plt #plotting
import pickle
import awkward as ak
import numpy as np
import os, sys
import seaborn as sns
import pickle
import numpy as np
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

f_energyLostEE= "/home/rusack/shared/pickles/HGCAL_TestBeam/0to1M_Energyabsorbed/energyLostEE.pickle"
energyLostEE_Pickle = open(f_energyLostEE, "rb")
energyLostEE_= np.asarray(pickle.load(energyLostEE_Pickle))
print(len(energyLostEE_))

f_energyLostFH= "/home/rusack/shared/pickles/HGCAL_TestBeam/0to1M_Energyabsorbed/energyLostFH.pickle"
energyLostFH_Pickle = open(f_energyLostFH, "rb")
energyLostFH_= np.asarray(pickle.load(energyLostFH_Pickle))
print(len(energyLostFH_))


f_energyLostBH= "/home/rusack/shared/pickles/HGCAL_TestBeam/0to1M_Energyabsorbed/energyLostBH.pickle"
energyLostBH_Pickle = open(f_energyLostBH, "rb")
energyLostBH_= np.asarray(pickle.load(energyLostBH_Pickle))
print(len(energyLostBH_))
trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/trueE.pickle"

trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
print(trueEn_pkl[0:836658])
trueEn_pkl=trueEn_pkl[0:836658]
#sys.exit()                                                                                                                                             
ratio_true ="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/ratio_target.pickle"

ratio_truePickle = open(ratio_true,"rb")
ratio_true_pkl = np.asarray(pickle.load(ratio_truePickle))
print(ratio_true_pkl[0:836658])
ratio_true_pkl=ratio_true_pkl[0:8336658]
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/recHitEn.pickle"
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

import ROOT
fout= ROOT.TFile("hist_ratio_beamE_5M_DownScaledAHCAL_92ep_updateindx.root", 'RECREATE')
import ROOT
hist_pred_categ1=[]
hist_true_categ1=[]
hist_pred_categ2=[]
hist_true_categ2=[]
hist_predTrue_categ1=[]
hist_norm_predTrue_categ1=[]
hist_predTrue_categ2=[]
hist_norm_predTrue_categ2=[]
hist_pred_categ3=[]
hist_true_categ3=[]
hist_predTrue_categ3=[]
hist_norm_predTrue_categ3=[]

Energy=[20,50,80,100,120,200,250,300]
M=8 # number of histograms
for i_hist in range(M):
    xhigh_pred= 3.0*Energy[i_hist]
    xhigh_true= 2.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(Energy[i_hist])#,u[i_hist],v[i_hist],typee[i_hist])
    hist_pred_categ1.append(ROOT.TH1F('categ1_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_categ1.append(ROOT.TH1F('categ1_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0,xhigh_true ))
    hist_pred_categ2.append(ROOT.TH1F('categ2_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_categ2.append(ROOT.TH1F('categ2_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_categ1.append(ROOT.TH1F('categ1_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_categ1.append(ROOT.TH1F('categ1_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_predTrue_categ2.append(ROOT.TH1F('categ2_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_categ2.append(ROOT.TH1F('categ2_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_pred_categ3.append(ROOT.TH1F('categ3_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
    hist_true_categ3.append(ROOT.TH1F('categ3_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_categ3.append(ROOT.TH1F('categ3_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_categ3.append(ROOT.TH1F('categ3_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))

for i in range(len(trueEn_pkl)):
    total_lost = energyLostEE_[i]+energyLostFH_[i]+energyLostBH_[i]
    ratio = total_lost/trueEn_pkl[i]
    if(ratio<=0.6):
        for ibin in range(8):
            if(trueEn_pkl[i]>=Energy[ibin]-2 and trueEn_pkl[i]<=Energy[ibin]+2 ):
                hist_pred_categ1[ibin].Fill(Pred_[i])
                hist_true_categ1[ibin].Fill(trueEn_pkl[i])
    elif(ratio>0.6 and ratio<=0.8):
        for ibin in range(8):
            if(trueEn_pkl[i]>=Energy[ibin]-2 and trueEn_pkl[i]<=Energy[ibin]+2 ):
                hist_pred_categ2[ibin].Fill(Pred_[i])
                hist_true_categ2[ibin].Fill(trueEn_pkl[i])
    elif(ratio>0.8):
        for ibin in range(8):
            if(trueEn_pkl[i]>=Energy[ibin]-2 and trueEn_pkl[i]<=Energy[ibin]+2 ):
                hist_pred_categ3[ibin].Fill(Pred_[i])
                hist_true_categ3[ibin].Fill(trueEn_pkl[i])

fout.cd()
for i in range(8):
    hist_pred_categ1[i].Write()
    hist_true_categ1[i].Write()
    hist_pred_categ2[i].Write()
    hist_true_categ2[i].Write()
    hist_pred_categ3[i].Write()
    hist_true_categ3[i].Write()

fout.Close()
