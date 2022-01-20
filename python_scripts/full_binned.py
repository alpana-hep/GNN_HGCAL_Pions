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
preds_ratio = np.asarray(pickle.load(predPickle))
# print(preds_ratio[preds_ratio>3])
# preds_ratio[preds_ratio>3] = 3
# print(preds_ratio[preds_ratio>3])

trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/ratio_target.pickle"
trueEnPickle = open(trueEn,"rb")
trueEn_ratio = np.asarray(pickle.load(trueEnPickle))
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

# hit_z ="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/2To3M/Hit_Z.pickle"
# hit_zPickle = open(hit_z,"rb")
# z =pickle.load(hit_zPickle)
# frac =  ((z<54)*0.0105) + (np.logical_and(z>54, z<154)*0.0789) + ((z>154)*0.0316)

rawE = ak.sum(RechitEn_pkl, axis=1)
print(preds_ratio[0],trueEn_ratio[0],rawE[0])
preds_trueEn = rawE*preds_ratio
trueEn_pkl = rawE*trueEn_ratio
# trueEn_pkl =trueEn_pkl*frac
# preds_trueEn = preds_trueEn*frac
valid_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/all_valididx.pickle"
train_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/all_trainidx.pickle"

valid_idx_f = open(valid_idx_file,"rb")
valid_idx = np.asarray(pickle.load(valid_idx_f))
print(len(valid_idx))

train_idx_f = open(train_idx_file,"rb")
train_idx = np.asarray(pickle.load(train_idx_f))
print(len(train_idx))
print(preds_trueEn[train_idx[1]])
print(trueEn_pkl[train_idx[1]])
print(preds_trueEn[valid_idx[1]])
print(trueEn_pkl[valid_idx[1]])
bin_range = np.arange(10,350,4)
print(bin_range)
print(len(bin_range))

import ROOT
fout= ROOT.TFile("hist_lr1e04_fixwt_ratio_FullBinned_rechitInGeV_100epoch.root", 'RECREATE')
import ROOT
hist_pred_Valid=[]
hist_true_Valid=[]
hist_pred_Train=[]
hist_true_Train=[]
hist_predTrue_Valid=[]
hist_norm_predTrue_Valid=[]
hist_predTrue_Train=[]
hist_norm_predTrue_Train=[]
hist_pred_Tbdata=[]
hist_true_Tbdata=[]
hist_predTrue_Tbdata=[]
hist_norm_predTrue_Tbdata=[]
Energy=[20,50,80,100,120,200,250,300]
M=35 # number of histograms
for i_hist in range(len(bin_range)):
    xhigh_pred= 3.0*bin_range[i_hist]
    xhigh_true= 2.0*bin_range[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    
    name1='TrueEn_%i_to_%i' %(bin_range[i_hist],bin_range[i_hist]+4)#,u[i_hist],v[i_hist],typee[i_hist])
    hist_pred_Valid.append(ROOT.TH1F('Valid_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_Valid.append(ROOT.TH1F('Valid_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0,xhigh_true ))
    hist_pred_Train.append(ROOT.TH1F('Train_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_Train.append(ROOT.TH1F('Train_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_Valid.append(ROOT.TH1F('Valid_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Valid.append(ROOT.TH1F('Valid_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_predTrue_Train.append(ROOT.TH1F('Train_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Train.append(ROOT.TH1F('Train_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))

valid_predEn_all=[]
valid_trueEn_all=[]
# for ibin in range(len(bin_range)):
#     #print(ibin)
#     if(ibin==0):
#         bin_range[0]=9.0
    #print(ibin, bin_range[ibin])
for i in range(len(valid_idx)):
    valid_trueEn=(trueEn_pkl[valid_idx[i]])
    valid_predEn=(preds_trueEn[valid_idx[i]])
    trueEn=np.empty(85,dtype='float')
    predEn=np.empty(85,dtype='float')
    for ibin in range(len(bin_range)):
        if(ibin==0):
            inext=5
        else:
            inext=4
        #f(ibin<len(bin_range)):
        if(valid_trueEn>=bin_range[ibin] and valid_trueEn <=bin_range[ibin]+4):
            #print(bin_range[ibin],bin_range[ibin+1])
            diff= valid_trueEn - valid_predEn
            norm = diff/valid_trueEn
            hist_pred_Valid[ibin].Fill(valid_predEn)
            hist_true_Valid[ibin].Fill(valid_trueEn)
            hist_predTrue_Valid[ibin].Fill(diff)
            hist_norm_predTrue_Valid[ibin].Fill(norm)
train_predEn_all=[]
train_trueEn_all=[]
# for ibin in range(len(bin_range)):
#     #print(ibin)
#     if(ibin==0):
#         bin_range[0]=9.0
    #print(ibin, bin_range[ibin])
for i in range(len(train_idx)):
    train_trueEn=(trueEn_pkl[train_idx[i]])
    train_predEn=(preds_trueEn[train_idx[i]])
    trueEn=np.empty(85,dtype='float')
    predEn=np.empty(85,dtype='float')
    for ibin in range(len(bin_range)):
        if(ibin==0):
            inext=5
        else:
            inext=4
        #if(ibin<len(bin_range)):
        if(train_trueEn>=bin_range[ibin] and train_trueEn <=bin_range[ibin]+4):
            diff= train_trueEn - train_predEn
            norm = diff/train_trueEn
            hist_pred_Train[ibin].Fill(train_predEn)
            hist_true_Train[ibin].Fill(train_trueEn)
            hist_predTrue_Train[ibin].Fill(diff)
            hist_norm_predTrue_Train[ibin].Fill(norm)

fout.cd()
for i in range(85):
    hist_pred_Valid[i].Write()
    hist_true_Valid[i].Write()
    hist_pred_Train[i].Write()
    hist_true_Train[i].Write()
    hist_predTrue_Valid[i].Write()
    hist_norm_predTrue_Valid[i].Write()
    hist_predTrue_Train[i].Write()
    hist_norm_predTrue_Train[i].Write()
fout.Close()
