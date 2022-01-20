#import the stuff
import pandas as pd #dataframes etc
import matplotlib.pyplot as plt #plotting
import pickle
import numpy as np
import os, sys
import seaborn as sns
#import the stuff
import pandas as pd #dataframes etc
import matplotlib.pyplot as plt #plotting
import pickle
import numpy as np
import os, sys
import awkward as ak
import pickle
import numpy as np
pred_v2 ="./valid_flat/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])


trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/ratio_target.pickle"
trueEnPickle = open(trueEn,"rb")
trueEn_ratio = np.asarray(pickle.load(trueEnPickle))
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)



rawE = ak.sum(RechitEn_pkl, axis=1)
preds_trueEn = rawE*preds_ratio
trueEn_pkl = rawE*trueEn_ratio
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

tb_valid_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/all_valididx.pickle"
tb_train_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/all_trainidx.pickle"
tb_pred_v2 ="./tb_data/pred_tb.pickle"
tb_predPickle = open(tb_pred_v2, "rb")
print(tb_predPickle)
tb_preds_ratio = np.asarray(pickle.load(tb_predPickle))
print(tb_preds_ratio[tb_preds_ratio>3])
tb_preds_ratio[tb_preds_ratio>3] = 3
print(tb_preds_ratio[tb_preds_ratio>3])

tb_trueEn= "/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/beamEn.pickle"
tb_trueEnPickle = open(tb_trueEn,"rb")
tb_trueEn_pkl = np.asarray(pickle.load(tb_trueEnPickle))
print(tb_trueEn_pkl[200000])
RechitEn_tb ="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/recHitEn.pickle"
RechitEn_tbPickle = open(RechitEn_tb,"rb")
RechitEn_tb_pkl =pickle.load(RechitEn_tbPickle)
tbhit_z ="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/Hit_Z.pickle"
tbhit_zPickle = open(tbhit_z,"rb")
z_tb =pickle.load(tbhit_zPickle)
frac =  ((z_tb<54)*1.035) + (np.logical_and(z_tb>54, z_tb<154)*1.095) + ((z_tb>154)*1.095)

rawE_tb = ak.sum(RechitEn_tb_pkl, axis=1)
tb_preds_trueEn = rawE_tb*tb_preds_ratio
tb_valid_idx_f = open(tb_valid_idx_file,"rb")
tb_valid_idx = np.asarray(pickle.load(tb_valid_idx_f))
print(len(tb_valid_idx))

tb_train_idx_f = open(tb_train_idx_file,"rb")
tb_train_idx = np.asarray(pickle.load(tb_train_idx_f))
print(len(tb_train_idx))


## reading shower start location eventwise
#SSLocation_file_sim = "/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_wt/SsLocation.pickle"
SSLocation_file_sim = "/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M/shower_start_loc.pickle"
SSLocation_sim =  open(SSLocation_file_sim,"rb")
SSLocation_arr_sim = np.asarray(pickle.load(SSLocation_sim))
print(SSLocation_arr_sim[0])
SSLocation_file_data = "/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/SsLocation.pickle"
SSLocation_data =  open(SSLocation_file_data,"rb")
SSLocation_arr_data = np.asarray(pickle.load(SSLocation_data))
print(SSLocation_arr_data[0])

## Booking the histograms
import ROOT
fout= ROOT.TFile("hist_ratio_fixwt_lr1e04_SSEvtCateg_vsResol_6bins_beamEn_100ep.root", 'RECREATE')

hist_pred_Valid_SSinEE=[]
hist_true_Valid_SSinEE=[]
hist_pred_Train_SSinEE=[]
hist_true_Train_SSinEE=[]
hist_predTrue_Valid_SSinEE=[]
hist_norm_predTrue_Valid_SSinEE=[]
hist_predTrue_Train_SSinEE=[]
hist_norm_predTrue_Train_SSinEE=[]
hist_pred_Tbdata_SSinEE=[]
hist_true_Tbdata_SSinEE=[]
hist_predTrue_Tbdata_SSinEE=[]
hist_norm_predTrue_Tbdata_SSinEE=[]
hist_valid_predVsSS=[]
hist_train_predVsSS=[]
hist_tbdata_predVsSS=[]
SS_location_arr=[10,20,28,32,36,40]
Energy=[20,50,80,100,120,200,250,300]
M=8 # number of histograms                                                                                                                  
m =6
for i_hist in range(M):
    xhigh_pred= 3.0*Energy[i_hist]
    xhigh_true= 2.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(Energy[i_hist])#,u[i_hist],v[i_hist],typee[i_hist])                                                                 
    hist_pred_Valid_SSinEE.append(ROOT.TH1F('SSinEE_Valid_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_Valid_SSinEE.append(ROOT.TH1F('SSinEE_Valid_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0,xhigh_true ))
    hist_pred_Train_SSinEE.append(ROOT.TH1F('SSinEE_Train_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_Train_SSinEE.append(ROOT.TH1F('SSinEE_Train_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_Valid_SSinEE.append(ROOT.TH1F('SSinEE_Valid_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Valid_SSinEE.append(ROOT.TH1F('SSinEE_Valid_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_predTrue_Train_SSinEE.append(ROOT.TH1F('SSinEE_Train_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Train_SSinEE.append(ROOT.TH1F('SSinEE_Train_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_pred_Tbdata_SSinEE.append(ROOT.TH1F('SSinEE_Tbdata_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
    hist_true_Tbdata_SSinEE.append(ROOT.TH1F('SSinEE_Tbdata_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_valid_temp=[]
    hist_train_temp=[]
    hist_tbdata_temp=[]
    for i_sloc in range(m):
        if(i_sloc==0):
            name1='Ss_%i_to_%i_En_%i' %(0,SS_location_arr[i_sloc],Energy[i_hist])
        elif (i_sloc<3 and i_sloc!=0):
            name1='Ss_%i_to_%i_En_%i' %(SS_location_arr[i_sloc-1],SS_location_arr[i_sloc],Energy[i_hist])
        else:
            name1='Ss_%i_to_%i_En_%i' %(SS_location_arr[i_sloc-1],SS_location_arr[i_sloc],Energy[i_hist])
        hist_valid_temp.append(ROOT.TH1F('Valid_Predi_%s' % name1, 'Predi_%s'% name1,  500, 0, xhigh_pred))
        hist_train_temp.append(ROOT.TH1F('Train_Predi_%s' % name1, 'Predi_%s' % name1,  500, 0, xhigh_pred))
        hist_tbdata_temp.append(ROOT.TH1F('Tbdata_Predi_%s' % name1, 'Predi_%s' % name1,  500, 0, xhigh_pred))
    hist_train_predVsSS.append(hist_train_temp)
    hist_valid_predVsSS.append(hist_valid_temp)
    hist_tbdata_predVsSS.append(hist_tbdata_temp)
hist_pred_Valid_MipsInEE=[]
hist_true_Valid_MipsInEE=[]
hist_pred_Train_MipsInEE=[]
hist_true_Train_MipsInEE=[]
hist_predTrue_Valid_MipsInEE=[]
hist_norm_predTrue_Valid_MipsInEE=[]
hist_predTrue_Train_MipsInEE=[]
hist_norm_predTrue_Train_MipsInEE=[]
hist_pred_Tbdata_MipsInEE=[]
hist_true_Tbdata_MipsInEE=[]
hist_predTrue_Tbdata_MipsInEE=[]
hist_norm_predTrue_Tbdata_MipsInEE=[]

for i_hist in range(M):
    xhigh_pred= 3.0*Energy[i_hist]
    xhigh_true= 2.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(Energy[i_hist])#,u[i_hist],v[i_hist],typee[i_hist])                                                                  
    hist_pred_Valid_MipsInEE.append(ROOT.TH1F('MipsInEE_Valid_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_Valid_MipsInEE.append(ROOT.TH1F('MipsInEE_Valid_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0,xhigh_true ))
    hist_pred_Train_MipsInEE.append(ROOT.TH1F('MipsInEE_Train_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_Train_MipsInEE.append(ROOT.TH1F('MipsInEE_Train_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_Valid_MipsInEE.append(ROOT.TH1F('MipsInEE_Valid_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Valid_MipsInEE.append(ROOT.TH1F('MipsInEE_Valid_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm))
    hist_predTrue_Train_MipsInEE.append(ROOT.TH1F('MipsInEE_Train_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Train_MipsInEE.append(ROOT.TH1F('MipsInEE_Train_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm))
    hist_pred_Tbdata_MipsInEE.append(ROOT.TH1F('MipsInEE_Tbdata_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
    hist_true_Tbdata_MipsInEE.append(ROOT.TH1F('MipsInEE_Tbdata_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_Tbdata_MipsInEE.append(ROOT.TH1F('MipsInEE_Tbdata_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Tbdata_MipsInEE.append(ROOT.TH1F('MipsInEE_Tbdata_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))



trueEn_Valid_MipsInEE=[]
PredEn_Valid_MipsInEE=[]
trueEn_Valid_SSinEE=[]
PredEn_Valid_SSinEE=[]
trueEn_Train_MipsInEE=[]
PredEn_Train_MipsInEE=[]
trueEn_Train_SSinEE=[]
PredEn_Train_SSinEE=[]


for i in range(len(valid_idx)):
    valid_trueEn=(trueEn_pkl[valid_idx[i]])
    valid_predEn=(preds_trueEn[valid_idx[i]])
    SS_location= SSLocation_arr_sim[valid_idx[i]]
    diff= valid_trueEn - valid_predEn
    norm = diff/valid_trueEn
    for i_sloc in range(m):
        if(i_sloc<5):
            if(SS_location>(SS_location_arr[i_sloc]-6) and SS_location<=(SS_location_arr[i_sloc])):               
                for ibin in range(8):
                    if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                        hist_valid_predVsSS[ibin][i_sloc].Fill(valid_predEn)
        else:
            if(SS_location>(SS_location_arr[i_sloc]-3) and SS_location<=(SS_location_arr[i_sloc])):
                for ibin in range(8):
                    if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                        hist_valid_predVsSS[ibin][i_sloc].Fill(valid_predEn)

    if(SS_location>28):
        for ibin in range(8):
            if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                #print(ibin)                                                                                                              
                hist_pred_Valid_MipsInEE[ibin].Fill(valid_predEn)
                hist_true_Valid_MipsInEE[ibin].Fill(valid_trueEn)
                hist_predTrue_Valid_MipsInEE[ibin].Fill(diff)
                hist_norm_predTrue_Valid_MipsInEE[ibin].Fill(norm)
                
    elif(SS_location<=28):
        for ibin in range(8):
            if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                hist_pred_Valid_SSinEE[ibin].Fill(valid_predEn)
                hist_true_Valid_SSinEE[ibin].Fill(valid_trueEn)
                hist_predTrue_Valid_SSinEE[ibin].Fill(diff)
                hist_norm_predTrue_Valid_SSinEE[ibin].Fill(norm)
    else:
        continue
for i in range(len(train_idx)):
    valid_trueEn=(trueEn_pkl[train_idx[i]])
    valid_predEn=(preds_trueEn[train_idx[i]])
    SS_location= SSLocation_arr_sim[train_idx[i]]
    diff= valid_trueEn - valid_predEn
    norm = diff/valid_trueEn
    for i_sloc in range(m):
        if(i_sloc<5):
            if(SS_location>(SS_location_arr[i_sloc]-6) and SS_location<=(SS_location_arr[i_sloc])):
                for ibin in range(8):
                    if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                        hist_train_predVsSS[ibin][i_sloc].Fill(valid_predEn)
        else:
            if(SS_location>(SS_location_arr[i_sloc]-3) and SS_location<=(SS_location_arr[i_sloc])):
                for ibin in range(8):
                    if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                        hist_train_predVsSS[ibin][i_sloc].Fill(valid_predEn)
                        
                        
    if(SS_location>28):
        for ibin in range(8):
            if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
               
                hist_pred_Train_MipsInEE[ibin].Fill(valid_predEn)
                hist_true_Train_MipsInEE[ibin].Fill(valid_trueEn)
                hist_predTrue_Train_MipsInEE[ibin].Fill(diff)
                hist_norm_predTrue_Train_MipsInEE[ibin].Fill(norm)
    elif(SS_location<=28):
        for ibin in range(8):
            if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                hist_pred_Train_SSinEE[ibin].Fill(valid_predEn)
                hist_true_Train_SSinEE[ibin].Fill(valid_trueEn)
                hist_predTrue_Train_SSinEE[ibin].Fill(diff)
                hist_norm_predTrue_Train_SSinEE[ibin].Fill(norm)
    else:
        continue
        
        

trueEn_TB_MipsInEE=[]
PredEn_TB_MipsInEE=[]
trueEn_TB_SSinEE=[]
PredEn_TB_SSinEE=[]
for i in range(len(tb_preds_trueEn)):
    valid_trueEn=(tb_trueEn_pkl[i])
    valid_predEn=(tb_preds_trueEn[i])
    SS_location= SSLocation_arr_data[i]
    diff= valid_trueEn - valid_predEn
    norm = diff/valid_trueEn
    for i_sloc in range(m):
        if(i_sloc<5):
            if(SS_location>(SS_location_arr[i_sloc]-6) and SS_location<=(SS_location_arr[i_sloc])):
                for ibin in range(8):
                    if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                        hist_tbdata_predVsSS[ibin][i_sloc].Fill(valid_predEn)
        else:
            if(SS_location>(SS_location_arr[i_sloc]-3) and SS_location<=(SS_location_arr[i_sloc])):
                for ibin in range(8):
                    if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                        hist_tbdata_predVsSS[ibin][i_sloc].Fill(valid_predEn)

    if(SS_location>28):
        for ibin in range(8):
            if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                hist_pred_Tbdata_MipsInEE[ibin].Fill(valid_predEn)
                hist_true_Tbdata_MipsInEE[ibin].Fill(valid_trueEn)
                #hist_predTrue_Tbdata_MipsInEE[ibin].Fill(diff)
                #hist_norm_predTrue_Tbdata_MipsInEE[ibin].Fill(norm)
    elif(SS_location<=28):
        for ibin in range(8):
            if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn<=Energy[ibin]+2 ):
                hist_pred_Tbdata_SSinEE[ibin].Fill(valid_predEn)
                hist_true_Tbdata_SSinEE[ibin].Fill(valid_trueEn)
                #hist_predTrue_Tbdata_SSinEE[ibin].Fill(diff)
                #hist_norm_predTrue_Tbdata_SSinEE[ibin].Fill(norm)
    else:
        continue







fout.cd()
for i in range(8):
    hist_pred_Valid_MipsInEE[i].Write()
    hist_true_Valid_MipsInEE[i].Write()
    hist_pred_Train_MipsInEE[i].Write()
    hist_true_Train_MipsInEE[i].Write()
    hist_pred_Tbdata_MipsInEE[i].Write()
    hist_true_Tbdata_MipsInEE[i].Write()
    hist_pred_Valid_SSinEE[i].Write()
    hist_true_Valid_SSinEE[i].Write()
    hist_pred_Train_SSinEE[i].Write()
    hist_true_Train_SSinEE[i].Write()
    hist_pred_Tbdata_SSinEE[i].Write()
    hist_true_Tbdata_SSinEE[i].Write()
    temp_train = hist_train_predVsSS[i]
    temp_valid = hist_valid_predVsSS[i]
    temp_tbdata = hist_tbdata_predVsSS[i]

    for i_sloc in range(6):
        
        temp_train[i_sloc].Write()
        temp_valid[i_sloc].Write()
        temp_tbdata[i_sloc].Write()
fout.Close()
