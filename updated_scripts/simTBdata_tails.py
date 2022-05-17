import pandas as pd #dataframes etc
import matplotlib.pyplot as plt #plotting
import pickle
import awkward as ak
import numpy as np
import os, sys
import seaborn as sns
import pickle
import numpy as np

folder = sys.argv[1]
print("input predictions",folder)
outfileName= sys.argv[2]
out_fname = './%s/%s'%(folder,outfileName)
print(out_fname, "output file is")
inpickle_folder =sys.argv[3]
print("input pickle files are picked from",inpickle_folder)

pred_v2 ="./%s/valid_flat/pred_tb.pickle" %folder
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])


trueEn ="%s/ratio_target.pickle"%inpickle_folder
trueEnPickle = open(trueEn,"rb")
trueEn_ratio = np.asarray(pickle.load(trueEnPickle))
RechitEn ="%s/recHitEn.pickle"%inpickle_folder
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)
hit_z ="%s/Hit_Z.pickle"%inpickle_folder
hit_zPickle = open(hit_z,"rb")
z =pickle.load(hit_zPickle)
# frac =  ((z<54)*0.0105) + (np.logical_and(z>54, z<154)*0.0789) + ((z>154)*0.0316)
print('sim',z)
rawE = ak.sum(RechitEn_pkl, axis=1)
preds_trueEn = rawE*preds_ratio
trueEn_pkl = rawE*trueEn_ratio
valid_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/Correct_training_5M/trimAhcal/NSM_infer/all_valididx.pickle"
train_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/Correct_training_5M/trimAhcal/NSM_infer/all_trainidx.pickle"
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
valid_trueEn_All=[]
valid_predEn_All=[]
##temp lists
valid_trueEn_20=[]
valid_predEn_20=[]
valid_trueEn_50=[]
valid_predEn_50=[]
valid_trueEn_80=[]
valid_predEn_80=[]
valid_trueEn_100=[]
valid_predEn_100=[]
valid_trueEn_120=[]
valid_predEn_120=[]
valid_trueEn_200=[]
valid_predEn_200=[]
valid_trueEn_250=[]
valid_predEn_250=[]
valid_trueEn_300=[]
valid_predEn_300=[]

train_trueEn_All=[]
train_predEn_All=[]
for i in range(len(valid_idx)):
    valid_trueEn=(trueEn_pkl[valid_idx[i]])
    valid_predEn=(preds_trueEn[valid_idx[i]])
    if(valid_trueEn>=18 and valid_trueEn<=22):
        valid_trueEn_20.append(valid_trueEn)
        valid_predEn_20.append(valid_predEn)
    elif(valid_trueEn>=48 and valid_trueEn<=52):
        valid_trueEn_50.append(valid_trueEn)
        valid_predEn_50.append(valid_predEn)
    elif(valid_trueEn>=78 and valid_trueEn<=82):
        valid_trueEn_80.append(valid_trueEn)
        valid_predEn_80.append(valid_predEn)
    elif(valid_trueEn>=98 and valid_trueEn<=102):
        valid_trueEn_100.append(valid_trueEn)
        valid_predEn_100.append(valid_predEn)
    elif(valid_trueEn>=118 and valid_trueEn<=122):
        valid_trueEn_120.append(valid_trueEn)
        valid_predEn_120.append(valid_predEn)
    elif(valid_trueEn>=198 and valid_trueEn<=202):
        valid_trueEn_200.append(valid_trueEn)
        valid_predEn_200.append(valid_predEn)
    elif(valid_trueEn>=248 and valid_trueEn<=252):
        valid_trueEn_250.append(valid_trueEn)
        valid_predEn_250.append(valid_predEn)
    elif(valid_trueEn>=298 and valid_trueEn<=302):
        valid_trueEn_300.append(valid_trueEn)
        valid_predEn_300.append(valid_predEn)
valid_trueEn_All.append(valid_trueEn_20)
valid_trueEn_All.append(valid_trueEn_50)
valid_trueEn_All.append(valid_trueEn_80)
valid_trueEn_All.append(valid_trueEn_100)
valid_trueEn_All.append(valid_trueEn_120)
valid_trueEn_All.append(valid_trueEn_200)
valid_trueEn_All.append(valid_trueEn_250)
valid_trueEn_All.append(valid_trueEn_300)

valid_predEn_All.append(valid_predEn_20)
valid_predEn_All.append(valid_predEn_50)
valid_predEn_All.append(valid_predEn_80)
valid_predEn_All.append(valid_predEn_100)
valid_predEn_All.append(valid_predEn_120)
valid_predEn_All.append(valid_predEn_200)
valid_predEn_All.append(valid_predEn_250)
valid_predEn_All.append(valid_predEn_300)
  
train_trueEn_20=[]
train_predEn_20=[]
train_trueEn_50=[]
train_predEn_50=[]
train_trueEn_80=[]
train_predEn_80=[]
train_trueEn_100=[]
train_predEn_100=[]
train_trueEn_120=[]
train_predEn_120=[]
train_trueEn_200=[]
train_predEn_200=[]
train_trueEn_250=[]
train_predEn_250=[]
train_trueEn_300=[]
train_predEn_300=[]

for i in range(len(train_idx)):
    train_trueEn=(trueEn_pkl[train_idx[i]])
    train_predEn=(preds_trueEn[train_idx[i]])
    if(train_trueEn>=18 and train_trueEn<=22):
        train_trueEn_20.append(train_trueEn)
        train_predEn_20.append(train_predEn)
    elif(train_trueEn>=48 and train_trueEn<=52):
        train_trueEn_50.append(train_trueEn)
        train_predEn_50.append(train_predEn)
    elif(train_trueEn>=78 and train_trueEn<=82):
        train_trueEn_80.append(train_trueEn)
        train_predEn_80.append(train_predEn)
    elif(train_trueEn>=98 and train_trueEn<=102):
        train_trueEn_100.append(train_trueEn)
        train_predEn_100.append(train_predEn)
    elif(train_trueEn>=118 and train_trueEn<=122):
        train_trueEn_120.append(train_trueEn)
        train_predEn_120.append(train_predEn)
    elif(train_trueEn>=198 and train_trueEn<=202):
        train_trueEn_200.append(train_trueEn)
        train_predEn_200.append(train_predEn)
    elif(train_trueEn>=248 and train_trueEn<=252):
        train_trueEn_250.append(train_trueEn)
        train_predEn_250.append(train_predEn)
    elif(train_trueEn>=298 and train_trueEn<=302):
        train_trueEn_300.append(train_trueEn)
        train_predEn_300.append(train_predEn)
train_trueEn_All.append(train_trueEn_20)
train_trueEn_All.append(train_trueEn_50)
train_trueEn_All.append(train_trueEn_80)
train_trueEn_All.append(train_trueEn_100)
train_trueEn_All.append(train_trueEn_120)
train_trueEn_All.append(train_trueEn_200)
train_trueEn_All.append(train_trueEn_250)
train_trueEn_All.append(train_trueEn_300)

train_predEn_All.append(train_predEn_20)
train_predEn_All.append(train_predEn_50)
train_predEn_All.append(train_predEn_80)
train_predEn_All.append(train_predEn_100)
train_predEn_All.append(train_predEn_120)
train_predEn_All.append(train_predEn_200)
train_predEn_All.append(train_predEn_250)
train_predEn_All.append(train_predEn_300)
    
    
np_valid_trueEn=np.array(valid_trueEn_All)
np_valid_predEn=np.array(valid_predEn_All)
np_train_trueEn=np.array(train_trueEn_All)
np_train_predEn=np.array(train_predEn_All)

SSLocation_file_sim = "/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_0to5M/SsLocation.pickle"
SSLocation_sim =  open(SSLocation_file_sim,"rb")
SSLocation_arr_sim = np.asarray(pickle.load(SSLocation_sim))
print(SSLocation_arr_sim[0])
SSLocation_file_data = "/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/fix_wt/SsLocation.pickle"
SSLocation_data =  open(SSLocation_file_data,"rb")
SSLocation_arr_data = np.asarray(pickle.load(SSLocation_data))
print(SSLocation_arr_data[0])


## reading tb data files
tbfolder = sys.argv[4]
tb_valid_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_updat_relwt_ahcaltrim_TB/all_valididx.pickle"
tb_train_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_updat_relwt_ahcaltrim_TB/all_trainidx.pickle"
tb_pred_v2 ="%s/tb_data_upscaled/pred_tb.pickle"%folder
tb_predPickle = open(tb_pred_v2, "rb")
print(tb_predPickle)
tb_preds_ratio = np.asarray(pickle.load(tb_predPickle))
print(tb_preds_ratio[tb_preds_ratio>3])
tb_preds_ratio[tb_preds_ratio>3] = 3
print(tb_preds_ratio[tb_preds_ratio>3])

tb_trueEn= "/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/tb_corrAlign/trim_Ahcal/beamEn.pickle"
tb_trueEnPickle = open(tb_trueEn,"rb")
tb_trueEn_pkl = np.asarray(pickle.load(tb_trueEnPickle))
print(tb_trueEn_pkl[200000])
RechitEn_tb ="%s/recHitEn.pickle"%tbfolder
RechitEn_tbPickle = open(RechitEn_tb,"rb")
RechitEn_tb_pkl =pickle.load(RechitEn_tbPickle)
tbhit_z ="%s/Hit_Z.pickle"%tbfolder

# tbhit_zPickle = open(tbhit_z,"rb")
# z_tb =pickle.load(tbhit_zPickle)
# print('data',z_tb)
# frac =  ((z_tb<54)*1.035) + (np.logical_and(z_tb>54, z_tb<154)*1.095) + ((z_tb>154)*1.095)
rawE_tb = ak.sum(RechitEn_tb_pkl, axis=1)
tb_preds_trueEn = rawE_tb*tb_preds_ratio
#trueEn_pkl = rawE*trueEn_ratio
print(tb_preds_trueEn[200000])
tb_valid_idx_f = open(tb_valid_idx_file,"rb")
tb_valid_idx = np.asarray(pickle.load(tb_valid_idx_f))
print(len(tb_valid_idx))

tb_train_idx_f = open(tb_train_idx_file,"rb")
tb_train_idx = np.asarray(pickle.load(tb_train_idx_f))
print(len(tb_train_idx))
tb_trueEn_All=[]
tb_predEn_All=[]
##temp lists
tb_trueEn_20=[]
tb_predEn_20=[]
tb_trueEn_50=[]
tb_predEn_50=[]
tb_trueEn_80=[]
tb_predEn_80=[]
tb_trueEn_100=[]
tb_predEn_100=[]
tb_trueEn_120=[]
tb_predEn_120=[]
tb_trueEn_200=[]
tb_predEn_200=[]
tb_trueEn_250=[]
tb_predEn_250=[]
tb_trueEn_300=[]
tb_predEn_300=[]

for i in range(len(tb_valid_idx)):
    tb_trueEn=(tb_trueEn_pkl[tb_valid_idx[i]])
    tb_predEn=(tb_preds_trueEn[tb_valid_idx[i]])
    if(tb_trueEn==20):
        tb_trueEn_20.append(tb_trueEn)
        tb_predEn_20.append(tb_predEn)
        #print(tb_trueEn)
    elif(tb_trueEn==50):
        tb_trueEn_50.append(tb_trueEn)
        tb_predEn_50.append(tb_predEn)
    elif(tb_trueEn==80):
        tb_trueEn_80.append(tb_trueEn)
        tb_predEn_80.append(tb_predEn)
    elif(tb_trueEn==100):
        tb_trueEn_100.append(tb_trueEn)
        tb_predEn_100.append(tb_predEn)
    elif(tb_trueEn==120):
        tb_trueEn_120.append(tb_trueEn)
        tb_predEn_120.append(tb_predEn)
    elif(tb_trueEn==200):
        tb_trueEn_200.append(tb_trueEn)
        tb_predEn_200.append(tb_predEn)
    elif(tb_trueEn==250):
        tb_trueEn_250.append(tb_trueEn)
        tb_predEn_250.append(tb_predEn)
    elif(tb_trueEn==300):
        tb_trueEn_300.append(tb_trueEn)
        tb_predEn_300.append(tb_predEn)
for i in range(len(tb_train_idx)):
    tb_trueEn=(tb_trueEn_pkl[tb_train_idx[i]])
    tb_predEn=(tb_preds_trueEn[tb_train_idx[i]])
    if(tb_trueEn==20.0):
        tb_trueEn_20.append(tb_trueEn)
        tb_predEn_20.append(tb_predEn)
    elif(tb_trueEn==50):
        tb_trueEn_50.append(tb_trueEn)
        tb_predEn_50.append(tb_predEn)
    elif(tb_trueEn==80):
        tb_trueEn_80.append(tb_trueEn)
        tb_predEn_80.append(tb_predEn)
    elif(tb_trueEn==100):
        tb_trueEn_100.append(tb_trueEn)
        tb_predEn_100.append(tb_predEn)
    elif(tb_trueEn==120):
        tb_trueEn_120.append(tb_trueEn)
        tb_predEn_120.append(tb_predEn)
    elif(tb_trueEn==200):
        tb_trueEn_200.append(tb_trueEn)
        tb_predEn_200.append(tb_predEn)
    elif(tb_trueEn==250):
        tb_trueEn_250.append(tb_trueEn)
        tb_predEn_250.append(tb_predEn)
    elif(tb_trueEn==300):
        tb_trueEn_300.append(tb_trueEn)
        tb_predEn_300.append(tb_predEn)
print(len(tb_predEn_20))
tb_trueEn_All.append(tb_trueEn_20)
tb_trueEn_All.append(tb_trueEn_50)
tb_trueEn_All.append(tb_trueEn_80)
tb_trueEn_All.append(tb_trueEn_100)
tb_trueEn_All.append(tb_trueEn_120)
tb_trueEn_All.append(tb_trueEn_200)
tb_trueEn_All.append(tb_trueEn_250)
tb_trueEn_All.append(tb_trueEn_300)

tb_predEn_All.append(tb_predEn_20)
tb_predEn_All.append(tb_predEn_50)
tb_predEn_All.append(tb_predEn_80)
tb_predEn_All.append(tb_predEn_100)
tb_predEn_All.append(tb_predEn_120)
tb_predEn_All.append(tb_predEn_200)
tb_predEn_All.append(tb_predEn_250)
tb_predEn_All.append(tb_predEn_300)

import ROOT
fout= ROOT.TFile(out_fname, 'RECREATE') # ROOT.TFile("./aggre2_maxlr6e04_75ep/constlr_4feat/ep100/hist_8beamE_uprelwt_ratio_constlr_maxlr1e04_2aggre_ep100_4feat_NSM.root", 'RECREATE')
# import ROOT
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

hist_pred_QGSP=[]
hist_true_QGSP=[]
hist_pred_FTFP=[]
hist_true_FTFP=[]
hist_predTrue_QGSP=[]
hist_norm_predTrue_QGSP=[]
hist_predTrue_FTFP=[]
hist_norm_predTrue_FTFP=[]
hist_2d= ROOT.TH2F("hist","true en vs pred en",300,0,400,300,0,400)
hist_2d_tb= ROOT.TH2F("hist_tb","true en vs pred en",300,0,400,300,0,400)
Energy=[20,50,80,100,120,200,250,300]
M=8 # number of histograms
for i_hist in range(M):
    if(Energy[i_hist]<100):
        xhigh_pred = 4.0*Energy[i_hist]
    else:
        xhigh_pred= 3.0*Energy[i_hist]
    xhigh_true= 2.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(Energy[i_hist])#,u[i_hist],v[i_hist],typee[i_hist])
    hist_pred_Valid.append(ROOT.TH1F('Valid_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_Valid.append(ROOT.TH1F('Valid_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0,xhigh_true ))
    hist_pred_Train.append(ROOT.TH1F('Train_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_Train.append(ROOT.TH1F('Train_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_Valid.append(ROOT.TH1F('Valid_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Valid.append(ROOT.TH1F('Valid_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_predTrue_Train.append(ROOT.TH1F('Train_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Train.append(ROOT.TH1F('Train_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_pred_Tbdata.append(ROOT.TH1F('Tbdata_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
    hist_true_Tbdata.append(ROOT.TH1F('Tbdata_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_Tbdata.append(ROOT.TH1F('Tbdata_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_Tbdata.append(ROOT.TH1F('Tbdata_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))
    hist_pred_QGSP.append(ROOT.TH1F('QGSP_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_QGSP.append(ROOT.TH1F('QGSP_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0,xhigh_true ))
    hist_pred_FTFP.append(ROOT.TH1F('FTFP_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_true_FTFP.append(ROOT.TH1F('FTFP_trueEn_%s' % name1, """:"true Beam energy in GeV":""", 500, 0, xhigh_true))
    hist_predTrue_QGSP.append(ROOT.TH1F('QGSP_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_QGSP.append(ROOT.TH1F('QGSP_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_predTrue_FTFP.append(ROOT.TH1F('FTFP_Diff_Predi_%s' % name1, """:"Predicted -true in GeV":""", 500, xlow_diff, xhigh_diff))
    hist_norm_predTrue_FTFP.append(ROOT.TH1F('FTFP_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))

hist_X_2sigma_Valid=[]
hist_Y_2sigma_Valid=[]
hist_Z_2sigma_Valid=[]
hist_X_high_2sigma_Valid=[]
hist_Y_high_2sigma_Valid=[]
hist_Z_high_2sigma_Valid=[]
hist_rechitEn_2sigma_Valid=[]
hist_ZvsX_high_2sigma_Valid=[]
hist_ZvsY_high_2sigma_Valid=[]
hist_ZvsX_2sigma_Valid=[]
hist_ZvsY_2sigma_Valid=[]
hist_sslocation_Valid=[]
hist_ssVsGNN_Valid=[]
hist_rechitEn_xhigh_2sigma_Valid=[]
hist_X_2sigma_Train=[]
hist_Y_2sigma_Train=[]
hist_Z_2sigma_Train=[]
hist_X_high_2sigma_Train=[]
hist_Y_high_2sigma_Train=[]
hist_Z_high_2sigma_Train=[]
hist_rechitEn_2sigma_Train=[]
hist_sslocation_Train=[]
hist_ssVsGNN_Train=[]
hist_rechitEn_xhigh_2sigma_Train=[]
hist_ZvsX_high_2sigma_Train=[]
hist_ZvsY_high_2sigma_Train=[]
hist_ZvsX_2sigma_Train=[]
hist_ZvsY_2sigma_Train=[]

hist_X_2sigma_Tbdata=[]
hist_Y_2sigma_Tbdata=[]
hist_Z_2sigma_Tbdata=[]

hist_X_high_2sigma_Tbdata=[]
hist_Y_high_2sigma_Tbdata=[]
hist_Z_high_2sigma_Tbdata=[]

hist_rechitEn_2sigma_Tbdata=[]
hist_sslocation_Tbdata=[]
hist_ssVsGNN_Tbdata=[]
hist_ZvsX_high_2sigma_Tbdata=[]
hist_ZvsY_high_2sigma_Tbdata=[]
hist_ZvsX_2sigma_Tbdata=[]
hist_ZvsY_2sigma_Tbdata=[]
hist_rechitEn_xhigh_2sigma_Tbdata=[]

hist_X_bulk_Tbdata=[]
hist_Y_bulk_Tbdata=[]
hist_Z_bulk_Tbdata=[]
hist_rechitEn_bulk_Tbdata=[]
hist_sslocation_bulk_Tbdata=[]
hist_sslocation_high_Tbdata=[]
hist_ZvsX_bulk_Tbdata=[]
hist_ZvsY_bulk_Tbdata=[]

hist_X_bulk_Train=[]
hist_Y_bulk_Train=[]
hist_Z_bulk_Train=[]
hist_rechitEn_bulk_Train=[]
hist_sslocation_bulk_Train=[]
hist_sslocation_high_Train=[]
hist_ZvsX_bulk_Train=[]
hist_ZvsY_bulk_Train=[]

hist_X_bulk_Valid=[]
hist_Y_bulk_Valid=[]
hist_Z_bulk_Valid=[]
hist_rechitEn_bulk_Valid=[]
hist_sslocation_bulk_Valid=[]
hist_sslocation_high_Valid=[]
hist_ZvsX_bulk_Valid=[]
hist_ZvsY_bulk_Valid=[]
hist_X_bulk_Valid_EE=[]
hist_Y_bulk_Valid_FH=[]
hist_X_bulk_Valid_FH=[]
hist_Y_bulk_Valid_EE=[]

hist_X_2sigma_Valid_EE=[]
hist_Y_2sigma_Valid_FH=[]
hist_X_2sigma_Valid_FH=[]
hist_Y_2sigma_Valid_EE=[]
hist_X_high_2sigma_Valid_EE=[]
hist_Y_high_2sigma_Valid_FH=[]
hist_X_high_2sigma_Valid_FH=[]
hist_Y_high_2sigma_Valid_EE=[]
#hist_Y_high_2sigma_Train_EE
hist_X_bulk_Train_FH=[]
hist_Y_bulk_Train_EE=[]
hist_X_2sigma_Train_FH=[]
hist_Y_2sigma_Train_EE=[]
hist_X_high_2sigma_Train_FH=[]
hist_Y_high_2sigma_Train_EE=[]

hist_X_bulk_Train_EE=[]
hist_Y_bulk_Train_FH=[]
hist_X_2sigma_Train_EE=[]
hist_Y_2sigma_Train_FH=[]
hist_X_high_2sigma_Train_EE=[]
hist_Y_high_2sigma_Train_FH=[]
hist_X_bulk_Tbdata_EE=[]
hist_Y_bulk_Tbdata_FH=[]
hist_X_2sigma_Tbdata_EE=[]
hist_Y_2sigma_Tbdata_FH=[]
hist_X_high_2sigma_Tbdata_EE=[]
hist_Y_high_2sigma_Tbdata_FH=[]
hist_X_bulk_Tbdata_FH=[]
hist_Y_bulk_Tbdata_EE=[]
hist_X_2sigma_Tbdata_FH=[]
hist_Y_2sigma_Tbdata_EE=[]
hist_X_high_2sigma_Tbdata_FH=[]
hist_Y_high_2sigma_Tbdata_EE=[]


#hist_ZvsX_bulk_2sigma_Valid
for i_hist in range(M):
    if(Energy[i_hist]<100):
        xhigh_pred = 4.0*Energy[i_hist]
    else:
        xhigh_pred= 2.0*Energy[i_hist]
    xhigh_true= 2.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    xhigh_X = 40
    xmin_X = -40
    xhigh_Z = 320
    xmin_Z = 0
    xbin_Z = 320
    xbin_X = 320
    name1='TrueEn_%i' %(Energy[i_hist])
    hist_ssVsGNN_Tbdata.append(ROOT.TH2F("hist_ssVsGNN_Tbdata_%s"%name1,"",500,0,xhigh_pred,40,0,40))
    hist_sslocation_Tbdata.append(ROOT.TH1F("hist_sslocation_Tbdata_%s"%name1,"",40,0,40))
    hist_rechitEn_2sigma_Tbdata.append(ROOT.TH1F("hist_rechitEn_2sigma_Tbdata_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_2sigma_Tbdata.append(ROOT.TH1F("hist_Z_2sigma_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_2sigma_Tbdata.append(ROOT.TH1F("hist_Y_2sigma_Tbdata_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Tbdata.append(ROOT.TH1F("hist_X_2sigma_Tbdata_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_rechitEn_xhigh_2sigma_Tbdata.append(ROOT.TH1F("hist_rechitEn_xhigh_2sigma_Tbdata_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_high_2sigma_Tbdata.append(ROOT.TH1F("hist_Z_xhigh_2sigma_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_high_2sigma_Tbdata.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Tbdata_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Tbdata.append(ROOT.TH1F("hist_X_xhigh_2sigma_Tbdata_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_high_2sigma_Tbdata.append(ROOT.TH2F("hist_ZvsX_high_2sigma_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_high_2sigma_Tbdata.append(ROOT.TH2F("hist_ZvsY_high_2sigma_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_2sigma_Tbdata.append(ROOT.TH2F("hist_ZvsX_2sigma_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_2sigma_Tbdata.append(ROOT.TH2F("hist_ZvsY_2sigma_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))

    hist_sslocation_high_Tbdata.append(ROOT.TH1F("hist_sslocation_high_Tbdata_%s"%name1,"",40,0,40))
    hist_rechitEn_bulk_Tbdata.append(ROOT.TH1F("hist_rechitEn_bulk_Tbdata_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_bulk_Tbdata.append(ROOT.TH1F("hist_Z_bulk_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_bulk_Tbdata.append(ROOT.TH1F("hist_Y_bulk_Tbdata_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Tbdata.append(ROOT.TH1F("hist_X_bulk_Tbdata_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_bulk_Tbdata.append(ROOT.TH2F("hist_ZvsX_bulk_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_bulk_Tbdata.append(ROOT.TH2F("hist_ZvsY_bulk_Tbdata_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_sslocation_bulk_Tbdata.append(ROOT.TH1F("hist_sslocation_bulk_Tbdata_%s"%name1,"",40,0,40))
    
    hist_ssVsGNN_Valid.append(ROOT.TH2F("hist_ssVsGNN_Valid_%s"%name1,"",500,0,xhigh_pred,40,0,40))
    hist_sslocation_Valid.append(ROOT.TH1F("hist_sslocation_Valid_%s"%name1,"",40,0,40))
    hist_rechitEn_2sigma_Valid.append(ROOT.TH1F("hist_rechitEn_2sigma_Valid_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_2sigma_Valid.append(ROOT.TH1F("hist_Z_2sigma_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_2sigma_Valid.append(ROOT.TH1F("hist_Y_2sigma_Valid_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Valid.append(ROOT.TH1F("hist_X_2sigma_Valid_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_rechitEn_xhigh_2sigma_Valid.append(ROOT.TH1F("hist_rechitEn_xhigh_2sigma_Valid_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_high_2sigma_Valid.append(ROOT.TH1F("hist_Z_xhigh_2sigma_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_high_2sigma_Valid.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Valid_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Valid.append(ROOT.TH1F("hist_X_xhigh_2sigma_Valid_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_high_2sigma_Valid.append(ROOT.TH2F("hist_ZvsX_high_2sigma_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_high_2sigma_Valid.append(ROOT.TH2F("hist_ZvsY_high_2sigma_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_2sigma_Valid.append(ROOT.TH2F("hist_ZvsX_2sigma_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_2sigma_Valid.append(ROOT.TH2F("hist_ZvsY_2sigma_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    
    hist_sslocation_high_Valid.append(ROOT.TH1F("hist_sslocation_high_Valid_%s"%name1,"",40,0,40))
    hist_rechitEn_bulk_Valid.append(ROOT.TH1F("hist_rechitEn_bulk_Valid_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_bulk_Valid.append(ROOT.TH1F("hist_Z_bulk_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_bulk_Valid.append(ROOT.TH1F("hist_Y_bulk_Valid_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Valid.append(ROOT.TH1F("hist_X_bulk_Valid_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_bulk_Valid.append(ROOT.TH2F("hist_ZvsX_bulk_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_bulk_Valid.append(ROOT.TH2F("hist_ZvsY_bulk_Valid_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_sslocation_bulk_Valid.append(ROOT.TH1F("hist_sslocation_bulk_Valid_%s"%name1,"",40,0,40))

    
    hist_sslocation_high_Train.append(ROOT.TH1F("hist_sslocation_high_Train_%s"%name1,"",40,0,40))
    hist_rechitEn_bulk_Train.append(ROOT.TH1F("hist_rechitEn_bulk_Train_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_bulk_Train.append(ROOT.TH1F("hist_Z_bulk_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_bulk_Train.append(ROOT.TH1F("hist_Y_bulk_Train_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Train.append(ROOT.TH1F("hist_X_bulk_Train_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_bulk_Train.append(ROOT.TH2F("hist_ZvsX_bulk_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_bulk_Train.append(ROOT.TH2F("hist_ZvsY_bulk_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_sslocation_bulk_Train.append(ROOT.TH1F("hist_sslocation_bulk_Train_%s"%name1,"",40,0,40))

    hist_ssVsGNN_Train.append(ROOT.TH2F("hist_ssVsGNN_Train_%s"%name1,"",500,0,xhigh_pred,40,0,40))
    hist_sslocation_Train.append(ROOT.TH1F("hist_sslocation_Train_%s"%name1,"",40,0,40))
    hist_rechitEn_2sigma_Train.append(ROOT.TH1F("hist_rechitEn_2sigma_Train_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_2sigma_Train.append(ROOT.TH1F("hist_Z_2sigma_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_2sigma_Train.append(ROOT.TH1F("hist_Y_2sigma_Train_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Train.append(ROOT.TH1F("hist_X_2sigma_Train_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_rechitEn_xhigh_2sigma_Train.append(ROOT.TH1F("hist_rechitEn_xhigh_2sigma_Train_%s"%name1,"",500,0,xhigh_pred))
    hist_Z_high_2sigma_Train.append(ROOT.TH1F("hist_Z_xhigh_2sigma_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z))
    hist_Y_high_2sigma_Train.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Train_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Train.append(ROOT.TH1F("hist_X_xhigh_2sigma_Train_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_high_2sigma_Train.append(ROOT.TH2F("hist_ZvsX_high_2sigma_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_high_2sigma_Train.append(ROOT.TH2F("hist_ZvsY_high_2sigma_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsX_2sigma_Train.append(ROOT.TH2F("hist_ZvsX_2sigma_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_ZvsY_2sigma_Train.append(ROOT.TH2F("hist_ZvsY_2sigma_Train_%s"%name1,"",xbin_Z,xmin_Z,xhigh_Z,xbin_X,xmin_X,xhigh_X))
    hist_Y_2sigma_Train_EE.append(ROOT.TH1F("hist_Y_2sigma_Train_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Train_EE.append(ROOT.TH1F("hist_X_2sigma_Train_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_2sigma_Train_FH.append(ROOT.TH1F("hist_Y_2sigma_Train_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Train_FH.append(ROOT.TH1F("hist_X_2sigma_Train_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_high_2sigma_Train_EE.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Train_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Train_EE.append(ROOT.TH1F("hist_X_xhigh_2sigma_Train_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_high_2sigma_Train_FH.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Train_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Train_FH.append(ROOT.TH1F("hist_X_xhigh_2sigma_Train_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_bulk_Train_EE.append(ROOT.TH1F("hist_Y_bulk_Train_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Train_EE.append(ROOT.TH1F("hist_X_bulk_Train_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_bulk_Train_FH.append(ROOT.TH1F("hist_Y_bulk_Train_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Train_FH.append(ROOT.TH1F("hist_X_bulk_Train_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))

    hist_Y_2sigma_Valid_EE.append(ROOT.TH1F("hist_Y_2sigma_Valid_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Valid_EE.append(ROOT.TH1F("hist_X_2sigma_Valid_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_2sigma_Valid_FH.append(ROOT.TH1F("hist_Y_2sigma_Valid_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Valid_FH.append(ROOT.TH1F("hist_X_2sigma_Valid_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_high_2sigma_Valid_EE.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Valid_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Valid_EE.append(ROOT.TH1F("hist_X_xhigh_2sigma_Valid_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_high_2sigma_Valid_FH.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Valid_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Valid_FH.append(ROOT.TH1F("hist_X_xhigh_2sigma_Valid_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_bulk_Valid_EE.append(ROOT.TH1F("hist_Y_bulk_Valid_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Valid_EE.append(ROOT.TH1F("hist_X_bulk_Valid_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_bulk_Valid_FH.append(ROOT.TH1F("hist_Y_bulk_Valid_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Valid_FH.append(ROOT.TH1F("hist_X_bulk_Valid_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    
    hist_Y_2sigma_Tbdata_EE.append(ROOT.TH1F("hist_Y_2sigma_Tbdata_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Tbdata_EE.append(ROOT.TH1F("hist_X_2sigma_Tbdata_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_2sigma_Tbdata_FH.append(ROOT.TH1F("hist_Y_2sigma_Tbdata_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_2sigma_Tbdata_FH.append(ROOT.TH1F("hist_X_2sigma_Tbdata_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_high_2sigma_Tbdata_EE.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Tbdata_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Tbdata_EE.append(ROOT.TH1F("hist_X_xhigh_2sigma_Tbdata_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_high_2sigma_Tbdata_FH.append(ROOT.TH1F("hist_Y_xhigh_2sigma_Tbdata_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_high_2sigma_Tbdata_FH.append(ROOT.TH1F("hist_X_xhigh_2sigma_Tbdata_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_bulk_Tbdata_EE.append(ROOT.TH1F("hist_Y_bulk_Tbdata_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Tbdata_EE.append(ROOT.TH1F("hist_X_bulk_Tbdata_EE_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_Y_bulk_Tbdata_FH.append(ROOT.TH1F("hist_Y_bulk_Tbdata_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))
    hist_X_bulk_Tbdata_FH.append(ROOT.TH1F("hist_X_bulk_Tbdata_FH_%s"%name1,"",xbin_X,xmin_X,xhigh_X))

    
#reading X,Y,Z coordinates 
hit_x="%s/Hit_X.pickle"%inpickle_folder
hit_xPickle = open(hit_x,"rb")
x =pickle.load(hit_xPickle)
hit_y ="%s/Hit_Y.pickle"%inpickle_folder
hit_yPickle = open(hit_y,"rb")
y =pickle.load(hit_yPickle)

hit_z ="%s/Hit_Z.pickle"%inpickle_folder
hit_zPickle = open(hit_z,"rb")
z =pickle.load(hit_zPickle)


##X,Y,Z for tb data
hit_x="%s/Hit_X.pickle"%tbfolder
hit_xPickle = open(hit_x,"rb")
x_tb =pickle.load(hit_xPickle)
hit_y ="%s/Hit_Y.pickle"%tbfolder
hit_yPickle = open(hit_y,"rb")
y_tb =pickle.load(hit_yPickle)

hit_z ="%s/Hit_Z.pickle"%tbfolder
hit_zPickle = open(hit_z,"rb")
z_tb =pickle.load(hit_zPickle)

energy_,Mean_,Sigma_ = np.loadtxt('./Fit_mean_sigma_4ipfeat_ep135.txt',usecols=(0,1, 2), unpack=True)
print(len(energy_))
print('after reading text file')


for i_en in range(len(trueEn_pkl)):
    hist_2d.Fill(trueEn_pkl[i_en],preds_trueEn[i_en])
for i_ev in range(len(tb_preds_trueEn)):
    hist_2d_tb.Fill(tb_trueEn_pkl[i_ev],tb_preds_trueEn[i_ev])

for i in range(len(valid_idx)):
    valid_trueEn=(trueEn_pkl[valid_idx[i]])
    valid_predEn=(preds_trueEn[valid_idx[i]])
    for ibin in range(8):
        if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn <=Energy[ibin]+2):
            hist_ssVsGNN_Valid[ibin].Fill(valid_predEn,SSLocation_arr_sim[valid_idx[i]])
            categ1 =  valid_predEn>(Mean_[ibin]+3.0*Sigma_[ibin])
            categ2 =  valid_predEn<(Mean_[ibin]-3.0*Sigma_[ibin])
            categ3 =  valid_predEn<=(Mean_[ibin]+3.0*Sigma_[ibin])
            if categ1:
                hist_sslocation_high_Valid[ibin].Fill(SSLocation_arr_sim[valid_idx[i]])
            elif categ2:
                hist_sslocation_Valid[ibin].Fill(SSLocation_arr_sim[valid_idx[i]])
            elif categ3:
                hist_sslocation_bulk_Valid[ibin].Fill(SSLocation_arr_sim[valid_idx[i]])
            Hit_X=x[valid_idx[i]]
            Hit_Z=z[valid_idx[i]]
            Hit_Y=y[valid_idx[i]]
            RechitEn_pkl1=RechitEn_pkl[valid_idx[i]]
            
            for j in range(len(Hit_Y)):
                if categ1:
                    hist_X_high_2sigma_Valid[ibin].Fill(Hit_X[j])
                    hist_Y_high_2sigma_Valid[ibin].Fill(Hit_Y[j])
                    hist_Z_high_2sigma_Valid[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_xhigh_2sigma_Valid[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_high_2sigma_Valid[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_high_2sigma_Valid[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_high_2sigma_Valid_EE[ibin].Fill(Hit_X[j])
                        hist_Y_high_2sigma_Valid_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_high_2sigma_Valid_FH[ibin].Fill(Hit_X[j])
                        hist_Y_high_2sigma_Valid_FH[ibin].Fill(Hit_Y[j])


                elif categ2:
                    hist_X_2sigma_Valid[ibin].Fill(Hit_X[j])
                    hist_Y_2sigma_Valid[ibin].Fill(Hit_Y[j])
                    hist_Z_2sigma_Valid[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_2sigma_Valid[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_2sigma_Valid[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_2sigma_Valid[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_2sigma_Valid_EE[ibin].Fill(Hit_X[j])
                        hist_Y_2sigma_Valid_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_2sigma_Valid_FH[ibin].Fill(Hit_X[j])
                        hist_Y_2sigma_Valid_FH[ibin].Fill(Hit_Y[j])

                elif categ3:
                    hist_X_bulk_Valid[ibin].Fill(Hit_X[j])
                    hist_Y_bulk_Valid[ibin].Fill(Hit_Y[j])
                    hist_Z_bulk_Valid[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_bulk_Valid[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_bulk_Valid[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_bulk_Valid[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_bulk_Valid_EE[ibin].Fill(Hit_X[j])
                        hist_Y_bulk_Valid_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_bulk_Valid_FH[ibin].Fill(Hit_X[j])
                        hist_Y_bulk_Valid_FH[ibin].Fill(Hit_Y[j])

                
for i in range(len(train_idx)):
    train_trueEn=(trueEn_pkl[train_idx[i]])
    train_predEn=(preds_trueEn[train_idx[i]])
    for ibin in range(8):
        if(train_trueEn>=Energy[ibin]-2 and train_trueEn <=Energy[ibin]+2):
            hist_ssVsGNN_Train[ibin].Fill(train_predEn,SSLocation_arr_sim[train_idx[i]])
            categ1 =  train_predEn>(Mean_[ibin+8]+3.0*Sigma_[ibin+8])
            categ2 =  train_predEn<(Mean_[ibin+8]-3.0*Sigma_[ibin+8])
            categ3 =  train_predEn<=(Mean_[ibin+8]+3.0*Sigma_[ibin+8])
            if categ1:
                hist_sslocation_high_Train[ibin].Fill(SSLocation_arr_sim[train_idx[i]])
            elif categ2:
                hist_sslocation_Train[ibin].Fill(SSLocation_arr_sim[train_idx[i]])
            elif categ3:
                hist_sslocation_bulk_Train[ibin].Fill(SSLocation_arr_sim[train_idx[i]])
            Hit_X=x[train_idx[i]]
            Hit_Z=z[train_idx[i]]
            Hit_Y=y[train_idx[i]]
            RechitEn_pkl1=RechitEn_pkl[train_idx[i]]


            for j in range(len(Hit_Y)):
                if categ1:
                    hist_X_high_2sigma_Train[ibin].Fill(Hit_X[j])
                    hist_Y_high_2sigma_Train[ibin].Fill(Hit_Y[j])
                    hist_Z_high_2sigma_Train[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_xhigh_2sigma_Train[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_high_2sigma_Train[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_high_2sigma_Train[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_high_2sigma_Train_EE[ibin].Fill(Hit_X[j])
                        hist_Y_high_2sigma_Train_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_high_2sigma_Train_FH[ibin].Fill(Hit_X[j])
                        hist_Y_high_2sigma_Train_FH[ibin].Fill(Hit_Y[j])

                elif categ2:
                    hist_X_2sigma_Train[ibin].Fill(Hit_X[j])
                    hist_Y_2sigma_Train[ibin].Fill(Hit_Y[j])
                    hist_Z_2sigma_Train[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_2sigma_Train[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_2sigma_Train[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_2sigma_Train[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_2sigma_Train_EE[ibin].Fill(Hit_X[j])
                        hist_Y_2sigma_Train_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_2sigma_Train_FH[ibin].Fill(Hit_X[j])
                        hist_Y_2sigma_Train_FH[ibin].Fill(Hit_Y[j])
                elif categ3:
                    hist_X_bulk_Train[ibin].Fill(Hit_X[j])
                    hist_Y_bulk_Train[ibin].Fill(Hit_Y[j])
                    hist_Z_bulk_Train[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_bulk_Train[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_bulk_Train[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_bulk_Train[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_bulk_Train_EE[ibin].Fill(Hit_X[j])
                        hist_Y_bulk_Train_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_bulk_Train_FH[ibin].Fill(Hit_X[j])
                        hist_Y_bulk_Train_FH[ibin].Fill(Hit_Y[j])


for i in range(len(tb_preds_trueEn)):
    valid_trueEn=(tb_trueEn_pkl[i])
    valid_predEn=(tb_preds_trueEn[i])
    for ibin in range(8):
        if(valid_trueEn>=Energy[ibin]-2 and valid_trueEn <=Energy[ibin]+2):
            hist_ssVsGNN_Tbdata[ibin].Fill(valid_predEn,SSLocation_arr_data[i])
            categ1 =  valid_predEn>(Mean_[ibin+16]+3.0*Sigma_[ibin+16])
            categ2 =  valid_predEn<(Mean_[ibin+16]-3.0*Sigma_[ibin+16])
            categ3 =  valid_predEn<=(Mean_[ibin+16]+3.0*Sigma_[ibin+16])
            if categ1:
                hist_sslocation_high_Tbdata[ibin].Fill(SSLocation_arr_data[i])
            elif categ2:
                hist_sslocation_Tbdata[ibin].Fill(SSLocation_arr_data[i])
            elif categ3:
                hist_sslocation_bulk_Tbdata[ibin].Fill(SSLocation_arr_data[i])
            Hit_X=x_tb[i]
            Hit_Z=z_tb[i]
            Hit_Y=y_tb[i]
            RechitEn_pkl1=RechitEn_tb_pkl[i]

            for j in range(len(Hit_Y)):
                if categ1:
                    hist_X_high_2sigma_Tbdata[ibin].Fill(Hit_X[j])
                    hist_Y_high_2sigma_Tbdata[ibin].Fill(Hit_Y[j])
                    hist_Z_high_2sigma_Tbdata[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_xhigh_2sigma_Tbdata[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_high_2sigma_Tbdata[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_high_2sigma_Tbdata[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_high_2sigma_Tbdata_EE[ibin].Fill(Hit_X[j])
                        hist_Y_high_2sigma_Tbdata_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_high_2sigma_Tbdata_FH[ibin].Fill(Hit_X[j])
                        hist_Y_high_2sigma_Tbdata_FH[ibin].Fill(Hit_Y[j])

                elif categ2:
                    hist_X_2sigma_Tbdata[ibin].Fill(Hit_X[j])
                    hist_Y_2sigma_Tbdata[ibin].Fill(Hit_Y[j])
                    hist_Z_2sigma_Tbdata[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_2sigma_Tbdata[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_2sigma_Tbdata[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_2sigma_Tbdata[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_2sigma_Tbdata_EE[ibin].Fill(Hit_X[j])
                        hist_Y_2sigma_Tbdata_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_2sigma_Tbdata_FH[ibin].Fill(Hit_X[j])
                        hist_Y_2sigma_Tbdata_FH[ibin].Fill(Hit_Y[j])

                elif categ3:
                    hist_X_bulk_Tbdata[ibin].Fill(Hit_X[j])
                    hist_Y_bulk_Tbdata[ibin].Fill(Hit_Y[j])
                    hist_Z_bulk_Tbdata[ibin].Fill(Hit_Z[j])
                    hist_rechitEn_bulk_Tbdata[ibin].Fill(RechitEn_pkl1[j])
                    hist_ZvsX_bulk_Tbdata[ibin].Fill(Hit_Z[j],Hit_X[j])
                    hist_ZvsY_bulk_Tbdata[ibin].Fill(Hit_Z[j],Hit_Y[j])
                    if(Hit_Z[j]<54):
                        hist_X_bulk_Tbdata_EE[ibin].Fill(Hit_X[j])
                        hist_Y_bulk_Tbdata_EE[ibin].Fill(Hit_Y[j])
                    elif(Hit_Z[j]<154 and Hit_Z[j]>54):
                        hist_X_bulk_Tbdata_FH[ibin].Fill(Hit_X[j])
                        hist_Y_bulk_Tbdata_FH[ibin].Fill(Hit_Y[j])

# QGSP_pred_v2 ="./QGSP/pred_tb.pickle"
# QGSP_predPickle = open(QGSP_pred_v2, "rb")
# print(QGSP_predPickle)
# QGSP_preds_trueEn = np.asarray(pickle.load(QGSP_predPickle))

# QGSP_trueEn= "/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/sim_QGSP_DiscreteEnergy/beamEn.pickle"
# QGSP_trueEnPickle = open(QGSP_trueEn,"rb")
# QGSP_trueEn_pkl = np.asarray(pickle.load(QGSP_trueEnPickle))
# print(QGSP_trueEn_pkl[200000])
# Energy=[20,50,80,100,120,200,250,300]
# for i in range(len(QGSP_trueEn_pkl)):
#     QGSP_trueEn=(QGSP_trueEn_pkl[i])
#     QGSP_predEn=(QGSP_preds_trueEn[i])
#     diff= QGSP_trueEn - QGSP_predEn
#     norm = diff/QGSP_trueEn
#     for ibin in range(8):
#         if(QGSP_trueEn==Energy[ibin]):
#             hist_pred_QGSP[ibin].Fill(QGSP_predEn)
#             hist_true_QGSP[ibin].Fill(QGSP_trueEn)
#             hist_predTrue_QGSP[ibin].Fill(diff)
#             hist_norm_predTrue_QGSP[ibin].Fill(norm)
# FTFP_pred_v2 ="./FTFP/pred_tb.pickle"
# FTFP_predPickle = open(FTFP_pred_v2, "rb")
# print(FTFP_predPickle)
# FTFP_preds_trueEn = np.asarray(pickle.load(FTFP_predPickle))

# FTFP_trueEn= "/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/sim_FTFP_DiscreteEnergy/beamEn.pickle"
# FTFP_trueEnPickle = open(FTFP_trueEn,"rb")
# FTFP_trueEn_pkl = np.asarray(pickle.load(FTFP_trueEnPickle))
# print(FTFP_trueEn_pkl[200000])
# Energy=[20,50,80,100,120,200,250,300]
# for i in range(len(FTFP_trueEn_pkl)):
#     FTFP_trueEn=(FTFP_trueEn_pkl[i])
#     FTFP_predEn=(FTFP_preds_trueEn[i])
#     diff= FTFP_trueEn - FTFP_predEn
#     norm = diff/FTFP_trueEn
#     for ibin in range(8):
#         if(FTFP_trueEn==Energy[ibin]):
#             hist_pred_FTFP[ibin].Fill(FTFP_predEn)
#             hist_true_FTFP[ibin].Fill(FTFP_trueEn)
#             hist_predTrue_FTFP[ibin].Fill(diff)
#             hist_norm_predTrue_FTFP[ibin].Fill(norm)
fout.cd()
for i in range(8):
    hist_X_2sigma_Valid[i].Write()
    hist_Y_2sigma_Valid[i].Write()
    hist_Z_2sigma_Valid[i].Write()
    hist_X_high_2sigma_Valid[i].Write()
    hist_Y_high_2sigma_Valid[i].Write()
    hist_Z_high_2sigma_Valid[i].Write()
    hist_rechitEn_2sigma_Valid[i].Write()
    hist_ZvsX_high_2sigma_Valid[i].Write()
    hist_ZvsY_high_2sigma_Valid[i].Write()
    hist_ZvsX_2sigma_Valid[i].Write()
    hist_ZvsY_2sigma_Valid[i].Write()
    hist_sslocation_Valid[i].Write()
    hist_ssVsGNN_Valid[i].Write()
    hist_rechitEn_xhigh_2sigma_Valid[i].Write()
    hist_X_2sigma_Train[i].Write()
    hist_Y_2sigma_Train[i].Write()
    hist_Z_2sigma_Train[i].Write()
    hist_X_high_2sigma_Train[i].Write()
    hist_Y_high_2sigma_Train[i].Write()
    hist_Z_high_2sigma_Train[i].Write()
    hist_rechitEn_2sigma_Train[i].Write()
    hist_sslocation_Train[i].Write()
    hist_ssVsGNN_Train[i].Write()
    hist_rechitEn_xhigh_2sigma_Train[i].Write()
    hist_ZvsX_high_2sigma_Train[i].Write()
    hist_ZvsY_high_2sigma_Train[i].Write()
    hist_ZvsX_2sigma_Train[i].Write()
    hist_ZvsY_2sigma_Train[i].Write()
    hist_X_2sigma_Tbdata[i].Write()
    hist_Y_2sigma_Tbdata[i].Write()
    hist_Z_2sigma_Tbdata[i].Write()

    hist_X_high_2sigma_Tbdata[i].Write()
    hist_Y_high_2sigma_Tbdata[i].Write()
    hist_Z_high_2sigma_Tbdata[i].Write()
    
    hist_rechitEn_2sigma_Tbdata[i].Write()
    hist_sslocation_Tbdata[i].Write()
    hist_ssVsGNN_Tbdata[i].Write()
    hist_ZvsX_high_2sigma_Tbdata[i].Write()
    hist_ZvsY_high_2sigma_Tbdata[i].Write()
    hist_ZvsX_2sigma_Tbdata[i].Write()
    hist_ZvsY_2sigma_Tbdata[i].Write()
    hist_rechitEn_xhigh_2sigma_Tbdata[i].Write()

    hist_X_bulk_Tbdata[i].Write()
    hist_Y_bulk_Tbdata[i].Write()
    hist_Z_bulk_Tbdata[i].Write()
    hist_rechitEn_bulk_Tbdata[i].Write()
    hist_sslocation_bulk_Tbdata[i].Write()
    hist_sslocation_high_Tbdata[i].Write()
    hist_ZvsX_bulk_Tbdata[i].Write()
    hist_ZvsY_bulk_Tbdata[i].Write()

    hist_X_bulk_Train[i].Write()
    hist_Y_bulk_Train[i].Write()
    hist_Z_bulk_Train[i].Write()
    hist_rechitEn_bulk_Train[i].Write()
    hist_sslocation_bulk_Train[i].Write()
    hist_sslocation_high_Train[i].Write()
    hist_ZvsX_bulk_Train[i].Write()
    hist_ZvsY_bulk_Train[i].Write()

    hist_X_bulk_Valid[i].Write()
    hist_Y_bulk_Valid[i].Write()
    hist_Z_bulk_Valid[i].Write()
    hist_rechitEn_bulk_Valid[i].Write()
    hist_sslocation_bulk_Valid[i].Write()
    hist_sslocation_high_Valid[i].Write()
    hist_ZvsX_bulk_Valid[i].Write()
    hist_ZvsY_bulk_Valid[i].Write()
    hist_X_bulk_Valid_EE[i].Write()
    hist_Y_bulk_Valid_FH[i].Write()
    hist_X_bulk_Valid_FH[i].Write()
    hist_Y_bulk_Valid_EE[i].Write()

    hist_X_2sigma_Valid_EE[i].Write()

    hist_Y_2sigma_Valid_EE[i].Write()
    hist_X_high_2sigma_Valid_EE[i].Write()
    hist_Y_high_2sigma_Valid_FH[i].Write()
    hist_X_high_2sigma_Valid_FH[i].Write()
    hist_Y_high_2sigma_Valid_EE[i].Write()

    hist_X_bulk_Train_FH[i].Write()
    hist_Y_bulk_Train_EE[i].Write()
    hist_X_2sigma_Train_FH[i].Write()
    hist_Y_2sigma_Train_EE[i].Write()
    hist_X_high_2sigma_Train_FH[i].Write()
    hist_Y_high_2sigma_Train_EE[i].Write()

    hist_X_bulk_Train_EE[i].Write()
    hist_Y_bulk_Train_FH[i].Write()
    hist_X_2sigma_Train_EE[i].Write()
    hist_Y_2sigma_Train_FH[i].Write()
    hist_X_high_2sigma_Train_EE[i].Write()
    hist_Y_high_2sigma_Train_FH[i].Write()
    hist_X_bulk_Tbdata_EE[i].Write()
    hist_Y_bulk_Tbdata_FH[i].Write()
    hist_X_2sigma_Tbdata_EE[i].Write()
    hist_Y_2sigma_Tbdata_FH[i].Write()
    hist_X_high_2sigma_Tbdata_EE[i].Write()
    hist_Y_high_2sigma_Tbdata_FH[i].Write()
    hist_X_bulk_Tbdata_FH[i].Write()
    hist_Y_bulk_Tbdata_EE[i].Write()
    hist_X_2sigma_Tbdata_FH[i].Write()
    hist_Y_2sigma_Tbdata_EE[i].Write()
    hist_X_high_2sigma_Tbdata_FH[i].Write()
    hist_Y_high_2sigma_Tbdata_EE[i].Write()

hist_2d.Write()
hist_2d_tb.Write()

fout.Close()
