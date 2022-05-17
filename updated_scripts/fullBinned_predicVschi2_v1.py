import uproot
from numba import jit
import numpy as np
import awkward as ak
from time import time
import pickle
import ROOT
folder = sys.argv[1]
folder2= sys.argv[4]
folder3= sys.argv[5]
print("input predictions",folder)
outfileName= sys.argv[2]
out_fname = '%s/%s'%(folder,outfileName)
print(out_fname, "output file is")
inpickle_folder =sys.argv[3]
print("input pickle files are picked from",inpickle_folder)
#inChi2 = sys.argv[5]

# path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/ntuple_chi2_reco_5M_firstFlatEn.root"
# treeName ="pion_variables_v1"
# outfolder ="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/ratio_updated/correct_inputs/epoch58_onwards/DownScale_Ahcal"
# tree = uproot.open("%s:%s"%(path, treeName))
# energyLostEE_ = tree['energyLostEE'].array()
# print(energyLostEE_)
# energyLostFH_ = tree['energyLostFH'].array()
# print(energyLostFH_)
# energyLostBH_ = tree['energyLostBH'].array()
# print(energyLostBH_)
# beamEnergy = tree['beamEnergy'].array()
# trueBeamEnergy = tree['trueBeamEnergy'].array()
# energyLeakTransverseEE_ =tree['energyLeakTransverseEE'].array()
# energyLeakTransverseFH_ =tree['energyLeakTransverseFH'].array()
# energyLeakTransverseAH_ =tree['energyLeakTransverseAH'].array()
# energyLeakLongitudinal_=tree['energyLeakLongitudinal'].array()
# energyLeakResidual_= tree['energyLeakResidual'].array()
# energyLeak_FH_AH_ = tree['energyLeak_FH_AH'].array()
# energyLeak_EE_FH_ = tree['energyLeak_EE_FH'].array()


# total_leakage = energyLeakTransverseEE_ +energyLeakTransverseFH_+energyLeakTransverseAH_+energyLeakResidual_+energyLeakLongitudinal_+energyLeak_FH_AH_+energyLeak_EE_FH_
# frac_leak = total_leakage/trueBeamEnergy
# path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/hadintmoreInfo/skimmed_ntuple_sim_discrete_chi2method01_3inOne.root"
# tree1 = uproot.open("%s:%s"%(path, treeName))

rechit_shower_start_layer=tree['rechit_shower_start_layer'].array()
#trimAhcal_chi2Reco = tree['trimAhcal_chi2Reco'].array()
trimAhcal_chi2Reco = np.asarray(pickle.load(open(folder,"rb"))) #tree['trimAhcal_chi2Reco'].array()                                                                                                                              
trimAhcal_chi2Reco[trimAhcal_chi2Reco>3]=3
trueBeamEnergy = beamEnergy
pred_v2 =folder2 #"./aggre2_maxlr6e04_75ep/constlr_2feat/valid_flat/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])
predic=preds_ratio#[0:836658]                                                                                                                
print(len(predic))


# pred_v2 ="./valid_flat/pred_tb.pickle"
# predPickle = open(pred_v2, "rb")
# preds_ratio = np.asarray(pickle.load(predPickle))
# print(preds_ratio[preds_ratio>3])
# preds_ratio[preds_ratio>3] = 3
# print(preds_ratio[preds_ratio>3])
# predic=preds_ratio
# print(len(predic))#pkl_Sim_50100300_hadinfor/trimAhcal
RechitEn ="%s/recHitEn.pickle"%inpickle_folder
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
#rawE= rawE[0:836658]
Pred_ = rawE * predic
trimAhcal_chi2Reco_v1 =rawE*trimAhcal_chi2Reco
trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_0to5M_fix_raw_ahcalTrim_up/trueE.pickle"

trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
#print(trueEn_pkl[0:836658])
#trueEn_pkl=trueEn_pkl[0:836658]

valid_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/Correct_training_5M/trimAhcal/NSM_infer/all_valididx.pickle"
train_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/Correct_training_5M/trimAhcal/NSM_infer/all_trainidx.pickle"
valid_idx_f = open(valid_idx_file,"rb")
valid_idx = np.asarray(pickle.load(valid_idx_f))
print(len(valid_idx))
#valid_idx[valid_idx>836658]
train_idx_f = open(train_idx_file,"rb")
train_idx = np.asarray(pickle.load(train_idx_f))
print(len(train_idx))

#print(numpy.array_equal(trueEn_pkl,trueBeamEnergy))

fout= ROOT.TFile(out_fname, 'RECREATE')
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
h1d_predModel_train=[]
h1d_predModel_valid=[]

h1d_predchi2=[]
h1d_predModel=[]

h1d_totalAbs=[]
h1d_fracAbs=[]
h1d_totalLeak=[]
h1d_fracleak=[]
h1d_fracleak_total=[]
h1d_fracvisi_true=[]
h2d_fracleakVsabs=[]
h2d_fracleakVsabs_v1=[]

Energy=[20,50,80,100,120,200,250,300]

for i_hist in range(len(Energy)):
    if(Energy[i_hist]<100):
        xhigh_pred = 4.0*Energy[i_hist]
    else:
        xhigh_pred= 3.0*Energy[i_hist]

    xhigh_true= 2.0*Energy[i_hist]
    #xhigh_pred = 3.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(Energy[i_hist])
    h1d_totalAbs.append(ROOT.TH1F("h1d_totalAbs_%s"%name1,"",500, 0, xhigh_pred))
    h1d_fracAbs.append(ROOT.TH1F("h1d_fracAbs_%s"%name1,"",500,0,2))
    h1d_totalLeak.append(ROOT.TH1F("h1d_totalLeak_%s"%name1,"",500,0,xhigh_pred))
    h1d_fracleak.append(ROOT.TH1F("h1d_fracleak_%s"%name1,"",500,0,2))
    h1d_fracleak_total.append(ROOT.TH1F("h1d_fracleak_total_%s"%name1,"",500,0,2))
    h1d_fracvisi_true.append(ROOT.TH1F("h1d_fracvisi_true_%s"%name1,"",500,0,2))
    h2d_fracleakVsabs.append(ROOT.TH2F("h2d_fracleakVsabs_%s"%name1,"",500,0,2,500,0,2))
    h2d_fracleakVsabs_v1.append(ROOT.TH2F("h2d_fracleakVsabs_v1_%s"%name1,"",500,0,2,500,0,2))
    
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
    h1d_predModel_train.append(ROOT.TH1F("h1d_predModel_train_%s"%name1,"h1d_predModel_train2",500,0,xhigh_pred))
    h1d_predModel_valid.append(ROOT.TH1F("h1d_predModel_valid_%s"%name1,"h1d_predModel_valid2",500,0,xhigh_pred))

#for i_en in range(len(Energy)):
# for i in range(len(train_idx)):
#     if(trueEn_pkl[train_idx[i]]>=(Energy[i_en]-2) and trueEn_pkl[train_idx[i]]<=(Energy[i_en]+2) ):
#         h1d_predModel_train[i_en].Fill(Pred_[train_idx[i]])
# for i in range(len(valid_idx)):
#     if(trueEn_pkl[valid_idx[i]]>=(Energy[i_en]-2) and trueEn_pkl[valid_idx[i]]<=(Energy[i_en]+2) ):
#         h1d_predModel_valid[i_en].Fill(Pred_[valid_idx[i]])

                                       
for i_en in range(len(Energy)):

    for i in range(len(trueEn_pkl)):
        if(trueEn_pkl[i]>=(Energy[i_en]-2) and trueEn_pkl[i]<=(Energy[i_en]+2) ):
            h2d_trueEnvstrueEn[i_en].Fill(trueEn_pkl[i],trueBeamEnergy[i])
            h2d_chi2vs_model[i_en].Fill(Pred_[i],trimAhcal_chi2Reco_v1[i])
            h2d_chi2vs_true[i_en].Fill(trueBeamEnergy[i],trimAhcal_chi2Reco_v1[i])
            h2d_modelvstrue[i_en].Fill(trueBeamEnergy[i],Pred_[i])
            total_lost = energyLostEE_[i]+energyLostFH_[i]+energyLostBH_[i]
            h2d_chi2vs_absorb[i_en].Fill(trimAhcal_chi2Reco_v1[i],total_lost)
            h2d_modelvs_absorb[i_en].Fill(Pred_[i],total_lost)
            ratio_gnn= Pred_[i]/trueEn_pkl[i]
            ratio_chi2 = trimAhcal_chi2Reco_v1[i]/trueEn_pkl[i]
            ratio_abs = total_lost/trueEn_pkl[i]
            h2d_ratioModel_vsAbsorb[i_en].Fill(ratio_gnn,ratio_abs)
            h2d_ratiochi2_vsAbsorb[i_en].Fill(ratio_chi2,ratio_abs)
            norm_gnn = (Pred_[i] - trimAhcal_chi2Reco_v1[i])/trueEn_pkl[i]            
            h2d_normchi2_vsAbsorb[i_en].Fill(norm_gnn,ratio_abs);
            norm_gnn = (Pred_[i] -trueEn_pkl[i])/trueEn_pkl[i]
            norm_chi2 = (trimAhcal_chi2Reco_v1[i]-trueEn_pkl[i])/trueEn_pkl[i]

            h1d_normchi2[i_en].Fill(norm_chi2)
            h1d_normModel[i_en].Fill(norm_gnn)
            h1d_predchi2[i_en].Fill(trimAhcal_chi2Reco_v1[i])
            h1d_predModel[i_en].Fill(Pred_[i])
            # frac_abs = total_lost/Energy[i_en]
            # frac_leak_v1 = total_leakage[i]/(total_lost+total_leakage[i])
            # h1d_totalAbs[i_en].Fill(total_lost)
            # h1d_fracAbs[i_en].Fill(frac_abs)
            # h1d_totalLeak[i_en].Fill(total_leakage[i])
            # h1d_fracleak[i_en].Fill(frac_leak[i])
            # h1d_fracleak_total[i_en].Fill(frac_leak_v1)
            # h1d_fracvisi_true[i_en].Fill((total_lost+total_leakage[i])/Energy[i_en])
            # h2d_fracleakVsabs[i_en].Fill(frac_abs,frac_leak[i])
            # h2d_fracleakVsabs_v1[i_en].Fill(frac_abs,frac_leak_v1 )


# hist_chi2_pred_categ1=[]
# hist_chi2_pred_categ2=[]
# hist_chi2_norm_predTrue_categ1=[]
# hist_chi2_norm_predTrue_categ2=[]
# hist_chi2_pred_categ3=[]
# hist_chi2_norm_predTrue_categ3=[]


# hist_model_pred_categ1=[]
# hist_model_pred_categ2=[]
# hist_model_norm_predTrue_categ1=[]
# hist_model_norm_predTrue_categ2=[]
# hist_model_pred_categ3=[]
# hist_model_norm_predTrue_categ3=[]

# hist_model_pred_categ4=[]
# hist_model_norm_predTrue_categ4=[]
# hist_chi2_pred_categ4=[]
# hist_chi2_norm_predTrue_categ4=[]

# h2d_fracleakVsAbs_categ1=[]
# h2d_fracleakVsAbsv1_categ1=[]
# h2d_fracleakVsAbs_categ2=[]
# h2d_fracleakVsAbsv1_categ2=[]
# h2d_fracleakVsAbs_categ3=[]
# h2d_fracleakVsAbs_categ4=[]
# h2d_fracleakVsAbsv1_categ4=[]
# h2d_fracleakVsAbsv1_categ3=[]

# h2d_fracLeakVsGnn_categ1=[]
# h2d_fracLeakVsGnn_categ2=[]
# h2d_fracLeakVsGnn_categ3=[]
# h2d_fracLeakVsGnn_categ4=[]

# h2d_fracLeakVsChi2_categ1=[]
# h2d_fracLeakVsChi2_categ2=[]
# h2d_fracLeakVsChi2_categ3=[]
# h2d_fracLeakVsChi2_categ4=[]

# M=8

# for i_hist in range(len(Energy)):
#     xhigh_pred= 3.0*Energy[i_hist]
#     xhigh_true= 2.0*Energy[i_hist]
#     xhigh_diff= 20
#     xlow_diff= -20
#     xhigh_norm= 5
#     xlow_norm= -5
#     name1='TrueEn_%i' %(Energy[i_hist])#,u[i_hist],v[i_hist],typee[i_hist])                                                                             
#     h2d_fracleakVsAbs_categ1.append(ROOT.TH2F('h2d_fracleakVsAbs_categ1_%s'%name1,"h2d_fracleakVsAbs_categ1",500,0,2,500,0,2))
#     h2d_fracleakVsAbsv1_categ1.append(ROOT.TH2F('h2d_fracleakVsAbsv1_categ1_%s'%name1,"h2d_fracleakVsAbsv1_categ1",500,0,2,500,0,2))
#     h2d_fracleakVsAbs_categ2.append(ROOT.TH2F('h2d_fracleakVsAbs_categ2_%s'%name1,"h2d_fracleakVsAbs_categ2",500,0,2,500,0,2))
#     h2d_fracleakVsAbsv1_categ2.append(ROOT.TH2F('h2d_fracleakVsAbsv1_categ2_%s'%name1,"h2d_fracleakVsAbsv1_categ2",500,0,2,500,0,2))
#     h2d_fracleakVsAbs_categ3.append(ROOT.TH2F('h2d_fracleakVsAbs_categ3_%s'%name1,"h2d_fracleakVsAbs_categ3",500,0,2,500,0,2))
#     h2d_fracleakVsAbs_categ4.append(ROOT.TH2F('h2d_fracleakVsAbs_categ4_%s'%name1,"h2d_fracleakVsAbs_categ4",500,0,2,500,0,2))
#     h2d_fracleakVsAbsv1_categ4.append(ROOT.TH2F('h2d_fracleakVsAbsv1_categ4_%s'%name1,"h2d_fracleakVsAbsv1_categ4",500,0,2,500,0,2))
#     h2d_fracleakVsAbsv1_categ3.append(ROOT.TH2F('h2d_fracleakVsAbsv1_categ3_%s'%name1,"h2d_fracleakVsAbsv1_categ3",500,0,2,500,0,2))
    
#     h2d_fracLeakVsGnn_categ1.append(ROOT.TH2F('h2d_fracLeakVsGnn_categ1_%s'%name1,"h2d_fracLeakVsGnn_categ1",500, 0, xhigh_pred,500,0,2))
#     h2d_fracLeakVsGnn_categ2.append(ROOT.TH2F('h2d_fracLeakVsGnn_categ2_%s'%name1,"h2d_fracLeakVsGnn_categ2",500, 0, xhigh_pred,500,0,2))
#     h2d_fracLeakVsGnn_categ3.append(ROOT.TH2F('h2d_fracLeakVsGnn_categ3_%s'%name1,"h2d_fracLeakVsGnn_categ3",500, 0, xhigh_pred,500,0,2))
#     h2d_fracLeakVsGnn_categ4.append(ROOT.TH2F('h2d_fracLeakVsGnn_categ4_%s'%name1,"h2d_fracLeakVsGnn_categ4",500, 0, xhigh_pred,500,0,2))
    
#     h2d_fracLeakVsChi2_categ1.append(ROOT.TH2F('h2d_fracLeakVsChi2_categ1_%s'%name1,"h2d_fracLeakVsChi2_categ1",500, 0, xhigh_pred,500,0,2))
#     h2d_fracLeakVsChi2_categ2.append(ROOT.TH2F('h2d_fracLeakVsChi2_categ2_%s'%name1,"h2d_fracLeakVsChi2_categ2",500, 0, xhigh_pred,500,0,2))
#     h2d_fracLeakVsChi2_categ3.append(ROOT.TH2F('h2d_fracLeakVsChi2_categ3_%s'%name1,"h2d_fracLeakVsChi2_categ3",500, 0, xhigh_pred,500,0,2))
#     h2d_fracLeakVsChi2_categ4.append(ROOT.TH2F('h2d_fracLeakVsChi2_categ4_%s'%name1,"h2d_fracLeakVsChi2_categ4",500, 0, xhigh_pred,500,0,2))
    

#     hist_chi2_pred_categ1.append(ROOT.TH1F('chi2_categ1_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
#     hist_chi2_pred_categ2.append(ROOT.TH1F('chi2_categ2_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
#     hist_chi2_norm_predTrue_categ1.append(ROOT.TH1F('chi2_categ1_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
#     hist_chi2_norm_predTrue_categ2.append(ROOT.TH1F('chi2_categ2_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
#     hist_chi2_pred_categ3.append(ROOT.TH1F('chi2_categ3_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
#     hist_chi2_norm_predTrue_categ3.append(ROOT.TH1F('chi2_categ3_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))

#     hist_chi2_pred_categ4.append(ROOT.TH1F('chi2_categ4_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
#     hist_chi2_norm_predTrue_categ4.append(ROOT.TH1F('chi2_categ4_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))
#     hist_model_pred_categ4.append(ROOT.TH1F('model_categ4_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
#     hist_model_norm_predTrue_categ4.append(ROOT.TH1F('model_categ4_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))

#     hist_model_pred_categ1.append(ROOT.TH1F('model_categ1_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
#     hist_model_pred_categ2.append(ROOT.TH1F('model_categ2_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
#     hist_model_norm_predTrue_categ1.append(ROOT.TH1F('model_categ1_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
#     hist_model_norm_predTrue_categ2.append(ROOT.TH1F('model_categ2_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
#     hist_model_pred_categ3.append(ROOT.TH1F('model_categ3_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
#     hist_model_norm_predTrue_categ3.append(ROOT.TH1F('model_categ3_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))

# #total_abs=(energyLostEE_[trueEn_pkl==50]+energyLostFH_[trueEn_pkl==50]+energyLostBH_[trueEn_pkl==50])/50

# mean_=[]
# stdDev =[]


# for i in range(len(Energy)):
#     total_abs=(energyLostEE_[np.logical_and(trueEn_pkl>=Energy[i]-2 ,trueEn_pkl<=Energy[i]+2)]+energyLostFH_[np.logical_and(trueEn_pkl>=Energy[i]-2 ,trueEn_pkl<=Energy[i]+2)]+energyLostBH_[np.logical_and(trueEn_pkl>=Energy[i]-2 ,trueEn_pkl<=Energy[i]+2)])/Energy[i]
#     mean_.append(np.mean(total_abs))
#     stdDev.append(np.std(total_abs))
#     print(np.mean(total_abs),np.std(total_abs))

# # total_abs=(energyLostEE_[trueEn_pkl==100]+energyLostFH_[trueEn_pkl==100]+energyLostBH_[trueEn_pkl==100])/100
# # mean_.append(np.mean(total_abs))
# # stdDev.append(np.std(total_abs))
# # print(np.mean(total_abs),np.std(total_abs))

# # total_abs=(energyLostEE_[trueEn_pkl==300]+energyLostFH_[trueEn_pkl==300]+energyLostBH_[trueEn_pkl==300])/300
# # mean_.append(np.mean(total_abs))
# # stdDev.append(np.std(total_abs))
# # print(np.mean(total_abs),np.std(total_abs))



# for i in range(len(trueEn_pkl)):
#     total_lost = energyLostEE_[i]+energyLostFH_[i]+energyLostBH_[i]
#     ratio = total_lost/trueEn_pkl[i]
#     norm_gnn = (Pred_[i] -trueEn_pkl[i])/trueEn_pkl[i]
#     norm_chi2 = (trimAhcal_chi2Reco[i]-trueEn_pkl[i])/trueEn_pkl[i]
#     for ibin in range(len(Energy)):
#         if(trueEn_pkl[i]>=(Energy[ibin]-2) and trueEn_pkl[i]<=(Energy[ibin]+2 )):
#             categ1 = ratio>(mean_[ibin]-stdDev[ibin]) and ratio<(mean_[ibin]+stdDev[ibin])
#             categ2 = ratio>(mean_[ibin]+stdDev[ibin])
#             categ3 = ratio<(mean_[ibin]-stdDev[ibin]) and ratio>(mean_[ibin]-2.0*stdDev[ibin])
#             categ4= ratio<(mean_[ibin]-2.0*stdDev[ibin])
#             #frac_leak_v1 = total_leakage[i]/(total_lost+total_leakage[i])
#             if(categ1):
#                 hist_chi2_pred_categ1[ibin].Fill(trimAhcal_chi2Reco[i])
#                 hist_model_pred_categ1[ibin].Fill(Pred_[i])
#                 hist_chi2_norm_predTrue_categ1[ibin].Fill(norm_chi2)
#                 hist_model_norm_predTrue_categ1[ibin].Fill(norm_gnn)
#                 #    h2d_fracleakVsAbs_categ1[ibin].Fill(ratio,frac_leak[i])
#                 #     h2d_fracleakVsAbsv1_categ1[ibin].Fill(ratio,frac_leak_v1)
#                 #     h2d_fracLeakVsGnn_categ1[ibin].Fill(Pred_[i],frac_leak[i])
#                 #     h2d_fracLeakVsChi2_categ1[ibin].Fill(trimAhcal_chi2Reco[i],frac_leak[i])
            
#             elif(categ2):
#                 hist_model_pred_categ2[ibin].Fill(Pred_[i])
#                 hist_chi2_pred_categ2[ibin].Fill(trimAhcal_chi2Reco[i])
#                 hist_chi2_norm_predTrue_categ2[ibin].Fill(norm_chi2)
#                 hist_model_norm_predTrue_categ2[ibin].Fill(norm_gnn)
#                 # h2d_fracleakVsAbs_categ2[ibin].Fill(ratio,frac_leak[i])
#                 # h2d_fracleakVsAbsv1_categ2[ibin].Fill(ratio,frac_leak_v1)
#                 # h2d_fracLeakVsGnn_categ2[ibin].Fill(Pred_[i],frac_leak[i])
#                 # h2d_fracLeakVsChi2_categ2[ibin].Fill(trimAhcal_chi2Reco[i],frac_leak[i])

#             elif(categ3):
#                 hist_model_pred_categ3[ibin].Fill(Pred_[i])
#                 hist_chi2_pred_categ3[ibin].Fill(trimAhcal_chi2Reco[i])
#                 hist_chi2_norm_predTrue_categ3[ibin].Fill(norm_chi2)
#                 hist_model_norm_predTrue_categ3[ibin].Fill(norm_gnn)
#                 # h2d_fracleakVsAbs_categ3[ibin].Fill(ratio,frac_leak[i])
#                 # h2d_fracleakVsAbsv1_categ3[ibin].Fill(ratio,frac_leak_v1)
#                 # h2d_fracLeakVsGnn_categ3[ibin].Fill(Pred_[i],frac_leak[i])
#                 # h2d_fracLeakVsChi2_categ3[ibin].Fill(trimAhcal_chi2Reco[i],frac_leak[i])
#             elif(categ4):
#                 hist_model_pred_categ4[ibin].Fill(Pred_[i])
#                 hist_chi2_pred_categ4[ibin].Fill(trimAhcal_chi2Reco[i])
#                 hist_chi2_norm_predTrue_categ4[ibin].Fill(norm_chi2)
#                 hist_model_norm_predTrue_categ4[ibin].Fill(norm_gnn)
#                 # h2d_fracleakVsAbs_categ4[ibin].Fill(ratio,frac_leak[i])
#                 # h2d_fracleakVsAbsv1_categ4[ibin].Fill(ratio,frac_leak_v1)
#                 # h2d_fracLeakVsGnn_categ4[ibin].Fill(Pred_[i],frac_leak[i])
#                 # h2d_fracLeakVsChi2_categ4[ibin].Fill(trimAhcal_chi2Reco[i],frac_leak[i])


fout.cd()
for i_en in range(len(Energy)):
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
    # hist_chi2_pred_categ1[i_en].Write()
    # hist_chi2_norm_predTrue_categ1[i_en].Write()
    # hist_chi2_pred_categ3[i_en].Write()
    # hist_chi2_norm_predTrue_categ3[i_en].Write()
    # hist_model_norm_predTrue_categ1[i_en].Write()
    # hist_model_pred_categ1[i_en].Write()

    # hist_model_norm_predTrue_categ3[i_en].Write()
    # hist_model_pred_categ3[i_en].Write()
    # hist_chi2_pred_categ2[i_en].Write()
    # hist_chi2_norm_predTrue_categ2[i_en].Write()
    # hist_model_norm_predTrue_categ2[i_en].Write()
    # hist_model_pred_categ2[i_en].Write()
    # hist_chi2_pred_categ4[i_en].Write()
    # hist_chi2_norm_predTrue_categ4[i_en].Write()
    # hist_model_norm_predTrue_categ4[i_en].Write()
    # hist_model_pred_categ4[i_en].Write()

    # h1d_predchi2[i_en].Write()
    # h1d_predModel[i_en].Write()
    # h1d_predModel_train[i_en].Write()
    # h1d_predModel_valid[i_en].Write()
    # h1d_totalAbs[i_en].Write()
    # h1d_fracAbs[i_en].Write()
    # h1d_totalLeak[i_en].Write()
    # h1d_fracleak[i_en].Write()
    # h1d_fracleak_total[i_en].Write()
    # h1d_fracvisi_true[i_en].Write()
    # h2d_fracleakVsabs[i_en].Write()
    # h2d_fracleakVsabs_v1[i_en].Write()
    # h2d_fracleakVsAbs_categ1[i_en].Write()
    # h2d_fracleakVsAbsv1_categ1[i_en].Write()
    # h2d_fracleakVsAbs_categ2[i_en].Write()
    # h2d_fracleakVsAbsv1_categ2[i_en].Write()
    # h2d_fracleakVsAbs_categ3[i_en].Write()
    # h2d_fracleakVsAbs_categ4[i_en].Write()
    # h2d_fracleakVsAbsv1_categ4[i_en].Write()
    # h2d_fracleakVsAbsv1_categ3[i_en].Write()

    # h2d_fracLeakVsGnn_categ1[i_en].Write()
    # h2d_fracLeakVsGnn_categ2[i_en].Write()
    # h2d_fracLeakVsGnn_categ3[i_en].Write()
    # h2d_fracLeakVsGnn_categ4[i_en].Write()

    # h2d_fracLeakVsChi2_categ1[i_en].Write()
    # h2d_fracLeakVsChi2_categ2[i_en].Write()
    # h2d_fracLeakVsChi2_categ3[i_en].Write()
    # h2d_fracLeakVsChi2_categ4[i_en].Write()

fout.Close()


