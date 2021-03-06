import uproot
from numba import jit
import numpy as np
import awkward as ak
from time import time
import pickle
import ROOT
import math
import sys
folder = sys.argv[1]
print("input predictions",folder)
outfileName= sys.argv[2]
out_fname = './%s/%s'%(folder,outfileName)
print(out_fname, "output file is")
inpickle_folder =sys.argv[3]
print("input pickle files are picked from",inpickle_folder)

pred_v2 ="./%s/DiscSim_leakageInfo/pred_tb.pickle" %folder

path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/hadintmoreInfo/cleanedSamples/skimmed_ntuple_sim_discrete_pionHadInfor_08.root"
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
energyLeakTransverseEE_ =tree['energyLeakTransverseEE'].array()
energyLeakTransverseFH_ =tree['energyLeakTransverseFH'].array()
energyLeakTransverseAH_ =tree['energyLeakTransverseAH'].array()
energyLeakLongitudinal_=tree['energyLeakLongitudinal'].array()
energyLeakResidual_= tree['energyLeakResidual'].array()
energyLeak_FH_AH_ = tree['energyLeak_FH_AH'].array()
energyLeak_EE_FH_ = tree['energyLeak_EE_FH'].array()


total_leakage = energyLeakTransverseEE_ +energyLeakTransverseFH_+energyLeakTransverseAH_+energyLeakResidual_+energyLeakLongitudinal_+energyLeak_FH_AH_+energyLeak_EE_FH_
Total_EE= energyLeakTransverseEE_+energyLeak_EE_FH_
Total_FH= energyLeakTransverseFH_+energyLeak_FH_AH_


frac_leak = total_leakage/trueBeamEnergy
path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/hadintmoreInfo/cleanedSamples/skimmed_ntuple_sim_discrete_chi2method01_0009.root"
tree1 = uproot.open("%s:%s"%(path, treeName))

rechit_shower_start_layer=tree1['rechit_shower_start_layer'].array()
trimAhcal_chi2Reco = tree1['trimAhcal_chi2Reco'].array()

# path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/hadintmoreInfo/pi0/allinOne_pi0_ntuple_.root"
# tree2 = uproot.open("%s:%s"%(path, treeName))

# En_activeLayer= tree2['rechit_En_GeV'].array()
# npi0 = tree2['npi0'].array()
# total_pi0KE =  tree2['total_pi0KE'].array()
# leadPi0_KE = tree2['leadPi0_KE'].array()


#pred_v2 ="./Hadinfor_inferen/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])
predic=preds_ratio
print(len(predic))#pkl_Sim_50100300_hadinfor/trimAhcal
RechitEn ="%s/recHitEn.pickle"%inpickle_folder
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
#rawE= rawE[0:836658]
Pred_ = rawE * predic
trueEn ="%s/beamEn.pickle"%inpickle_folder
trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
print(trueEn_pkl[0:836658])
#trueEn_pkl=trueEn_pkl[0:836658]

# Hit_X ="%s/trimAhcal/Hit_X.pickle"
# Hit_XPickle = open(Hit_X,"rb")
# Hit_X_pkl =pickle.load(Hit_XPickle)
# print(Hit_X_pkl)

# Hit_Y ="%s/trimAhcal/Hit_Y.pickle"
# Hit_YPickle = open(Hit_Y,"rb")
# Hit_Y_pkl =pickle.load(Hit_YPickle)

# print(Hit_Y_pkl)
# Hit_Z ="%s/trimAhcal/Hit_Z.pickle"
# Hit_ZPickle = open(Hit_Z,"rb")
# Hit_Z_pkl =pickle.load(Hit_ZPickle)
# print(Hit_Z_pkl)


# valid_idx_file="%s/all_valididx.pickle"
# train_idx_file="%s/all_trainidx.pickle"
# valid_idx_f = open(valid_idx_file,"rb")
# valid_idx = np.asarray(pickle.load(valid_idx_f))
# print(len(valid_idx))
# #valid_idx[valid_idx>836658]
# train_idx_f = open(train_idx_file,"rb")
# train_idx = np.asarray(pickle.load(train_idx_f))
# print(len(train_idx))

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
h2d_fracleakVsGnn=[]
h2d_fracleakVsChi2=[]
h1d_lekageEE=[]
h1d_lekageFH=[]
h1d_lekageAH=[]
h1d_beforeEE=[]
h1d_long=[]
Energy=[20,50,80,100,120,200,250,300]
h1d_SSloc=[]
h1d_SSloc_v1=[]
h2d_leakvsGnn=[]
h2d_fracleakvsGnn=[]
h2d_vfracleakvsGnn=[]
h2d_leakvsChi2=[]
baseline = ["beforeEE","lekageEE","lekageFH","lekageAH","longi"]
baseline1 = ["stdDevEE","stdDevFH","stdDevAH"]
baseline2 = ["leak_stdDevEE","leak_stdDevFH","leak_stdDevAh","leakStdAh_longi","stdDevEE_beforeEE"]
h2d_gnnvsStdDev=[]
h2d_chi2vsStdDev=[]
h2d_stdDevvsleak=[]
h1d_totalleak=[]
h1d_fracleak=[]
h2d_gnnVspi0=[]
h2d_chi2vspi0=[]
h2d_Absvspi0=[]
h2d_pi0vsLeak=[]
h2d_pi0vsstdDev=[]
h2d_fracAbs_pi0=[]
h1d_totalVisi=[]
h1d_fractotalVisi=[]
for i_hist in range(len(Energy)):
    if(Energy[i_hist]<100):
        xhigh_pred = 4.0*Energy[i_hist]
    else:
        xhigh_pred= 3.0*Energy[i_hist]

    print(xhigh_pred)
    xhigh = xhigh_pred
    xhigh_true= 2.0*Energy[i_hist]
    #xhigh_pred = 3.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(Energy[i_hist])
    #xhigh = 3.0*Energy[i_hist]
    h1d_fractotalVisi.append(ROOT.TH1F("h1d_fractotalVisi_%s"%name1,"",500,0,2))
    h1d_totalVisi.append(ROOT.TH1F("h1d_totalVisi_%s"%name1,"",500,0,xhigh))
    h2d_gnnVspi0.append(ROOT.TH2F("h2d_gnnVspi0_%s"%name1,"",500,0,xhigh,500,0,Energy[i_hist]+20))
    h2d_chi2vspi0.append(ROOT.TH2F("h2d_chi2vspi0_%s"%name1,"",500,0,xhigh,500,0,Energy[i_hist]+20))
    h2d_Absvspi0.append(ROOT.TH2F("h2d_Absvspi0_%s"%name1,"",500,0,Energy[i_hist]+20,500,0,Energy[i_hist]+20))
    h2d_fracAbs_pi0.append(ROOT.TH2F("h2d_fracAbsvspi0_%s"%name1,"frac Abs vs total_pi0/abs",500,0,2,500,0,2))
    h1d_fracleak.append(ROOT.TH1F("h1d_fracLeak_%s"%name1,"",500,0,1.1))
    h1d_totalleak.append(ROOT.TH1F("h1d_totalleak_%s"%name1,"",500,0,Energy[i_hist]+20))
    xhigh = Energy[i_hist]
    h1d_lekageEE.append(ROOT.TH1F("h1d_lekageEE_%s"%name1,"",500,0,xhigh))
    h1d_lekageFH.append(ROOT.TH1F("h1d_lekageFH_%s"%name1,"",500,0,xhigh))
    h1d_lekageAH.append(ROOT.TH1F("h1d_lekageAH_%s"%name1,"",500,0,xhigh))
    h1d_beforeEE.append(ROOT.TH1F("h1d_beforeEE_%s"%name1,"",500,0,xhigh))
    h1d_long.append(ROOT.TH1F("h1d_long_%s"%name1,"",500,0,xhigh))
    h1d_SSloc.append(ROOT.TH1F("h1d_SSloc_%s"%name1,"",50,0,50))
    h1d_SSloc_v1.append(ROOT.TH1F("h1d_SSloc_v1_%s"%name1,"",50,0,50))
    h2d_fracleakVsGnn.append(ROOT.TH2F("h2d_fracleakVsGnn_%s"%name1,"",500, 0, xhigh_pred,500,0,2))
    h2d_fracleakVsChi2.append(ROOT.TH2F("h2d_fracleakVsChi2_%s"%name1,"",500, 0, xhigh_pred,500,0,2))
    h1d_totalAbs.append(ROOT.TH1F("h1d_totalAbs_%s"%name1,"",500, 0, xhigh_pred))
    h1d_fracAbs.append(ROOT.TH1F("h1d_fracAbs_%s"%name1,"",500,0,2))
    h1d_totalLeak.append(ROOT.TH1F("h1d_totalLeak_%s"%name1,"",500,0,xhigh))
    #h1d_fracleak.append(ROOT.TH1F("h1d_fracleak_%s"%name1,"",500,0,2))
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
    temp_gnn=[]
    temp_chi2=[]
    temp_pi0=[]
    f_gnn=[]
    f1_gnn=[]
    xhigh = 3.0*Energy[i_hist]
    for i_b in range(len(baseline)):
        name = '%s_TrueEn_%i'%(baseline[i_b],Energy[i_hist])
        temp_gnn.append(ROOT.TH2F("h2d_Gnnvs_%s"%name,"",500,0,xhigh,200,0,Energy[i_hist]+20))
        temp_chi2.append(ROOT.TH2F("h2d_Chi2vs_%s"%name,"",500,0,xhigh,200,0,Energy[i_hist]+20))
        temp_pi0.append(ROOT.TH2F("h2d_leadpi0Vs_%s"%name,"",500,0,Energy[i_hist]+20,500,0,Energy[i_hist]+20))
        f_gnn.append(ROOT.TH2F("h2d_fracGnnvs_%s"%name,"",500,0,xhigh,500,0,1))
        f1_gnn.append(ROOT.TH2F("h2d_v1fracGnnvs_%s"%name,"",500,0,xhigh,500,0,1))
        
    h2d_leakvsGnn.append(temp_gnn)
    h2d_leakvsChi2.append(temp_chi2)
    h2d_pi0vsLeak.append(temp_pi0)
    h2d_fracleakvsGnn.append(f_gnn)
    h2d_vfracleakvsGnn.append(f1_gnn)
    temp_gnn=[]
    temp_chi2=[]
    temp_pi0=[]
    for i_b in range(len(baseline1)):
        name = '%s_TrueEn_%i'%(baseline1[i_b],Energy[i_hist])
        temp_gnn.append(ROOT.TH2F("h2d_Gnnvs_%s"%name,"",500,0,xhigh,500,0,100))
        temp_chi2.append(ROOT.TH2F("h2d_Chi2vs_%s"%name,"",500,0,xhigh,500,0,100))
        temp_pi0.append(ROOT.TH2F("h2d_leadpi0Vs_%s"%name,"",500,0,Energy[i_hist]+20,500,0,100))
    h2d_gnnvsStdDev.append(temp_gnn)
    h2d_chi2vsStdDev.append(temp_chi2)
    h2d_pi0vsstdDev.append(temp_pi0)
    temp=[]
    for i_b in range(len(baseline2)):
        name = '%s_TrueEn_%i'%(baseline2[i_b],Energy[i_hist])
        temp.append(ROOT.TH2F("h2d_%s"%name,"",500,0,100,500,0,Energy[i_hist]+20))
        
    h2d_stdDevvsleak.append(temp)






for i_en in range(len(Energy)):
    # for i in range(len(train_idx)):
    #     if(trueEn_pkl[train_idx[i]]>=(Energy[i_en]-2) and trueEn_pkl[train_idx[i]]<=(Energy[i_en]+2) ):
    #         h1d_predModel_train[i_en].Fill(Pred_[train_idx[i]])
    # for i in range(len(valid_idx)):
    #     if(trueEn_pkl[valid_idx[i]]>=(Energy[i_en]-2) and trueEn_pkl[valid_idx[i]]<=(Energy[i_en]+2) ):
    #         h1d_predModel_valid[i_en].Fill(Pred_[valid_idx[i]])

                                       

    for i in range(len(trueEn_pkl)):
        # x_pion = np.array(Hit_X_pkl[i])
        # y_pion = np.array(Hit_Y_pkl[i])
        # z_pion =np.array(Hit_Z_pkl[i])
        # rec_en = RechitEn_pkl[i]
        # stdDev_x_EE= np.std(x_pion[z_pion<54])
        # stdDev_y_EE= np.std(y_pion[z_pion<54])
        # stdDev_z_EE= np.std(z_pion[z_pion<54])
        # stdDev_x_FH= np.std(x_pion[np.logical_and(z_pion>54 , z_pion<154)])
        # stdDev_y_FH= np.std(y_pion[np.logical_and(z_pion>54 ,z_pion<154)])
        # stdDev_z_FH= np.std(z_pion[np.logical_and(z_pion>54 ,z_pion<154)])
        # stdDev_x_AH= np.std(x_pion[z_pion>154])
        # stdDev_y_AH= np.std(y_pion[z_pion>154])
        # stdDev_z_AH= np.std(z_pion[z_pion>154])
        # stdDev_2d_EE = math.sqrt((stdDev_x_EE*stdDev_x_EE)+(stdDev_y_EE*stdDev_y_EE))
        # stdDev_2d_FH = math.sqrt((stdDev_x_FH*stdDev_x_FH)+(stdDev_y_FH*stdDev_y_FH))
        # stdDev_2d_AH = math.sqrt((stdDev_x_AH*stdDev_x_AH)+(stdDev_y_AH*stdDev_y_AH))
        
        if(trueEn_pkl[i]>=(Energy[i_en]-2) and trueEn_pkl[i]<=(Energy[i_en]+2)):
            h2d_trueEnvstrueEn[i_en].Fill(trueEn_pkl[i],trueBeamEnergy[i])
            #if(energyLeakLongitudinal_[i]>(0.5*trueEn_pkl[i])):
            h1d_SSloc_v1[i_en].Fill(rechit_shower_start_layer[i])
            
            #if(energyLeakResidual_[i]>20):
            h1d_SSloc[i_en].Fill(rechit_shower_start_layer[i])
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
            h1d_totalVisi[i_en].Fill(total_lost+total_leakage[i])
            h1d_fractotalVisi[i_en].Fill((total_lost+total_leakage[i])/trueEn_pkl[i])
            h1d_normchi2[i_en].Fill(norm_chi2)
            h1d_normModel[i_en].Fill(norm_gnn)
            h1d_predchi2[i_en].Fill(trimAhcal_chi2Reco[i])
            h1d_predModel[i_en].Fill(Pred_[i])
            frac_abs = total_lost/Energy[i_en]
            frac_leak_v1 = total_leakage[i]/(total_lost+total_leakage[i])
            h1d_totalAbs[i_en].Fill(total_lost)
            h1d_fracAbs[i_en].Fill(frac_abs)
            h1d_totalLeak[i_en].Fill(total_leakage[i])
            h1d_fracleak[i_en].Fill(frac_leak[i])
            h1d_fracleak_total[i_en].Fill(frac_leak_v1)
            h1d_fracvisi_true[i_en].Fill((total_lost+total_leakage[i])/Energy[i_en])
            h2d_fracleakVsabs[i_en].Fill(frac_abs,frac_leak[i])
            h2d_fracleakVsabs_v1[i_en].Fill(frac_abs,frac_leak_v1 )
            h2d_fracleakVsGnn[i_en].Fill(Pred_[i],frac_leak[i])
            h2d_fracleakVsChi2[i_en].Fill(trimAhcal_chi2Reco[i],frac_leak[i])
            h1d_lekageEE[i_en].Fill(Total_EE[i])
            h1d_lekageFH[i_en].Fill(Total_FH[i])
            h1d_lekageAH[i_en].Fill(energyLeakTransverseAH_[i])
            h1d_beforeEE[i_en].Fill(energyLeakResidual_[i])
            h1d_long[i_en].Fill(energyLeakLongitudinal_[i])
            
            h2d_leakvsGnn[i_en][0].Fill(Pred_[i],energyLeakResidual_[i])
            h2d_leakvsGnn[i_en][1].Fill(Pred_[i],Total_EE[i])
            h2d_leakvsGnn[i_en][2].Fill(Pred_[i],Total_FH[i])
            h2d_leakvsGnn[i_en][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
            h2d_leakvsGnn[i_en][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
            
            h2d_fracleakvsGnn[i_en][0].Fill(Pred_[i],energyLeakResidual_[i]/Energy[i_en])
            h2d_fracleakvsGnn[i_en][1].Fill(Pred_[i],Total_EE[i]/Energy[i_en])
            h2d_fracleakvsGnn[i_en][2].Fill(Pred_[i],Total_FH[i]/Energy[i_en])
            h2d_fracleakvsGnn[i_en][3].Fill(Pred_[i],energyLeakTransverseAH_[i]/Energy[i_en])
            h2d_fracleakvsGnn[i_en][4].Fill(Pred_[i],energyLeakLongitudinal_[i]/Energy[i_en])

            h2d_vfracleakvsGnn[i_en][0].Fill(Pred_[i],energyLeakResidual_[i]/total_leakage[i])
            h2d_vfracleakvsGnn[i_en][1].Fill(Pred_[i],Total_EE[i]/total_leakage[i])
            h2d_vfracleakvsGnn[i_en][2].Fill(Pred_[i],Total_FH[i]/total_leakage[i])
            h2d_vfracleakvsGnn[i_en][3].Fill(Pred_[i],energyLeakTransverseAH_[i]/total_leakage[i])
            h2d_vfracleakvsGnn[i_en][4].Fill(Pred_[i],energyLeakLongitudinal_[i]/total_leakage[i])

            h2d_leakvsChi2[i_en][0].Fill(trimAhcal_chi2Reco[i],energyLeakResidual_[i])
            h2d_leakvsChi2[i_en][1].Fill(trimAhcal_chi2Reco[i],Total_EE[i])
            h2d_leakvsChi2[i_en][2].Fill(trimAhcal_chi2Reco[i],Total_FH[i])
            h2d_leakvsChi2[i_en][3].Fill(trimAhcal_chi2Reco[i],energyLeakTransverseAH_[i])
            h2d_leakvsChi2[i_en][4].Fill(trimAhcal_chi2Reco[i],energyLeakLongitudinal_[i])
            
            # h2d_gnnVspi0[i_en].Fill(Pred_[i],total_pi0KE[i])
            # h2d_chi2vspi0[i_en].Fill(trimAhcal_chi2Reco[i],total_pi0KE[i])
            # h2d_Absvspi0[i_en].Fill(total_lost,total_pi0KE[i])
            # h2d_pi0vsLeak[i_en][0].Fill(total_pi0KE[i],energyLeakResidual_[i])
            # h2d_pi0vsLeak[i_en][1].Fill(total_pi0KE[i],Total_EE[i])
            # h2d_pi0vsLeak[i_en][2].Fill(total_pi0KE[i],Total_FH[i])
            # h2d_pi0vsLeak[i_en][3].Fill(total_pi0KE[i],energyLeakTransverseAH_[i])
            # h2d_pi0vsLeak[i_en][4].Fill(total_pi0KE[i],energyLeakLongitudinal_[i])
            
            # h2d_gnnvsStdDev[i_en][0].Fill(Pred_[i],stdDev_2d_EE)
            # h2d_gnnvsStdDev[i_en][1].Fill(Pred_[i],stdDev_2d_FH)
            # h2d_gnnvsStdDev[i_en][2].Fill(Pred_[i],stdDev_2d_AH)
            # h2d_chi2vsStdDev[i_en][0].Fill(Pred_[i],stdDev_2d_EE)
            # h2d_chi2vsStdDev[i_en][1].Fill(Pred_[i],stdDev_2d_FH)
            # h2d_chi2vsStdDev[i_en][2].Fill(Pred_[i],stdDev_2d_AH)
            
            # h2d_stdDevvsleak[i_en][0].Fill(stdDev_2d_EE,Total_EE[i])
            # h2d_stdDevvsleak[i_en][1].Fill(stdDev_2d_FH,Total_FH[i])
            # h2d_stdDevvsleak[i_en][2].Fill(stdDev_2d_AH,energyLeakTransverseAH_[i])
            # h2d_stdDevvsleak[i_en][3].Fill(stdDev_2d_AH,energyLeakLongitudinal_[i])
            # h2d_stdDevvsleak[i_en][4].Fill(stdDev_2d_EE,energyLeakResidual_[i])

            # h2d_pi0vsstdDev[i_en][0].Fill(total_pi0KE[i],stdDev_2d_EE)
            # h2d_pi0vsstdDev[i_en][1].Fill(total_pi0KE[i],stdDev_2d_FH)
            # h2d_pi0vsstdDev[i_en][2].Fill(total_pi0KE[i],stdDev_2d_AH)
            # #h2d_pi0vsstdDev[i_en][0].Fill()
            # h2d_fracAbs_pi0[i_en].Fill(ratio_abs,total_pi0KE[i]/total_lost)


hist_chi2_pred_categ1=[]
hist_chi2_pred_categ2=[]
hist_chi2_norm_predTrue_categ1=[]
hist_chi2_norm_predTrue_categ2=[]
hist_chi2_pred_categ3=[]
hist_chi2_norm_predTrue_categ3=[]


hist_model_pred_categ1=[]
hist_model_pred_categ2=[]
hist_model_norm_predTrue_categ1=[]
hist_model_norm_predTrue_categ2=[]
hist_model_pred_categ3=[]
hist_model_norm_predTrue_categ3=[]

hist_model_pred_categ4=[]
hist_model_norm_predTrue_categ4=[]
hist_chi2_pred_categ4=[]
hist_chi2_norm_predTrue_categ4=[]

h2d_fracleakVsAbs_categ1=[]
h2d_fracleakVsAbsv1_categ1=[]
h2d_fracleakVsAbs_categ2=[]
h2d_fracleakVsAbsv1_categ2=[]
h2d_fracleakVsAbs_categ3=[]
h2d_fracleakVsAbs_categ4=[]
h2d_fracleakVsAbsv1_categ4=[]
h2d_fracleakVsAbsv1_categ3=[]

h2d_fracLeakVsGnn_categ1=[]
h2d_fracLeakVsGnn_categ2=[]
h2d_fracLeakVsGnn_categ3=[]
h2d_fracLeakVsGnn_categ4=[]

h2d_fracLeakVsChi2_categ1=[]
h2d_fracLeakVsChi2_categ2=[]
h2d_fracLeakVsChi2_categ3=[]
h2d_fracLeakVsChi2_categ4=[]
h1d_lekageEE_categ1=[]
h1d_lekageFH_categ1=[]
h1d_lekageAH_categ1=[]
h1d_beforeEE_categ1=[]
h1d_long_categ1=[]

h1d_lekageEE_categ2=[]
h1d_lekageFH_categ2=[]
h1d_lekageAH_categ2=[]
h1d_beforeEE_categ2=[]
h1d_long_categ2=[]

h1d_lekageEE_categ3=[]
h1d_lekageFH_categ3=[]
h1d_lekageAH_categ3=[]
h1d_beforeEE_categ3=[]
h1d_long_categ3=[]
h1d_lekageEE_categ4=[]
h1d_lekageFH_categ4=[]
h1d_lekageAH_categ4=[]
h1d_beforeEE_categ4=[]
h1d_long_categ4=[]

h2d_leakvsGnn_categ=[]
h2d_leakvsChi2_categ=[]
baseline0=["categ1","categ2","categ3","categ4"]
# baseline = ["beforeEE","lekageEE","lekageFH","lekageAH","longi"]
# baseline1 = ["stdDevEE","stdDevFH","stdDevAH"]
# baseline2 = ["leak_stdDevEE","leak_stdDevFH","leak_stdDevAh","leakStdAh_longi","stdDevEE_beforeEE"]
h2d_gnnvsStdDev_categ=[]
h2d_chi2vsStdDev_categ=[]
h2d_stdDevvsleak_categ=[]
h2d_gnnVspi0_categ=[]
h2d_chi2vspi0_categ=[]
h2d_Absvspi0_categ=[]
h2d_pi0vsLeak_categ=[]
h2d_pi0vsstdDev_categ=[]
h2d_fracAbs_pi0_categ=[]
h1d_totalleak_categ=[]
h1d_fracleak_categ=[]


M=8

for i_hist in range(len(Energy)):
    if(Energy[i_hist]<100):
        xhigh_pred = 4.0*Energy[i_hist]
    else:
        xhigh_pred= 3.0*Energy[i_hist]

    #xhigh_pred= 3.0*Energy[i_hist]
    xhigh_true= 2.0*Energy[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(Energy[i_hist])#,u[i_hist],v[i_hist],typee[i_hist])                                                                            
    xhigh = Energy[i_hist]
    h1d_lekageEE_categ4.append(ROOT.TH1F('h1d_lekageEE_categ4_%s'%name1,"h1d_lekageEE_categ4_",500,0,xhigh))
    h1d_lekageFH_categ4.append(ROOT.TH1F('h1d_lekageFH_categ4_%s'%name1,"h1d_lekageFH_categ4_",500,0,xhigh))
    h1d_lekageAH_categ4.append(ROOT.TH1F('h1d_lekageAH_categ4_%s'%name1,"h1d_lekageAH_categ4_",500,0,xhigh))
    h1d_beforeEE_categ4.append(ROOT.TH1F('h1d_beforeEE_categ4_%s'%name1,"h1d_beforeEE_categ4",500,0,xhigh))
    h1d_long_categ4.append(ROOT.TH1F('h1d_long_categ4_%s'%name1,"h1d_long_categ4",500,0,xhigh))


    h1d_lekageEE_categ3.append(ROOT.TH1F('h1d_lekageEE_categ3_%s'%name1,"h1d_lekageEE_categ3_",500,0,xhigh))
    h1d_lekageFH_categ3.append(ROOT.TH1F('h1d_lekageFH_categ3_%s'%name1,"h1d_lekageFH_categ3_",500,0,xhigh))
    h1d_lekageAH_categ3.append(ROOT.TH1F('h1d_lekageAH_categ3_%s'%name1,"h1d_lekageAH_categ3_",500,0,xhigh))
    h1d_beforeEE_categ3.append(ROOT.TH1F('h1d_beforeEE_categ3_%s'%name1,"h1d_beforeEE_categ3",500,0,xhigh))
    h1d_long_categ3.append(ROOT.TH1F('h1d_long_categ3_%s'%name1,"h1d_long_categ3",500,0,xhigh))

    h1d_lekageEE_categ2.append(ROOT.TH1F('h1d_lekageEE_categ2_%s'%name1,"h1d_lekageEE_categ2_",500,0,xhigh))
    h1d_lekageFH_categ2.append(ROOT.TH1F('h1d_lekageFH_categ2_%s'%name1,"h1d_lekageFH_categ2_",500,0,xhigh))
    h1d_lekageAH_categ2.append(ROOT.TH1F('h1d_lekageAH_categ2_%s'%name1,"h1d_lekageAH_categ2_",500,0,xhigh))
    h1d_beforeEE_categ2.append(ROOT.TH1F('h1d_beforeEE_categ2_%s'%name1,"h1d_beforeEE_categ2",500,0,xhigh))
    h1d_long_categ2.append(ROOT.TH1F('h1d_long_categ2_%s'%name1,"h1d_long_categ2",500,0,xhigh))

    h1d_lekageEE_categ1.append(ROOT.TH1F('h1d_lekageEE_categ1_%s'%name1,"h1d_lekageEE_categ1_",500,0,xhigh))
    h1d_lekageFH_categ1.append(ROOT.TH1F('h1d_lekageFH_categ1_%s'%name1,"h1d_lekageFH_categ1_",500,0,xhigh))
    h1d_lekageAH_categ1.append(ROOT.TH1F('h1d_lekageAH_categ1_%s'%name1,"h1d_lekageAH_categ1_",500,0,xhigh))
    h1d_beforeEE_categ1.append(ROOT.TH1F('h1d_beforeEE_categ1_%s'%name1,"h1d_beforeEE_categ1",500,0,xhigh))
    h1d_long_categ1.append(ROOT.TH1F('h1d_long_categ1_%s'%name1,"h1d_long_categ1",500,0,xhigh))


    h2d_fracleakVsAbs_categ1.append(ROOT.TH2F('h2d_fracleakVsAbs_categ1_%s'%name1,"h2d_fracleakVsAbs_categ1",500,0,2,500,0,2))
    h2d_fracleakVsAbsv1_categ1.append(ROOT.TH2F('h2d_fracleakVsAbsv1_categ1_%s'%name1,"h2d_fracleakVsAbsv1_categ1",500,0,2,500,0,2))
    h2d_fracleakVsAbs_categ2.append(ROOT.TH2F('h2d_fracleakVsAbs_categ2_%s'%name1,"h2d_fracleakVsAbs_categ2",500,0,2,500,0,2))
    h2d_fracleakVsAbsv1_categ2.append(ROOT.TH2F('h2d_fracleakVsAbsv1_categ2_%s'%name1,"h2d_fracleakVsAbsv1_categ2",500,0,2,500,0,2))
    h2d_fracleakVsAbs_categ3.append(ROOT.TH2F('h2d_fracleakVsAbs_categ3_%s'%name1,"h2d_fracleakVsAbs_categ3",500,0,2,500,0,2))
    h2d_fracleakVsAbs_categ4.append(ROOT.TH2F('h2d_fracleakVsAbs_categ4_%s'%name1,"h2d_fracleakVsAbs_categ4",500,0,2,500,0,2))
    h2d_fracleakVsAbsv1_categ4.append(ROOT.TH2F('h2d_fracleakVsAbsv1_categ4_%s'%name1,"h2d_fracleakVsAbsv1_categ4",500,0,2,500,0,2))
    h2d_fracleakVsAbsv1_categ3.append(ROOT.TH2F('h2d_fracleakVsAbsv1_categ3_%s'%name1,"h2d_fracleakVsAbsv1_categ3",500,0,2,500,0,2))
    
    h2d_fracLeakVsGnn_categ1.append(ROOT.TH2F('h2d_fracLeakVsGnn_categ1_%s'%name1,"h2d_fracLeakVsGnn_categ1",500, 0, xhigh_pred,500,0,2))
    h2d_fracLeakVsGnn_categ2.append(ROOT.TH2F('h2d_fracLeakVsGnn_categ2_%s'%name1,"h2d_fracLeakVsGnn_categ2",500, 0, xhigh_pred,500,0,2))
    h2d_fracLeakVsGnn_categ3.append(ROOT.TH2F('h2d_fracLeakVsGnn_categ3_%s'%name1,"h2d_fracLeakVsGnn_categ3",500, 0, xhigh_pred,500,0,2))
    h2d_fracLeakVsGnn_categ4.append(ROOT.TH2F('h2d_fracLeakVsGnn_categ4_%s'%name1,"h2d_fracLeakVsGnn_categ4",500, 0, xhigh_pred,500,0,2))
    
    h2d_fracLeakVsChi2_categ1.append(ROOT.TH2F('h2d_fracLeakVsChi2_categ1_%s'%name1,"h2d_fracLeakVsChi2_categ1",500, 0, xhigh_pred,500,0,2))
    h2d_fracLeakVsChi2_categ2.append(ROOT.TH2F('h2d_fracLeakVsChi2_categ2_%s'%name1,"h2d_fracLeakVsChi2_categ2",500, 0, xhigh_pred,500,0,2))
    h2d_fracLeakVsChi2_categ3.append(ROOT.TH2F('h2d_fracLeakVsChi2_categ3_%s'%name1,"h2d_fracLeakVsChi2_categ3",500, 0, xhigh_pred,500,0,2))
    h2d_fracLeakVsChi2_categ4.append(ROOT.TH2F('h2d_fracLeakVsChi2_categ4_%s'%name1,"h2d_fracLeakVsChi2_categ4",500, 0, xhigh_pred,500,0,2))
    

    hist_chi2_pred_categ1.append(ROOT.TH1F('chi2_categ1_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_chi2_pred_categ2.append(ROOT.TH1F('chi2_categ2_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_chi2_norm_predTrue_categ1.append(ROOT.TH1F('chi2_categ1_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_chi2_norm_predTrue_categ2.append(ROOT.TH1F('chi2_categ2_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_chi2_pred_categ3.append(ROOT.TH1F('chi2_categ3_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
    hist_chi2_norm_predTrue_categ3.append(ROOT.TH1F('chi2_categ3_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))

    hist_chi2_pred_categ4.append(ROOT.TH1F('chi2_categ4_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
    hist_chi2_norm_predTrue_categ4.append(ROOT.TH1F('chi2_categ4_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))
    hist_model_pred_categ4.append(ROOT.TH1F('model_categ4_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
    hist_model_norm_predTrue_categ4.append(ROOT.TH1F('model_categ4_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))

    hist_model_pred_categ1.append(ROOT.TH1F('model_categ1_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_model_pred_categ2.append(ROOT.TH1F('model_categ2_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0, xhigh_pred))
    hist_model_norm_predTrue_categ1.append(ROOT.TH1F('model_categ1_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_model_norm_predTrue_categ2.append(ROOT.TH1F('model_categ2_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500, xlow_norm, xhigh_norm))
    hist_model_pred_categ3.append(ROOT.TH1F('model_categ3_Predi_%s' % name1, """:"Predicted energy in GeV":""", 500, 0,xhigh_pred ))
    hist_model_norm_predTrue_categ3.append(ROOT.TH1F('model_categ3_norm_pred_trueEn_%s' % name1, """:"(pred-true)/true in GeV":""", 500,xlow_norm, xhigh_norm ))
    xhigh = xhigh_pred #3.0*Energy[i_hist]
    h2d_leakvsGnn_categ_temp=[]
    h2d_leakvsChi2_categ_temp=[]
    h2d_gnnvsStdDev_categ_temp=[]
    h2d_chi2vsStdDev_categ_temp=[]
    h2d_stdDevvsleak_categ_temp=[]
    h1d_fracleak_categ_temp=[]
    h1d_totalleak_categ_temp=[]
    h2d_gnnVspi0_categ_temp=[]
    h2d_chi2vspi0_categ_temp=[]
    h2d_Absvspi0_categ_temp=[]
    h2d_pi0vsLeak_categ_temp=[]
    h2d_pi0vsstdDev_categ_temp=[]
    h2d_fracAbs_pi0_categ_temp=[]

    for i_c in range(len(baseline0)):
        temp_gnn=[]
        temp_chi2=[]
        h2d_gnnVspi0_categ_temp.append(ROOT.TH2F("h2d_gnnVspi0_%s_%s"%(baseline0[i_c],name1),"",500,0,xhigh,500,0,Energy[i_hist]+20))
        h2d_chi2vspi0_categ_temp.append(ROOT.TH2F("h2d_chi2vspi0_%s_%s"%(baseline0[i_c],name1),"",500,0,xhigh,500,0,Energy[i_hist]+20))
        h2d_Absvspi0_categ_temp.append(ROOT.TH2F("h2d_Absvspi0_%s_%s"%(baseline0[i_c],name1),"",500,0,Energy[i_hist]+20,500,0,Energy[i_hist]+20))
        h2d_fracAbs_pi0_categ_temp.append(ROOT.TH2F("h2d_fracAbsvspi0_%s_%s"%(baseline0[i_c],name1),"frac Abs vs total_pi0/abs",500,0,2,500,0,2))
        h1d_fracleak_categ_temp.append(ROOT.TH1F("h1d_fracleak_%s_%s"%(baseline0[i_c],name1),"",500,0,1.1))
        h1d_totalleak_categ_temp.append(ROOT.TH1F("h1d_totalleak_%s_%s"%(baseline0[i_c],name1),"",500,0,Energy[i_hist]+20))

        temp_pi0=[]
        for i_b in range(len(baseline)):
            name = '%s_%s_TrueEn_%i'%(baseline0[i_c],baseline[i_b],Energy[i_hist])
            temp_gnn.append(ROOT.TH2F("h2d_Gnnvs_%s"%name,"",500,0,xhigh,200,0,Energy[i_hist]+20))
            temp_chi2.append(ROOT.TH2F("h2d_Chi2vs_%s"%name,"",500,0,xhigh,200,0,Energy[i_hist]+20))
            temp_pi0.append(ROOT.TH2F("h2d_leadpi0Vs_%s"%name,"",500,0,Energy[i_hist]+20,500,0,Energy[i_hist]+20))

        h2d_leakvsGnn_categ_temp.append(temp_gnn)
        h2d_leakvsChi2_categ_temp.append(temp_chi2)
        h2d_pi0vsLeak_categ_temp.append(temp_pi0)
        temp_gnn=[]
        temp_chi2=[]
        temp_pi0=[]

        for i_b in range(len(baseline1)):
            name = '%s_%s_TrueEn_%i'%(baseline0[i_c],baseline1[i_b],Energy[i_hist])
            temp_gnn.append(ROOT.TH2F("h2d_Gnnvs_%s"%name,"",500,0,xhigh,500,0,100))
            temp_chi2.append(ROOT.TH2F("h2d_Chi2vs_%s"%name,"",500,0,xhigh,500,0,100))
            temp_pi0.append(ROOT.TH2F("h2d_leadpi0Vs_%s"%name,"",500,0,100,500,0,Energy[i_hist]))
        
        h2d_gnnvsStdDev_categ_temp.append(temp_gnn)
        h2d_chi2vsStdDev_categ_temp.append(temp_chi2)
        h2d_pi0vsstdDev_categ_temp.append(temp_pi0)
        temp=[]
        for i_b in range(len(baseline2)):
            name = '%s_%s_TrueEn_%i'%(baseline0[i_c],baseline2[i_b],Energy[i_hist])
            temp.append(ROOT.TH2F("h2d_%s"%name,"",500,0,100,500,0,Energy[i_hist]+20))
        
        h2d_stdDevvsleak_categ_temp.append(temp)

    h2d_leakvsGnn_categ.append(h2d_leakvsGnn_categ_temp)
    h2d_leakvsChi2_categ.append(h2d_leakvsChi2_categ_temp)
    h2d_gnnvsStdDev_categ.append(h2d_gnnvsStdDev_categ_temp)
    h2d_chi2vsStdDev_categ.append(h2d_chi2vsStdDev_categ_temp)
    h2d_stdDevvsleak_categ.append(h2d_stdDevvsleak_categ_temp)
    h2d_pi0vsLeak_categ.append(h2d_pi0vsLeak_categ_temp)
    h2d_pi0vsstdDev_categ.append(h2d_pi0vsstdDev_categ_temp)
    h2d_fracAbs_pi0_categ.append(h2d_fracAbs_pi0_categ_temp)
    h2d_gnnVspi0_categ.append(h2d_gnnVspi0_categ_temp)
    h1d_totalleak_categ.append(h1d_totalleak_categ_temp)
    h1d_fracleak_categ.append(h1d_fracleak_categ_temp)
    h2d_chi2vspi0_categ.append(h2d_chi2vspi0_categ_temp)
    h2d_Absvspi0_categ.append(h2d_chi2vspi0_categ_temp)
    

#Mean_sigma = np.load('./FracAbs_meanSigma_gausFit_8Enpoints.txt')
# mean_ = Mean_sigma[1]
# sigma_ = Mean_sigma[2]
energy_,Mean_,Sigma_ = np.loadtxt('./Absorber_resolutionFrac.txt',usecols=(0,1, 2), unpack=True)
# total_abs=total_leakage[trueEn_pkl==50]/50 #(energyLostEE_[trueEn_pkl==50]+energyLostFH_[trueEn_pkl==50]+energyLostBH_[trueEn_pkl==50])/50
mean_=Mean_ #[]
stdDev =Sigma_ #[]
# mean_.append(Mean_[energy_==50])
# stdDev.append(Sigma_[energy_==50])
# mean_.append(Mean_[energy_==100])
# stdDev.append(Sigma_[energy_==100])
# mean_.append(Mean_[energy_==300])
# stdDev.append(Sigma_[energy_==300])

print(mean_,stdDev)

# mean_.append(np.mean(total_abs))
# stdDev.append(np.std(total_abs))
# print(np.mean(total_abs),np.std(total_abs))

# total_abs=total_leakage[trueEn_pkl==100]/100#(energyLostEE_[trueEn_pkl==100]+energyLostFH_[trueEn_pkl==100]+energyLostBH_[trueEn_pkl==100])/100
# mean_.append(np.mean(total_abs))
# stdDev.append(np.std(total_abs))
# print(np.mean(total_abs),np.std(total_abs))

# total_abs=total_leakage[trueEn_pkl==300]/300#(energyLostEE_[trueEn_pkl==300]+energyLostFH_[trueEn_pkl==300]+energyLostBH_[trueEn_pkl==300])/300
# mean_.append(np.mean(total_abs))
# stdDev.append(np.std(total_abs))
# print(np.mean(total_abs),np.std(total_abs))



for i in range(len(trueEn_pkl)):
    total_lost = energyLostEE_[i]+energyLostFH_[i]+energyLostBH_[i]
    ratio = total_lost/trueEn_pkl[i]
    norm_gnn = (Pred_[i] -trueEn_pkl[i])/trueEn_pkl[i]
    norm_chi2 = (trimAhcal_chi2Reco[i]-trueEn_pkl[i])/trueEn_pkl[i]
    # x_pion = np.array(Hit_X_pkl[i])
    # y_pion = np.array(Hit_Y_pkl[i])
    # z_pion =np.array(Hit_Z_pkl[i])
    # rec_en = RechitEn_pkl[i]
    # stdDev_x_EE= np.std(x_pion[z_pion<54])
    # stdDev_y_EE= np.std(y_pion[z_pion<54])
    # stdDev_z_EE= np.std(z_pion[z_pion<54])
    # stdDev_x_FH= np.std(x_pion[np.logical_and(z_pion>54 , z_pion<154)])
    # stdDev_y_FH= np.std(y_pion[np.logical_and(z_pion>54 ,z_pion<154)])
    # stdDev_z_FH= np.std(z_pion[np.logical_and(z_pion>54 ,z_pion<154)])
    # stdDev_x_AH= np.std(x_pion[z_pion>154])
    # stdDev_y_AH= np.std(y_pion[z_pion>154])
    # stdDev_z_AH= np.std(z_pion[z_pion>154])
    # stdDev_2d_EE = math.sqrt((stdDev_x_EE*stdDev_x_EE)+(stdDev_y_EE*stdDev_y_EE))
    # stdDev_2d_FH = math.sqrt((stdDev_x_FH*stdDev_x_FH)+(stdDev_y_FH*stdDev_y_FH))
    # stdDev_2d_AH = math.sqrt((stdDev_x_AH*stdDev_x_AH)+(stdDev_y_AH*stdDev_y_AH))

    for ibin in range(len(Energy)):
        if(trueEn_pkl[i]>=(Energy[ibin]-2) and trueEn_pkl[i]<=(Energy[ibin]+2 )):
            categ1 = ratio>(mean_[ibin]-stdDev[ibin]) and ratio<(mean_[ibin]+stdDev[ibin])
            categ2 = ratio>(mean_[ibin]+stdDev[ibin])
            categ3 = ratio<(mean_[ibin]-stdDev[ibin]) and ratio>(mean_[ibin]-2.0*stdDev[ibin])
            categ4= ratio<(mean_[ibin]-2.0*stdDev[ibin])
            frac_leak_v1 = total_leakage[i]/(total_lost+total_leakage[i])
            #ratio=total_lost/trueEn_pkl[i]
            if(categ1):
                hist_chi2_pred_categ1[ibin].Fill(trimAhcal_chi2Reco[i])
                hist_model_pred_categ1[ibin].Fill(Pred_[i])
                hist_chi2_norm_predTrue_categ1[ibin].Fill(norm_chi2)
                hist_model_norm_predTrue_categ1[ibin].Fill(norm_gnn)
                h2d_fracleakVsAbs_categ1[ibin].Fill(ratio,frac_leak[i])
                h2d_fracleakVsAbsv1_categ1[ibin].Fill(ratio,frac_leak_v1)
                h2d_fracLeakVsGnn_categ1[ibin].Fill(Pred_[i],frac_leak[i])
                h2d_fracLeakVsChi2_categ1[ibin].Fill(trimAhcal_chi2Reco[i],frac_leak[i])
                h1d_lekageEE_categ1[ibin].Fill(Total_EE[i])
                h1d_lekageFH_categ1[ibin].Fill(Total_FH[i])
                h1d_lekageAH_categ1[ibin].Fill(energyLeakTransverseAH_[i])
                h1d_beforeEE_categ1[ibin].Fill(energyLeakResidual_[i])
                h1d_long_categ1[ibin].Fill(energyLeakLongitudinal_[i])
                h2d_leakvsGnn_categ[ibin][0][0].Fill(Pred_[i],energyLeakResidual_[i])
                h2d_leakvsGnn_categ[ibin][0][1].Fill(Pred_[i],Total_EE[i])
                h2d_leakvsGnn_categ[ibin][0][2].Fill(Pred_[i],Total_FH[i])
                h2d_leakvsGnn_categ[ibin][0][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
                h2d_leakvsGnn_categ[ibin][0][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
                h2d_leakvsChi2_categ[ibin][0][0].Fill(Pred_[i],energyLeakResidual_[i])
                h2d_leakvsChi2_categ[ibin][0][1].Fill(Pred_[i],Total_EE[i])
                h2d_leakvsChi2_categ[ibin][0][2].Fill(Pred_[i],Total_FH[i])
                h2d_leakvsChi2_categ[ibin][0][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
                h2d_leakvsChi2_categ[ibin][0][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
                # h2d_pi0vsLeak_categ[ibin][0][0].Fill(total_pi0KE[i],energyLeakResidual_[i])
                # h2d_pi0vsLeak_categ[ibin][0][1].Fill(total_pi0KE[i],Total_EE[i])
                # h2d_pi0vsLeak_categ[ibin][0][2].Fill(total_pi0KE[i],Total_FH[i])
                # h2d_pi0vsLeak_categ[ibin][0][3].Fill(total_pi0KE[i],energyLeakTransverseAH_[i])
                # h2d_pi0vsLeak_categ[ibin][0][4].Fill(total_pi0KE[i],energyLeakLongitudinal_[i])
                
                # h2d_gnnVspi0_categ[ibin][0].Fill(Pred_[i],total_pi0KE[i])
                # h2d_chi2vspi0_categ[ibin][0].Fill(trimAhcal_chi2Reco[i],total_pi0KE[i])
                # h2d_Absvspi0_categ[ibin][0].Fill(total_lost,total_pi0KE[i])
                # h2d_fracAbs_pi0_categ[ibin][0].Fill(ratio_abs,total_pi0KE[i]/total_lost)
                h1d_totalleak_categ[ibin][0].Fill( total_leakage[i])
                h1d_fracleak_categ[ibin][0].Fill(frac_leak_v1)
                # h2d_gnnvsStdDev_categ[ibin][0][0].Fill(Pred_[i],stdDev_2d_EE)
                # h2d_gnnvsStdDev_categ[ibin][0][1].Fill(Pred_[i],stdDev_2d_FH)
                # h2d_gnnvsStdDev_categ[ibin][0][2].Fill(Pred_[i],stdDev_2d_AH)
                # h2d_chi2vsStdDev_categ[ibin][0][0].Fill(Pred_[i],stdDev_2d_EE)
                # h2d_chi2vsStdDev_categ[ibin][0][1].Fill(Pred_[i],stdDev_2d_FH)
                # h2d_chi2vsStdDev_categ[ibin][0][2].Fill(Pred_[i],stdDev_2d_AH)
                
                # h2d_stdDevvsleak_categ[ibin][0][0].Fill(stdDev_2d_EE,Total_EE[i])
                # h2d_stdDevvsleak_categ[ibin][0][1].Fill(stdDev_2d_FH,Total_FH[i])
                # h2d_stdDevvsleak_categ[ibin][0][2].Fill(stdDev_2d_AH,energyLeakTransverseAH_[i])
                # h2d_stdDevvsleak_categ[ibin][0][3].Fill(stdDev_2d_AH,energyLeakLongitudinal_[i])
                # h2d_stdDevvsleak_categ[ibin][0][4].Fill(stdDev_2d_EE,energyLeakResidual_[i])
            

                # h2d_pi0vsstdDev_categ[ibin][0][0].Fill(total_pi0KE[i],stdDev_2d_EE)
                # h2d_pi0vsstdDev_categ[ibin][0][1].Fill(total_pi0KE[i],stdDev_2d_FH)
                # h2d_pi0vsstdDev_categ[ibin][0][2].Fill(total_pi0KE[i],stdDev_2d_AH)

            elif(categ2):
                hist_model_pred_categ2[ibin].Fill(Pred_[i])
                hist_chi2_pred_categ2[ibin].Fill(trimAhcal_chi2Reco[i])
                hist_chi2_norm_predTrue_categ2[ibin].Fill(norm_chi2)
                hist_model_norm_predTrue_categ2[ibin].Fill(norm_gnn)
                h2d_fracleakVsAbs_categ2[ibin].Fill(ratio,frac_leak[i])
                h2d_fracleakVsAbsv1_categ2[ibin].Fill(ratio,frac_leak_v1)
                h2d_fracLeakVsGnn_categ2[ibin].Fill(Pred_[i],frac_leak[i])
                h2d_fracLeakVsChi2_categ2[ibin].Fill(trimAhcal_chi2Reco[i],frac_leak[i])
                h1d_lekageEE_categ2[ibin].Fill(Total_EE[i])
                h1d_lekageFH_categ2[ibin].Fill(Total_FH[i])
                h1d_lekageAH_categ2[ibin].Fill(energyLeakTransverseAH_[i])
                h1d_beforeEE_categ2[ibin].Fill(energyLeakResidual_[i])
                h1d_long_categ2[ibin].Fill(energyLeakLongitudinal_[i])
                
                h2d_leakvsGnn_categ[ibin][1][0].Fill(Pred_[i],energyLeakResidual_[i])
                h2d_leakvsGnn_categ[ibin][1][1].Fill(Pred_[i],Total_EE[i])
                h2d_leakvsGnn_categ[ibin][1][2].Fill(Pred_[i],Total_FH[i])
                h2d_leakvsGnn_categ[ibin][1][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
                h2d_leakvsGnn_categ[ibin][1][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
                h2d_leakvsChi2_categ[ibin][1][0].Fill(Pred_[i],energyLeakResidual_[i])
                h2d_leakvsChi2_categ[ibin][1][1].Fill(Pred_[i],Total_EE[i])
                h2d_leakvsChi2_categ[ibin][1][2].Fill(Pred_[i],Total_FH[i])
                h2d_leakvsChi2_categ[ibin][1][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
                h2d_leakvsChi2_categ[ibin][1][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
                # h2d_pi0vsLeak_categ[ibin][1][0].Fill(total_pi0KE[i],energyLeakResidual_[i])
                # h2d_pi0vsLeak_categ[ibin][1][1].Fill(total_pi0KE[i],Total_EE[i])
                # h2d_pi0vsLeak_categ[ibin][1][2].Fill(total_pi0KE[i],Total_FH[i])
                # h2d_pi0vsLeak_categ[ibin][1][3].Fill(total_pi0KE[i],energyLeakTransverseAH_[i])
                # h2d_pi0vsLeak_categ[ibin][1][4].Fill(total_pi0KE[i],energyLeakLongitudinal_[i])

                # h2d_gnnVspi0_categ[ibin][1].Fill(Pred_[i],total_pi0KE[i])
                # h2d_chi2vspi0_categ[ibin][1].Fill(trimAhcal_chi2Reco[i],total_pi0KE[i])
                # h2d_Absvspi0_categ[ibin][1].Fill(total_lost,total_pi0KE[i])
                # h2d_fracAbs_pi0_categ[ibin][1].Fill(ratio_abs,total_pi0KE[i]/total_lost)
                h1d_totalleak_categ[ibin][1].Fill(total_leakage[i])
                h1d_fracleak_categ[ibin][1].Fill(frac_leak_v1)
                # h2d_gnnvsStdDev_categ[ibin][1][0].Fill(Pred_[i],stdDev_2d_EE)
                # h2d_gnnvsStdDev_categ[ibin][1][1].Fill(Pred_[i],stdDev_2d_FH)
                # h2d_gnnvsStdDev_categ[ibin][1][2].Fill(Pred_[i],stdDev_2d_AH)
                # h2d_chi2vsStdDev_categ[ibin][1][0].Fill(Pred_[i],stdDev_2d_EE)
                # h2d_chi2vsStdDev_categ[ibin][1][1].Fill(Pred_[i],stdDev_2d_FH)
                # h2d_chi2vsStdDev_categ[ibin][1][2].Fill(Pred_[i],stdDev_2d_AH)
                # h2d_stdDevvsleak_categ[ibin][1][0].Fill(stdDev_2d_EE,Total_EE[i])
                # h2d_stdDevvsleak_categ[ibin][1][1].Fill(stdDev_2d_FH,Total_FH[i])
                # h2d_stdDevvsleak_categ[ibin][1][2].Fill(stdDev_2d_AH,energyLeakTransverseAH_[i])
                # h2d_stdDevvsleak_categ[ibin][1][3].Fill(stdDev_2d_AH,energyLeakLongitudinal_[i])
                # h2d_stdDevvsleak_categ[ibin][1][4].Fill(stdDev_2d_EE,energyLeakResidual_[i])

                # h2d_stdDevvsleak_categ[ibin][1][0].Fill(Total_EE[i],stdDev_2d_EE)
                # h2d_stdDevvsleak_categ[ibin][1][1].Fill(Total_FH[i],stdDev_2d_FH)
                # h2d_stdDevvsleak_categ[ibin][1][2].Fill(energyLeakTransverseAH_[i],stdDev_2d_AH)
                # h2d_stdDevvsleak_categ[ibin][1][3].Fill(stdDev_2d_AH,energyLeakLongitudinal_[i])
                # h2d_stdDevvsleak_categ[ibin][1][4].Fill(stdDev_2d_EE,energyLeakResidual_[i])

                # h2d_pi0vsstdDev_categ[ibin][1][0].Fill(total_pi0KE[i],stdDev_2d_EE)
                # h2d_pi0vsstdDev_categ[ibin][1][1].Fill(total_pi0KE[i],stdDev_2d_FH)
                # h2d_pi0vsstdDev_categ[ibin][1][2].Fill(total_pi0KE[i],stdDev_2d_AH)


            elif(categ3):
                hist_model_pred_categ3[ibin].Fill(Pred_[i])
                hist_chi2_pred_categ3[ibin].Fill(trimAhcal_chi2Reco[i])
                hist_chi2_norm_predTrue_categ3[ibin].Fill(norm_chi2)
                hist_model_norm_predTrue_categ3[ibin].Fill(norm_gnn)
                h2d_fracleakVsAbs_categ3[ibin].Fill(ratio,frac_leak[i])
                h2d_fracleakVsAbsv1_categ3[ibin].Fill(ratio,frac_leak_v1)
                h2d_fracLeakVsGnn_categ3[ibin].Fill(Pred_[i],frac_leak[i])
                h2d_fracLeakVsChi2_categ3[ibin].Fill(trimAhcal_chi2Reco[i],frac_leak[i])
                h1d_lekageEE_categ3[ibin].Fill(Total_EE[i])
                h1d_lekageFH_categ3[ibin].Fill(Total_FH[i])
                h1d_lekageAH_categ3[ibin].Fill(energyLeakTransverseAH_[i])
                h1d_beforeEE_categ3[ibin].Fill(energyLeakResidual_[i])
                h1d_long_categ3[ibin].Fill(energyLeakLongitudinal_[i])
                
                h2d_leakvsGnn_categ[ibin][2][0].Fill(Pred_[i],energyLeakResidual_[i])
                h2d_leakvsGnn_categ[ibin][2][1].Fill(Pred_[i],Total_EE[i])
                h2d_leakvsGnn_categ[ibin][2][2].Fill(Pred_[i],Total_FH[i])
                h2d_leakvsGnn_categ[ibin][2][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
                h2d_leakvsGnn_categ[ibin][2][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
                h2d_leakvsChi2_categ[ibin][2][0].Fill(Pred_[i],energyLeakResidual_[i])
                h2d_leakvsChi2_categ[ibin][2][1].Fill(Pred_[i],Total_EE[i])
                h2d_leakvsChi2_categ[ibin][2][2].Fill(Pred_[i],Total_FH[i])
                h2d_leakvsChi2_categ[ibin][2][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
                h2d_leakvsChi2_categ[ibin][2][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
                # h2d_pi0vsLeak_categ[ibin][2][0].Fill(total_pi0KE[i],energyLeakResidual_[i])
                # h2d_pi0vsLeak_categ[ibin][2][1].Fill(total_pi0KE[i],Total_EE[i])
                # h2d_pi0vsLeak_categ[ibin][2][2].Fill(total_pi0KE[i],Total_FH[i])
                # h2d_pi0vsLeak_categ[ibin][2][3].Fill(total_pi0KE[i],energyLeakTransverseAH_[i])
                # h2d_pi0vsLeak_categ[ibin][2][4].Fill(total_pi0KE[i],energyLeakLongitudinal_[i])

                # h2d_gnnVspi0_categ[ibin][2].Fill(Pred_[i],total_pi0KE[i])
                # h2d_chi2vspi0_categ[ibin][2].Fill(trimAhcal_chi2Reco[i],total_pi0KE[i])
                # h2d_Absvspi0_categ[ibin][2].Fill(total_lost,total_pi0KE[i])
                # h2d_fracAbs_pi0_categ[ibin][2].Fill(ratio_abs,total_pi0KE[i]/total_lost)
                h1d_totalleak_categ[ibin][2].Fill( total_leakage[i])
                h1d_fracleak_categ[ibin][2].Fill(frac_leak_v1)
                # h2d_gnnvsStdDev_categ[ibin][2][0].Fill(Pred_[i],stdDev_2d_EE)
                # h2d_gnnvsStdDev_categ[ibin][2][1].Fill(Pred_[i],stdDev_2d_FH)
                # h2d_gnnvsStdDev_categ[ibin][2][2].Fill(Pred_[i],stdDev_2d_AH)
                # h2d_chi2vsStdDev_categ[ibin][2][0].Fill(Pred_[i],stdDev_2d_EE)
                # h2d_chi2vsStdDev_categ[ibin][2][1].Fill(Pred_[i],stdDev_2d_FH)
                # h2d_chi2vsStdDev_categ[ibin][2][2].Fill(Pred_[i],stdDev_2d_AH)
                # h2d_stdDevvsleak_categ[ibin][2][0].Fill(Total_EE[i],stdDev_2d_EE)
                # h2d_stdDevvsleak_categ[ibin][2][1].Fill(Total_FH[i],stdDev_2d_FH)
                # h2d_stdDevvsleak_categ[ibin][2][2].Fill(energyLeakTransverseAH_[i],stdDev_2d_AH)
                # h2d_stdDevvsleak_categ[ibin][2][3].Fill(stdDev_2d_AH,energyLeakLongitudinal_[i])
                # h2d_stdDevvsleak_categ[ibin][2][4].Fill(stdDev_2d_EE,energyLeakResidual_[i])


                # h2d_pi0vsstdDev_categ[ibin][2][0].Fill(total_pi0KE[i],stdDev_2d_EE)
                # h2d_pi0vsstdDev_categ[ibin][2][1].Fill(total_pi0KE[i],stdDev_2d_FH)
                # h2d_pi0vsstdDev_categ[ibin][2][2].Fill(total_pi0KE[i],stdDev_2d_AH)

            elif(categ4):
                hist_model_pred_categ4[ibin].Fill(Pred_[i])
                hist_chi2_pred_categ4[ibin].Fill(trimAhcal_chi2Reco[i])
                hist_chi2_norm_predTrue_categ4[ibin].Fill(norm_chi2)
                hist_model_norm_predTrue_categ4[ibin].Fill(norm_gnn)
                h2d_fracleakVsAbs_categ4[ibin].Fill(ratio,frac_leak[i])
                h2d_fracleakVsAbsv1_categ4[ibin].Fill(ratio,frac_leak_v1)
                h2d_fracLeakVsGnn_categ4[ibin].Fill(Pred_[i],frac_leak[i])
                h2d_fracLeakVsChi2_categ4[ibin].Fill(trimAhcal_chi2Reco[i],frac_leak[i])
                h1d_lekageEE_categ4[ibin].Fill(Total_EE[i])
                h1d_lekageFH_categ4[ibin].Fill(Total_FH[i])
                h1d_lekageAH_categ4[ibin].Fill(energyLeakTransverseAH_[i])
                h1d_beforeEE_categ4[ibin].Fill(energyLeakResidual_[i])
                h1d_long_categ4[ibin].Fill(energyLeakLongitudinal_[i])
                h2d_leakvsGnn_categ[ibin][3][0].Fill(Pred_[i],energyLeakResidual_[i])
                h2d_leakvsGnn_categ[ibin][3][1].Fill(Pred_[i],Total_EE[i])
                h2d_leakvsGnn_categ[ibin][3][2].Fill(Pred_[i],Total_FH[i])
                h2d_leakvsGnn_categ[ibin][3][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
                h2d_leakvsGnn_categ[ibin][3][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
                h2d_leakvsChi2_categ[ibin][3][0].Fill(Pred_[i],energyLeakResidual_[i])
                h2d_leakvsChi2_categ[ibin][3][1].Fill(Pred_[i],Total_EE[i])
                h2d_leakvsChi2_categ[ibin][3][2].Fill(Pred_[i],Total_FH[i])
                h2d_leakvsChi2_categ[ibin][3][3].Fill(Pred_[i],energyLeakTransverseAH_[i])
                h2d_leakvsChi2_categ[ibin][3][4].Fill(Pred_[i],energyLeakLongitudinal_[i])
                # h2d_pi0vsLeak_categ[ibin][3][0].Fill(total_pi0KE[i],energyLeakResidual_[i])
                # h2d_pi0vsLeak_categ[ibin][3][1].Fill(total_pi0KE[i],Total_EE[i])
                # h2d_pi0vsLeak_categ[ibin][3][2].Fill(total_pi0KE[i],Total_FH[i])
                # h2d_pi0vsLeak_categ[ibin][3][3].Fill(total_pi0KE[i],energyLeakTransverseAH_[i])
                # h2d_pi0vsLeak_categ[ibin][3][4].Fill(total_pi0KE[i],energyLeakLongitudinal_[i])

                # h2d_gnnVspi0_categ[ibin][3].Fill(Pred_[i],total_pi0KE[i])
                # h2d_chi2vspi0_categ[ibin][3].Fill(trimAhcal_chi2Reco[i],total_pi0KE[i])
                # h2d_Absvspi0_categ[ibin][3].Fill(total_lost,total_pi0KE[i])
                # h2d_fracAbs_pi0_categ[ibin][3].Fill(ratio_abs,total_pi0KE[i]/total_lost)
                h1d_totalleak_categ[ibin][3].Fill( total_leakage[i])
                h1d_fracleak_categ[ibin][3].Fill(frac_leak_v1)
                # h2d_gnnvsStdDev_categ[ibin][3][0].Fill(Pred_[i],stdDev_2d_EE)
                # h2d_gnnvsStdDev_categ[ibin][3][1].Fill(Pred_[i],stdDev_2d_FH)
                # h2d_gnnvsStdDev_categ[ibin][3][2].Fill(Pred_[i],stdDev_2d_AH)
                # h2d_chi2vsStdDev_categ[ibin][3][0].Fill(Pred_[i],stdDev_2d_EE)
                # h2d_chi2vsStdDev_categ[ibin][3][1].Fill(Pred_[i],stdDev_2d_FH)
                # h2d_chi2vsStdDev_categ[ibin][3][2].Fill(Pred_[i],stdDev_2d_AH)
                
                # h2d_stdDevvsleak_categ[ibin][3][0].Fill(Total_EE[i],stdDev_2d_EE)
                # h2d_stdDevvsleak_categ[ibin][3][1].Fill(Total_FH[i],stdDev_2d_FH)
                # h2d_stdDevvsleak_categ[ibin][3][2].Fill(energyLeakTransverseAH_[i],stdDev_2d_AH)
                # h2d_stdDevvsleak_categ[ibin][3][3].Fill(stdDev_2d_AH,energyLeakLongitudinal_[i])
                # h2d_stdDevvsleak_categ[ibin][3][4].Fill(stdDev_2d_EE,energyLeakResidual_[i])


                # h2d_stdDevvsleak_categ[ibin][3][0].Fill(Total_EE[i],stdDev_2d_EE)
                # h2d_stdDevvsleak_categ[ibin][3][1].Fill(Total_FH[i],stdDev_2d_FH)
                # h2d_stdDevvsleak_categ[ibin][3][2].Fill(energyLeakTransverseAH_[i],stdDev_2d_AH)

                # h2d_pi0vsstdDev_categ[ibin][3][0].Fill(total_pi0KE[i],stdDev_2d_EE)
                # h2d_pi0vsstdDev_categ[ibin][3][1].Fill(total_pi0KE[i],stdDev_2d_FH)
                # h2d_pi0vsstdDev_categ[ibin][3][2].Fill(total_pi0KE[i],stdDev_2d_AH)

                
fout.cd()
for i_en in range(len(Energy)):
    h1d_totalVisi[i_en].Write()
    h1d_fractotalVisi[i_en].Write()
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
    hist_chi2_pred_categ1[i_en].Write()
    hist_chi2_norm_predTrue_categ1[i_en].Write()
    hist_chi2_pred_categ3[i_en].Write()
    hist_chi2_norm_predTrue_categ3[i_en].Write()
    hist_model_norm_predTrue_categ1[i_en].Write()
    hist_model_pred_categ1[i_en].Write()

    hist_model_norm_predTrue_categ3[i_en].Write()
    hist_model_pred_categ3[i_en].Write()
    hist_chi2_pred_categ2[i_en].Write()
    hist_chi2_norm_predTrue_categ2[i_en].Write()
    hist_model_norm_predTrue_categ2[i_en].Write()
    hist_model_pred_categ2[i_en].Write()
    hist_chi2_pred_categ4[i_en].Write()
    hist_chi2_norm_predTrue_categ4[i_en].Write()
    hist_model_norm_predTrue_categ4[i_en].Write()
    hist_model_pred_categ4[i_en].Write()

    h1d_predchi2[i_en].Write()
    h1d_predModel[i_en].Write()
    h1d_predModel_train[i_en].Write()
    h1d_predModel_valid[i_en].Write()
    h1d_totalAbs[i_en].Write()
    h1d_fracAbs[i_en].Write()
    h1d_totalLeak[i_en].Write()
    h1d_fracleak[i_en].Write()
    h1d_fracleak_total[i_en].Write()
    h1d_fracvisi_true[i_en].Write()
    h2d_fracleakVsabs[i_en].Write()
    h2d_fracleakVsabs_v1[i_en].Write()
    h2d_fracleakVsAbs_categ1[i_en].Write()
    h2d_fracleakVsAbsv1_categ1[i_en].Write()
    h2d_fracleakVsAbs_categ2[i_en].Write()
    h2d_fracleakVsAbsv1_categ2[i_en].Write()
    h2d_fracleakVsAbs_categ3[i_en].Write()
    h2d_fracleakVsAbs_categ4[i_en].Write()
    h2d_fracleakVsAbsv1_categ4[i_en].Write()
    h2d_fracleakVsAbsv1_categ3[i_en].Write()

    h2d_fracLeakVsGnn_categ1[i_en].Write()
    h2d_fracLeakVsGnn_categ2[i_en].Write()
    h2d_fracLeakVsGnn_categ3[i_en].Write()
    h2d_fracLeakVsGnn_categ4[i_en].Write()

    h2d_fracLeakVsChi2_categ1[i_en].Write()
    h2d_fracLeakVsChi2_categ2[i_en].Write()
    h2d_fracLeakVsChi2_categ3[i_en].Write()
    h2d_fracLeakVsChi2_categ4[i_en].Write()
    h2d_fracleakVsGnn[i_en].Write()
    h2d_fracleakVsChi2[i_en].Write()
    h1d_lekageEE_categ1[i_en].Write()
    h1d_lekageFH_categ1[i_en].Write()
    h1d_lekageAH_categ1[i_en].Write()
    h1d_beforeEE_categ1[i_en].Write()
    h1d_long_categ1[i_en].Write()

    h1d_lekageEE_categ2[i_en].Write()
    h1d_lekageFH_categ2[i_en].Write()
    h1d_lekageAH_categ2[i_en].Write()
    h1d_beforeEE_categ2[i_en].Write()
    h1d_long_categ2[i_en].Write()

    h1d_lekageEE_categ3[i_en].Write()
    h1d_lekageFH_categ3[i_en].Write()
    h1d_lekageAH_categ3[i_en].Write()
    h1d_beforeEE_categ3[i_en].Write()
    h1d_long_categ3[i_en].Write()
    h1d_lekageEE_categ4[i_en].Write()
    h1d_lekageFH_categ4[i_en].Write()
    h1d_lekageAH_categ4[i_en].Write()
    h1d_beforeEE_categ4[i_en].Write()
    h1d_long_categ4[i_en].Write()
    
    h1d_lekageEE[i_en].Write()
    h1d_lekageFH[i_en].Write()
    h1d_lekageAH[i_en].Write()
    h1d_beforeEE[i_en].Write()
    h1d_long[i_en].Write()
    h1d_SSloc[i_en].Write()
    h1d_SSloc_v1[i_en].Write()
    for i_b in range(len(baseline)):
        h2d_leakvsGnn[i_en][i_b].Write()
        h2d_fracleakvsGnn[i_en][i_b].Write()
        h2d_vfracleakvsGnn[i_en][i_b].Write()
        h2d_leakvsChi2[i_en][i_b].Write()
        h2d_pi0vsLeak[i_en][i_b].Write()
    for i_b in range(len(baseline1)):
        h2d_gnnvsStdDev[i_en][i_b].Write()
        h2d_chi2vsStdDev[i_en][i_b].Write()
        h2d_pi0vsstdDev[i_en][i_b].Write()
    for i_b in range(len(baseline2)):
        h2d_stdDevvsleak[i_en][i_b].Write()
        
    h2d_gnnVspi0[i_en].Write()
    h2d_chi2vspi0[i_en].Write()
    h2d_Absvspi0[i_en].Write()
    h2d_fracAbs_pi0[i_en].Write()
    
    for i_c in range(len(baseline0)):
        for i_b in range(len(baseline)):
            h2d_leakvsGnn_categ[i_en][i_c][i_b].Write()
            h2d_leakvsChi2_categ[i_en][i_c][i_b].Write()
            h2d_pi0vsLeak_categ[i_en][i_c][i_b].Write()
        for i_b in range(len(baseline1)):
            h2d_gnnvsStdDev_categ[i_en][i_c][i_b].Write()
            h2d_chi2vsStdDev_categ[i_en][i_c][i_b].Write()
            h2d_pi0vsstdDev_categ[i_en][i_c][i_b].Write()
        for i_b in range(len(baseline2)):
            h2d_stdDevvsleak_categ[i_en][i_c][i_b].Write()
        h2d_gnnVspi0_categ[i_en][i_c].Write()
        h2d_chi2vspi0_categ[i_en][i_c].Write()
        h2d_Absvspi0_categ[i_en][i_c].Write()
        h2d_fracAbs_pi0_categ[i_en][i_c].Write()


        h1d_totalleak_categ[i_en][i_c].Write()
        h1d_fracleak_categ[i_en][i_c].Write()



fout.Close()


