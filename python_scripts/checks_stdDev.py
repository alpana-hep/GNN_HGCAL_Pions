import pandas as pd
import numpy as np
import uproot
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import awkward as ak
import ROOT
import math
path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/hadintmoreInfo/skimmed_ntuple_sim_discrete_pionHadInfor_3inone.root"
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
frac_leak = total_leakage/trueBeamEnergy
path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/hadintmoreInfo/skimmed_ntuple_sim_discrete_chi2method01_3inOne.root"
tree1 = uproot.open("%s:%s"%(path, treeName))

rechit_shower_start_layer=tree1['rechit_shower_start_layer'].array()
trimAhcal_chi2Reco = tree1['trimAhcal_chi2Reco'].array()


pred_v2 ="./Hadinfor_inferen/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])
predic=preds_ratio
print(len(predic))#test_0to5M_fix_raw_ahcalTrim_up                                                                                                      
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_Sim_50100300_hadinfor/trimAhcal/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
#rawE= rawE[0:836658]
Pred_ = rawE * predic
trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_Sim_50100300_hadinfor/trimAhcal/trueE.pickle"
Hit_X ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_Sim_50100300_hadinfor/trimAhcal/Hit_X.pickle"
Hit_XPickle = open(Hit_X,"rb")
Hit_X_pkl =pickle.load(Hit_XPickle)
print(Hit_X_pkl)

Hit_Y ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_Sim_50100300_hadinfor/trimAhcal/Hit_Y.pickle"
Hit_YPickle = open(Hit_Y,"rb")
Hit_Y_pkl =pickle.load(Hit_YPickle)

print(Hit_Y_pkl)
Hit_Z ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_Sim_50100300_hadinfor/trimAhcal/Hit_Z.pickle"
Hit_ZPickle = open(Hit_Z,"rb")
Hit_Z_pkl =pickle.load(Hit_ZPickle)
print(Hit_Z_pkl)

trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
print(trueEn_pkl[0:836658])
#trueEn_pkl=trueEn_pkl[0:836658]
gnnresp= Pred_/trueEn_pkl
chi2resp= trimAhcal_chi2Reco/trueEn_pkl
frac_abs=(energyLostEE_+energyLostFH_+energyLostBH_)/trueEn_pkl
total_abs=energyLostEE_+energyLostFH_+energyLostBH_
print(gnnresp)
print(chi2resp)
print(frac_abs)
#eneFracAbs= 
print(len(trueEn_pkl))
eveList= np.arange(len(trueEn_pkl),1)
print(len(eveList),"total length")
#if(gnnresp>0.9 and np.logical_and(gnnresp<1.1 and chi2resp<0.5)
en_list=[50]#,100,300]
h1d_absr=[]
h1d_leak=[]
h1d_fracabsr=[]
h1d_fracleak=[]
h2d_absr_vsGnn=[]
h2d_chi2vsGnn=[]
h2d_absr_vsChi2=[]
h2d_stdX=[]
h2d_stdY=[]
h2d_stdZ=[]
h2d_Gnnvs3dStddev=[]
h2d_absVsstd=[]
h1d_dR_min=[]
h1d_dR_max=[]
h2d_dR_min_vsGnn=[]
h2d_dR_max_vsGnn=[]
h1d_dR=[]
h2d_dR_vsGnn=[]
h1d_dR_4d_min=[]
h1d_dR_4d_max=[]
h2d_dR_4d_min_vsGnn=[]
h2d_dR_4d_max_vsGnn=[]
h1d_dR_4d=[]
h2d_dR_4d_vsGnn=[]

import ROOT
fout= ROOT.TFile("hist_checks_dR_86epochs.root", 'RECREATE')
for i_hist in range(len(en_list)):
    xhigh = 3.0*en_list[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(en_list[i_hist])
    h1d_dR_min.append(ROOT.TH1F("h1d_dR_min_%s"%name1,"",500,0,500))
    h1d_dR_max.append(ROOT.TH1F("h1d_dR_max_%s"%name1,"",500,0,500))
    h2d_dR_min_vsGnn.append(ROOT.TH2F("h2d_dR_min_vsGnn_%s"%name1,"",500,0,500,500,0,500))
    h2d_dR_max_vsGnn.append(ROOT.TH2F("h2d_dR_max_vsGnn_%s"%name1,"",500,0,500,500,0,500))
    h1d_dR.append(ROOT.TH1F("h1d_dR_%s"%name1,"",500,0,500))
    h2d_dR_vsGnn.append(ROOT.TH2F("h2d_dR_vsGnn_%s"%name1,"",500,0,500,500,0,500))
    h1d_dR_4d_min.append(ROOT.TH1F("h1d_dR_4d_min_%s"%name1,"",500,0,500))
    h1d_dR_4d_max.append(ROOT.TH1F("h1d_dR_4d_max_%s"%name1,"",500,0,500))
    h2d_dR_4d_min_vsGnn.append(ROOT.TH2F("h2d_dR_4d_min_vsGnn_%s"%name1,"",500,0,500,500,0,500))
    h2d_dR_4d_max_vsGnn.append(ROOT.TH2F("h2d_dR_4d_max_vsGnn_%s"%name1,"",500,0,500,500,0,500))
    h1d_dR_4d.append(ROOT.TH1F("h1d_dR_4d_%s"%name1,"",500,0,500))
    h2d_dR_4d_vsGnn.append(ROOT.TH2F("h2d_dR_4d_vsGnn_%s"%name1,"",500,0,500,500,0,500))

    h2d_absVsstd.append(ROOT.TH2F("h2d_absVsstd_%s"%name1,"",500,0,500,500,0,100))
    h2d_stdX.append(ROOT.TH2F("h2d_stdX_%s"%name1,"h2d_stdX_",500,0,500,500,0,100))
    h2d_stdY.append(ROOT.TH2F("h2d_stdY_%s"%name1,"h2d_stdY_",500,0,500,500,0,100))
    h2d_stdZ.append(ROOT.TH2F("h2d_stdZ_%s"%name1,"h2d_stdZ_",500,0,500,500,0,100))
    
    h1d_absr.append(ROOT.TH1F("h1d_absrb_%s"%name1,"h1d_absrb",500,0,xhigh))
    h1d_leak.append(ROOT.TH1F("h1d_leak_%s"%name1,"h1d_leak",500,0,xhigh))
    h1d_fracabsr.append(ROOT.TH1F("h1d_fracabsrb_%s"%name1,"h1d_fracabsrb",500,0,2))
    h1d_fracleak.append(ROOT.TH1F("h1d_fracleak_%s"%name1,"h1d_fracleak",500,0,2))
    h2d_absr_vsGnn.append(ROOT.TH2F("h2d_absr_vsGnn_%s"%name1,"h2d_absr_vsGnn",500,0,xhigh,500,0,xhigh))
    h2d_absr_vsChi2.append(ROOT.TH2F("h2d_absr_vsChi2_%s"%name1,"h2d_absr_vsChi2",500,0,xhigh,500,0,xhigh))
    h2d_chi2vsGnn.append(ROOT.TH2F("h2d_chi2vsGnn_%s"%name1,"h2d_chi2vsGnn",500,0,xhigh,500,0,xhigh))
    h2d_Gnnvs3dStddev.append(ROOT.TH2F("h2d_Gnnvs3dStddev_%s"%name1,"h2d_Gnnvs3dStddev_",500,0,xhigh,500,0,100))

for k in range(len(trueEn_pkl)):
    x_pion = np.array(Hit_X_pkl[k])
    y_pion = np.array(Hit_Y_pkl[k])
    z_pion =np.array(Hit_Z_pkl[k])
    rec_en = RechitEn_pkl[k]    
    en= rec_en
    frac_e= frac_abs[k]
    frac_l= frac_leak[k]
    e = trueEn_pkl[k]
    e_gnn= Pred_[k]
    e_chi2=trimAhcal_chi2Reco[k]
    #z_ss= z_pion[
    X_ien= np.sum(x_pion*rec_en)/(np.mean(rec_en)*len(x_pion))
    Y_ien = np.sum(y_pion*rec_en)/(np.mean(rec_en)*len(y_pion))
    Z_ien = np.sum(z_pion*rec_en)/(np.mean(rec_en)*len(z_pion))
    #print(X_ien,Y_ien,Z_ien)
    std_Xien= math.sqrt(np.sum(np.square((x_pion-X_ien))/len(x_pion)))
    std_Yien= math.sqrt(np.sum(np.square((y_pion-Y_ien))/len(y_pion)))
    std_Zien= math.sqrt(np.sum(np.square((z_pion-Z_ien))/len(z_pion)))
    #print(std_Xien,std_Yien,std_Zien)
    stdDev_x= np.std(x_pion)
    stdDev_y= np.std(y_pion)
    stdDev_z= np.std(z_pion)
    stdDev_en=np.std(rec_en)
    std_3d = math.sqrt((stdDev_x*stdDev_x)+(stdDev_y*stdDev_y)+(stdDev_z*stdDev_z)+(stdDev_en*stdDev_en))
    dR = np.square(x_pion)+np.square(y_pion)+np.square(z_pion)
    dR_4d = np.square(x_pion)+np.square(y_pion)+np.square(z_pion)+np.square(rec_en)
    for i in range(len(en_list)):
        if(e==en_list[i]):
            h1d_dR_min[i].Fill(math.sqrt(np.min(dR)))
            h1d_dR_max[i].Fill(math.sqrt(np.max(dR)))
            h2d_dR_min_vsGnn[i].Fill(e_gnn,math.sqrt(np.min(dR)))
            h2d_dR_max_vsGnn[i].Fill(e_gnn,math.sqrt(np.max(dR)))
            h1d_dR[i].Fill(math.sqrt(np.max(dR))-math.sqrt(np.min(dR)))
            h2d_dR_vsGnn[i].Fill(e_gnn,math.sqrt(np.max(dR))-math.sqrt(np.min(dR)))
            
            h1d_dR_4d_min[i].Fill(math.sqrt(np.min(dR_4d)))
            h1d_dR_4d_max[i].Fill(math.sqrt(np.max(dR_4d)))
            h2d_dR_4d_min_vsGnn[i].Fill(e_gnn,math.sqrt(np.min(dR_4d)))
            h2d_dR_4d_max_vsGnn[i].Fill(e_gnn,math.sqrt(np.max(dR_4d)))
            h1d_dR_4d[i].Fill(math.sqrt(np.max(dR_4d))-math.sqrt(np.min(dR_4d)))
            h2d_dR_4d_vsGnn[i].Fill(e_gnn,math.sqrt(np.max(dR_4d))-math.sqrt(np.min(dR_4d)))
                              
            h2d_absr_vsChi2[i].Fill(e_chi2,total_abs[k])
            h2d_absr_vsGnn[i].Fill(e_gnn,total_abs[k])
            h1d_fracleak[i].Fill(frac_l)
            h1d_fracabsr[i].Fill(frac_e)
            h1d_absr[i].Fill(total_abs[k])
            h1d_leak[i].Fill(total_leakage[k])
            h2d_chi2vsGnn[i].Fill(e_gnn,e_chi2)
            h2d_stdX[i].Fill(e_gnn,std_Xien)
            h2d_stdY[i].Fill(e_gnn,std_Yien)
            h2d_stdZ[i].Fill(e_gnn,std_Zien)
            
            h2d_Gnnvs3dStddev[i].Fill(e_gnn,std_3d)
            h2d_absVsstd[i].Fill(total_abs[k],std_3d)
        #     fig = plt.figure(figsize = (15, 10)) 
        #     ax = plt.axes(projection ="3d") 
            
        #     #plt.rcParams["figure.figsize"] = [7.50, 7.50]
        #     #plt.rcParams["figure.autolayout"] = True
        #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            
        #     ax.view_init(elev=5,azim=-80)          
        #     e = trueEn_pkl[k]
        #     #l = "trueE="+str(e)[:3]+ " GeV" +"\n"+"GNN="+str(e_gnn)[:3]+" GeV"+"\n"
        #     l1= "trueE="+str(e)[:3] 
        #     l2="GNN="+str(e_gnn)[:3]
        #     l3="Chi2="+str(e_chi2)[:3]+"GeV"
        #     l4="frac_abs="+str(frac_e)[:3]
        #     l5="frac_leak="+str(frac_l)[:3]
        #     l=l1+" "+l2+" "+l3+" "+l4+" "+l5
        #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="jet",vmin=0,vmax=5)#color="blue")
        #     points = np.array([[13, -7.5,-7.5],
        #                        [13, 7.5, -7.5 ],
        #                        [13, 7.5, 7.5],
        #                        [13,-7.5, 7.5],
        #                        [54,-7.5, -7.5],
        #                        [54, 7.5, -7.5],
        #                        [54, 7.5, 7.5],
        #                        [54, -7.5, 7.5 ]])
        #     Z = points
        #     verts = [[Z[0],Z[1],Z[2],Z[3]],
        #              [Z[4],Z[5],Z[6],Z[7]],
        #              [Z[0],Z[1],Z[5],Z[4]],
        #              [Z[2],Z[3],Z[7],Z[6]],
        #              [Z[1],Z[2],Z[6],Z[5]],
        #              [Z[4],Z[7],Z[3],Z[0]]]
        #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
        #     points = np.array([[64, -18.5,-18.5],
        #                        [64, 18.5, -18.5 ],
        #                        [64, 18.5, 18.5],
        #                        [64,-18.5, 18.5],
        #                        [152.5,-18.5, -18.5],
        #                        [152.5, 18.5, -18.5],
        #                        [152.5, 18.5, 18.5],
        #                        [152.5, -18.5, 18.5 ]])
        #     Z = points
        #     verts = [[Z[0],Z[1],Z[2],Z[3]],
        #              [Z[4],Z[5],Z[6],Z[7]],
        #              [Z[0],Z[1],Z[5],Z[4]],
        #              [Z[2],Z[3],Z[7],Z[6]],
        #              [Z[1],Z[2],Z[6],Z[5]],
        #              [Z[4],Z[7],Z[3],Z[0]]]
        #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
        #     points = np.array([[159, -36,-36],
        #                        [159, 36, -36 ],
        #                        [159, 36, 36],
        #                        [159,-36, 36],
        #                        [264,-36, -36],
        #                        [264, 36, -36],
        #                        [264, 36, 36],
        #                        [264, -36, 36 ]])
        #     Z = points
        #     verts = [[Z[0],Z[1],Z[2],Z[3]],
        #              [Z[4],Z[5],Z[6],Z[7]],
        #              [Z[0],Z[1],Z[5],Z[4]],
        #              [Z[2],Z[3],Z[7],Z[6]],
        #              [Z[1],Z[2],Z[6],Z[5]],
        #              [Z[4],Z[7],Z[3],Z[0]]]
        #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))

        #     ax.set_xlabel('z (cm)')
        #     ax.set_ylabel('x (cm)')
        #     ax.set_zlabel('y (cm)')        
        #     ax.set_xlim([0,280])
        #     r = 36.5
        #     ax.set_ylim([-r,r])
        #     ax.set_zlim([-r,r])        
        #     cbar = fig.colorbar(sc,shrink = 0.5, aspect = 10)
        #     cbar.set_label('rechit en (GeV)')
        #     ax.text(0, 40, 40, l, fontsize = 15, color="r")
        #     fp = "./Results_Hadmoreinfor/binWise_chi2GnnChecks/"+str(k)+"_evedisplay_Gnnunity_chi2bad_"+str(e)+".png"
            
        #     plt.savefig(fp)
        # else:
        #     continue


fout.cd()
for i in range(len(en_list)):
    h1d_absr[i].Write()
    h1d_leak[i].Write()
    h1d_fracabsr[i].Write()
    h1d_fracleak[i].Write()
    h2d_absr_vsGnn[i].Write()
    h2d_absr_vsChi2[i].Write()
    h2d_chi2vsGnn[i].Write()
    h2d_stdX[i].Write()
    h2d_stdY[i].Write()
    h2d_stdZ[i].Write()
    h2d_Gnnvs3dStddev[i].Write()
    h2d_absVsstd[i].Write()
    h1d_dR_min[i].Write()
    h1d_dR_max[i].Write()
    h2d_dR_min_vsGnn[i].Write()
    h2d_dR_max_vsGnn[i].Write()
    h1d_dR[i].Write()
    h2d_dR_vsGnn[i].Write()
    h1d_dR_4d_min[i].Write()
    h1d_dR_4d_max[i].Write()
    h2d_dR_4d_min_vsGnn[i].Write()
    h2d_dR_4d_max_vsGnn[i].Write()
    h1d_dR_4d[i].Write()
    h2d_dR_4d_vsGnn[i].Write()

fout.Close()
    # elif(gnnresp[k]<0.7 and chi2resp[k]>0.9 and chi2resp[k]<1.1):
    #     fig = plt.figure(figsize = (15, 10))
    #     ax = plt.axes(projection ="3d")
    #     # plt.rcParams["figure.figsize"] = [7.50, 7.50]
    #     # plt.rcParams["figure.autolayout"] = True
    #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    #     ax.view_init(elev=5,azim=-80)
    #     e = trueEn_pkl[k]
    #     #l = "trueE="+str(e)[:3]+ " GeV"
    #     l1= "trueE="+str(e)[:3]
    #     l2="GNN="+str(e_gnn)[:3]
    #     l3="Chi2="+str(e_chi2)[:3]+" GeV"
    #     l4="frac_abs="+str(frac_e)[:3]
    #     l5="frac_leak="+str(frac_l)[:3]
    #     l=l1+" "+l2+" "+l3+" "+l4+" "+l5

    #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="jet",vmin=0,vmax=5) #color="blue")
    #     points = np.array([[13, -7.5,-7.5],
    #               [13, 7.5, -7.5 ],
    #               [13, 7.5, 7.5],
    #               [13,-7.5, 7.5],
    #               [54,-7.5, -7.5],
    #               [54, 7.5, -7.5],
    #               [54, 7.5, 7.5],
    #               [54, -7.5, 7.5 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
    #     points = np.array([[64, -18.5,-18.5],
    #                        [64, 18.5, -18.5 ],
    #                        [64, 18.5, 18.5],
    #                        [64,-18.5, 18.5],
    #                        [152.5,-18.5, -18.5],
    #                        [152.5, 18.5, -18.5],
    #                        [152.5, 18.5, 18.5],
    #                        [152.5, -18.5, 18.5 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
    #     points = np.array([[159, -36,-36],
    #               [159, 36, -36 ],
    #               [159, 36, 36],
    #               [159,-36, 36],
    #               [264,-36, -36],
    #               [264, 36, -36],
    #               [264, 36, 36],
    #               [264, -36, 36 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))

    #     ax.set_xlabel('z (cm)')
    #     ax.set_ylabel('x (cm)')
    #     ax.set_zlabel('y (cm)')
    #     ax.set_xlim([0,280])
    #     r = 36.5
    #     ax.set_ylim([-r,r])
    #     ax.set_zlim([-r,r])
    #     fig.colorbar(sc,shrink = 0.5, aspect = 10)
                        
    #     ax.text(0, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./Results_Hadmoreinfor/binWise_chi2GnnChecks/"+str(k)+"_evedisplay_Gnn_bad_chi2Good.png"

    #     plt.savefig(fp)

    # elif(gnnresp[k]>1.5 and chi2resp[k]>0.9 and chi2resp[k]<1.1):
    #     fig = plt.figure(figsize = (15, 10))
    #     ax = plt.axes(projection ="3d")
    #     # plt.rcParams["figure.figsize"] = [7.50, 7.50]
    #     # plt.rcParams["figure.autolayout"] = True
    #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
    #     ax.view_init(elev=5,azim=-80)
    #     e = trueEn_pkl[k]
    #     #l = "trueE="+str(e)[:3]+ " GeV"
    #     l1= "trueE="+str(e)[:3]
    #     l2="GNN="+str(e_gnn)[:3]
    #     l3="Chi2="+str(e_chi2)[:3]+" GeV"
    #     l4="frac_abs="+str(frac_e)[:3]
    #     l5="frac_leak="+str(frac_l)[:3]
    #     l=l1+" "+l2+" "+l3+" "+l4+" "+l5

    #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="jet",vmin=0,vmax=5)#color="blue")
    #     points = np.array([[13, -7.5,-7.5],
    #               [13, 7.5, -7.5 ],
    #               [13, 7.5, 7.5],
    #               [13,-7.5, 7.5],
    #               [54,-7.5, -7.5],
    #               [54, 7.5, -7.5],
    #               [54, 7.5, 7.5],
    #               [54, -7.5, 7.5 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
    #     points = np.array([[64, -18.5,-18.5],
    #                        [64, 18.5, -18.5 ],
    #                        [64, 18.5, 18.5],
    #                        [64,-18.5, 18.5],
    #                        [152.5,-18.5, -18.5],
    #                        [152.5, 18.5, -18.5],
    #                        [152.5, 18.5, 18.5],
    #                        [152.5, -18.5, 18.5 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
    #     points = np.array([[159, -36,-36],
    #               [159, 36, -36 ],
    #               [159, 36, 36],
    #               [159,-36, 36],
    #               [264,-36, -36],
    #               [264, 36, -36],
    #               [264, 36, 36],
    #               [264, -36, 36 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))

    #     ax.set_xlabel('z (cm)')
    #     ax.set_ylabel('x (cm)')
    #     ax.set_zlabel('y (cm)')
    #     ax.set_xlim([0,280])
    #     r = 36.5
    #     ax.set_ylim([-r,r])
    #     ax.set_zlim([-r,r])
    #     fig.colorbar(sc,shrink = 0.5, aspect = 10)
    #     ax.text(0, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./Results_Hadmoreinfor/binWise_chi2GnnChecks/"+str(k)+"_evedisplay_Gnnhigh_chi2good.png"

    #     plt.savefig(fp)

    # elif(gnnresp[k]>0.9 and gnnresp[k]<1.1 and chi2resp[k]>1.5):
    #     fig = plt.figure(figsize = (15, 10))
    #     ax = plt.axes(projection ="3d")
    #     # plt.rcParams["figure.figsize"] = [7.50, 7.50]
    #     # plt.rcParams["figure.autolayout"] = True
    #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
    #     ax.view_init(elev=5,azim=-80)
    #     e = trueEn_pkl[k]
    #     #l = "trueE="+str(e)[:3]+ " GeV"
    #     l1= "trueE="+str(e)[:3]
    #     l2="GNN="+str(e_gnn)[:3]
    #     l3="Chi2="+str(e_chi2)[:3]+" GeV"
    #     l4="frac_abs="+str(frac_e)[:3]
    #     l5="frac_leak="+str(frac_l)[:3]    
    #     l = l1+" "+l2+" "+l3+" "+l4+" "+l5

        
    #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="jet",vmin=0,vmax=5) #color="blue")
    #     points = np.array([[13, -7.5,-7.5],
    #               [13, 7.5, -7.5 ],
    #               [13, 7.5, 7.5],
    #               [13,-7.5, 7.5],
    #               [54,-7.5, -7.5],
    #               [54, 7.5, -7.5],
    #               [54, 7.5, 7.5],
    #               [54, -7.5, 7.5 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
    #     points = np.array([[64, -18.5,-18.5],
    #                        [64, 18.5, -18.5 ],
    #                        [64, 18.5, 18.5],
    #                        [64,-18.5, 18.5],
    #                        [152.5,-18.5, -18.5],
    #                        [152.5, 18.5, -18.5],
    #                        [152.5, 18.5, 18.5],
    #                        [152.5, -18.5, 18.5 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
    #     points = np.array([[159, -36,-36],
    #               [159, 36, -36 ],
    #               [159, 36, 36],
    #               [159,-36, 36],
    #               [264,-36, -36],
    #               [264, 36, -36],
    #               [264, 36, 36],
    #               [264, -36, 36 ]])
    #     Z = points
    #     verts = [[Z[0],Z[1],Z[2],Z[3]],
    #              [Z[4],Z[5],Z[6],Z[7]],
    #              [Z[0],Z[1],Z[5],Z[4]],
    #              [Z[2],Z[3],Z[7],Z[6]],
    #              [Z[1],Z[2],Z[6],Z[5]],
    #              [Z[4],Z[7],Z[3],Z[0]]]
    #     ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))

    #     ax.set_xlabel('z (cm)')
    #     ax.set_ylabel('x (cm)')
    #     ax.set_zlabel('y (cm)')
    #     ax.set_xlim([0,280])
    #     r = 36.5
    #     ax.set_ylim([-r,r])
    #     ax.set_zlim([-r,r])
    #     fig.colorbar(sc,shrink = 0.5, aspect = 10)
    #     ax.text(0, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./Results_Hadmoreinfor/binWise_chi2GnnChecks/"+str(k)+"_evedisplay_Gnnunity_chi2high.png"

    #     plt.savefig(fp)
    # else:
    #     continue
# for k in range(eveList1.trueBeamEnergy.size):
#     x_pion = eveList1.combined_rechit_x.values[k]
#     y_pion = eveList1.combined_rechit_y.values[k]
#     z_pion = eveList1.combined_rechit_z.values[k]
#     #en = eve.combined_rechits_energy.values[k]

#     fig = plt.figure(figsize = (15, 10)) 

#     ax = plt.axes(projection ="3d") 
#     ax.view_init(elev=5,azim=-80)  

#     e = eveList1.trueBeamEnergy.values[k]

#     #l = "trueE="+str(ssl)+" (z~"+str(lay_zs[ssl-1])[:3]+"GeV)"
#     l = "trueE="+str(e)[:3]+ " GeV"

#     ax.scatter3D(z_pion, x_pion, y_pion, color = "blue",label=l)



#     ax.set_xlabel('z (cm)')
#     ax.set_ylabel('x (cm)')
#     ax.set_zlabel('y (cm)')

#     ax.set_xlim([0,280])
#     r = 36.5
#     ax.set_ylim([-r,r])
#     ax.set_zlim([-r,r])

#     #plt.text(0.5, 0.5, "aa")#, fontsize=12)
#     #plt.text(0.5, 0.5, 'matplotlib', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#     ax.text(0, 40, 40, l, fontsize = 13, color="r")
  
#     c= plt.colorbar(sc, ax=ax)
#     #plt.clim(0, 150)
#     plt.title("")
#     plt.show()
#     fp = "./Results_Hadmoreinfor/binWise_chi2GnnChecks/"+str(k)+"_image.png"
    
#     plt.savefig(fp)
#     #plt.show()


# eveList2 = df.loc[(df.gnnResp[K]onse>0.9) & (df.gnnResp[K]onse<1.1) & (df.chi2Resp[K]onse>1.5)]
# eveList2["eneFracAbs"] = eveList2.apply(eneFracAbs, axis=1)
# print(eveList2.trueBeamEnergy.size)
# eveList2.head()
