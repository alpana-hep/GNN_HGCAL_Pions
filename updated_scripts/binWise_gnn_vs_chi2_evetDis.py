import pandas as pd
import numpy as np
import uproot
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import awkward as ak
import ROOT
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


leakageInEE = energyLeakTransverseEE_+energyLeak_EE_FH_
leakageInFH = energyLeakTransverseFH_+energyLeak_FH_AH_
frac_EE = leakageInEE/total_leakage
frac_FH = leakageInFH/total_leakage
frac_AH= energyLeakTransverseAH_/total_leakage
frac_long = energyLeakLongitudinal_/total_leakage
frac_res = energyLeakResidual_/total_leakage

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
#print(trueEn_pkl[0:836658])
Hit_X ="%s/Hit_X.pickle"%inpickle_folder
Hit_XPickle = open(Hit_X,"rb")
Hit_X_pkl =pickle.load(Hit_XPickle)
print(Hit_X_pkl)

Hit_Y ="%s/Hit_Y.pickle"%inpickle_folder
Hit_YPickle = open(Hit_Y,"rb")
Hit_Y_pkl =pickle.load(Hit_YPickle)

print(Hit_Y_pkl)
Hit_Z ="%s/Hit_Z.pickle"%inpickle_folder
Hit_ZPickle = open(Hit_Z,"rb")
Hit_Z_pkl =pickle.load(Hit_ZPickle)
print(Hit_Z_pkl)

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
en_list=[20,50,80,100,120,200,250,300]
#total_abs=(energyLostEE_[trueEn_pkl==50]+energyLostFH_[trueEn_pkl==50]+energyLostBH_[trueEn_pkl==50])/50
# mean_=[]
# stdDev =[]
# mean_.append(np.mean(total_abs))
# stdDev.append(np.std(total_abs))
# print(np.mean(total_abs),np.std(total_abs))

# total_abs=(energyLostEE_[trueEn_pkl==100]+energyLostFH_[trueEn_pkl==100]+energyLostBH_[trueEn_pkl==100])/100
# mean_.append(np.mean(total_abs))
# stdDev.append(np.std(total_abs))
# print(np.mean(total_abs),np.std(total_abs))

# total_abs=(energyLostEE_[trueEn_pkl==300]+energyLostFH_[trueEn_pkl==300]+energyLostBH_[trueEn_pkl==300])/300
# mean_.append(np.mean(total_abs))
# stdDev.append(np.std(total_abs))
# print(np.mean(total_abs),np.std(total_abs))
energy_,mean_,stdDev = np.loadtxt('./Absorber_resolutionFrac.txt',usecols=(0,1, 2), unpack=True)
min_ = int(sys.argv[4])
max_ = int(sys.argv[5])
count_ev=0
for k in range(min_, max_):
    x_pion = np.array(Hit_X_pkl[k])
    y_pion = np.array(Hit_Y_pkl[k])
    z_pion =np.array(Hit_Z_pkl[k])
    rec_en = RechitEn_pkl[k]    
    en= rec_en
    frac_e= frac_abs[k]
    ratio = frac_e
    frac_l= frac_leak[k]
    e = trueEn_pkl[k]
    e_gnn= Pred_[k]
    e_chi2=trimAhcal_chi2Reco[k]
    for i in range(len(en_list)):
        if(e==en_list[i]):
            categ1 = ratio>(mean_[i]-stdDev[i]) and ratio<(mean_[i]+stdDev[i])
            categ2 = ratio>(mean_[i]+stdDev[i])
            categ3 = ratio<(mean_[i]-stdDev[i]) and ratio>(mean_[i]-2.0*stdDev[i])
            categ4= ratio<(mean_[i]-2.0*stdDev[i])
            if(frac_l>0.4):
                count_ev+=1
                fig = plt.figure(figsize = (15, 10)) 
                ax = plt.axes(projection ="3d") 
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                ax.view_init(elev=5,azim=-80)          
                e = trueEn_pkl[k]
                l1= "trueE="+str(e)[:3] 
                l2="GNN="+str(e_gnn)[:3]
                l3="Chi2="+str(e_chi2)[:3]+"GeV"
                l4="frac_abs="+str(frac_e)[:3]
                l5="frac_leak="+str(frac_l)[:3]
                l=l1+" "+l2+" "+l3+" "+l4+" "+l5
                sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool",vmin=0,vmax=1.0)
                points = np.array([[13, -7.5,-7.5],
                                   [13, 7.5, -7.5 ],
                                   [13, 7.5, 7.5],
                                   [13,-7.5, 7.5],
                                   [54,-7.5, -7.5],
                                   [54, 7.5, -7.5],
                                   [54, 7.5, 7.5],
                                   [54, -7.5, 7.5 ]])
                Z = points
                verts = [[Z[0],Z[1],Z[2],Z[3]],
                         [Z[4],Z[5],Z[6],Z[7]],
                         [Z[0],Z[1],Z[5],Z[4]],
                         [Z[2],Z[3],Z[7],Z[6]],
                         [Z[1],Z[2],Z[6],Z[5]],
                         [Z[4],Z[7],Z[3],Z[0]]]
                ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
                points = np.array([[64, -18.5,-18.5],
                                   [64, 18.5, -18.5 ],
                                   [64, 18.5, 18.5],
                                   [64,-18.5, 18.5],
                                   [152.5,-18.5, -18.5],
                                   [152.5, 18.5, -18.5],
                                   [152.5, 18.5, 18.5],
                                   [152.5, -18.5, 18.5 ]])
                Z = points
                verts = [[Z[0],Z[1],Z[2],Z[3]],
                         [Z[4],Z[5],Z[6],Z[7]],
                         [Z[0],Z[1],Z[5],Z[4]],
                         [Z[2],Z[3],Z[7],Z[6]],
                         [Z[1],Z[2],Z[6],Z[5]],
                         [Z[4],Z[7],Z[3],Z[0]]]
                ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
                points = np.array([[159, -36,-36],
                                   [159, 36, -36 ],
                                   [159, 36, 36],
                                   [159,-36, 36],
                                   [264,-36, -36],
                                   [264, 36, -36],
                                   [264, 36, 36],
                                   [264, -36, 36 ]])
                Z = points
                verts = [[Z[0],Z[1],Z[2],Z[3]],
                         [Z[4],Z[5],Z[6],Z[7]],
                         [Z[0],Z[1],Z[5],Z[4]],
                         [Z[2],Z[3],Z[7],Z[6]],
                         [Z[1],Z[2],Z[6],Z[5]],
                         [Z[4],Z[7],Z[3],Z[0]]]
                ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))

                ax.set_xlabel('z (cm)')
                ax.set_ylabel('x (cm)')
                ax.set_zlabel('y (cm)')        
                ax.set_xlim([0,280])
                r = 36.5
                ax.set_ylim([-r,r])
                ax.set_zlim([-r,r])        
                cbar = fig.colorbar(sc,shrink = 0.5, aspect = 10)
                cbar.set_label('rechit en (GeV)')
                ax.text(-10, 45, 45, l, fontsize = 20, color="r")
                # L = "leak[beforeCE-E =" +str(frac_res[k])[:4]+", in CE-E="+str(frac_EE[k])[:4]+",in CE-H="+str(frac_FH[k])[:4]+",in AH="+str(frac_AH[k])[:4]+",Longi="+str(frac_long[k])[:4]+"]" 

                #ax.text(0,-48,-48,L,fontsize =17,color="b")
                resp =""
                resp1=""
                resp2 = ""
                resp3 =""
                resp4 =""
                resp5 =""
                if(gnnresp[k]<0.6):
                    resp1 ="gnn_bad"
                elif(gnnresp[k]>1.5):
                    resp2 = "high_gnn"
                elif(chi2resp[k]<0.6):
                    resp3 ="chi2_bad"
                elif(chi2resp[k]>1.5):
                    resp4="chi2_high"
                else:
                    resp5 = "Godd_chi2_gnn"
                resp = resp1+resp2+resp3+resp4+resp5
                fp = "./EventDisplay/"+str(en_list[i])+"_energy/"+str(k)+"GNN_"+str(e_gnn)[:2]+"Chi2_"+str(e_chi2)[:2]+"Abs_"+str(frac_e)[:3]+"leak_"+str(frac_l)[:3]+".png"                
                print(fp)
                plt.savefig(fp)
    if(count_ev>500):
        break
            # elif(categ2):
                
            #     fig = plt.figure(figsize = (15, 10))
            #     ax = plt.axes(projection ="3d")
            #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            #     ax.view_init(elev=5,azim=-80)
            #     e = trueEn_pkl[k]
            #     l1= "trueE="+str(e)[:3]
            #     l2="GNN="+str(e_gnn)[:3]
            #     l3="Chi2="+str(e_chi2)[:3]+"GeV"
            #     l4="frac_abs="+str(frac_e)[:3]
            #     l5="frac_leak="+str(frac_l)[:3]
            #     l=l1+" "+l2+" "+l3+" "+l4+" "+l5
            #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool",vmin=0,vmax=1.0)
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
            #                                                         [264,-36, -36],
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
            #     ax.text(-10, 40, 40, l, fontsize = 15, color="r")
            #     L = "leak[beforeCE-E =" +str(frac_res[k])[:4]+", in CE-E="+str(frac_EE[k])[:4]+",in CE-H="+str(frac_FH[k])[:4]+",in AH="+str(frac_AH[k])[:4]+",Longi="+str(frac_long[k])[:4]+"]"

            #     ax.text(0,-48,-48,L,fontsize =17,color="b")
            #     resp =""
            #     resp1=""
            #     resp2 = ""
            #     resp3 =""
            #     resp4 =""
            #     resp5 =""
            #     if(gnnresp[k]<0.6):
            #         resp1 ="gnn_bad"
            #     elif(gnnresp[k]>1.5):
            #         resp2 = "high_gnn"
            #     elif(chi2resp[k]<0.6):
            #         resp3 ="chi2_bad"
            #     elif(chi2resp[k]>1.5):
            #         resp4="chi2_high"
            #     else:
            #         resp5 = "Godd_chi2_gnn"
            #     resp = resp1+resp2+resp3+resp4+resp5
            #     fp = "./EventDisplay/categ_wise/categ2/"+str(k)+"_categ2_evedisplay_"+resp+str(e)+".png"
            #     plt.savefig(fp)
            # elif(categ3):
            #     fig = plt.figure(figsize = (15, 10))
            #     ax = plt.axes(projection ="3d")
            #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            #     ax.view_init(elev=5,azim=-80)
            #     e = trueEn_pkl[k]

            #     l1= "trueE="+str(e)[:3]
            #     l2="GNN="+str(e_gnn)[:3]
            #     l3="Chi2="+str(e_chi2)[:3]+"GeV"
            #     l4="frac_abs="+str(frac_e)[:3]
            #     l5="frac_leak="+str(frac_l)[:3]
            #     l=l1+" "+l2+" "+l3+" "+l4+" "+l5
            #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool",vmin=0,vmax=1.0)
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
            #     ax.text(-10, 40, 40, l, fontsize = 15, color="r")
            #     L = "leak[beforeCE-E =" +str(frac_res[k])[:4]+", in CE-E="+str(frac_EE[k])[:4]+",in CE-H="+str(frac_FH[k])[:4]+",in AH="+str(frac_AH[k])[:4]+",Longi="+str(frac_long[k])[:4]+"]"

            #     ax.text(0,-48,-48,L,fontsize =17,color="b")
            #     resp =""
            #     resp1=""
            #     resp2 = ""
            #     resp3 =""
            #     resp4 =""
            #     resp5 =""
            #     if(gnnresp[k]<0.6):
            #         resp1 ="gnn_bad"
            #     elif(gnnresp[k]>1.5):
            #         resp2 = "high_gnn"
            #     elif(chi2resp[k]<0.6):
            #         resp3 ="chi2_bad"
            #     elif(chi2resp[k]>1.5):
            #         resp4="chi2_high"
            #     else:
            #         resp5 = "Godd_chi2_gnn"
            #     resp = resp1+resp2+resp3+resp4+resp5
            #     fp = "./EventDisplay/categ_wise/categ3/"+str(k)+"_categ3_evedisplay_"+resp+str(e)+".png"
            #     plt.savefig(fp)
            # elif(categ4):
            #     fig = plt.figure(figsize = (15, 10))
            #     ax = plt.axes(projection ="3d")
            #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            #     ax.view_init(elev=5,azim=-80)
            #     e = trueEn_pkl[k]

            #     l1= "trueE="+str(e)[:3]
            #     l2="GNN="+str(e_gnn)[:3]
            #     l3="Chi2="+str(e_chi2)[:3]+"GeV"
            #     l4="frac_abs="+str(frac_e)[:3]
            #     l5="frac_leak="+str(frac_l)[:3]
            #     l=l1+" "+l2+" "+l3+" "+l4+" "+l5
            #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool",vmin=0,vmax=1.0)
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
            #     ax.text(-10, 40, 40, l, fontsize = 15, color="r")
            #     L = "leak[beforeCE-E =" +str(frac_res[k])[:4]+", in CE-E="+str(frac_EE[k])[:4]+",in CE-H="+str(frac_FH[k])[:4]+",in AH="+str(frac_AH[k])[:4]+",Longi="+str(frac_long[k])[:4]+"]"

            #     ax.text(0,-48,-48,L,fontsize =17,color="b")
            #     resp =""
            #     resp1=""
            #     resp2 = ""
            #     resp3 =""

            #     resp4 =""
            #     resp5 =""
            #     if(gnnresp[k]<0.6):
            #         resp1 ="gnn_bad"
            #     elif(gnnresp[k]>1.5):
            #         resp2 = "high_gnn"
            #     elif(chi2resp[k]<0.6):
            #         resp3 ="chi2_bad"
            #     elif(chi2resp[k]>1.5):
            #         resp4="chi2_high"
            #     else:
            #         resp5 = "Godd_chi2_gnn"
            #     resp = resp1+resp2+resp3+resp4+resp5
            #     fp = "./EventDisplay/categ_wise/categ4/"+str(k)+"_categ4_evedisplay_"+resp+str(e)+".png"
            #     plt.savefig(fp)
            # else:
            #     continue


# fout.cd()
# for i in range(len(en_list)):
#     h1d_absr[i].Write()
#     h1d_leak[i].Write()
#     h1d_fracabsr[i].Write()
#     h1d_fracleak[i].Write()
#     h2d_absr_vsGnn[i].Write()
#     h2d_absr_vsChi2[i].Write()
#     h2d_chi2vsGnn[i].Write()
# fout.Close()
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

    #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool",vmin=0,vmax=1.0) #color="blue")
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
                        
    #     ax.text(-10, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./EventDisplay/binWise_chi2GnnChecks/"+str(k)+"_evedisplay_Gnn_bad_chi2Good.png"

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

    #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool",vmin=0,vmax=1.0)#color="blue")
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
    #     ax.text(-10, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./EventDisplay/binWise_chi2GnnChecks/"+str(k)+"_evedisplay_Gnnhigh_chi2good.png"

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

        
    #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool",vmin=0,vmax=1.0) #color="blue")
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
    #     ax.text(-10, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./EventDisplay/binWise_chi2GnnChecks/"+str(k)+"_evedisplay_Gnnunity_chi2high.png"

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
#     ax.text(-10, 40, 40, l, fontsize = 13, color="r")
  
#     c= plt.colorbar(sc, ax=ax)
#     #plt.clim(0, 150)
#     plt.title("")
#     plt.show()
#     fp = "./EventDisplay/binWise_chi2GnnChecks/"+str(k)+"_image.png"
    
#     plt.savefig(fp)
#     #plt.show()


# eveList2 = df.loc[(df.gnnResp[K]onse>0.9) & (df.gnnResp[K]onse<1.1) & (df.chi2Resp[K]onse>1.5)]
# eveList2["eneFracAbs"] = eveList2.apply(eneFracAbs, axis=1)
# print(eveList2.trueBeamEnergy.size)
# eveList2.head()
