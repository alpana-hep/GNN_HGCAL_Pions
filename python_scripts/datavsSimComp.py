import pandas as pd
import numpy as np
import uproot
import pickle
import matplotlib.pyplot as plt
import awkward as ak
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import ROOT
import math
data = uproot.open("/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/ntuple_chi2_reco_5M_firstFlatEn.root")
#data["pion_variables_v1"].pandas.df( flatten=False, entrystart=0, entrystop=1).columns
path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/ntuple_chi2_reco_5M_firstFlatEn.root"
treeName ="pion_variables_v1"
#outfolder ="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/ratio_updated/correct_inputs/epoch58_onwards/DownScale_Ahcal"
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
pred_v2 ="./valid_flat/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])
predic=preds_ratio
print(len(predic))#test_0to5M_fix_raw_ahcalTrim_up                                                                                                      
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_0to5M_fix_raw_ahcalTrim_up/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
#rawE= rawE[0:836658]
Pred_ = rawE * predic
trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_0to5M_fix_raw_ahcalTrim_up/trueE.pickle"

en_list=[20,50,80,100,120,200,250,300]
h1d_absr=[]
h1d_fracabsr=[]
h2d_absr_vsGnn=[]
h2d_absr_vsChi2=[]
h2d_chi2vsGnn=[]
import ROOT
fout= ROOT.TFile("hist_checks_chi2vsGnn_86epochs_8enpoints.root", 'RECREATE')
for i_hist in range(len(en_list)):
    xhigh_true= 2.0*en_list[i_hist]
    xhigh = 3.0*en_list[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(en_list[i_hist])
    h1d_absr.append(ROOT.TH1F("h1d_absrb_%s"%name1,"h1d_absrb",500,0,xhigh))
    h1d_fracabsr.append(ROOT.TH1F("h1d_fracabsrb_%s"%name1,"h1d_fracabsrb",500,0,2))
    h2d_absr_vsGnn.append(ROOT.TH2F("h2d_absr_vsGnn_%s"%name1,"h2d_absr_vsGnn",500,0,xhigh,500,0,xhigh))
    h2d_absr_vsChi2.append(ROOT.TH2F("h2d_absr_vsChi2_%s"%name1,"h2d_absr_vsChi2",500,0,xhigh,500,0,xhigh))
    h2d_chi2vsGnn.append(ROOT.TH2F("h2d_chi2vsGnn_%s"%name1,"h2d_chi2vsGnn_",500,0,xhigh,500,0,xhigh))
    h_abs_inc.append(ROOT.TH1F("h_abs_inc_%s"%name1,"h_abs_inc_",500,0,xhigh))
    h_fracabs_inc.append(ROOT.TH1F("h_fracabs_inc_%s"%name1,"h_fracabs_inc_",500,0,2))
    h1d_stddev.append(ROOT.TH1F("h1d_stddev_%s"%name1,"h1d_stddev",500,0,100))
    h1d_3dstddev.append(ROOT.TH1F("h1d_3dstddev_%s"%name1,"h1d_3dstddev",500,0,100))
    h2d_GnnvsStddev.append(ROOT.TH2F("h2d_GnnvsStddev_%s"%name1,"h2d_GnnvsStddev_",500,0,xhigh,500,0,100))
    h2d_Gnnvs3dStddev.append(ROOT.TH2F("h2d_Gnnvs3dStddev_%s"%name1,"h2d_Gnnvs3dStddev_",500,0,xhigh,500,0,100))

for i in range(len(trueEn_pkl)):
    
    # x_pion = np.array(Hit_X_pkl[k])
#     y_pion = np.array(Hit_Y_pkl[k])
#     z_pion =np.array(Hit_Z_pkl[k])
#     rec_en = RechitEn_pkl[k]    
#     en= rec_en
#     frac_e= frac_abs[k]
#     e = trueEn_pkl[k]
#     e_gnn= Pred_[k]
#     e_chi2=trimAhcal_chi2Reco[k]
#     stdDev_x= np.std(x_pion)
#     stdDev_y= np.std(y_pion)
#     stdDev_z= np.std(z_pion)
#     std_2d= math.sqrt((stdDev_x*stdDev_x)+(stdDev_y*stdDev_y))
#     std_3d = math.sqrt((stdDev_x*stdDev_x)+(stdDev_y*stdDev_y)+(stdDev_z*stdDev_z))

#     for i in range(len(en_list)):
#         if(e>=en_list[i]-2 and e<=en_list[i]+2):
#             h_abs_inc[i].Fill(total_abs[k])
#             h_fracabs_inc[i].Fill(frac_e)
#             if((gnnresp[k]>0.8 and gnnresp[k]<=1.2 and chi2resp[k]<0.4) or (gnnresp[k]>1.2 and chi2resp[k]>0.7 and chi2resp[k]<1))  :
#                 h2d_absr_vsChi2[i].Fill(e_chi2,total_abs[k])
#                 h2d_absr_vsGnn[i].Fill(e_gnn,total_abs[k])
#                 h1d_fracabsr[i].Fill(frac_e)
#                 h1d_absr[i].Fill(total_abs[k])
#                 h2d_chi2vsGnn[i].Fill(e_gnn,e_chi2)
#                 h1d_3dstddev[i].Fill(std_3d)
#                 h1d_stddev[i].Fill(std_2d)
#                 h2d_GnnvsStddev[i].Fill(e_gnn,std_2d)
#                 h2d_Gnnvs3dStddev[i].Fill(e_gnn,std_3d)
                
#                 #print(k,e)
#                 fig = plt.figure(figsize = (15, 10))
#                 ax = plt.axes(projection ="3d")
#                 plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

#                 ax.view_init(elev=5,azim=-80)
#                 e = trueEn_pkl[k]
#                 l1= "trueE="+str(e)[:3]
#                 l2="GNN="+str(e_gnn)[:3]
#                 l3="Chi2="+str(e_chi2)[:3]+"GeV"
#                 l4="frac_abs="+str(frac_e)[:3]
#                 #l5="frac_leak="+str(frac_l)[:3]
#                 l=l1+" "+l2+" "+l3+" "+l4
#                 sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="jet",vmin=0,vmax=5)#color="blue")                                                                 
#                 points = np.array([[13, -7.5,-7.5],
#                                    [13, 7.5, -7.5 ],
#                                    [13, 7.5, 7.5],
#                                    [13,-7.5, 7.5],
#                                    [54,-7.5, -7.5],
#                                    [54, 7.5, -7.5],
#                                    [54, 7.5, 7.5],
#                                    [54, -7.5, 7.5 ]])
#                 Z = points
#                 verts = [[Z[0],Z[1],Z[2],Z[3]],
#                         [Z[4],Z[5],Z[6],Z[7]],
#                          [Z[0],Z[1],Z[5],Z[4]],
#                          [Z[2],Z[3],Z[7],Z[6]],
#                          [Z[1],Z[2],Z[6],Z[5]],
#                          [Z[4],Z[7],Z[3],Z[0]]]
#                 ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
#                 points = np.array([[64, -18.5,-18.5],
#                                    [64, 18.5, -18.5 ],
#                                    [64, 18.5, 18.5],
#                                    [64,-18.5, 18.5],
#                                    [152.5,-18.5, -18.5],
#                                    [152.5, 18.5, -18.5],
#                                    [152.5, 18.5, 18.5],
#                                    [152.5, -18.5, 18.5 ]])
#                 Z = points
#                 verts = [[Z[0],Z[1],Z[2],Z[3]],
#                          [Z[4],Z[5],Z[6],Z[7]],
#                          [Z[0],Z[1],Z[5],Z[4]],
#                          [Z[2],Z[3],Z[7],Z[6]],
#                          [Z[1],Z[2],Z[6],Z[5]],
#                          [Z[4],Z[7],Z[3],Z[0]]]
#                 ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))
#                 points = np.array([[159, -36,-36],
#                                    [159, 36, -36 ],
#                                    [159, 36, 36],
#                                    [159,-36, 36],
#                                    [264,-36, -36],
#                                    [264, 36, -36],
#                                    [264, 36, 36],
#                                    [264, -36, 36 ]])
#                 Z = points
#                 verts = [[Z[0],Z[1],Z[2],Z[3]],
#                          [Z[4],Z[5],Z[6],Z[7]],
#                          [Z[0],Z[1],Z[5],Z[4]],
#                          [Z[2],Z[3],Z[7],Z[6]],
#                          [Z[1],Z[2],Z[6],Z[5]],
#                          [Z[4],Z[7],Z[3],Z[0]]]
#                 ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='black', alpha=.0))

#                 ax.set_xlabel('z (cm)')
#                 ax.set_ylabel('x (cm)')
#                 ax.set_zlabel('y (cm)')
#                 ax.set_xlim([0,280])
#                 r = 36.5
#                 ax.set_ylim([-r,r])
#                 ax.set_zlim([-r,r])
#                 cbar = fig.colorbar(sc,shrink = 0.5, aspect = 10)
#                 cbar.set_label('rechit en (GeV)')
#                 ax.text(0, 40, 40, l, fontsize = 15, color="r")
#                 fp = "./Results_v1/8enpoints/TrueEn_"+str(en_list[i])+"/"+str(k)+"_evedisplay_Gnnunity_chi2bad_"+str(e)+".png"

#                 plt.savefig(fp)
#             else:
#                 continue

# fout.cd()
# for i in range(len(en_list)):
#     h1d_absr[i].Write()
#     h1d_fracabsr[i].Write()
#     h2d_absr_vsGnn[i].Write()
#     h2d_absr_vsChi2[i].Write()
#     h_fracabs_inc[i].Write()
#     h_abs_inc[i].Write()
#     h2d_chi2vsGnn[i].Write()
#     h1d_3dstddev[i].Write()
#     h1d_stddev[i].Write()
#     h2d_GnnvsStddev[i].Write()
#     h2d_Gnnvs3dStddev[i].Write()

# fout.Close()


    # if(gnnresp[k]>0.9 and gnnresp[k]<1.1 and chi2resp[k]<0.5):
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
    #     l = l1+" "+l2+" "+l3+" "+l4
    #     sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool")#color="blue")
    #     ax.set_xlabel('z (cm)')
    #     ax.set_ylabel('x (cm)')
    #     ax.set_zlabel('y (cm)')        
    #     ax.set_xlim([0,270])
    #     r = 35
    #     ax.set_ylim([-r,r])
    #     ax.set_zlim([-r,r])        
    #     fig.colorbar(sc)
    #     ax.text(0, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./Results_v1/"+str(k)+"_evedisplay_Gnnunity_chi2bad.png"
    
    #     plt.savefig(fp)
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
    #     l = l1+" "+l2+" "+l3+" "+l4

    #     sc=ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool") #color="blue")
    #     ax.set_xlabel('z (cm)')
    #     ax.set_ylabel('x (cm)')
    #     ax.set_zlabel('y (cm)')
    #     ax.set_xlim([0,270])
    #     r = 35
    #     ax.set_ylim([-r,r])
    #     ax.set_zlim([-r,r])
    #     fig.colorbar(sc)
                        
    #     ax.text(0, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./Results_v1/"+str(k)+"_evedisplay_Gnn_bad_chi2Good.png"

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
    #     l = l1+" "+l2+" "+l3+" "+l4

    #     sc=ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool")#color="blue")
    #     ax.set_xlabel('z (cm)')
    #     ax.set_ylabel('x (cm)')
    #     ax.set_zlabel('y (cm)')
    #     ax.set_xlim([0,270])
    #     r = 35
    #     ax.set_ylim([-r,r])
    #     ax.set_zlim([-r,r])
    #     fig.colorbar(sc)
    #     ax.text(0, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./Results_v1/"+str(k)+"_evedisplay_Gnnhigh_chi2good.png"

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
    #     l = l1+" "+l2+" "+l3+" "+l4
        
    #     sc=ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool") #color="blue")
    #     ax.set_xlabel('z (cm)')
    #     ax.set_ylabel('x (cm)')
    #     ax.set_zlabel('y (cm)')
    #     ax.set_xlim([0,270])
    #     r = 35
    #     ax.set_ylim([-r,r])
    #     ax.set_zlim([-r,r])
    #     fig.colorbar(sc)
    #     ax.text(0, 40, 40, l, fontsize = 15, color="r")
    #     fp = "./Results_v1/"+str(k)+"_evedisplay_Gnnunity_chi2high.png"

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

#     ax.set_xlim([0,270])
#     r = 35
#     ax.set_ylim([-r,r])
#     ax.set_zlim([-r,r])

#     #plt.text(0.5, 0.5, "aa")#, fontsize=12)
#     #plt.text(0.5, 0.5, 'matplotlib', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#     ax.text(0, 40, 40, l, fontsize = 13, color="r")
  
#     c= plt.colorbar(sc, ax=ax)
#     #plt.clim(0, 150)
#     plt.title("")
#     plt.show()
#     fp = "./Results_v1/"+str(k)+"_image.png"
    
#     plt.savefig(fp)
#     #plt.show()


# eveList2 = df.loc[(df.gnnResp[K]onse>0.9) & (df.gnnResp[K]onse<1.1) & (df.chi2Resp[K]onse>1.5)]
# eveList2["eneFracAbs"] = eveList2.apply(eneFracAbs, axis=1)
# print(eveList2.trueBeamEnergy.size)
# eveList2.head()
