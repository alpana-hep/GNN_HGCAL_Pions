import pandas as pd
import numpy as np
import uproot
import pickle
import matplotlib.pyplot as plt
import awkward as ak
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import ROOT
import math
import sys
folder = sys.argv[1]
folder1 = sys.argv[2]
folder2 = sys.argv[3]
print("input predictions",folder)
outfileName= sys.argv[4]
out_fname = '%s/%s'%(folder,outfileName)
print(out_fname, "output file is")
inpickle_folder =sys.argv[5]
print("input pickle files are picked from",inpickle_folder)
inChi2 = sys.argv[6]
inTBChi2 = sys.argv[7]
inTBpickle = sys.argv[8]
#/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files
data = uproot.open(inChi2) #"%s/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/ntuple_chi2_reco_5M_firstFlatEn.root")

#data["pion_variables_v1"].pandas.df( flatten=False, entrystart=0, entrystop=1).columns
path=inChi2 #"/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/ntuple_chi2_reco_5M_firstFlatEn.root"
treeName ="pion_variables_v1"
#outfolder ="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/ratio_updated/correct_inputs/epoch58_onwards/DownScale_Ahcal"
tree = uproot.open("%s:%s"%(path, treeName))
# energyLostEE_ = tree['energyLostEE'].array()
# print(energyLostEE_)
# energyLostFH_ = tree['energyLostFH'].array()
# print(energyLostFH_)
# energyLostBH_ = tree['energyLostBH'].array()
# print(energyLostBH_)
# beamEnergy = tree['beamEnergy'].array()
# trueBeamEnergy = tree['trueBeamEnergy'].array()
# rechit_shower_start_layer=tree['rechit_shower_start_layer'].array()
# trimAhcal_chi2Reco = tree['trimAhcal_chi2Reco'].array()
pred_v1 ="%s/valid_flat/pred_tb.pickle"%folder
predPickle = open(pred_v1, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
preds_ratio[preds_ratio>3] = 3
predic1=preds_ratio

pred_v1 ="%s/valid_flat/pred_tb.pickle"%folder1
predPickle = open(pred_v1, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
preds_ratio[preds_ratio>3] = 3
predic2=preds_ratio

pred_v1 ="%s/valid_flat/pred_tb.pickle"%folder2
predPickle = open(pred_v1, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
preds_ratio[preds_ratio>3] = 3
predic3=preds_ratio


#print(len(predic))#test_0to5M_fix_raw_ahcalTrim_up                     
#total_abs = energyLostEE_ + energyLostFH_ + energyLostBH_             
    
RechitEn ="%s/recHitEn.pickle"%inpickle_folder
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
Pred_4 = rawE * predic1
Pred_2 = rawE * predic2
Pred_1 = rawE * predic3

trueEn ="%s/trueE.pickle"%inpickle_folder
trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
en_list=[20,50,80,100,120,200,250,300]

valid_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/Correct_training_5M/trimAhcal/NSM_infer/all_valididx.pickle"
train_idx_file="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/Correct_training_5M/trimAhcal/NSM_infer/all_trainidx.pickle"
valid_idx_f = open(valid_idx_file,"rb")
valid_idx = np.asarray(pickle.load(valid_idx_f))
print(len(valid_idx))

train_idx_f = open(train_idx_file,"rb")
train_idx = np.asarray(pickle.load(train_idx_f))
print(len(train_idx))

h2d_Gnn4_vsGnn2=[]
h2d_Gnn2_vsGnn1=[]
h2d_Gnn4_vsGnn1=[]

import ROOT
fout= ROOT.TFile(out_fname, 'RECREATE')
for i_hist in range(len(en_list)):
    if(en_list[i_hist]<100):
        xhigh_pred = 4.0*en_list[i_hist]
    else:
        xhigh_pred= 3.0*en_list[i_hist]
    xhigh_true= 2.0*en_list[i_hist]
    xhigh = xhigh_pred #3.0*en_list[i_hist]
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='TrueEn_%i' %(en_list[i_hist])
    h2d_Gnn4_vsGnn2.append(ROOT.TH2F("h2d_Gnn4_vsGnn2_%s"%name1,"h2d_Gnn4_vsGnn2_",500,0,xhigh,500,0,xhigh))
    h2d_Gnn2_vsGnn1.append(ROOT.TH2F("h2d_Gnn2_vsGnn1_%s"%name1,"h2d_Gnn2_vsGnn1_",500,0,xhigh,500,0,xhigh))
    h2d_Gnn4_vsGnn1.append(ROOT.TH2F("h2d_Gnn4_vsGnn1_%s"%name1,"h2d_Gnn4_vsGnn1_",500,0,xhigh,500,0,xhigh))

for k in range(len(trueEn_pkl)):
    e_gnn4=Pred_4[k]
    e_gnn2=Pred_2[k]
    e_gnn1=Pred_1[k]
    e =trueEn_pkl[k]
    for i in range(len(en_list)):
        if(e>=en_list[i]-2 and e<=en_list[i]+2):
            h2d_Gnn4_vsGnn2[i].Fill(e_gnn4,e_gnn2)
            h2d_Gnn4_vsGnn1[i].Fill(e_gnn4,e_gnn1)
            h2d_Gnn2_vsGnn1[i].Fill(e_gnn2,e_gnn1)


h2d_Gnn4_vsGnn2_Valid=[]
h2d_Gnn2_vsGnn1_Valid=[]
h2d_Gnn4_vsGnn1_Valid=[]

for i_hist in range(len(en_list)):
    if(en_list[i_hist]<100):
        xhigh_pred = 4.0*en_list[i_hist]
    else:
        xhigh_pred= 3.0*en_list[i_hist]
    xhigh_true= 2.0*en_list[i_hist]
    xhigh = xhigh_pred #3.0*en_list[i_hist]                                                                                                       
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='Valid_TrueEn_%i' %(en_list[i_hist])
    h2d_Gnn4_vsGnn2_Valid.append(ROOT.TH2F("h2d_Gnn4_vsGnn2_%s"%name1,"h2d_Gnn4_vsGnn2_",500,0,xhigh,500,0,xhigh))
    h2d_Gnn2_vsGnn1_Valid.append(ROOT.TH2F("h2d_Gnn2_vsGnn1_%s"%name1,"h2d_Gnn2_vsGnn1_",500,0,xhigh,500,0,xhigh))
    h2d_Gnn4_vsGnn1_Valid.append(ROOT.TH2F("h2d_Gnn4_vsGnn1_%s"%name1,"h2d_Gnn4_vsGnn1_",500,0,xhigh,500,0,xhigh))


#for k in range(len(trueEn_pkl)):
for k in range(len(valid_idx)):
    e_gnn4=Pred_4[valid_idx[k]]
    e_gnn2=Pred_2[valid_idx[k]]
    e_gnn1=Pred_1[valid_idx[k]]
    e =trueEn_pkl[valid_idx[k]]
    for i in range(len(en_list)):
        if(e>=en_list[i]-2 and e<=en_list[i]+2):
            h2d_Gnn4_vsGnn2_Valid[i].Fill(e_gnn4,e_gnn2)
            h2d_Gnn4_vsGnn1_Valid[i].Fill(e_gnn4,e_gnn1)
            h2d_Gnn2_vsGnn1_Valid[i].Fill(e_gnn2,e_gnn1)

h2d_Gnn4_vsGnn2_Train=[]
h2d_Gnn2_vsGnn1_Train=[]
h2d_Gnn4_vsGnn1_Train=[]

for i_hist in range(len(en_list)):
    if(en_list[i_hist]<100):
        xhigh_pred = 4.0*en_list[i_hist]
    else:
        xhigh_pred= 3.0*en_list[i_hist]
    xhigh_true= 2.0*en_list[i_hist]
    xhigh = xhigh_pred #3.0*en_list[i_hist]                                                                                                                                                                        
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='Train_TrueEn_%i' %(en_list[i_hist])
    h2d_Gnn4_vsGnn2_Train.append(ROOT.TH2F("h2d_Gnn4_vsGnn2_%s"%name1,"h2d_Gnn4_vsGnn2_",500,0,xhigh,500,0,xhigh))
    h2d_Gnn2_vsGnn1_Train.append(ROOT.TH2F("h2d_Gnn2_vsGnn1_%s"%name1,"h2d_Gnn2_vsGnn1_",500,0,xhigh,500,0,xhigh))
    h2d_Gnn4_vsGnn1_Train.append(ROOT.TH2F("h2d_Gnn4_vsGnn1_%s"%name1,"h2d_Gnn4_vsGnn1_",500,0,xhigh,500,0,xhigh))


for k in range(len(train_idx)):
    e_gnn4=Pred_4[train_idx[k]]
    e_gnn2=Pred_2[train_idx[k]]
    e_gnn1=Pred_1[train_idx[k]]
    e =trueEn_pkl[train_idx[k]]
    for i in range(len(en_list)):
        if(e>=en_list[i]-2 and e<=en_list[i]+2):
            h2d_Gnn4_vsGnn2_Train[i].Fill(e_gnn4,e_gnn2)
            h2d_Gnn4_vsGnn1_Train[i].Fill(e_gnn4,e_gnn1)
            h2d_Gnn2_vsGnn1_Train[i].Fill(e_gnn2,e_gnn1)




#data1 = uproot.open(inChi2)
path=inTBChi2 
tree = uproot.open("%s:%s"%(path, treeName))
beamEnergy = tree['beamEnergy'].array()
rechit_shower_start_layer=tree['rechit_shower_start_layer'].array()
trimAhcal_chi2Reco = tree['trimAhcal_chi2Reco'].array()
pred_v2 ="%s/tb_data_upscaled/pred_tb.pickle"%folder
predPickle_TB = open(pred_v2, "rb")
preds_ratio_TB = np.asarray(pickle.load(predPickle_TB))
preds_ratio_TB[preds_ratio_TB>3] = 3
predic=preds_ratio_TB
pred_v2 ="%s/tb_data_upscaled/pred_tb.pickle"%folder1
predPickle_TB = open(pred_v2, "rb")
preds_ratio_TB1 = np.asarray(pickle.load(predPickle_TB))
preds_ratio_TB1[preds_ratio_TB1>3] = 3
predic1=preds_ratio_TB1

pred_v2 ="%s/tb_data_upscaled/pred_tb.pickle"%folder2
predPickle_TB = open(pred_v2, "rb")
preds_ratio_TB2 = np.asarray(pickle.load(predPickle_TB))
preds_ratio_TB2[preds_ratio_TB2>3] = 3
predic2=preds_ratio_TB2

RechitEn ="%s/recHitEn.pickle"%inTBpickle
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)
rawE = ak.sum(RechitEn_pkl, axis=1)
Pred_4 = rawE * predic
Pred_2 = rawE * predic1
Pred_1 = rawE * predic2


trueEn ="%s/beamEn.pickle"%inTBpickle #%tbpickle_folder
trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))

h2d_Gnn4_vsGnn2_Tbdata=[]
h2d_Gnn2_vsGnn1_Tbdata=[]
h2d_Gnn4_vsGnn1_Tbdata=[]

for i_hist in range(len(en_list)):
    if(en_list[i_hist]<100):
        xhigh_pred = 4.0*en_list[i_hist]
    else:
        xhigh_pred= 3.0*en_list[i_hist]
    xhigh_true= 2.0*en_list[i_hist]
    xhigh = xhigh_pred #3.0*en_list[i_hist]                                                                                                                                                                        
    xhigh_diff= 20
    xlow_diff= -20
    xhigh_norm= 5
    xlow_norm= -5
    name1='Tbdata_TrueEn_%i' %(en_list[i_hist])
    h2d_Gnn4_vsGnn2_Tbdata.append(ROOT.TH2F("h2d_Gnn4_vsGnn2_%s"%name1,"h2d_Gnn4_vsGnn2_",500,0,xhigh,500,0,xhigh))
    h2d_Gnn2_vsGnn1_Tbdata.append(ROOT.TH2F("h2d_Gnn2_vsGnn1_%s"%name1,"h2d_Gnn2_vsGnn1_",500,0,xhigh,500,0,xhigh))
    h2d_Gnn4_vsGnn1_Tbdata.append(ROOT.TH2F("h2d_Gnn4_vsGnn1_%s"%name1,"h2d_Gnn4_vsGnn1_",500,0,xhigh,500,0,xhigh))

for k in range(len(trueEn_pkl)):
    e_gnn4=Pred_4[k]
    e_gnn2=Pred_2[k]
    e_gnn1=Pred_1[k]
    e = trueEn_pkl[k]
    for i in range(len(en_list)):
        if(e==en_list[i]): #-2 and e<=en_list[i]+2):
            h2d_Gnn4_vsGnn2_Tbdata[i].Fill(e_gnn4,e_gnn2)
            h2d_Gnn4_vsGnn1_Tbdata[i].Fill(e_gnn4,e_gnn1)
            h2d_Gnn2_vsGnn1_Tbdata[i].Fill(e_gnn2,e_gnn1)

fout.cd()
for i in range(len(en_list)):
    h2d_Gnn4_vsGnn2_Tbdata[i].Write()
    h2d_Gnn4_vsGnn1_Tbdata[i].Write()
    h2d_Gnn2_vsGnn1_Tbdata[i].Write()
    h2d_Gnn4_vsGnn2_Train[i].Write()
    h2d_Gnn4_vsGnn1_Train[i].Write()
    h2d_Gnn2_vsGnn1_Train[i].Write()
    h2d_Gnn4_vsGnn2_Valid[i].Write()
    h2d_Gnn4_vsGnn1_Valid[i].Write()
    h2d_Gnn2_vsGnn1_Valid[i].Write()

    h2d_Gnn4_vsGnn2[i].Write()
    h2d_Gnn4_vsGnn1[i].Write()

    h2d_Gnn2_vsGnn1[i].Write()

fout.Close()

#             if((gnnresp[k]>0.8 and gnnresp[k]<=1.2 and chi2resp[k]<0.4) or (gnnresp[k]>1.2 and chi2resp[k]>0.7 and chi2resp[k]<1))  :
                # h2d_absr_vsChi2[i].Fill(e_chi2,total_abs[k])
                # h2d_absr_vsGnn[i].Fill(e_gnn,total_abs[k])
                # h1d_fracabsr[i].Fill(frac_e)
                # h1d_absr[i].Fill(total_abs[k])
                # h2d_chi2vsGnn[i].Fill(e_gnn,e_chi2)
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
# #     h2d_chi2vsGnn[i].Write()
# #     h1d_3dstddev[i].Write()
# #     h1d_stddev[i].Write()
# #     h2d_GnnvsStddev[i].Write()
# #     h2d_Gnnvs3dStddev[i].Write()

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
