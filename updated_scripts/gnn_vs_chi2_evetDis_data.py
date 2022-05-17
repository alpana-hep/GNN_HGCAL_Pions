import pandas as pd
import numpy as np
import uproot
import pickle
import matplotlib.pyplot as plt
import awkward as ak
import ROOT
data = uproot.open("/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/skimmed_ntuple_data_chi2method01_0000.root")
#data["pion_variables_v1"].pandas.df( flatten=False, entrystart=0, entrystop=1).columns
path="/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/skimmed_ntuple_data_chi2method01_0000.root"
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
pred_v2 ="./tb_data/pred_tb.pickle"
predPickle = open(pred_v2, "rb")
preds_ratio = np.asarray(pickle.load(predPickle))
print(preds_ratio[preds_ratio>3])
preds_ratio[preds_ratio>3] = 3
print(preds_ratio[preds_ratio>3])
predic=preds_ratio
print(len(predic))#test_0to5M_fix_wt                                                                                                      
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_tb_fix_raw_ahcalTrim/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
#rawE= rawE[0:836658]
Pred_ = rawE * predic
trueEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_tb_fix_raw_ahcalTrim/beamEn.pickle"
Hit_X ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_tb_fix_raw_ahcalTrim/Hit_X.pickle"
Hit_XPickle = open(Hit_X,"rb")
Hit_X_pkl =pickle.load(Hit_XPickle)
print(Hit_X_pkl)

Hit_Y ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_tb_fix_raw_ahcalTrim/Hit_Y.pickle"
Hit_YPickle = open(Hit_Y,"rb")
Hit_Y_pkl =pickle.load(Hit_YPickle)

print(Hit_Y_pkl)
Hit_Z ="/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/test_tb_fix_raw_ahcalTrim/Hit_Z.pickle"
Hit_ZPickle = open(Hit_Z,"rb")
Hit_Z_pkl =pickle.load(Hit_ZPickle)
print(Hit_Z_pkl)

trueEnPickle = open(trueEn,"rb")
trueEn_pkl = np.asarray(pickle.load(trueEnPickle))
print(trueEn_pkl[0:836658])
#trueEn_pkl=trueEn_pkl[0:836658]
gnnresp= Pred_/trueEn_pkl
chi2resp= trimAhcal_chi2Reco/trueEn_pkl
# frac_abs=(energyLostEE_+energyLostFH_+energyLostBH_)/trueEn_pkl
print(gnnresp)
print(chi2resp)
# print(frac_abs)
#eneFracAbs= 
print(len(trueEn_pkl))
eveList= np.arange(len(trueEn_pkl),1)
print(len(eveList),"total length")
#if(gnnresp>0.9 and np.logical_and(gnnresp<1.1 and chi2resp<0.5)

for k in range(len(trueEn_pkl)):
    x_pion = np.array(Hit_X_pkl[k])
    y_pion = np.array(Hit_Y_pkl[k])
    z_pion =np.array(Hit_Z_pkl[k])
    rec_en = RechitEn_pkl[k]
    en= rec_en
    #frac_e= frac_abs[k]
    e = trueEn_pkl[k]
    e_gnn= Pred_[k]
    e_chi2=trimAhcal_chi2Reco[k]
    if(gnnresp[k]>0.9 and gnnresp[k]<1.1 and chi2resp[k]<0.5):
        fig = plt.figure(figsize = (15, 10)) 
        ax = plt.axes(projection ="3d") 
        #plt.rcParams["figure.figsize"] = [7.50, 7.50]
        #plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        ax.view_init(elev=5,azim=-80)          
        e = trueEn_pkl[k]
        #l = "trueE="+str(e)[:3]+ " GeV" +"\n"+"GNN="+str(e_gnn)[:3]+" GeV"+"\n"
        l1= "trueE="+str(e)[:3] 
        l2="GNN="+str(e_gnn)[:3]
        l3="Chi2="+str(e_chi2)[:3]+"GeV"
        # l4="frac_abs="+str(frac_e)[:3]
        l = l1+" "+l2+" "+l3
        sc= ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool")#color="blue")
        ax.set_xlabel('z (cm)')
        ax.set_ylabel('x (cm)')
        ax.set_zlabel('y (cm)')        
        ax.set_xlim([0,270])
        r = 35
        ax.set_ylim([-r,r])
        ax.set_zlim([-r,r])        
        fig.colorbar(sc)
        ax.text(0, 40, 40, l, fontsize = 15, color="r")
        fp = "./Results_data/"+str(k)+"_evedisplay_Gnnunity_chi2bad.png"
    
        plt.savefig(fp)
    elif(gnnresp[k]<0.7 and chi2resp[k]>0.9 and chi2resp[k]<1.1):
        fig = plt.figure(figsize = (15, 10))
        ax = plt.axes(projection ="3d")
        # plt.rcParams["figure.figsize"] = [7.50, 7.50]
        # plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        ax.view_init(elev=5,azim=-80)
        e = trueEn_pkl[k]
        #l = "trueE="+str(e)[:3]+ " GeV"
        l1= "trueE="+str(e)[:3]
        l2="GNN="+str(e_gnn)[:3]
        l3="Chi2="+str(e_chi2)[:3]+" GeV"
        l = l1+" "+l2+" "+l3

        sc=ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool") #color="blue")
        ax.set_xlabel('z (cm)')
        ax.set_ylabel('x (cm)')
        ax.set_zlabel('y (cm)')
        ax.set_xlim([0,270])
        r = 35
        ax.set_ylim([-r,r])
        ax.set_zlim([-r,r])
        fig.colorbar(sc)
                        
        ax.text(0, 40, 40, l, fontsize = 15, color="r")
        fp = "./Results_data/"+str(k)+"_evedisplay_Gnn_bad_chi2Good.png"

        plt.savefig(fp)

    elif(gnnresp[k]>1.5 and chi2resp[k]>0.9 and chi2resp[k]<1.1):
        fig = plt.figure(figsize = (15, 10))
        ax = plt.axes(projection ="3d")
        # plt.rcParams["figure.figsize"] = [7.50, 7.50]
        # plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        ax.view_init(elev=5,azim=-80)
        e = trueEn_pkl[k]
        #l = "trueE="+str(e)[:3]+ " GeV"
        l1= "trueE="+str(e)[:3]
        l2="GNN="+str(e_gnn)[:3]
        l3="Chi2="+str(e_chi2)[:3]+" GeV"
        l = l1+" "+l2+" "+l3

        sc=ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool")#color="blue")
        ax.set_xlabel('z (cm)')
        ax.set_ylabel('x (cm)')
        ax.set_zlabel('y (cm)')
        ax.set_xlim([0,270])
        r = 35
        ax.set_ylim([-r,r])
        ax.set_zlim([-r,r])
        fig.colorbar(sc)
        ax.text(0, 40, 40, l, fontsize = 15, color="r")
        fp = "./Results_data/"+str(k)+"_evedisplay_Gnnhigh_chi2good.png"

        plt.savefig(fp)

    elif(gnnresp[k]>0.9 and gnnresp[k]<1.1 and chi2resp[k]>1.5):
        fig = plt.figure(figsize = (15, 10))
        ax = plt.axes(projection ="3d")
        # plt.rcParams["figure.figsize"] = [7.50, 7.50]
        # plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        ax.view_init(elev=5,azim=-80)
        e = trueEn_pkl[k]
        #l = "trueE="+str(e)[:3]+ " GeV"
        l1= "trueE="+str(e)[:3]
        l2="GNN="+str(e_gnn)[:3]
        l3="Chi2="+str(e_chi2)[:3]+" GeV"
        l = l1+" "+l2+" "+l3
        
        sc=ax.scatter3D(z_pion, x_pion, y_pion,s=40,c = en,cmap="cool") #color="blue")
        ax.set_xlabel('z (cm)')
        ax.set_ylabel('x (cm)')
        ax.set_zlabel('y (cm)')
        ax.set_xlim([0,270])
        r = 35
        ax.set_ylim([-r,r])
        ax.set_zlim([-r,r])
        fig.colorbar(sc)
        ax.text(0, 40, 40, l, fontsize = 15, color="r")
        fp = "./Results_data/"+str(k)+"_evedisplay_Gnnunity_chi2high.png"

        plt.savefig(fp)
    else:
        continue
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
