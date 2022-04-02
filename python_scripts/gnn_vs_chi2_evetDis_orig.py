import pandas as pd
import numpy as np
import uproot
import pickle
import matplotlib.pyplot as plt

import ROOT
data = uproot.open("/home/rusack/shared/nTuples/HGCAL_TestBeam/PionSamples/cleanedNtuples/simmed_files/skimmed_ntuple_sim_chi2method01_0000.root")
#data["pion_variables_v1"].pandas.df( flatten=False, entrystart=0, entrystop=1).columns
req_list = ['energyLostEE', 'energyLostFH', 'energyLostBH', 'energyLostBeam','trueBeamEnergy','trimAhcal_chi2Reco',]

def totEneLost(dff):
    return dff.energyLostEE + dff.energyLostFH + dff.energyLostBH

df = data["pion_variables_v1"].pandas.df(req_list, flatten=False)#, entrystart=0, entrystop=20000)
df["energyLostTotal"] = df.apply(totEneLost, axis=1)
df.head()
df.trueBeamEnergy.values.size

with open("./valid_flat/pred_tb.pickle","rb") as f:
    gnn_pred = pickle.load(f)
    
RechitEn ="/home/rusack/shared/pickles/HGCAL_TestBeam/test_0to5M_fix_raw_ahcalTrim_up/recHitEn.pickle"
RechitEnPickle = open(RechitEn,"rb")
RechitEn_pkl =pickle.load(RechitEnPickle)

rawE = ak.sum(RechitEn_pkl, axis=1)
raw= rawE[0:836658]
gnn_ene = gnn_pred[:836658]*raw[:836658]
df["gnn_pred"] = gnn_ene

gp = df.gnn_pred.values
cp = df.trimAhcal_chi2Reco.values
true = df.trueBeamEnergy.values
def gnnResp(dff):
    return dff.gnn_pred/dff.trueBeamEnergy

def chi2Resp(dff):
    return dff.trimAhcal_chi2Reco/dff.trueBeamEnergy

def eneFracAbs(dff):
    return (dff.energyLostEE+dff.energyLostFH+dff.energyLostBH)/dff.trueBeamEnergy

df["gnnResponse"] = df.apply(gnnResp, axis=1)
df["chi2Response"] = df.apply(chi2Resp, axis=1)
df.head()



eveList1 = df.loc[(df.gnnResponse>0.9) & (df.gnnResponse<1.1) & (df.chi2Response<0.5)]
eveList1["eneFracAbs"] = eveList1.apply(eneFracAbs, axis=1)
print(eveList1.trueBeamEnergy.size)
eveList1.head()

for k in range(5):
    x_pion = eveList1.comb_rechit_x_trimAhcal.values[k]
    y_pion = eveList1.comb_rechit_y_trimAhcal.values[k]
    z_pion = eveList1.comb_rechit_z_trimAhcal.values[k]
    en = eveList1.rechitEn_trimAhcal.values[k]

    fig = plt.figure(figsize = (15, 10)) 

    ax = plt.axes(projection ="3d") 
    ax.view_init(elev=5,azim=-80)  

    e = eveList1.trueBeamEnergy.values[k]
    l = "trueE="+str(e)[:5]+ " GeV"

    sc = ax.scatter3D(z_pion, x_pion, y_pion,c = en, cmap = "hot")#,label=l)

    #my_cmap = plt.get_cmap('Blues')

    ax.set_xlabel('z (cm)')
    ax.set_ylabel('x (cm)')
    ax.set_zlabel('y (cm)')

    ax.set_xlim([0,270])
    r = 35
    ax.set_ylim([-r,r])
    ax.set_zlim([-r,r])

    #plt.text(0.5, 0.5, "aa")#, fontsize=12)
    #plt.text(0.5, 0.5, 'matplotlib', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.text(20, 0.8, 20, l, fontsize = 15, color="r")
    
    c= plt.colorbar(sc, ax=ax)
    #plt.clim(0, 150)
    plt.title("")
    fp = "./Results/"+str(k)+"_image.png"
    
    plt.savefig(fp)

    #plt.show()

for k in range(eveList1.trueBeamEnergy.size):
    x_pion = eveList1.combined_rechit_x.values[k]
    y_pion = eveList1.combined_rechit_y.values[k]
    z_pion = eveList1.combined_rechit_z.values[k]
    #en = eve.combined_rechits_energy.values[k]

    fig = plt.figure(figsize = (15, 10)) 

    ax = plt.axes(projection ="3d") 
    ax.view_init(elev=5,azim=-80)  

    e = eveList1.trueBeamEnergy.values[k]

    #l = "trueE="+str(ssl)+" (z~"+str(lay_zs[ssl-1])[:5]+"GeV)"
    l = "trueE="+str(e)[:5]+ " GeV"

    ax.scatter3D(z_pion, x_pion, y_pion, color = "blue",label=l)



    ax.set_xlabel('z (cm)')
    ax.set_ylabel('x (cm)')
    ax.set_zlabel('y (cm)')

    ax.set_xlim([0,270])
    r = 35
    ax.set_ylim([-r,r])
    ax.set_zlim([-r,r])

    #plt.text(0.5, 0.5, "aa")#, fontsize=12)
    #plt.text(0.5, 0.5, 'matplotlib', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.text(20, 0.8, 20, l, fontsize = 13, color="r")
  
    c= plt.colorbar(sc, ax=ax)
    #plt.clim(0, 150)
    plt.title("")
    plt.show()
    fp = "./Results/"+str(k)+"_image.png"
    
    plt.savefig(fp)
    #plt.show()


eveList2 = df.loc[(df.gnnResponse>0.9) & (df.gnnResponse<1.1) & (df.chi2Response>1.5)]
eveList2["eneFracAbs"] = eveList2.apply(eneFracAbs, axis=1)
print(eveList2.trueBeamEnergy.size)
eveList2.head()
