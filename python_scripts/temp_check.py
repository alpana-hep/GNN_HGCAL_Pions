import pandas as pd #dataframes etc                                                                                                       
import matplotlib.pyplot as plt #plotting                                                                                                 
import pickle
import awkward as ak
import numpy as np
import os, sys
import seaborn as sns
import pickle
import numpy as np


tb_pred_v2 ="./tb_data/pred_tb.pickle"
tb_predPickle = open(tb_pred_v2, "rb")
print(tb_predPickle)
tb_preds_ratio = np.asarray(pickle.load(tb_predPickle))
tb_trueEn= "/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/ratio_target.pickle"
tb_trueEnPickle = open(tb_trueEn,"rb")
tb_trueEn_pkl = np.asarray(pickle.load(tb_trueEnPickle))
print(tb_trueEn_pkl)
print(tb_preds_ratio)

RechitEn_tb ="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/tb_corrAlign/trim_Ahcal/Hit_Y.pickle"
RechitEn_tb1 ="/home/rusack/shared/pickles/HGCAL_TestBeam/Test_alps/tb_data/tb_corrAlign/Hit_Y.pickle"

RechitEn_tbPickle = open(RechitEn_tb,"rb")
RechitEn_tb_pkl =pickle.load(RechitEn_tbPickle)
print(RechitEn_tb_pkl)

RechitEn_tbPickle1 = open(RechitEn_tb1,"rb")
RechitEn_tb_pkl1 =pickle.load(RechitEn_tbPickle1)
print(RechitEn_tb_pkl1)
sys.exit()
print(RechitEn_tb_pkl[np.logical_and(RechitEn_tb_pkl1<54, RechitEn_tb_pkl>7.5)])
frac = np.logical_and(RechitEn_tb_pkl1<54, RechitEn_tb_pkl>7.5)
print(frac)
#frac[frac]=7.5
RechitEn_tb_pkl[frac]=7.5*frac #np.logical_and(RechitEn_tb_pkl1<54, RechitEn_tb_pkl>7.5)] =frac
print(RechitEn_tb_pkl[np.logical_and(RechitEn_tb_pkl1<54, RechitEn_tb_pkl>7.5)])
rawE_tb = ak.sum(RechitEn_tb_pkl, axis=1)
tb_preds_trueEn = rawE_tb*tb_preds_ratio
print(tb_preds_trueEn,' pred')
print(rawE_tb,'raw')
