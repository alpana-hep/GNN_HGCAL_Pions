import os
import shutil
path = '/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/Correct_training_5M/trimAhcal/infer_epoch80/Results_v1'
# Source path                                                                                                                                                             
#src = '/home/alps/Work/BE_DAQ/Pseudo_samples/Results/n30/v1/'
#Results_v1/8enpoints/
en_list=[20,50,80,100,120,200,250,300]
for i in range(len(en_list)):
         name = '/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/fix_wt_5M/Correct_training_5M/trimAhcal/infer_epoch80/Results_v1/8enpoints/TrueEn_%02d' %en_list[i]
         os.mkdir(name)
