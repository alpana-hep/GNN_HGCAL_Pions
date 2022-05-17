import numpy as np
energy,mean_,sigma_ = np.loadtxt('./FracAbs_meanSigma_gausFit_8Enpoints.txt',usecols=(0,1, 2), unpack=True)
#print(ean_sigma)
# mean_ =Mean_sigma[1]
# sigma_ = Mean_sigma[2]
print(energy,mean_, sigma_)


Mean_=[]
stdDev =[]
Mean_.append(mean_[energy==50])
stdDev.append(sigma_[energy==50])
print(Mean_, stdDev)
