# TFM

Mejoras programa: ideas para mejorar el programa (falta desarrollar implementación)

INITIAL GUESSES:
mean of a variable=sum(variable)/size(variable) or np.mean(list)
R_i == mean distance (initial)
ssN_i == ??
xN_i == color_i,color_i+binsize,color_i+2*binsize,color_i+3*binsize,color_f
(same values as the bins)
sMN_i == ??
muAlphaMean and muDeltaMean == mean proper motions (initial)
sigmaMuAlphaMean and sigmaMuDeltaMean == ??

BINS
bins = [[bin1],[bin2],[bin3],[bin4]]
binN=binN_i,binN_f
#color_i==min value for color index
#color_f==max value for color index
binsize=(color_f-color_i)/4 #-> homogeneous bins
bin1=color_i,color_i+binsize
bin2=color_i+binsize,color_i+2*binsize
bin3=color_i+2*binsize,color_i+3*binsize
bin4=color_i+3*binsize,color_f


