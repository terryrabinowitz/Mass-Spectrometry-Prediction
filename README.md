# Mass-Spectrometry-Prediction
Two deep learning tensorflow models:
1) spectra2formula:  predict molecular formula from GC-MS spectra
2) spectra2smiles: predict SMILES (simplified molecular-input line-entry system notation) from GC-MS spectra

# Data
This supervised training relies on the NIST data set which contains 242466 spectrum-label pairs. This training data contains a total of 71 different elements/isotopes and to initially reduce the complexity of the problem, only covalently bound compounds containing (C, O, H, N) were selected reducing the number of samples to 116354.  Out of these 116354 samples, 6354 were saved for a final testing set and 10000 were used as a validation set in order to assist in hyper-parameter picking and to prevent overfitting.  This resulted in 100000 final training samples. 

1) Spectrum Input:  The spectral input is represented as a two-dimensional numerical matrix with the rows equal to the number of samples and the number of columns equal to the highest mass integer in the spectra.  The intensities for each spectral peak in the NIST data range from 0.0 to 999.0 and are subsequently normalized to a range of 0.0 to 1.0.  

2) Formula Labels:

3) SMILES Labels:  The canonical SMILES maintains all the structural information of the molecule within a string. The SMILES label is represented as a three-dimensional matrix in the form (number of samples x length of SMILES x vocabulary size of SMILES). For the NIST data, the length of the longest SMILES is 185 and there are 17 possible choices for the vocabulary.  Two points should be made. Firstly, covalent Hydrogen is not needed as an explicit character since the location of hydrogen in a compound is easily determined from the canonical SMILES and elementary bonding rules.  Secondly, on average, each chemical compound is dominated by carbon which makes up nearly 50% of the SMILES string.  

