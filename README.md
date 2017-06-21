# Mass-Spectrometry-Prediction
Two deep learning tensorflow models:
1) spectra2formula:  predict molecular formula from GC-MS spectra
2) spectra2smiles: predict SMILES (simplified molecular-input line-entry system notation) from GC-MS spectra

# Data
This supervised training relies on the NIST data set which contains 242466 spectrum-label pairs. This training data contains a total of 71 different elements/isotopes and to initially reduce the complexity of the problem, only covalently bound compounds containing (C, O, H, N) were selected reducing the number of samples to 116354.  Out of these 116354 samples, 6354 were saved for a final testing set and 10000 were used as a validation set in order to assist in hyper-parameter picking and to prevent overfitting.  This resulted in 100000 final training samples. 

1) Spectrum Input:  The spectral input is represented as a two-dimensional numerical matrix with the rows equal to the number of samples and the number of columns equal to the highest mass integer in the spectra.  The intensities for each spectral peak in the NIST data range from 0.0 to 999.0 and are subsequently normalized to a range of 0.0 to 1.0.  

2) Formula Labels:  The formula label is represented as a two-dimensional matrix with the rows equal to number of samples and the number of columns equal to the number of chemical elements.  For the purpose of predicting only the formulas from organic compounds, the number of elements equals 4 (Carbon, Hydrogen, Nitrogen, Oxygen).  The value of the matrix is an integer representing the number of atoms of the respective element contained in the compound and thus, predicting the formula is a multilabel regression problem.

3) SMILES Labels:  The canonical SMILES maintains all the structural information of the molecule within a string (http://www.daylight.com/dayhtml/doc/theory/theory.smiles.html) The SMILES label is represented as a three-dimensional matrix in the form (number of samples x length of SMILES x vocabulary size of SMILES). A value of 1.0 is added if the character is present in that location in the SMILES string and a 0.0 elsewise.  For the NIST data, the length of the longest SMILES is 185 and there are 17 possible choices for the vocabulary.  Note that covalent Hydrogen is not needed as an explicit character since the location of hydrogen in a compound is easily determined from the canonical SMILES and elementary bonding rules. 
The 17 characters used are: C, O, N, =, #, (, ), 1, 2, 3, 4, 5, 6, 7, 8, 9, End of String

# Model 
(see spectra2formula_model.pdf and spectra2smiles_model.pdf for visual)

1) spectra2formula:  The spectra input is fed into a variable level convolution subnetwork that follows the following repeating structure:  Convolution-> Pooling-> Batch Norm-> Dropout-> Elu Activation.  The resulting state is fed into a fully connected variable level dense subnetwork that follows the following repeating structure:  State-> Batch Norm -> Dropout -> Elu Activation.  A final set of weights transforms the last state to the regression results for the four elements.

2) spectra2smiles:  The spectra input is fed into a variable level convolution subnetwork that follows the following repeating structure:  Convolution-> Pooling-> Batch Norm-> Dropout-> Elu Activation.  The state from each convolution layer is fed into a fully connected variable level "translation" subnetwork.  The translated convolution states are then fed into their corresponding LSTM layer.  In addition to the translated convolution states, the LSTM subnetwork is input with a zero matrix the size of the SMILES labels.  A final set of weights transforms the output of the LSTM subnetwork to the SMILES prediction. 
