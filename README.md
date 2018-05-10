# Bank-Note-authenticity-dterminayion-using-Deep-Neural-Networks-Tensor-Flow-and-Random-Forest-tree

Dataset used:  Bank Authentication Data Set from the UCI repository

link: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

The data consists of 5 columns:

    variance of Wavelet Transformed image (continuous)
    
    skewness of Wavelet Transformed image (continuous)
    
    curtosis of Wavelet Transformed image (continuous)
    
    entropy of image (continuous)
    
    class (integer)
    
    Where class indicates whether or not a Bank Note was authentic.
    
    The classifier used is DNNClassifier imported from Tensorflow's contriblearn library. 
    Fiest, the classifification prediction is done using neural networks, and then by the Random forest algorithm. And then on comparing 
    the accuracy, neural networks prove to be more accurate.
