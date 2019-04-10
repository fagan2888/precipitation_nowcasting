# Precipitation Nowcasting

Some of the source code to train the models for precipitation nowcasting of my degree project. The spatio temporal dataset is stored in HDF5 format which is then read from disk to memory to train the network. The PyTorch framework is used to train the models. The main objective was to evaluate if these networks could time extrapolate radar images and therefore the ad hoc structuring of the dataset.

# Prediction sample on test data
![Alt Text](https://media.giphy.com/media/ZbZNj4GBCHBpSCHfZV/giphy.gif) <br />
Left is the raw probability of the prediction, middle rounded at threshold .5, right is the ground truth.
