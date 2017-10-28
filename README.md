# ADLxMLDS2017

|Author|劉家維
|---|---
|學號|b05902052

***

## hw1
* ### Using Packages
    * `numpy 1.13.3`
    * `keras 2.0.8`
    * `tensorflow-gpu 1.3.0`
    * `h5py 2.7.1`
    
* ### Training Code
    * #### *RNN model*
        * ##### result model: `models/big_RNN_model.h5`
        * `$ python model_rnn.py`

    * #### *CNN+RNN model*
        * ##### result model: `models/CRNN_model.h5`
        * `$ python model_cnn.py`
    
    * #### *CNN+RNN+CNN model*
        * ##### result model: `models/very_deep_RNN_model.h5`
        * `$ python model_crc.py`
    
    * #### best model
        * ##### The best model is ensembled from the above three models.
          ##### So I don't have the training code of best model.
        * ##### To ensemble, just run:
          `$ python hw1_best.py`
* ### Another Data
    * #### in `dict` folders:
        * ##### `num2char` : map index to characters
        * ##### `phone2num` : map phone to index
