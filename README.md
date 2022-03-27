# 4203-Intrusion-Detection-Project

**Presentation Video**\
https://youtu.be/X9NMJFhzqqQ 

**Refferenced Paper**\
Satam, & Hariri, S. (2021). WIDS: An Anomaly Based Intrusion Detection System for Wi-Fi (IEEE 802.11) Protocol. IEEE eTransactions on Network and Service Management, 18(1), 1077–1091. https://doi.org/10.1109/TNSM.2020.3036138

**Important Info From Paper**
* Live-Testing of the model:
  * n-gram size of 4 -> Performs the best
  * observation time of 10s
  * restrict observed n-grams to < 5000
* Training of the model:
  * 30,000 entries -> Using classification algorithm
  * Used 15 abnormal entries from the attack dataset to train the abnormal definition
  * This is (one of) the datasets used by the paper:
    * https://icsdweb.aegean.gr/awid/
* Possible Machine Learning (classification) algorithms:
  * **Paper shows that C.45 and AdaBoost Perform the best**
  * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
  * https://github.com/geerk/C45algorithm
  * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
  * https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

  



**Other Links**
* https://www.tensorflow.org/ 
* https://keras.io/about/ 
* https://numpy.org/ 
* https://www.wireshark.org/ 
* https://github.com/KimiNewt/pyshark/ 
* https://matplotlib.org/ 

**Wireshark Cheatsheets**
* https://semfionetworks.com/wp-content/uploads/2021/04/wireshark_802.11_filters_-_reference_sheet.pdf
* https://www.wireshark.org/docs/dfref/f/frame.html

**Datasets**
* https://www.unb.ca/cic/datasets/ids-2017.html
* https://ocslab.hksecurity.net/Datasets/iot-network-intrusion-dataset

**Related Github Projects**
* https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/MTH_IDS_IoTJ.ipynb
* ~~https://github.com/shibha20/Intrusion-Detection-in-Wireless-Network~~ [REMOVED]
* https://github.com/kahramankostas/Anomaly-Detection-in-Networks-Using-Machine-Learning
* https://github.com/Bee-Mar/AWID-Intrusion-Detection
* https://github.com/ymirsky/Kitsune-py
* https://github.com/vinayakumarr/Network-Intrusion-Detection

**Attack Generation**
* https://github.com/tklab-tud/ID2T

**Steps:**

1. Gather data frames to be used to train the network\
  1.1. Using WireShark or pyshark to capture normal and abnormal activity on a network\
  1.2. Label this data as either normal or abnormal
  
2. Build the ML model using tensorflow\
  2.1 Re-read the paper to determine the type of model

3. Train the model on the collected datasets and labels.

4. Test the model against real time traffic.\
  4.1. Compare results against the papers. Using metrics such as accuracy, detection time, etc.
