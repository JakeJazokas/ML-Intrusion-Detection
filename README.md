# 4203-Intrusion-Detection-Project

**Presentation Video**
https://youtu.be/X9NMJFhzqqQ 

**Refferenced Paper**
Satam, & Hariri, S. (2021). WIDS: An Anomaly Based Intrusion Detection System for Wi-Fi (IEEE 802.11) Protocol. IEEE eTransactions on Network and Service Management, 18(1), 1077–1091. https://doi.org/10.1109/TNSM.2020.3036138

**Other Links**
* https://www.tensorflow.org/ 
* https://keras.io/about/ 
* https://numpy.org/ 
* https://www.wireshark.org/ 
* https://github.com/KimiNewt/pyshark/ 
* https://matplotlib.org/ 

**Steps:**

1. Gather data frames to be used to train the network\
  1.1. Using WireShark or pyshark to capture normal and abnormal activity on a network\
  1.2. Label this data as either normal or abnormal
  
2. Build the ML model using tensorflow\
  2.1 Re-read the paper to determine the type of model

3. Train the model on the collected datasets and labels.

4. Test the model against real time traffic.\
  4.1. Compare results against the papers. Using metrics such as accuracy, detection time, etc.
