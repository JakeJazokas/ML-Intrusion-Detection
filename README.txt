Project: Intrusion Detection for Wi-Fi networks
Authors: Joseph Vinson - 101126637, Jake Jazokas - 101083496, AJ Ricketts - 101084146

File Locations:

    - 4203-Project-Report-Intrusion-Detection.pdf -> The pdf for the completed project report.

    - Attack Dataset files not included as they are very large, however all datasets were created from the AWID3 dataset.

    - /training/ Directory
        - custom_dataset.csv -> The CSV dataset we used to train the models

    - /code/ Directory
        - CaptureToFlow.py -> Helper functions that translate captures or datasets to n-gram flows
        - FlowToFeatures.py -> Helper functions to extract probabilitiy features from n-gram flows
        - WIDSNetwork.py -> Functions to train, save, and test ML models
    
    - /results/ Directory
        - example_live_output.txt -> Output from using our AdaBoost model to classify live capture
        - trace.pcap -> The live capture file generated
        - ResultsGraphs.png -> Visual accuracy results for the AdaBoost model
        - ResultsGraphs2.png -> Visual accuracy results for the RandomForest model
    
    - /trained-models/ Directory
        - smallDsModel5000.pkl -> Saved (pickle) model using the AdaBoost classifer
        - smallDsModel5000-Random.pkl -> Saved (pickle) model using the RandomForest classifer

Execution:

    - Run 'python code\WIDSNetwork.py' from the main directory, make sure you have the required python modules 
        - this runs a demo and prints the general accuracy the RandomForest classifer after training