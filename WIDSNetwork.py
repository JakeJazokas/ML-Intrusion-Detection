'''
Initial input is a stream of data frames

These frames are organized into a structure called an observation flow

Then, the n-grams are extracted from the observation flow

Finally, a binary classification (machine learning) algorithm is used to determine if the flow is normal. 
This assumes that all n-grams within the flow are used for the classification.

Output layer of the binary classifier -> Returns 1 if abnormal, 0 otherwise
'''
