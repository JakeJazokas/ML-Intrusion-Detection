import pyshark
import time
import numpy as np
import pandas as pd
import subprocess

class CaptureToFlow():

    def __init__(self):
        pass

    def hash_observation_features(self, observation_array):
        output_array = np.empty(shape=(0,6))
        for o in observation_array:
            if(o[4] == None or o[5] == None):
                o[4] = 0
                o[5] = 0
            else:
                o[4] = int(o[4].replace(":", ""), 16)
                o[5] = int(o[5].replace(":", ""), 16)
                output_array = np.vstack((output_array, o))
        return output_array

    def extract_feature_set_from_capture_path(self, captrue_path):
        captrue_array = pyshark.FileCapture(captrue_path)
        feature_set = np.empty(shape=(0,6))
        # print(dir(captrue_array[1]['wlan']))
        # print(captrue_array[1]['wlan']._all_fields)
        for capture in captrue_array:
            if 'WLAN' in capture and 'wlan.fc.type_subtype' in capture.wlan._all_fields:
                capture_fields = capture.wlan._all_fields
                # print(capture_fields)
                capture_fields['wlan.fc.type_subtype'] = str(int(capture_fields['wlan.fc.type_subtype'], 16))
                if(capture_fields['wlan.fc.type_subtype'] == '29'):
                    # No src address for an ACK frame only dst
                    # We want to include this in the n-gram
                    feature_array = []
                    # Epoch time
                    feature_array.append(capture.frame_info.time_epoch)
                    # Source = none
                    # feature_array.append(np.nan)
                    feature_array.append(0)
                    # Recieve = dst address
                    feature_array.append(capture_fields['wlan.ra'])
                    # Frame type
                    feature_array.append(capture_fields['wlan.fc.type'])
                    # Frame subtype
                    feature_array.append(capture_fields['wlan.fc.subtype'])
                    # Hash the type and subtype togeather
                    feature_array.append(capture_fields['wlan.fc.type_subtype'])
                    # Add to total numpy array
                    feature_set = np.vstack((feature_set, feature_array))
                elif(capture_fields['wlan.fc.type_subtype'] == '33'):
                    # Malformed packet, retransmission nessicary
                    # We don't care about these
                    continue
                elif(capture_fields['wlan.fc.type_subtype'] == '49'):
                    # Fragmented frame, contains data
                    # We don't care about these currently, but we might need to
                    continue
                elif(capture_fields['wlan.fc.type_subtype'] == '59' or capture_fields['wlan.fc.type_subtype'] == '51'
                        or capture_fields['wlan.fc.type_subtype'] == '57' or capture_fields['wlan.fc.type_subtype'] == '55'):
                    # Unrecognized frame
                    # We don't care about these
                    continue
                elif(capture_fields['wlan.fc.type_subtype'] == '28'):
                    # CTS/Clear to send frame
                    # We don't care about these currently, but we might need to
                    continue
                else:
                    feature_array = []
                    # Epoch time
                    feature_array.append(capture.frame_info.time_epoch)
                    # Transmit = src address
                    if(not 'wlan.ta' in capture_fields):
                        # feature_array.append(np.nan)
                        feature_array.append(0)
                    else:
                        feature_array.append(capture_fields['wlan.ta'])
                    # Recieve = dst address
                    feature_array.append(capture_fields['wlan.ra'])
                    # Frame type
                    feature_array.append(capture_fields['wlan.fc.type'])
                    # Frame subtype
                    feature_array.append(capture_fields['wlan.fc.subtype'])
                    # Hash the type and subtype togeather
                    feature_array.append(capture_fields['wlan.fc.type_subtype'])
                    # Add to total numpy array
                    feature_set = np.vstack((feature_set, feature_array))
        captrue_array.close()
        return(feature_set)

    def extract_feature_set_from_capture(self, captrue):
        feature_set = np.empty(shape=(0,6))
        for i in range(len(captrue)):
            if 'WLAN' in captrue[i]:
                print(captrue[i])
                capture_fields = captrue[i].wlan._all_fields
                print(capture_fields['wlan.fc.type_subtype'])
                capture_fields['wlan.fc.type_subtype'] = int(capture_fields['wlan.fc.type_subtype'], 16)
                if(capture_fields['wlan.fc.type_subtype'] == 29):
                    # No src address for an ACK frame only dst
                    # We want to include this in the n-gram
                    feature_array = []
                    # Epoch time
                    feature_array.append(captrue[i].frame_info.time_epoch)
                    # Source = none
                    # feature_array.append(None)
                    feature_array.append(0)
                    # Recieve = dst address
                    feature_array.append(capture_fields['wlan.ra'])
                    # Frame type
                    feature_array.append(capture_fields['wlan.fc.type'])
                    # Frame subtype
                    feature_array.append(capture_fields['wlan.fc.subtype'])
                    # Hash the type and subtype togeather
                    feature_array.append(capture_fields['wlan.fc.type_subtype'])
                    # Add to total numpy array
                    feature_set = np.vstack((feature_set, feature_array))
                elif(capture_fields['wlan.fc.type_subtype'] == 33):
                    # Malformed packet, retransmission nessicary
                    # We don't care about these
                    continue
                elif(capture_fields['wlan.fc.type_subtype'] == 49):
                    # Fragmented frame, contains data
                    # We don't care about these currently, but we might need to
                    continue
                elif(capture_fields['wlan.fc.type_subtype'] == 59 or capture_fields['wlan.fc.type_subtype'] == 51
                        or capture_fields['wlan.fc.type_subtype'] == 57 or capture_fields['wlan.fc.type_subtype'] == 55):
                    # Unrecognized frame
                    # We don't care about these
                    continue
                elif(capture_fields['wlan.fc.type_subtype'] == 28):
                    # CTS/Clear to send frame
                    # We don't care about these currently, but we might need to
                    continue
                else:
                    feature_array = []
                    # Epoch time
                    feature_array.append(captrue[i].frame_info.time_epoch)
                    # Transmit = src address
                    feature_array.append(capture_fields['wlan.ta'])
                    # Recieve = dst address
                    feature_array.append(capture_fields['wlan.ra'])
                    # Frame type
                    feature_array.append(capture_fields['wlan.fc.type'])
                    # Frame subtype
                    feature_array.append(capture_fields['wlan.fc.subtype'])
                    # Hash the type and subtype togeather
                    feature_array.append(capture_fields['wlan.fc.type_subtype'])
                    # Add to total numpy array
                    feature_set = np.vstack((feature_set, feature_array))
        return(feature_set)

    def create_n_grams_from_observed_features(self, features):
        '''
        Each element of the features 2d array is formatted as follows:
        -   [Epoch time, Soruce Address, Destination Address, Frame Type, Frame Subtype, Type/Subtype Hash]

        This function use t (time) to create n-gram flows of frames (Auth, Asso, Data, Deauth)
        '''
        pattern_length = 0
        four_gram_pattern = []
        # all_n_grams = np.empty(shape=(4,6))
        all_n_grams = []
        for feature in features:
            # Hash the MAC adresses
            if isinstance(feature[1], float) or feature[1] == 'nan':
                feature[1] == 0
            elif isinstance(feature[1], str):
                feature[1] = int(feature[1].replace(":", ""), 16)
            if isinstance(feature[2], float) or feature[2] == 'nan':
                feature[2] == 0
            elif isinstance(feature[2], str):
                feature[2] = int(feature[2].replace(":", ""), 16)
            # Only valid start frames are: Auth, Asso Req, and Data
            if(pattern_length == 0):
                # Authentication (Hash=11)
                # Deauthentication (Hash=12)
                if(feature[5] == '11'):
                    four_gram_pattern.append(feature)
                    pattern_length += 1
                # Association Request (Hash=0)
                elif(feature[5] == '0'):
                    four_gram_pattern.append(feature)
                    pattern_length += 1
                # Data Frame (Type=2)
                # TODO Fix this
                # elif(feature[3] == '2'):
                #     four_gram_pattern.append(feature)
                #     pattern_length += 1
            elif(pattern_length == 1):
                # If frame 1 was Auth, next must be Auth or Asso Req
                if(four_gram_pattern[0][5] == '11'):
                    if(feature[5] == '11' or feature[5] == '0'):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
                # If frame 1 was Asso Req or Data, next must be Data
                elif(four_gram_pattern[0][5] == '0' or four_gram_pattern[0][3] == '2'):
                    if(feature[3] == '2'):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
            elif(pattern_length == 2):
                # If frame 2 was Auth, next must be Asso Req
                if(four_gram_pattern[1][5] == '11'):
                    if(feature[5] == '0'):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
                # If frame 2 was Asso Req, next must be Data
                elif(four_gram_pattern[1][5] == '0'):
                    if(feature[3] == '2'):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
                # If frame 2 was Data, next must be Data or Deauth
                elif(four_gram_pattern[1][3] == '2'):
                    if(feature[3] == '2' or feature[5] == '12'):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
            elif(pattern_length == 3):
                # If frame 3 was Asso Req, last must be Data
                if(four_gram_pattern[2][5] == '0'):
                    if(feature[3] == '2'):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
                # If frame 3 was Data, last must be Data or Deauth (When all others are data)
                elif(four_gram_pattern[2][3] == '2'):
                    if(four_gram_pattern[0][3] == '2' and four_gram_pattern[1][3] == '2'):
                        if(feature[5] == '12'):
                            four_gram_pattern.append(feature)
                            pattern_length += 1
                    else:
                        if(feature[3] == '2'):
                            four_gram_pattern.append(feature)
                            pattern_length += 1
                # If frame 3 was Deauth, last must be Deauth
                elif(four_gram_pattern[2][5] == '12'):
                    if(feature[5] == '12'):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
            elif(pattern_length == 4):
                # Add to total and reset
                # all_n_grams = np.vstack((all_n_grams, four_gram_pattern))
                all_n_grams.append(np.array(four_gram_pattern))
                # print(all_n_grams)
                four_gram_pattern = []
                pattern_length = 0
            # print(pattern_length)
            # print(len(four_gram_pattern))
        return(np.asarray(all_n_grams))
    
    def create_n_grams_from_dataset_features(self, features):
        '''
        Each element of the features 2d array is formatted as follows:
        -   [frame.time_epoch,wlan.fc.type_subtype,wlan.fc.type,wlan.fc.subtype,wlan.ra,wlan.ta,class]

        This function use t (time) to create n-gram flows of frames (Auth, Asso, Data, Deauth)
        '''
        pattern_length = 0
        four_gram_pattern = []
        all_n_grams = []
        for feature in features:
            feature = np.array(feature)
            type_subtype_hash = int(feature[1], 16)
            type_number = feature[2]
            feature[1] = type_subtype_hash
            # Hash the MAC adresses
            if isinstance(feature[4], str):
                feature[4] == 0
            elif isinstance(feature[4], str):
                feature[4] = int(feature[4].replace(":", ""), 16)
            if isinstance(feature[5], float):
                feature[5] == 0
            elif isinstance(feature[5], str):
                feature[5] = int(feature[5].replace(":", ""), 16)
            # Only valid start frames are: Auth, Asso Req, and Data
            if(pattern_length == 0):
                # Authentication (Hash=11)
                # Deauthentication (Hash=12)
                if(type_subtype_hash == 11):
                    four_gram_pattern.append(feature)
                    pattern_length += 1
                # Association Request (Hash=0)
                elif(type_subtype_hash == 0):
                    four_gram_pattern.append(feature)
                    pattern_length += 1
                # Data Frame (Type=2)
                elif(type_number == 2):
                    four_gram_pattern.append(feature)
                    pattern_length += 1
            elif(pattern_length == 1):
                # If frame 1 was Auth, next must be Auth or Asso Req
                # if(int(four_gram_pattern[0][1], 16) == 11):
                if(four_gram_pattern[0][1] == 11):
                    if(type_subtype_hash == 11 or type_subtype_hash == 0):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
                # If frame 1 was Asso Req or Data, next must be Data
                # elif(int(four_gram_pattern[0][1], 16) == 0 or four_gram_pattern[0][2] == 2):
                elif(four_gram_pattern[0][1] == 0 or four_gram_pattern[0][2] == 2):
                    if(type_number == 2):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
            elif(pattern_length == 2):
                # If frame 2 was Auth, next must be Asso Req
                # if(int(four_gram_pattern[1][1], 16) == 11):
                if(four_gram_pattern[1][1] == 11):
                    if(type_subtype_hash == 0):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
                # If frame 2 was Asso Req, next must be Data
                # elif(int(four_gram_pattern[1][1], 16) == 0):
                elif(four_gram_pattern[1][1] == 0):
                    if(type_number == 2):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
                # If frame 2 was Data, next must be Data or Deauth
                elif(four_gram_pattern[1][2] == 2):
                    if(type_number == 2 or type_subtype_hash == 12):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
            elif(pattern_length == 3):
                # If frame 3 was Asso Req, last must be Data
                # if(int(four_gram_pattern[2][1], 16) == 0):
                if(four_gram_pattern[2][1] == 0):
                    if(type_number == 2):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
                # If frame 3 was Data, last must be Data or Deauth (When all others are data)
                elif(four_gram_pattern[2][2] == 2):
                    if(four_gram_pattern[0][2] == 2 and four_gram_pattern[1][2] == 2):
                        if(type_subtype_hash == 12):
                            four_gram_pattern.append(feature)
                            pattern_length += 1
                    else:
                        if(type_number == 2):
                            four_gram_pattern.append(feature)
                            pattern_length += 1
                # If frame 3 was Deauth, last must be Deauth
                # elif(int(four_gram_pattern[2][1], 16) == 12):
                elif(four_gram_pattern[2][1] == 12):
                    if(type_subtype_hash == 12):
                        four_gram_pattern.append(feature)
                        pattern_length += 1
            elif(pattern_length == 4):
                # Add to total and reset
                all_n_grams.append(np.array(four_gram_pattern))
                four_gram_pattern = []
                pattern_length = 0
        return(np.array(all_n_grams))
    
    def generate_live_pcap(self, filename):
        p = subprocess.Popen(['tcpdump', '-In', '-i', 'en0', '-w', filename], stdout=subprocess.PIPE)
        time.sleep(10)
        p.terminate()

    def extract_feature_set_from_live_capture(self, timeout):
        '''
        Implementation for live capture using MAC OSX
        '''
        #sudo tcpdump -In -i en0 -w trace.pcap
        capture = pyshark.LiveCapture(
            interface='en0',
            monitor_mode=True, 
            decryption_key='37272F911391:BELL397', 
            encryption_type='wpa-pwd', 
            output_file='datadump1_pyshark.pcap'
        ) # Capture on given interfaces # NEED MONITOR MODE
        capture.set_debug()
        capture.sniff(timeout=timeout)
        capture.close()
        return self.extract_feature_set_from_capture(capture)

# f = extract_feature_set_from_capture('Wireshark_802_11.pcap')
# n = create_n_grams_from_observed_features(f)
# print(n)

# Change the interface before use
# extract_feature_set_from_live_capture_to_file('TestCap.pcap', r'\\Device\\NPF_{8615F95B-43AB-4DB1-A4FF-35172DFE1D57}', 10)
# f2 = extract_feature_set_from_live_capture(10)
# # Extract and return the feature set from the capture
# # f2 = extract_feature_set_from_capture('TestCap.pcap')
# n2 = create_n_grams_from_observed_features(f2)
# print(n2)
