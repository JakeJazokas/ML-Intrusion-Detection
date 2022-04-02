import pyshark
import time
import numpy as np

def extract_feature_set_from_capture_path(captrue_path):
    captrue_array = pyshark.FileCapture(captrue_path)
    feature_set = np.empty(shape=(0,6))
    # print(dir(captrue_array[1]['wlan']))
    # print(captrue_array[1]['wlan']._all_fields)
    for capture in captrue_array:
        # print(capture)
        if 'WLAN' in capture:
            capture_fields = capture.wlan._all_fields
            if(capture_fields['wlan.fc.type_subtype'] == '29'):
                # No src address for an ACK frame only dst
                # We want to include this in the n-gram
                feature_array = []
                # Epoch time
                feature_array.append(capture.frame_info.time_epoch)
                # Source = none
                feature_array.append(None)
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
    # print(feature_set)
    return(feature_set)

def extract_feature_set_from_capture(captrue):
    feature_set = np.empty(shape=(0,6))

    # with open('output.txt', 'w') as f:
    #     for x in captrue:
    #         f.write(str(x))

    for i in range(len(captrue)):
        if 'WLAN' in captrue[i]:
            print(captrue[i])
            capture_fields = captrue[i].wlan._all_fields
            if(capture_fields['wlan.fc.type_subtype'] == '29'):
                # No src address for an ACK frame only dst
                # We want to include this in the n-gram
                feature_array = []
                # Epoch time
                feature_array.append(captrue[i].frame_info.time_epoch)
                # Source = none
                feature_array.append(None)
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

def create_n_grams_from_observed_features(features):
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
            elif(feature[3] == '2'):
                four_gram_pattern.append(feature)
                pattern_length += 1
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
            all_n_grams.append(four_gram_pattern)
            four_gram_pattern = []
            pattern_length = 0
    return(np.asarray(all_n_grams))

def extract_feature_set_from_live_capture(timeout):
    # TODO Update to use Wi-Fi and Monitor mode
    capture = pyshark.LiveCapture(interface='eth', monitor_mode=True) # Capture on given interfaces # NEED MONITOR MODE
    capture.set_debug()
    capture.sniff(timeout=timeout)
    capture.close()
    return extract_feature_set_from_capture(capture)

def extract_feature_set_from_live_capture_to_file(output_path, interface_name, timeout):
    # TODO Update to use Wi-Fi and Monitor mode
    capture = pyshark.LiveCapture(interface='eth', output_file=output_path) # Capture on given interfaces to the given path
    count = 0
    start = time.time()
    capture.set_debug()
    # for item in capture.sniff_continuously():
        # # print(item)
        # if timeout and time.time() - start > timeout:
        #     break
    # Create a capture file with n observations, each taking 10 seconds
    # while count < iterations:
    capture.sniff(timeout=10) # Capture for 10 Seconds
        # count += 1# Increase count
    # Try to close the file and return if succesful
    # try:
    #     capture.close()
    # except:
    #     return False
    # return True
    # capture.clear()
    print(capture[-1])
    capture.close()
    return

# f = extract_feature_set_from_capture('Wireshark_802_11.pcap')
# n = create_n_grams_from_observed_features(f)
# print(n)

# Change the interface before use
# extract_feature_set_from_live_capture_to_file('TestCap.pcap', r'\\Device\\NPF_{8615F95B-43AB-4DB1-A4FF-35172DFE1D57}', 10)
f2 = extract_feature_set_from_live_capture(10)
# Extract and return the feature set from the capture
# f2 = extract_feature_set_from_capture('TestCap.pcap')
n2 = create_n_grams_from_observed_features(f2)
print(n2)