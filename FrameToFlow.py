import pyshark

def extract_feature_set_from_capture(captrue_path):
    captrue_array = pyshark.FileCapture(captrue_path)
    feature_set = dict()
    # print(dir(captrue_array[1]['wlan']))
    # print(captrue_array[1]['wlan']._all_fields)
    for capture in captrue_array:
        # print(capture)
        capture_fields = capture.wlan._all_fields
        if(capture_fields['wlan.fc.type_subtype'] == '29'):
            # No src address for an ACK frame only dst
            # We want to include this in the n-gram
            feature_array = []
            # Source = none
            feature_array.append(None)
            # Recieve = dst address
            feature_array.append(capture_fields['wlan.ra'])
            # Frame type
            feature_array.append(capture_fields['wlan.fc.type'])
            # Frame subtype
            feature_array.append(capture_fields['wlan.fc.subtype'])
            feature_set[capture.frame_info.time_epoch] = feature_array
        elif(capture_fields['wlan.fc.type_subtype'] == '33'):
            # Malformed packet, retransmission nessicary
            continue
        elif(capture_fields['wlan.fc.type_subtype'] == '49'):
            # Fragmented frame, contains data
            continue
        elif(capture_fields['wlan.fc.type_subtype'] == '59' or capture_fields['wlan.fc.type_subtype'] == '51'
                or capture_fields['wlan.fc.type_subtype'] == '57' or capture_fields['wlan.fc.type_subtype'] == '55'):
            # Unrecognized frame
            continue
        elif(capture_fields['wlan.fc.type_subtype'] == '28'):
            # CTS/Clear to send frame
            continue
        else:
            # print(capture_fields)
            feature_array = []
            # Transmit = src address
            feature_array.append(capture_fields['wlan.ta'])
            # Recieve = dst address
            feature_array.append(capture_fields['wlan.ra'])
            # Frame type
            feature_array.append(capture_fields['wlan.fc.type'])
            # Frame subtype
            feature_array.append(capture_fields['wlan.fc.subtype'])
            feature_set[capture.frame_info.time_epoch] = feature_array
            # TODO Hash the type and subtype togeather
            # TODO use t (time) to create n-gram flows of frames (Auth, Asso, Data, Deauth)
    captrue_array.close()
    print(feature_set)
    return(feature_set)

extract_feature_set_from_capture('Wireshark_802_11.pcap')