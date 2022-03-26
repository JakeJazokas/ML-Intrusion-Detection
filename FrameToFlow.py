import pyshark

def capture_to_n_gram_mapping(captrue_path):
    captrue_array = pyshark.FileCapture(captrue_path)
    n_gram_map = dict()
    # print(dir(captrue_array[1]['wlan']))
    # print(captrue_array[1]['wlan']._all_fields)
    for capture in captrue_array:
        capture_fields = capture.wlan._all_fields
        if(capture_fields['wlan.fc.type_subtype'] == '29'):
            # No src address for an ACK frame
            continue
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
            n_gram_map[capture.frame_info.time_epoch] = feature_array
            # TODO Hash the type and subtype togeather, and use t (time) to create n-gram flows of frames
    captrue_array.close()
    # print(n_gram_map)
    return(n_gram_map)

capture_to_n_gram_mapping('Project/Wireshark_802_11.pcap')