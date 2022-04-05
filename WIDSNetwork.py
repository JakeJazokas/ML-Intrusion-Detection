from sklearn.ensemble import AdaBoostClassifier
from CaptureToFlow import CaptureToFlow
import numpy as np
import pandas as pd
import os
import pickle
from CaptureToFlow import CaptureToFlow
'''
Initial input is a stream of data frames

These frames are organized into a structure called an observation flow

Then, the n-grams are extracted from the observation flow

Finally, a binary classification (machine learning) algorithm is used to determine if the flow is normal. 
This assumes that all n-grams within the flow are used for the classification.

Output layer of the binary classifier -> Returns 1 if abnormal, 0 otherwise
'''

def get_custom_data_from_dataset():
    # frame.time_epoch, wlan.ta, wlan.ra, wlan.fc.type, wlan.fc.subtype, wlan.fc.type_subtype, class
    desired_cols = [3, 77, 75, 65, 66, 63, 154]

    # Read specific columns of CSV file from ZIP file in resources directory
    data = pd.read_csv(
        "full_dataset.zip",
        sep=',',
        header=None,
        compression='zip',
        usecols=desired_cols
    )
    
    dataset_col_name = [
        "frame.interface_id", "frame.dlt", "frame.offset_shift", "frame.time_epoch", "frame.time_delta", "frame.time_delta_displayed", 
        "frame.time_relative", "frame.len", "frame.cap_len", "frame.marked", "frame.ignored", "radiotap.version", "radiotap.pad", 
        "radiotap.length", "radiotap.present.tsft", "radiotap.present.flags", "radiotap.present.rate", "radiotap.present.channel", 
        "radiotap.present.fhss", "radiotap.present.dbm_antsignal", "radiotap.present.dbm_antnoise", "radiotap.present.lock_quality", 
        "radiotap.present.tx_attenuation", "radiotap.present.db_tx_attenuation", "radiotap.present.dbm_tx_power", "radiotap.present.antenna", 
        "radiotap.present.db_antsignal", "radiotap.present.db_antnoise", "radiotap.present.rxflags", "radiotap.present.xchannel", 
        "radiotap.present.mcs", "radiotap.present.ampdu", "radiotap.present.vht", "radiotap.present.reserved", "radiotap.present.rtap_ns", 
        "radiotap.present.vendor_ns", "radiotap.present.ext", "radiotap.mactime", "radiotap.flags.cfp", "radiotap.flags.preamble", 
        "radiotap.flags.wep", "radiotap.flags.frag", "radiotap.flags.fcs", "radiotap.flags.datapad", "radiotap.flags.badfcs", 
        "radiotap.flags.shortgi", "radiotap.datarate", "radiotap.channel.freq", "radiotap.channel.type.turbo", "radiotap.channel.type.cck", 
        "radiotap.channel.type.ofdm", "radiotap.channel.type.2ghz", "radiotap.channel.type.5ghz", "radiotap.channel.type.passive", 
        "radiotap.channel.type.dynamic", "radiotap.channel.type.gfsk", "radiotap.channel.type.gsm", "radiotap.channel.type.sturbo", 
        "radiotap.channel.type.half", "radiotap.channel.type.quarter", "radiotap.dbm_antsignal", "radiotap.antenna", 
        "radiotap.rxflags.badplcp", "wlan.fc.type_subtype", "wlan.fc.version", "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", 
        "wlan.fc.frag", "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "wlan.fc.order", "wlan.duration", 
        "wlan.ra", "wlan.da", "wlan.ta", "wlan.sa", "wlan.bssid", "wlan.frag", "wlan.seq", "wlan.bar.type", "wlan.ba.control.ackpolicy", 
        "wlan.ba.control.multitid", "wlan.ba.control.cbitmap", "wlan.bar.compressed.tidinfo", "wlan.ba.bm", "wlan.fcs_good", 
        "wlan_mgt.fixed.capabilities.ess", "wlan_mgt.fixed.capabilities.ibss", "wlan_mgt.fixed.capabilities.cfpoll.ap", 
        "wlan_mgt.fixed.capabilities.privacy", "wlan_mgt.fixed.capabilities.preamble", "wlan_mgt.fixed.capabilities.pbcc", 
        "wlan_mgt.fixed.capabilities.agility", "wlan_mgt.fixed.capabilities.spec_man", "wlan_mgt.fixed.capabilities.short_slot_time", 
        "wlan_mgt.fixed.capabilities.apsd", "wlan_mgt.fixed.capabilities.radio_measurement", "wlan_mgt.fixed.capabilities.dsss_ofdm", 
        "wlan_mgt.fixed.capabilities.del_blk_ack", "wlan_mgt.fixed.capabilities.imm_blk_ack", "wlan_mgt.fixed.listen_ival", 
        "wlan_mgt.fixed.current_ap", "wlan_mgt.fixed.status_code", "wlan_mgt.fixed.timestamp", "wlan_mgt.fixed.beacon", 
        "wlan_mgt.fixed.aid", "wlan_mgt.fixed.reason_code", "wlan_mgt.fixed.auth.alg", "wlan_mgt.fixed.auth_seq", 
        "wlan_mgt.fixed.category_code", "wlan_mgt.fixed.htact", "wlan_mgt.fixed.chanwidth", "wlan_mgt.fixed.fragment", 
        "wlan_mgt.fixed.sequence", "wlan_mgt.tagged.all", "wlan_mgt.ssid", "wlan_mgt.ds.current_channel", "wlan_mgt.tim.dtim_count", 
        "wlan_mgt.tim.dtim_period", "wlan_mgt.tim.bmapctl.multicast", "wlan_mgt.tim.bmapctl.offset", "wlan_mgt.country_info.environment", 
        "wlan_mgt.rsn.version", "wlan_mgt.rsn.gcs.type", "wlan_mgt.rsn.pcs.count", "wlan_mgt.rsn.akms.count", "wlan_mgt.rsn.akms.type", 
        "wlan_mgt.rsn.capabilities.preauth", "wlan_mgt.rsn.capabilities.no_pairwise", "wlan_mgt.rsn.capabilities.ptksa_replay_counter", 
        "wlan_mgt.rsn.capabilities.gtksa_replay_counter", "wlan_mgt.rsn.capabilities.mfpr", "wlan_mgt.rsn.capabilities.mfpc", 
        "wlan_mgt.rsn.capabilities.peerkey", "wlan_mgt.tcprep.trsmt_pow", "wlan_mgt.tcprep.link_mrg", "wlan.wep.iv", "wlan.wep.key", 
        "wlan.wep.icv", "wlan.tkip.extiv", "wlan.ccmp.extiv", "wlan.qos.tid", "wlan.qos.priority", "wlan.qos.eosp", "wlan.qos.ack", 
        "wlan.qos.amsdupresent", "wlan.qos.buf_state_indicated", "wlan.qos.bit4", "wlan.qos.txop_dur_req", "wlan.qos.buf_state_indicated", 
        "data.len", "class"
    ]
    col_names = []

    for line_num, name in enumerate(dataset_col_name):
        if line_num in desired_cols:
            col_names.append(name.rstrip())

    # Set the column headers to the names from the Wireshark frame
    data.columns = col_names

    data = data.replace('?', np.nan)

    # Output the minimized dataset to a CSV file (with no index column added)
    data.to_csv("custom_dataset.csv", sep=',', index=False)

def get_n_grams_from_custom_dataset():
    data = pd.read_csv(
        "custom_dataset.csv",
        sep=',',
        header=0,
    )
    # frame.time_epoch,wlan.fc.type_subtype,wlan.fc.type,wlan.fc.subtype,wlan.ra,wlan.ta,class
    all_n_gram_flows = CaptureToFlow().create_n_grams_from_dataset_features(data.values)
    classified_n_gram_flows = []
    n_gram_flow_labels = []

    for n_gram in all_n_gram_flows:
        label_array = n_gram[:,6]
        count = np.count_nonzero(label_array == 'normal')
        n_gram_array = np.asarray(n_gram[:,:6])
        if(count < 4):
            # classified_n_gram_flows.append([n_gram_array, 1])
            # classified_n_gram_flows.append(np.append(n_gram_array, 1))
            classified_n_gram_flows.append(n_gram_array)
            n_gram_flow_labels.append(1)
            # classified_n_gram_flows.append(1)
        else:
            # classified_n_gram_flows.append([n_gram_array, 0])    
            # classified_n_gram_flows.append(np.append(n_gram_array, 0))
            classified_n_gram_flows.append(n_gram_array)
            n_gram_flow_labels.append(0)
            # classified_n_gram_flows.append(0)
    # return np.asarray(classified_n_gram_flows)
    return classified_n_gram_flows, n_gram_flow_labels

def train_network_from_classified_flows(classified_n_gram_flows, n_gram_flow_labels):
    model = AdaBoostClassifier(n_estimators=100)
    # print(classified_n_gram_flows[0])
    # values = np.array(classified_n_gram_flows[:, 0])
    # labels = classified_n_gram_flows[:, 1]
    # print(np.array(classified_n_gram_flows).shape)
    # print(np.array(classified_n_gram_flows))
    # test_ngram = np.array(classified_n_gram_flows)[0].reshape(1, 6*4)
    # print(test_ngram)
    classified_n_gram_flows = np.array(classified_n_gram_flows).reshape(9821, 6*4)
    model.fit(classified_n_gram_flows, n_gram_flow_labels)
    # print(history)
    # print(model.predict(test_ngram))
    if not os.path.exists('smallDsModel.pkl'):
        with open('smallDsModel.pkl', 'wb') as f:
            pickle.dump(model, f)

def used_trained_model_to_predit_flow(model_path, n_gram_flows):
    model = pickle.load(open(model_path, 'rb'))
    predictions = model.predict(n_gram_flows)
    return(predictions)

def predict_live_capture(model_path):
    features = CaptureToFlow().extract_feature_set_from_live_capture(timeout=10)
    n_gram_flows = CaptureToFlow().create_n_grams_from_dataset_features(features)
    return used_trained_model_to_predit_flow(model_path,n_gram_flows)

if __name__ == "__main__":
    # get_custom_data_from_dataset() and save to file
    # x, y = get_n_grams_from_custom_dataset()
    # train_network_from_classified_flows(x, y)
    print(predict_live_capture('smallDsModel.pkl'))

# # 

# # make predictions
# expected = testlabel
# predicted = model.predict(testdata)
# # summarize the fit of the model

# cm = metrics.confusion_matrix(expected, predicted)
# print(cm)
# tpr = float(cm[0][0])/np.sum(cm[0])
# fpr = float(cm[1][1])/np.sum(cm[1])
# print("%.3f" %tpr)
# print("%.3f" %fpr)
# print("Accuracy")
# print("%.3f" %accuracy)
# print("precision")
# print("%.3f" %precision)
# print("recall")
# print("%.3f" %recall)
# print("f-score")
# print("%.3f" %f1)
# print("fpr")
# print("%.3f" %fpr)
# print("tpr")
# print("%.3f" %tpr)
# print("***************************************************************")
