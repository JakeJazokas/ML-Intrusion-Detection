import matplotlib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from CaptureToFlow import CaptureToFlow
import numpy as np
import pandas as pd
import os
import pickle
from CaptureToFlow import CaptureToFlow
import matplotlib.pyplot as plt

'''
Initial input is a stream of data frames

These frames are organized into a structure called an observation flow

Then, the n-grams are extracted from the observation flow

Finally, a binary classification (machine learning) algorithm is used to determine if the flow is normal. 
This assumes that all n-grams within the flow are used for the classification.

Output layer of the binary classifier -> Returns 1 if abnormal, 0 otherwise
'''

def get_custom_data_from_dataset(input_dataset, compression_type, output_dataset, trainingBool):
    # Different column names depending on the dataset
    if(trainingBool):
        # frame.time_epoch, wlan.ta, wlan.ra, wlan.fc.type, wlan.fc.subtype, wlan.fc.type_subtype, class
        desired_cols = [3, 77, 75, 65, 66, 63, 154]
        # Read specific columns of CSV file from ZIP file in resources directory
        data = pd.read_csv(
            # "full_dataset.zip",
            input_dataset,
            sep=',',
            header=None,
            compression=compression_type, #'zip'
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
    else:
        # frame.time_epoch, wlan.ta, wlan.ra, wlan.fc.type, wlan.fc.subtype, wlan.fc.type_subtype, class aka. Label
        desired_cols = [6, 49, 41, 32, 34, 253]
        # Read specific columns of CSV file from ZIP file in resources directory
        data = pd.read_csv(
            input_dataset,
            sep=',',
            header=None,
            compression=compression_type, #'zip'
            usecols=desired_cols,
            dtype=object
            # dtype={
            #     "frame.time_epoch" : str,
            #     "wlan.ta" : str, 
            #     "wlan.ra" : str, 
            #     "wlan.fc.type" : int, 
            #     "wlan.fc.subtype" : int,
            #     "Label" : str
            # }
        )
        dataset_col_name = [
            "frame.encap_type","frame.len","frame.number","frame.time","frame.time_delta","frame.time_delta_displayed","frame.time_epoch",
            "frame.time_relative","radiotap.channel.flags.cck","radiotap.channel.flags.ofdm","radiotap.channel.freq","radiotap.datarate",
            "radiotap.dbm_antsignal","radiotap.length","radiotap.mactime","radiotap.present.tsft","radiotap.rxflags","radiotap.timestamp.ts",
            "radiotap.vendor_oui","wlan.duration","wlan.analysis.kck","wlan.analysis.kek","wlan.bssid","wlan.country_info.fnm",
            "wlan.country_info.code","wlan.da","wlan.fc.ds","wlan.fc.frag","wlan.fc.order","wlan.fc.moredata","wlan.fc.protected",
            "wlan.fc.pwrmgt","wlan.fc.type","wlan.fc.retry","wlan.fc.subtype","wlan.fcs.bad_checksum","wlan.fixed.beacon",
            "wlan.fixed.capabilities.ess","wlan.fixed.capabilities.ibss","wlan.fixed.reason_code","wlan.fixed.timestamp",
            "wlan.ra","wlan_radio.duration","wlan.rsn.ie.gtk.key","wlan.rsn.ie.igtk.key","wlan.rsn.ie.pmkid","wlan.sa","wlan.seq",
            "wlan.ssid","wlan.ta","wlan.tag","wlan.tag.length","wlan_radio.channel","wlan_radio.data_rate","wlan_radio.end_tsf",
            "wlan_radio.frequency","wlan_radio.signal_dbm","wlan_radio.start_tsf","wlan_radio.phy","wlan_radio.timestamp",
            "wlan.rsn.capabilities.mfpc","wlan_rsna_eapol.keydes.msgnr","wlan_rsna_eapol.keydes.data","wlan_rsna_eapol.keydes.data_len",
            "wlan_rsna_eapol.keydes.key_info.key_mic","wlan_rsna_eapol.keydes.nonce","eapol.keydes.key_len","eapol.keydes.replay_counter",
            "eapol.len","eapol.type","llc","arp","arp.hw.type","arp.proto.type","arp.hw.size","arp.proto.size","arp.opcode","arp.src.hw_mac",
            "arp.src.proto_ipv4","arp.dst.hw_mac","arp.dst.proto_ipv4","ip.dst","ip.proto","ip.src","ip.ttl","ip.version","data.data","data.len",
            "icmpv6.mldr.nb_mcast_records","icmpv6.ni.nonce","tcp.ack","tcp.ack_raw","tcp.analysis","tcp.analysis.flags",
            "tcp.analysis.retransmission","tcp.analysis.reused_ports","tcp.analysis.rto_frame","tcp.checksum","tcp.checksum.status",
            "tcp.flags.syn","tcp.dstport","tcp.flags.ack","tcp.flags.fin","tcp.flags.push","tcp.flags.reset","tcp.option_len",
            "tcp.payload","tcp.seq","tcp.seq_raw","tcp.srcport","tcp.time_delta","tcp.time_relative","udp.dstport","udp.srcport",
            "udp.length","udp.payload","udp.time_relative","udp.time_delta","nbns","nbss.continuation_data","nbss.type","nbss.length",
            "ldap","smb.access.generic_execute","smb.access.generic_read","smb.access.generic_write","smb.flags.notify",
            "smb.flags.response","smb.flags2.nt_error","smb.flags2.sec_sig","smb.mid","smb.nt_status","smb.server_component",
            "smb.pid.high","smb.tid","smb2.acct","smb2.auth_frame","smb2.buffer_code","smb2.cmd","smb2.data_offset","smb2.domain",
            "smb2.fid","smb2.filename","smb2.header_len","smb2.host","smb2.msg_id","smb2.pid","smb2.previous_sesid","smb2.protocol_id",
            "smb2.sesid","smb2.session_flags","smb2.tid","smb2.write_length","dhcp","dhcp.client_id.duid_ll_hw_type","dhcp.cookie",
            "dhcp.hw.addr_padding","dhcp.hw.mac_addr","dhcp.id","dhcp.ip.client","dhcp.ip.relay","dhcp.ip.server","dhcp.option.broadcast_address",
            "dhcp.option.dhcp_server_id","dhcp.option.router","dhcp.option.vendor.bsdp.message_type","mdns","dns","dns.a","dns.count.add_rr",
            "dns.count.answers","dns.count.auth_rr","dns.count.labels","dns.count.queries","dns.flags.authoritative","dns.flags.checkdisable",
            "dns.flags.opcode","dns.flags.response","dns.id","dns.ptr.domain_name","dns.qry.name","dns.qry.name.len","dns.resp.len",
            "dns.resp.name","dns.resp.ttl","dns.resp.len.1","dns.retransmit_request","dns.retransmit_response","dns.time","ssdp",
            "http.connection","http.content_length","http.content_type","http.date","http.file_data","http.host","http.last_modified",
            "http.location","http.next_request_in","http.next_response_in","http.request.full_uri","http.request.line","http.request.method",
            "http.request.uri.path","http.request.uri.query","http.request.uri.query.parameter","http.request.version","http.request_in",
            "http.response.code","http.response.code.desc","http.response.line","http.response.phrase","http.response.version",
            "http.response_for.uri","http.response_in","http.referer","http.time","http.server","json.value.string","json.key",
            "ssh.cookie","ssh.compression_algorithms_client_to_server_length","ssh.compression_algorithms_server_to_client_length",
            "ssh.direction","ssh.dh_gex.max","ssh.dh_gex.min","ssh.dh_gex.nbits","ssh.encryption_algorithms_client_to_server_length",
            "ssh.encryption_algorithms_server_to_client_length","ssh.host_key.length","ssh.host_key.type_length","ssh.kex_algorithms_length",
            "ssh.mac_algorithms_client_to_server_length","ssh.mac_algorithms_server_to_client_length","ssh.message_code","ssh.mpint_length",
            "ssh.packet_length","ssh.packet_length_encrypted","ssh.padding_length","ssh.padding_string","ssh.protocol",
            "ssh.server_host_key_algorithms_length","tls.alert_message.desc","tls.alert_message.level","tls.app_data_proto",
            "tls.compress_certificate.compressed_certificate_message.length","tls.connection_id","tls.handshake.extension.type",
            "tls.handshake.extensions_key_share_group","tls.handshake.session_ticket_length","tls.handshake.version","tls.record.content_type",
            "tls.record.version","Label"
        ]
    # Get columns for output file
    col_names = []
    for line_num, name in enumerate(dataset_col_name):
        if line_num in desired_cols:
            col_names.append(name.rstrip())
    # Set the column headers to the names from the Wireshark frame
    data.columns = col_names
    data = data.replace('?', np.nan)
    # Add the type_subtype column if using AWID
    if not trainingBool:
        data = data[1:]
        new_values = []
        for idx, type_value in enumerate(data["wlan.fc.type"].values):
            subtype_value = data['wlan.fc.subtype'].values[idx]
            hex_string = f"0x{hex(int(type_value))[2:]}{hex(int(subtype_value))[2:]}"
            new_values.append(hex_string)
        data.insert(1, 'wlan.fc.type_subtype', new_values)
    # Output the minimized dataset to a CSV file (with no index column added)
    data.to_csv(output_dataset, sep=',', index=False)

def get_n_grams_from_custom_dataset(custom_dataset_path):
    data = pd.read_csv(
        # "custom_dataset.csv",
        custom_dataset_path,
        sep=',',
        header=0,
    )
    # frame.time_epoch,wlan.fc.type_subtype,wlan.fc.type,wlan.fc.subtype,wlan.ra,wlan.ta,class
    all_n_gram_flows = CaptureToFlow().create_n_grams_from_dataset_features(data.values)
    classified_n_gram_flows = []
    n_gram_flow_labels = []
    for n_gram in all_n_gram_flows:
        label_array = n_gram[:,6]
        count = np.count_nonzero(label_array == 'normal') + np.count_nonzero(label_array == 'Normal')
        n_gram_array = np.asarray(n_gram[:,:6])
        if(count < 4):
            classified_n_gram_flows.append(n_gram_array)
            n_gram_flow_labels.append(1)
        else:
            classified_n_gram_flows.append(n_gram_array)
            n_gram_flow_labels.append(0)
    return classified_n_gram_flows, n_gram_flow_labels

def train_network_from_classified_flows(classified_n_gram_flows, n_gram_flow_labels):
    model = AdaBoostClassifier(n_estimators=100)
    classified_n_gram_flows = np.array(classified_n_gram_flows).reshape(len(classified_n_gram_flows), 6*4)
    model.fit(classified_n_gram_flows, n_gram_flow_labels)
    # Save the trained model to a file
    if not os.path.exists('smallDsModel5000.pkl'):
        with open('smallDsModel5000.pkl', 'wb') as f:
            pickle.dump(model, f)

def train_network2_from_classified_flows(classified_n_gram_flows, n_gram_flow_labels):
    model = RandomForestClassifier(n_estimators=100)
    classified_n_gram_flows = np.array(classified_n_gram_flows).reshape(len(classified_n_gram_flows), 6*4)
    model.fit(classified_n_gram_flows, n_gram_flow_labels)
    # Save the trained model to a file
    if not os.path.exists('smallDsModel5000-Random.pkl'):
        with open('smallDsModel5000-Random.pkl', 'wb') as f:
            pickle.dump(model, f)

def train_network3_from_classified_flows(classified_n_gram_flows, n_gram_flow_labels):
    model = DecisionTreeClassifier()
    classified_n_gram_flows = np.array(classified_n_gram_flows).reshape(len(classified_n_gram_flows), 6*4)
    model.fit(classified_n_gram_flows, n_gram_flow_labels)
    # Save the trained model to a file
    if not os.path.exists('smallDsModel5000-Tree.pkl'):
        with open('smallDsModel5000-Tree.pkl', 'wb') as f:
            pickle.dump(model, f)

def used_trained_model_to_predit_flow(model_path, n_gram_flows):
    n_gram_flows = np.array(n_gram_flows).reshape(n_gram_flows.shape[0], 6*4)
    model = pickle.load(open(model_path, 'rb'))
    predictions = model.predict(n_gram_flows)
    return(predictions)

def predict_live_capture(model_path):
    features = CaptureToFlow().extract_feature_set_from_live_capture(timeout=10)
    n_gram_flows = CaptureToFlow().create_n_grams_from_observed_features(features)
    return used_trained_model_to_predit_flow(model_path,n_gram_flows)

def demonstrate_model_performance():
    featureFlow = CaptureToFlow().extract_feature_set_from_capture_path('trace.pcap')
    print(f"Extraced Features:\n {featureFlow}\n")
    n_gram_flows = CaptureToFlow().create_n_grams_from_observed_features(featureFlow)
    print(f"N-Gram Flows:\n {n_gram_flows}\n")
    predictionArray = used_trained_model_to_predit_flow('smallDsModel.pkl', n_gram_flows)
    print(f"Predictions:\n {predictionArray}\n")

def generate_test_accuracy(predicted, actual):
    counter = 0
    for idx, value in enumerate(predicted):
        if(value == actual[idx]):
            counter += 1
    return(counter/len(predicted))

def generate_false_positives(predicted, actual):
    fp_counter = 0
    for idx, value in enumerate(predicted):
        if(value == 1 and actual[idx] == 0):
            fp_counter += 1
    return(fp_counter/len(predicted))

def generate_true_positives(predicted, actual):
    tp_counter = 0
    for idx, value in enumerate(predicted):
        if(value == 1 and actual[idx] == 1):
            tp_counter += 1
    return(tp_counter/len(predicted))

def generate_custom_attack_csv_files():
    get_custom_data_from_dataset("datasets\(Re)Assoc_29.csv", None, "datasets\output\AssocAttack.csv", False)
    get_custom_data_from_dataset("datasets\Botnet_18.csv", None, "datasets\output\BotnetAttack.csv", False)
    get_custom_data_from_dataset("datasets\Deauth_22.csv", None, "datasets\output\DeauthAttack.csv", False)
    get_custom_data_from_dataset("datasets\Disas_38.csv", None, "datasets\output\DisasAttack.csv", False)
    get_custom_data_from_dataset("datasets\Evil_Twin_28.csv", None, "datasets\output\EvilTwinAttack.csv", False)
    # get_custom_data_from_dataset("datasets\Kr00k_34.csv", None, "datasets\output\Kr00kAttack.csv", False)
    get_custom_data_from_dataset("datasets\Kr00k_0.csv", None, "datasets\output\Kr00kAttack.csv", False)
    # get_custom_data_from_dataset("datasets\Krack_28.csv", None, "datasets\output\KrackAttack.csv", False)
    get_custom_data_from_dataset("datasets\Krack_27.csv", None, "datasets\output\KrackAttack.csv", False)
    get_custom_data_from_dataset("datasets\Malware_0.csv", None, "datasets\output\MalwareAttack.csv", False)
    get_custom_data_from_dataset("datasets\RogueAP_0.csv", None, "datasets\output\RougeAPAttack.csv", False)
    get_custom_data_from_dataset("datasets\SQL_Injection_0.csv", None, "datasets\output\SQLAttack.csv", False)
    get_custom_data_from_dataset("datasets\SSDP_18.csv", None, "datasets\output\SSDPAttack.csv", False)
    get_custom_data_from_dataset("datasets\SSH_0.csv", None, "datasets\output\SSHAttack.csv", False)
    get_custom_data_from_dataset("datasets\Website_spoofing_0.csv", None, "datasets\output\WebsiteSpoofAttack.csv", False)

def generate_attack_prediction_accuracy(model, attack_dataset):
    x, y = get_n_grams_from_custom_dataset(attack_dataset)
    predicted_values = used_trained_model_to_predit_flow(model, np.asarray(x))
    return generate_test_accuracy(predicted_values, y), generate_false_positives(predicted_values, y), generate_true_positives(predicted_values, y)

def generate_predictions_for_all_attacks(model):
    p1,fp1,tp1 = generate_attack_prediction_accuracy(model, "datasets\output\AssocAttack.csv")
    p2,fp2,tp2 = generate_attack_prediction_accuracy(model, "datasets\output\BotnetAttack.csv")
    p3,fp3,tp3 = generate_attack_prediction_accuracy(model, "datasets\output\DeauthAttack.csv")
    p4,fp4,tp4 = generate_attack_prediction_accuracy(model, "datasets\output\DisasAttack.csv")
    p5,fp5,tp5 = generate_attack_prediction_accuracy(model, "datasets\output\EvilTwinAttack.csv")
    p6,fp6,tp6 = generate_attack_prediction_accuracy(model, "datasets\output\Kr00kAttack.csv")
    p7,fp7,tp7 = generate_attack_prediction_accuracy(model, "datasets\output\KrackAttack.csv")
    p8,fp8,tp8 = generate_attack_prediction_accuracy(model, "datasets\output\MalwareAttack.csv")
    p9,fp9,tp9 = generate_attack_prediction_accuracy(model, "datasets\output\RougeAPAttack.csv")
    p10,fp10,tp10 = generate_attack_prediction_accuracy(model, "datasets\output\SQLAttack.csv")
    p11,fp11,tp11 = generate_attack_prediction_accuracy(model, "datasets\output\SSDPAttack.csv")
    p12,fp12,tp12 = generate_attack_prediction_accuracy(model, "datasets\output\SSHAttack.csv")
    p13,fp13,tp13 = generate_attack_prediction_accuracy(model, "datasets\output\WebsiteSpoofAttack.csv")
    return (
        [p1*100,p2*100,p3*100,p4*100,p5*100,p6*100,p7*100,p8*100,p9*100,p10*100,p11*100,p12*100,p13*100], 
        [fp1*100,fp2*100,fp3*100,fp4*100,fp5*100,fp6*100,fp7*100,fp8*100,fp9*100,fp11*100,fp10*100,fp12*100,fp13*100],
        [tp1*100,tp2*100,tp3*100,tp4*100,tp5*100,tp6*100,tp7*100,tp8*100,tp9*100,tp11*100,tp10*100,tp12*100,tp13*100]
    )

def generate_prediction_graphs(pred_accuracy_array, false_positive_array, true_positive_array):
    attack_names = [
        'Re-Assoc','Botnet','Deauth','DisAssoc','EvilTwin',
        'Kr00k','Krack','Malware','RougeAP','SQL','SSDP',
        'SSH','WebsiteSpoof'
    ]
    common_names = ['Deauth', 'DisAssoc','EvilTwin']
    false_positive_array = false_positive_array[2:5]
    true_positive_array = true_positive_array[2:5]
    # Plots
    fig = plt.figure()
    fig.set_figwidth(13)
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax0 = fig.add_subplot(gs[1, :])
    # Set grids
    ax0.set_axisbelow(True)
    ax0.grid(color='gray', linestyle='dashed')
    ax1.set_axisbelow(True)
    ax1.grid(color='gray', linestyle='dashed')
    ax2.set_axisbelow(True)
    ax2.grid(color='gray', linestyle='dashed')
    # Set titles and values
    ax0.set_title('Test accuracy % per attack in AWID3 attack database')
    ax0.bar(attack_names, pred_accuracy_array, color=[(0, (191-(10*x))/255, 1) for x in range(14)], edgecolor='black')
    ax1.set_title('False Positive % per common attack with paper')
    ax1.bar(common_names, false_positive_array, color=[((255-(20*x))/255, 3/255, 27/255) for x in range(3)], edgecolor='black')
    ax2.set_title('True Positive % per common attack with paper')
    ax2.bar(common_names, true_positive_array, color=[(0, (191-(20*x))/255, 0) for x in range(3)], edgecolor='black')
    # Axis values
    ax0.set_ylim([0, 100])
    ax1.set_ylim([0, 100])
    ax2.set_ylim([0, 100])
    ax0.set_yticks(np.arange(0,101,10))
    ax1.set_yticks(np.arange(0,101,10))
    ax2.set_yticks(np.arange(0,101,10))
    plt.tight_layout()
    # plt.show()
    plt.savefig('ResultsGraphs2.png') # change depending on model

def generate_test_train_accuracy(model):
    x, y = get_n_grams_from_custom_dataset('custom_dataset.csv')
    test_x = x[5000:9821]
    test_y = y[5000:9821]
    predicted_values = used_trained_model_to_predit_flow(model, np.asarray(test_x))
    return(generate_test_accuracy(predicted_values, test_y)*100)

if __name__ == "__main__":
    # Get ngrams from the total dataset
    # x, y = get_n_grams_from_custom_dataset('custom_dataset.csv')
    # train_x = x[0:5000]
    # train_y = y[0:5000]

    # Use the functions below to generate and save the trained models
    # train_network_from_classified_flows(train_x, train_y) 
    # train_network2_from_classified_flows(train_x, train_y)
    # train_network3_from_classified_flows(train_x, train_y)

    # Change depending on the model
    model_name = 'smallDsModel5000-Random.pkl'

    # General accuracy on data
    print(f"General test set accuracy: {generate_test_train_accuracy(model_name)}")

    # Accuracy against specific attacks
    # generate_custom_attack_csv_files()
    attack_accuracy_array, false_positives_array, true_positives_array = generate_predictions_for_all_attacks(model_name)
    print(f"Accuracy against specific attacks: {attack_accuracy_array}")
    print(f"False positives for specific attacks: {false_positives_array}")
    print(f"True positives for specific attacks: {true_positives_array}")

    # Visually display the accuracy against specific attacks
    generate_prediction_graphs(attack_accuracy_array, false_positives_array, true_positives_array)
    
    # Accuracy against live capture
    # demonstrate_model_performance()
    
