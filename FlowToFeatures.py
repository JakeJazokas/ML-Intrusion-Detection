from cProfile import label
from sklearn.naive_bayes import BernoulliNB
from CaptureToFlow import CaptureToFlow
# Features used to build the model

# 1) Flow Probability - Jelinek-Mercer smoothing model - (Lambda = 0.5)
def calculate_flow_probability(n_gram, n_gram_flow):
    # TODO Determine P(n_gram|n_gram_flow)
    # Maybe a useful link: https://github.com/scikit-learn/scikit-learn/issues/12862
    return 1

# 2) Total Frames in Flow (Each n-gram flow is made up of 4 frames)
def calculate_total_frames_in_flow(n_gram_flow):
    return(len(n_gram_flow)*4)

# 3) New n-grams in flow (Difference from last input)
def calculate_new_n_grams_in_flow(n_gram_flow_prev, n_gram_flow_new):
    return(len(n_gram_flow_new)-len(n_gram_flow_prev))

# 4) Ratio of number of management frames to total frames (Auth and Deauth)
def calculate_management_frames_in_flow(n_gram_flow):
    management_frame_counter = 0
    for n_gram in n_gram_flow:
        for frame in n_gram:
            if(frame[5] == '11' or [5] == '12'):
                management_frame_counter += 1
    return(management_frame_counter/calculate_total_frames_in_flow(n_gram_flow))

# 5) Ratio of number of control frames to total frames TODO Frames other than Asso Reqs
def calculate_control_frames_in_flow(n_gram_flow):
    control_frame_counter = 0
    for n_gram in n_gram_flow:
        for frame in n_gram:
            if(frame[5] == '0'):
                control_frame_counter += 1
    return(control_frame_counter/calculate_total_frames_in_flow(n_gram_flow))

# 6) Ratio of number of data frames to total frames
def calculate_data_frames_in_flow(n_gram_flow):
    data_frame_counter = 0
    for n_gram in n_gram_flow:
        for frame in n_gram:
            if(frame[3] == '2'):
                data_frame_counter += 1
    return(data_frame_counter/calculate_total_frames_in_flow(n_gram_flow))

if __name__ == "__main__":
    featureFlow = CaptureToFlow().extract_feature_set_from_capture_path('Wireshark_802_11.pcap')
    hahsedFeatureFlow = CaptureToFlow().hash_observation_features(featureFlow)

    # f = extract_feature_set_from_capture('Wireshark_802_11.pcap')
    # n = create_n_grams_from_observed_features(f)
    # print(n)
    # print(hahsedFeatureFlow)
    # print(calculate_flow_probability(featureFlow[0], featureFlow))
