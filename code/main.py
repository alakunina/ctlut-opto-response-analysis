'''
Run optotagging analysis on assets attached to capsule
'''

import glob
import numpy as np
import re
import optotagging_analysis as oa
import plotting_funcs as pf

# global variables
FLIP_NIDAQ = True # if there was an issue with alignment and sync signal was flipped
IGNORE_ONSET = True # ignore onset and offset of laser stim during calculations, good when you have lots of artifacts

# find all the raw ephys folders attached to the capsule
recording_clipped_folder = glob.glob(f"/data/ecephys_*/**/ecephys_clipped/", recursive=True)

for this_recording in recording_clipped_folder:
    this_session = re.search('\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', this_recording).group(0)
    recording_sorted_folder = glob.glob(f'{this_recording[:this_recording.find(this_session)+len(this_session)]}_sorted*')
    if len(recording_sorted_folder) == 0:
        print(f"No sorted asset for session {this_session}!")
        continue
    else:
        recording_sorted_folder = recording_sorted_folder[0]
    
    # create analyzer object
    print("Constructing the analyzer object!")
    analyzer = oa.OptotaggingAnalysis(this_recording, recording_sorted_folder, flip_NIDAQ=FLIP_NIDAQ)

    # dictionary of parameters to query in the trials table
    trials_query = {'type' : ['external_red', 'external_blue'],
                'param_group' : ['train'],
                'power' : np.unique(analyzer.trial_ids.power)}
    # for naming columns
    suffixes = [None, None, 'mW']

    ap_streams = analyzer.get_stream_names()
    for this_stream in ap_streams:
        print(f"Loading data for session {this_session}, {this_stream}.")
        timestamps, sorting_data = analyzer.get_sorting_output(this_stream)

        # DUMB FIX FOR ANNA'S DATA
        # Anna was not consistent in naming the emission location in their trial tables...
        probe_name = this_stream
        if probe_name[-3:] == '-AP':
            probe_name = probe_name[:-3]
        if 'emission_location' in analyzer.trial_ids.columns:
            all_emission_locations = np.unique(analyzer.trial_ids.emission_location)
            if probe_name not in all_emission_locations:
                probe_name = probe_name[:5]+' '+probe_name[-1]
                if probe_name not in all_emission_locations:
                    for location in all_emission_locations:
                        if probe_name in location:
                            probe_name = location

        # pre-stim interval to use to calculate pre-stim ISI
        stimulus_start = analyzer.laser_onset_times[0] - 1
        session_start = timestamps[0]
        pre_stim_duration = stimulus_start - session_start

        laser_response_metrics = analyzer.one_probe_laser_responses(timestamps, sorting_data, trials_query, probe_name, suffixes=suffixes, ignore_onset_offset=IGNORE_ONSET, pre_opto_duration=pre_stim_duration)

        # calculate the best power out of all the powers presented during session, save metrics
        for group_name in list(laser_response_metrics.keys()):
            this_laser_response_metrics = laser_response_metrics[group_name]
            trial_types = ['external_red_train', 'external_blue_train']

            col_list = list(this_laser_response_metrics.columns)
            for trial_type in trial_types:
                this_col_list = [item for item in col_list if trial_type in item and 'num_sig_pulses' in item]
                for ind_unit in this_laser_response_metrics.index.tolist():
                    best_response = 0
                    best_col = None
                    for col in this_col_list:
                        this_val = this_laser_response_metrics.at[ind_unit, col]
                        if this_val > best_response:
                            best_response = this_val
                            best_col = col
                    if best_col is not None:
                        parts = best_col.split('_')
                        best_power = [part for part in parts if 'mW' in part][0]
                        this_laser_response_metrics.at[ind_unit, f'{trial_type}_best_power'] = float(best_power[:-2])
                        this_laser_response_metrics.at[ind_unit, f'{trial_type}_max_num_sig_pulses'] = this_laser_response_metrics.at[ind_unit, f'{trial_type}_{best_power}_num_sig_pulses']
                        this_laser_response_metrics.at[ind_unit, f'{trial_type}_best_mean_latency'] = this_laser_response_metrics.at[ind_unit, f'{trial_type}_{best_power}_mean_latency']
                        this_laser_response_metrics.at[ind_unit, f'{trial_type}_best_latency_range'] = this_laser_response_metrics.at[ind_unit, f'{trial_type}_{best_power}_latency_range']
                        this_laser_response_metrics.at[ind_unit, f'{trial_type}_best_mean_time_to_first_spike'] = this_laser_response_metrics.at[ind_unit, f'{trial_type}_{best_power}_mean_time_to_first_spike']
                        this_laser_response_metrics.at[ind_unit, f'{trial_type}_best_mean_jitter'] = this_laser_response_metrics.at[ind_unit, f'{trial_type}_{best_power}_mean_jitter']
                        this_laser_response_metrics.at[ind_unit, f'{trial_type}_best_mean_reliability'] = this_laser_response_metrics.at[ind_unit, f'{trial_type}_{best_power}_mean_reliability']

            this_laser_response_metrics.to_csv(f'/results/{this_session}_{this_stream}_{group_name}_laser_response_metrics.csv')

        for group_name in list(laser_response_metrics.keys()):
            decoder_label = sorting_data[group_name]['sorting_output'].get_property('decoder_label')

            blue_responsive = laser_response_metrics[group_name].query('external_blue_train_max_num_sig_pulses == 5 and external_blue_train_best_mean_jitter < 0.005 and pre_stim_isi_ratio < 0.5')
            red_responsive = laser_response_metrics[group_name].query('external_red_train_max_num_sig_pulses >= 4 and external_red_train_best_mean_jitter < 0.005 and pre_stim_isi_ratio < 0.5')

            mask = ~blue_responsive.index.isin(red_responsive.index)
            blue_responsive = blue_responsive[mask]

            print(f'Channel group {group_name}:')

            good_ext_red_units = ~(decoder_label[red_responsive.index.tolist()] == 'noise')
            print(f"{sum(good_ext_red_units)} externally tagged red units: {np.array(red_responsive.unit_id.tolist())[good_ext_red_units]}")

            good_ext_blue_units = ~(decoder_label[blue_responsive.index.tolist()] == 'noise')
            print(f"{sum(good_ext_blue_units)} externally tagged blue units: {np.array(blue_responsive.unit_id.tolist())[good_ext_blue_units]}")

            if sum(good_ext_red_units) > 0:
                fig_title = f"{this_session}_{this_stream}_{group_name}_red_responsive"
                trial_types = np.unique(analyzer.trial_ids['type'])
                pf.multi_unit_raster_plot(np.array(red_responsive.unit_id.tolist())[good_ext_red_units], sorting_data[group_name]['sorting_output'], timestamps, sorting_data[group_name]['waveform_extractor'], analyzer.trial_ids, analyzer.laser_onset_times, trial_types, probe_name, fig_title)
                pf.multi_unit_pulse_plot(np.array(red_responsive.unit_id.tolist())[good_ext_red_units], sorting_data[group_name]['sorting_output'], timestamps, laser_response_metrics[group_name], analyzer.laser_onset_times, analyzer.trial_ids, trial_types, probe_name, fig_title+"_pulse_plot")
            
            if sum(good_ext_blue_units) > 0:
                fig_title = f"{this_session}_{this_stream}_{group_name}_blue_responsive"
                trial_types = np.unique(analyzer.trial_ids['type'])
                pf.multi_unit_raster_plot(np.array(blue_responsive.unit_id.tolist())[good_ext_blue_units], sorting_data[group_name]['sorting_output'], timestamps, sorting_data[group_name]['waveform_extractor'], analyzer.trial_ids, analyzer.laser_onset_times, trial_types, probe_name, fig_title)
                pf.multi_unit_pulse_plot(np.array(blue_responsive.unit_id.tolist())[good_ext_blue_units], sorting_data[group_name]['sorting_output'], timestamps, laser_response_metrics[group_name], analyzer.laser_onset_times, analyzer.trial_ids, trial_types, probe_name, fig_title+"_pulse_plot")
