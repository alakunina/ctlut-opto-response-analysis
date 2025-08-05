import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
import glob
from tqdm import tqdm
from xml.dom import minidom

from aind_ephys_utils import align, sort
from open_ephys.analysis import Session
import plotting_funcs as pf

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

class OptotaggingAnalysis:
    '''
    Calculates laser response metrics for one recording session. 

    Parameters:
    recording_clipped_folder: The raw ephys recording folder
    recording_sorted_folder: The spike-sorted ephys folder, derived from raw ephys folder if none.
    opto_recording: Optional. Index of recording with laser presentations, default 0 

    Output:
    csv files with laser response metrics
    raster plots of putative tagged units
    '''

    def __init__(self, recording_clipped_folder, recording_sorted_folder=None, trials_csv=None, opto_recording=0, laser_event_id='2'):
        self.opto_recording = opto_recording
        self.recording_clipped_folder = recording_clipped_folder
        self.session = re.search('\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', recording_clipped_folder).group(0)
        if recording_sorted_folder is None:
            self.recording_sorted_folder = glob.glob(f'{recording_clipped_folder[:recording_clipped_folder.find(self.session)+len(self.session)]}_sorted*')[0]
        else:
            self.recording_sorted_folder = recording_sorted_folder
        if trials_csv is None:
            trials_csv = glob.glob(f"{recording_clipped_folder}/*opto.csv")[0]
        self.trial_ids = pd.read_csv(trials_csv, index_col=0)
        self._get_laser_onset_times(laser_event_id)

    

    def _get_laser_onset_times(self, event_id):
        event = se.read_openephys_event(self.recording_clipped_folder, block_index=0)
        events = event.get_events(channel_id="PXIe-6341Digital Input Line", segment_index=self.opto_recording)
        laser_pulses = events[events["label"] == event_id]
        self.laser_onset_times = laser_pulses['time']
    
    def get_stream_names(self):
        settings_file = glob.glob(f"{self.recording_clipped_folder}/Record Node ???/settings.xml")[0]
        settings = minidom.parse(settings_file)
        processors = settings.getElementsByTagName('PROCESSOR')
        np_pxi = [processor for processor in processors if processor.getAttribute('name')=='Neuropix-PXI'][0]
        streams = np_pxi.getElementsByTagName('STREAM')
        ap_streams = [stream.getAttribute('name') for stream in streams if stream.getAttribute('sample_rate')=='30000.0']
        return ap_streams

    def one_probe_laser_responses(self, probe, timestamps, sorting_outputs, waveform_extractors, extremum_channels):
        print(f'Now processing session {self.session}, {probe}')
        all_laser_response_metrics = []
        param_group = 'train'
        trial_types = np.unique(self.trial_ids.type)
        #timestamps, sorting_outputs, waveform_extractors, extremum_channels = self._get_sorting_output(probe)

        if probe[-3:] == '-AP':
            this_probe = probe[:-3]
        else:
            this_probe = probe

        # for the different ways probe might be named in the trials table...
        if 'emission_location' in self.trial_ids.columns:
            all_emission_locations = np.unique(self.trial_ids.emission_location)
            if this_probe not in all_emission_locations:
                this_probe = this_probe[:5]+' '+this_probe[-1]
                if this_probe not in all_emission_locations:
                    for location in all_emission_locations:
                        if this_probe in location:
                            this_probe = location

        for ind_sorting, sorting_output in enumerate(sorting_outputs):
            print(f'Channel group {ind_sorting}')
            laser_response_metrics = pd.DataFrame({'unit_id':sorting_output.unit_ids.flatten()})
            for ind_unit, unit in enumerate(tqdm(sorting_output.unit_ids)):
                sample_numbers = sorting_output.get_unit_spike_train(unit, segment_index=self.opto_recording)
                unit_spike_times = timestamps[sample_numbers]

                # save best channel
                peak_channel = int(extremum_channels[ind_sorting][unit][2:])
                laser_response_metrics.at[ind_unit, 'peak_channel'] = peak_channel

                for trial_type in trial_types:
                    # do stats for all laser powers
                    this_powers = np.unique(self.trial_ids.query('type == @trial_type').power)
                    # collect averaged stats for all, only save the best one and indicate best power
                    all_num_sig_pulses = np.zeros(len(this_powers))
                    all_mean_latencies = np.zeros(len(this_powers))
                    all_mean_jitters = np.zeros(len(this_powers))
                    all_mean_reliability = np.zeros(len(this_powers))
                    all_best_sites = np.zeros(len(this_powers))
                    all_channel_diffs = np.zeros(len(this_powers))

                    for ind_power, power in enumerate(this_powers):
                        this_sites = list(np.unique(self.trial_ids.query('type == @trial_type and power == @power').site))
                        if 'emission_location' in self.trial_ids.columns:
                            tag_trials = self.trial_ids.query('param_group == @param_group and site == @this_sites and power == @power and type == @trial_type and emission_location == @this_probe')
                        else:
                            tag_trials = self.trial_ids.query('num_pulses == 5 and site == @this_sites and power == @power and type == @trial_type')
                        all_tag_trials_timestamps = self.laser_onset_times[tag_trials.index.tolist()]

                        duration = np.unique(tag_trials.duration)[0]
                        num_pulses = np.unique(tag_trials.num_pulses)[0]
                        pulse_interval = np.unique(tag_trials.pulse_interval)[0]
                        total_duration = (duration*num_pulses)+(pulse_interval*num_pulses)

                        # --- find best stim site by finding maximum positive response during laser train ---
                        laser_total_time_range = [0, (total_duration+pulse_interval)/1000]
                        min_interval = np.min(tag_trials['interval'])
                        baseline_time_range = [-min_interval, 0]
                        unneeded_bins, baseline_spike_counts, unneeded_ids = align.to_events(unit_spike_times, all_tag_trials_timestamps, baseline_time_range, bin_size=np.diff(baseline_time_range)[0])
                        baseline_spike_rate = np.mean(baseline_spike_counts)/(baseline_time_range[-1]-baseline_time_range[0])

                        # for latency calculations later, using longer baseline to get better estimate of mean + stdev
                        baseline_rate_stdev = np.std(baseline_spike_counts/(baseline_time_range[-1]-baseline_time_range[0]))
                        threshold_spike_rate = baseline_spike_rate + 2*baseline_rate_stdev
                        # time_range_raster = [-(duration * 2) / 1000, (duration + pulse_interval) / 1000]

                        # smoothing params for psth for estimating max response
                        win = np.concatenate((np.zeros(3), np.ones(3)))  # square (causal)
                        win = win/np.sum(win)
                        bin_size = 1 # in ms
                        bin_edges_full = np.arange(laser_total_time_range[0],laser_total_time_range[-1], bin_size/1000)
                        
                        all_responses = np.zeros(len(this_sites))
                        # all_responses_o = np.zeros(len(this_sites))

                        for ind_site, site in enumerate(this_sites):
                            this_tag_trials = tag_trials.query('site == @site')
                            this_tag_laser_event_timestamps = self.laser_onset_times[this_tag_trials.index.tolist()]

                            # create smoothed PSTH of response to entire train
                            unneeded_bins, laser_spike_counts, unneeded_ids = align.to_events(unit_spike_times, this_tag_laser_event_timestamps, laser_total_time_range, bin_size=bin_size/1000)

                            average_response = np.mean(laser_spike_counts,axis=1)/(bin_size/1000)
                            smooth_PSTH = np.convolve(average_response, win, mode='same')
                            normalised_PSTH = smooth_PSTH - baseline_spike_rate
                            # only looking for greatest increase in firing rate to indicate tagging
                            max_response = np.max(normalised_PSTH)

                            all_responses[ind_site] = max_response

                        best_arg = np.argmax(all_responses)
                        best_site = this_sites[best_arg]
                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_best_site'] = int(best_site)

                        ### calculate latency, jitter, etc. from all pulses
                        pulse_time_range = [0, duration/1000]
                        pulse_offset = (duration + pulse_interval)/1000
                        pulse_latency_time_range = [0, pulse_offset]

                        if 'emission_location' in self.trial_ids.columns:
                            best_site_tag_trials = self.trial_ids.query('param_group == @param_group and site == @best_site and power == @power and type == @trial_type and emission_location == @this_probe')
                        else:
                            best_site_tag_trials = self.trial_ids.query('num_pulses == 5 and site == @best_site and power == @power and type == @trial_type')
                        best_site_event_timestamps = self.laser_onset_times[best_site_tag_trials.index.tolist()]

                        all_pvals_unpaired = np.ones(num_pulses)
                        all_pvals_paired = np.ones(num_pulses)
                        all_jitter = np.zeros(num_pulses)
                        all_latencies = np.zeros(num_pulses)
                        all_reliability = np.zeros(num_pulses)

                        # baseline counts over same time range as laser stim
                        short_baseline_time_range = [-duration/1000, 0]
                        unneeded_bins, short_baseline_spike_counts, unneeded_ids = align.to_events(unit_spike_times, best_site_event_timestamps, short_baseline_time_range, bin_size=np.diff(short_baseline_time_range)[0])


                        # for ANOVA later
                        all_pulse_spike_counts = [short_baseline_spike_counts.flatten()]

                        for ind_pulse in range(num_pulses):
                            this_laser_time_range = [pulse_time_range[0]+ind_pulse*pulse_offset, pulse_time_range[1]+ind_pulse*pulse_offset]
                            unneeded_bins, this_pulse_spike_counts, unneeded_ids = align.to_events(unit_spike_times, best_site_event_timestamps, this_laser_time_range, bin_size=np.diff(this_laser_time_range)[0])


                            all_pulse_spike_counts.append(this_pulse_spike_counts.flatten())

                            # paired test
                            try:
                                statistic, pVal = stats.wilcoxon(this_pulse_spike_counts.flatten()/np.diff(this_laser_time_range),
                                                        short_baseline_spike_counts.flatten()/np.diff(short_baseline_time_range),
                                                        alternative='greater')
                            except(ValueError):  # wilcoxon test doesn't like it when there's no difference between passed values
                                statistic = 0
                                pVal = 1
                            all_pvals_paired[ind_pulse] = pVal
                            #laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_pulse_{ind_pulse+1}_pVal_unpaired'] = pVal

                            # unpaired test
                            statistic, pVal = stats.ranksums(this_pulse_spike_counts.flatten()/np.diff(this_laser_time_range),
                                                        short_baseline_spike_counts.flatten()/np.diff(short_baseline_time_range),
                                                        alternative='greater')
                            all_pvals_unpaired[ind_pulse] = pVal

                            # reliability
                            all_reliability[ind_pulse] = np.count_nonzero(this_pulse_spike_counts)/len(this_pulse_spike_counts)

                            # latency
                            this_latency_time_range = [pulse_latency_time_range[0]+ind_pulse*pulse_offset, pulse_latency_time_range[1]+ind_pulse*pulse_offset]
                            this_bin_edges = np.arange(this_latency_time_range[0],this_latency_time_range[-1], bin_size/1000)
                            
                            # this_pulse_latency_locked_timestamps = af.event_locked_timestamps(unit_spike_times, best_site_event_timestamps, this_latency_time_range)
                            this_pulse_latency_locked_timestamps, latency_event_ids, unneeded_ids = align.to_events(unit_spike_times, best_site_event_timestamps, this_latency_time_range)
                            # this_pulse_latency_spike_counts = af.timestamps_to_spike_counts(this_pulse_latency_locked_timestamps, this_bin_edges)
                            unneeded_bins, this_pulse_latency_spike_counts, unneeded_ids = align.to_events(unit_spike_times, best_site_event_timestamps, this_latency_time_range, bin_size=(bin_size/1000))
                            average_response = np.mean(this_pulse_latency_spike_counts,axis=1)/(bin_size/1000)
                            smooth_PSTH = np.convolve(average_response,win, mode='same')
                            responsive_inds = np.flatnonzero(smooth_PSTH>threshold_spike_rate)
                            if len(responsive_inds)>0:
                                first_responsive_ind = responsive_inds[0]
                                y_diff = (smooth_PSTH[first_responsive_ind]-smooth_PSTH[first_responsive_ind-1])
                                y_fraction = (threshold_spike_rate-smooth_PSTH[first_responsive_ind-1])/ y_diff
                                response_latency = this_bin_edges[first_responsive_ind] + y_fraction*(bin_size/1000) - this_latency_time_range[0]
                            else:
                                response_latency = None
                            all_latencies[ind_pulse] = response_latency

                            # jitter
                            if response_latency is not None and len(this_pulse_latency_locked_timestamps) > 0:
                                pulse_timestamps_filtered = []
                                for ind in np.unique(latency_event_ids):
                                    this_pulse_first_spike_ind = np.where(latency_event_ids == ind)[0][0]
                                    pulse_timestamps_filtered.append(this_pulse_latency_locked_timestamps[this_pulse_first_spike_ind])
                                # pulse_timestamps_filtered = [timestamps[0] for timestamps in this_pulse_latency_locked_timestamps if len(timestamps)>0]
                                if len(pulse_timestamps_filtered) > 1:
                                    jitter = np.std(pulse_timestamps_filtered)
                                else:
                                    jitter = None
                            else:
                                jitter = None
                            all_jitter[ind_pulse] = jitter

                        # corrected pvals
                        (responsive_sites_unpaired, corrected_pVals_unpaired, alphaSidak, alphaBonf) = multipletests(all_pvals_unpaired, method='holm')
                        (responsive_sites_paired, corrected_pVals_paired, alphaSidak, alphaBonf) = multipletests(all_pvals_paired, method='holm')

                        # calculate p-value from kruskal-wallis test (unpaired) and friedman test (paired) over all laser pulses
                        try:
                            statistic, pVal_unpaired = stats.kruskal(*all_pulse_spike_counts)
                        except(ValueError):  # kruskal test ALSO doesn't like it when there's no difference between passed values
                            statistic = 0
                            pVal_unpaired = 1
                        statistic, pVal_paired = stats.friedmanchisquare(*all_pulse_spike_counts)

                        # save all the things
                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_overall_pVal_unpaired'] = pVal_unpaired
                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_overall_pVal_paired'] = pVal_paired

                        # laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_all_pVals_unpaired'] = corrected_pVals_unpaired.tolist()
                        # laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_all_pVals_paired'] = corrected_pVals_paired.tolist()
                        # laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_all_latencies'] = all_latencies.tolist()
                        # laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_all_jitters'] = all_jitter.tolist()
                        # laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_all_reliability'] = all_reliability.tolist()

                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_mean_latency'] = np.mean(all_latencies)
                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_mean_jitter'] = np.mean(all_jitter)
                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_mean_reliability'] = np.mean(all_reliability)

                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_num_sig_pulses_unpaired'] = np.sum(responsive_sites_unpaired)
                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_num_sig_pulses_paired'] = np.sum(responsive_sites_paired)

                        #### compare peak channel to peak emission site
                        peak_channel_diff = abs(peak_channel - 10*best_site)
                        laser_response_metrics.at[ind_unit, f'{trial_type}_{power}mW_{param_group}_channel_diff'] = int(peak_channel_diff)

                        # keep track of stats I usually use for unit selection so I can save the best ones in their own column
                        all_num_sig_pulses[ind_power] = np.sum(responsive_sites_paired)
                        all_mean_latencies[ind_power] = np.mean(all_latencies)
                        all_mean_jitters[ind_power] = np.mean(all_jitter)
                        all_mean_reliability[ind_power] = np.mean(all_reliability)
                        all_best_sites[ind_power] = best_site
                        all_channel_diffs[ind_power] = peak_channel_diff

                    # select best power as highest power with max number of responsive pulses
                    best_power_ind = np.asarray(all_num_sig_pulses==np.max(all_num_sig_pulses)).nonzero()[0][-1]

                    laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_best_power'] = this_powers[best_power_ind]
                    laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_best_site'] = all_best_sites[best_power_ind]
                    laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_max_num_sig_pulses'] = all_num_sig_pulses[best_power_ind]
                    laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_best_mean_latency'] = all_mean_latencies[best_power_ind]
                    laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_best_mean_jitter'] = all_mean_jitters[best_power_ind]
                    laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_best_mean_reliability'] = all_mean_reliability[best_power_ind]
                    laser_response_metrics.at[ind_unit, f'{trial_type}_{param_group}_best_channel_diff'] = all_channel_diffs[best_power_ind]

            all_laser_response_metrics.append(laser_response_metrics)
        return all_laser_response_metrics
            
    def get_responsive_neurons(self, all_laser_response_metrics, sorting_outputs, this_probe, unit_criteria=None):
        if unit_criteria is None:
            unit_criteria = self.default_laser_response_criteria(all_laser_response_metrics[0])
        responsive_units = {}
        for shank, laser_response_metrics in enumerate(all_laser_response_metrics):
            print(f"{self.session}, {this_probe}, shank {shank}:")

            default_qc = sorting_outputs[shank].get_property('default_qc')
            decoder_label = sorting_outputs[shank].get_property('decoder_label')

            for trial_type in unit_criteria.keys():
                these_responsive_units = laser_response_metrics.query(unit_criteria[trial_type])
                good_units = (default_qc[these_responsive_units.index.tolist()] == True) & ~(decoder_label[these_responsive_units.index.tolist()] == 'noise')
                good_unit_ids = np.array(these_responsive_units.unit_id.tolist())[good_units]
                print(f"{sum(good_units)} tagged {trial_type} units: {good_unit_ids}")
                if len(good_unit_ids) > 0:
                    responsive_units[shank] = {trial_type: good_unit_ids}

        return responsive_units

    def default_laser_response_criteria(self, laser_response_metrics):
        queries = {}
        if 'internal_red_train_max_num_sig_pulses' in laser_response_metrics.columns:
            # at least 4 responsive pulses for red units as chrmine sometimes fails to respond to first pulse
            queries['internal_red'] = "internal_red_train_max_num_sig_pulses > 3 and internal_red_train_best_channel_diff < 25"

        if 'external_red_train_max_num_sig_pulses' in laser_response_metrics.columns:
            # at least 4 responsive pulses for red units as chrmine sometimes fails to respond to first pulse
            queries['external_red'] = "external_red_train_max_num_sig_pulses > 3"

        if 'internal_blue_train_max_num_sig_pulses' in laser_response_metrics.columns:
            if 'internal_red_train_max_num_sig_pulses' in laser_response_metrics.columns:
                # less than 2 responsive red pulses for blue tagged units again because of chrmine...
                queries['internal_blue'] = "internal_red_train_max_num_sig_pulses < 2 and internal_blue_train_max_num_sig_pulses == 5 and internal_blue_train_best_channel_diff < 25"
            else:
                queries['internal_blue'] = "internal_blue_train_max_num_sig_pulses == 5 and internal_blue_train_best_channel_diff < 25"
    
        if 'external_blue_train_max_num_sig_pulses' in laser_response_metrics.columns:
            if 'external_red_train_max_num_sig_pulses' in laser_response_metrics.columns:
                # less than 2 responsive red pulses for blue tagged units again because of chrmine...
                queries['external_blue'] = "external_red_train_max_num_sig_pulses < 2 and external_blue_train_max_num_sig_pulses == 5"
            else:
                queries['external_blue'] = "external_blue_train_max_num_sig_pulses == 5"

        return queries



    def get_sorting_output(self, probe):
        sorting_folder = glob.glob(f"{self.recording_sorted_folder}/*curated/experiment1_Record Node ???#Neuropix-PXI-???.{probe}_recording{self.opto_recording+1}")
        waveform_folder = glob.glob(f"{self.recording_sorted_folder}/postprocessed/experiment1_Record Node ???#Neuropix-PXI-???.{probe}_recording{self.opto_recording+1}*")
        if len(sorting_folder) == 0:
            sorting_folder = glob.glob(f"{self.recording_sorted_folder}/*curated/experiment1_Record Node ???#Neuropix-PXI-???.{probe}_recording{self.opto_recording+1}_group*")
            waveform_folder = glob.glob(f"{self.recording_sorted_folder}/postprocessed/experiment1_Record Node ???#Neuropix-PXI-???.{probe}_recording{self.opto_recording+1}_group*")
        stream_folder = glob.glob(f"{self.recording_clipped_folder}/Record Node ???/experiment1/recording{self.opto_recording+1}/continuous/Neuropix-PXI-???.{probe}")[0]
        if Path(f'{stream_folder}/sample_numbers.npy').is_file():
            timestamps = np.load(f'{stream_folder}/timestamps.npy')
        else:
            timestamps = np.load(f'{stream_folder}/synchronized_timestamps.npy')

        sorting_folder.sort()
        waveform_folder.sort()

        sorting_outputs = []
        waveform_extractors = []
        extremum_channels = []

        for ind_sorting, sorting in enumerate(sorting_folder):
            this_sorting_output = si.load_extractor(sorting)
            this_waveform_extractor = si.load_sorting_analyzer_or_waveforms(waveform_folder[ind_sorting], sorting=this_sorting_output)

            sorting_outputs.append(this_sorting_output)
            waveform_extractors.append(this_waveform_extractor)
            extremum_channels.append(si.get_template_extremum_channel(this_waveform_extractor))

        return timestamps, sorting_outputs, waveform_extractors, extremum_channels
