import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors

import analysis_funcs as af

def shiftedColorMap(cmap, min_val, max_val, name):
    '''Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero. Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Input
    -----
      cmap : The matplotlib colormap to be altered.
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.'''
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    try:
        midpoint = 1.0 - max_val/(max_val + abs(min_val))
    except(ZeroDivisionError):
        midpoint = 1.0
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    matplotlib.colormaps.register(cmap=newcmap, force=True)
    return newcmap


def raster_plot(event_locked_spike_times, time_range, cond_each_trial=None, raster=None, color='k', cond_colors = None, trial_start=0, ms=3, **kwargs):
    '''
    :param event_locked_spike_times: spike timestamps each trial relative to an event
    :param cond_each_trial: (OPTIONAL) some sort of label for each trial so that trials with the same parameters can be grouped together.
    :return: a cool raster plot
    '''
    if raster is None:
        raster = []

    if cond_each_trial is not None:
        conds = np.unique(cond_each_trial)

        if type(color) == str:
            color = np.tile(color, len(conds))
        if cond_colors is None:
            cond_colors = np.tile(['0.5', '0.75'], int(np.ceil(len(conds)/2)))

        total_trials = 0
        cond_lines = []
        cond_bars = []

        for indcond, cond in enumerate(conds):
            this_event_locked_spike_times = np.array(event_locked_spike_times, dtype=object)[cond_each_trial == cond]
            raster, none_cond_lines, none_cond_bars = raster_plot(this_event_locked_spike_times, time_range, raster=raster, color=color[indcond], trial_start=total_trials, ms=ms, **kwargs)
            total_trials += len(this_event_locked_spike_times)

            cond_line = plt.axhline(total_trials, color='0.7', zorder=-100)
            cond_lines.append(cond_line)

            xpos = [time_range[0]-0.03*(time_range[1]-time_range[0]),time_range[0]]
            ybot = [total_trials-len(this_event_locked_spike_times), total_trials-len(this_event_locked_spike_times)]
            ytop = [total_trials, total_trials]
            cond_bar = plt.fill_between(xpos, ybot, ytop,ec='none',fc=cond_colors[indcond], clip_on=False)
            cond_bars.append(cond_bar)


        trials_per_cond = total_trials/len(conds)
        plt.yticks(np.arange(trials_per_cond/2, total_trials, trials_per_cond), [f'{cond}' for cond in conds])
        plt.gca().tick_params('y', length=0, pad=8)

    else:
        Ntrials = len(event_locked_spike_times)
        for trial in range(Ntrials):
            this_raster = plt.plot(event_locked_spike_times[trial],
                                   (trial + 1 + trial_start) * np.ones(len(event_locked_spike_times[trial])),
                                   '.', color=color, rasterized=False, ms=ms, **kwargs)
            raster.append(this_raster)

        cond_lines = None
        cond_bars = None
        plt.ylim(0,Ntrials+2+trial_start)
        #zline = plt.axvline(0, color='0.8', zorder=-100)

    plt.xlim(time_range)
    return raster, cond_lines, cond_bars

def multi_unit_raster_plot(unit_ids, sorting_output, timestamps, waveform_extractor, event_ids, laser_onset_times, trial_types, probe, fig_title, segment=0):
    '''
    Makes the output plots showing a summary of all the good tagged units
    '''
    width = np.ceil(np.sqrt(len(unit_ids)))
    height = np.ceil(len(unit_ids)/width)

    plt.clf()
    gs = gridspec.GridSpec(int(height), int(width), hspace=0.7)
    #gs.update(top=1-(height*0.01), bottom=0+(height*0.02), left=0+(width*0.015), right=1-(width*0.015), wspace=0.5, hspace=0.7)
    gs.update(wspace=0.5, hspace=0.7)


    for ind_unit, unit in enumerate(unit_ids):
        sample_numbers = sorting_output.get_unit_spike_train(unit, segment_index=segment)
        #sample_numbers = sample_numbers[sample_numbers<len(timestamps)]
        unit_spike_times = timestamps[sample_numbers]
        #unit_waveform = waveform_extractor.get_unit_template(unit)
        template_ext = waveform_extractor.get_extension("templates")
        unit_waveform = template_ext.get_unit_template(unit)
        #unit_metrics = laser_response_metrics.query('unit_id == @unit')
        
        gs_this_unit = gridspec.GridSpecFromSubplotSpec(2, len(trial_types)+1, subplot_spec=gs[int(ind_unit//width), int(ind_unit%width)], wspace=0.9, hspace=0.6, height_ratios=[0.005,1])
        #gs_this_cool_unit = gridspec.GridSpecFromSubplotSpec(2, len(width_ratios), subplot_spec=gs_cool_units[ind_unit//num_cols, np.mod(ind_unit, num_cols)], wspace=0.2, hspace=0.5, height_ratios=[0.005,1])

        for ind_type, trial_type in enumerate(trial_types):
            # plot npopto stim by site
            if 'internal' in trial_type:
                max_power = max(event_ids.query('type == @trial_type').power)
                sites = list(np.unique(event_ids.query('type == @trial_type').site))
                tag_trials = event_ids.query('param_group == "train" and site == @sites and power == @max_power and type == @trial_type and emission_location == @probe')
                y_axis = tag_trials.site.tolist()
                y_label = 'Emission site'
                y_ticks = sites
            elif 'external' in trial_type:
                tag_trials = event_ids.query('param_group == "train" and site == 0 and type == @trial_type and emission_location == @probe')
                powers = list(np.unique(event_ids.query('type == @trial_type').power))
                y_axis = tag_trials.power.tolist()
                y_label = 'Power (mW)'
                y_ticks = powers
            x_label = 'Time from laser onset (s)'

            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]
            total_duration = (duration*num_pulses)+(pulse_interval*num_pulses)
            raster_time_range = [-(total_duration/2)/1000, (1.5*total_duration)/1000]
            wavelength = np.unique(tag_trials.wavelength)[0]

            # plot responses to train of pulses
            ax_raster = plt.subplot(gs_this_unit[1 + ind_type//2, ind_type%2])
            this_event_timestamps = laser_onset_times[tag_trials.index.tolist()]
            event_locked_timestamps = af.event_locked_timestamps(unit_spike_times, this_event_timestamps, raster_time_range)
            raster_plot(event_locked_timestamps, raster_time_range, y_axis, ms=2.5, markeredgecolor='none')
            #plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xlim(raster_time_range)
            ax_raster.set_yticklabels(y_ticks)
            plt.gca().tick_params('both', labelsize=8)
            plt.title(f'{wavelength} nm')
            
            # add xlabel to last units
            plt.xlabel("Time from laser onset (s)")


            # patches showing laser presentation
            yLims = np.array(plt.ylim())
            laser_color = 'tomato' if 'red' in trial_type else 'skyblue'
            for pulse in range(num_pulses):
                rect = patches.Rectangle((pulse * (duration+pulse_interval)/1000, yLims[0]), duration / 1000, yLims[1] - yLims[0], linewidth=1, edgecolor=laser_color, facecolor=laser_color, alpha=0.35, clip_on=False)
                ax_raster.add_patch(rect)

        # plot waveform
        ax_waveform = plt.subplot(gs_this_unit[1,len(trial_types)])
        cmap = matplotlib.cm.PRGn
        #cmap = matplotlib.cm.viridis
        shifted_cmap = shiftedColorMap(cmap, np.min(unit_waveform[40:160,:150]), np.max(unit_waveform[40:160,:150]), 'shifted_PRGn')
        waveform_peak = np.min(unit_waveform)
        plt.imshow(unit_waveform[40:160,:150].T, aspect='auto', cmap=shifted_cmap)
        ax_waveform.invert_yaxis()
        # unit_peak_channel = laser_response_metrics.peak_channel[laser_response_metrics.unit_id==unit].tolist()[0]
        # channel_list = np.array([0, 5, 10, 15])
        # channel_labels = np.array(channel_list + unit_peak_channel - 10, dtype=int)
        # ax_waveform.set_yticks(channel_list)
        # ax_waveform.set_yticklabels(channel_labels)
        plt.ylabel('Channel')
        # add xlabel to last units
        plt.xlabel('Sample number')
        cbar = plt.colorbar()
        cbar.set_label('Voltage (uV)')

        # inset with peak channel waveform
        waveform_peak = np.min(unit_waveform)
        peak_ind = np.where(unit_waveform==waveform_peak)
        peak_waveform = unit_waveform[40:160,peak_ind[1][0]]

        ax_inset = inset_axes(ax_waveform, width="30%", height="25%", loc=1, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax_waveform.transAxes)
        plt.plot(peak_waveform, lw=2, c='k')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        # Add ghost axes so I can add a title with cluster number...
        ax_title = plt.subplot(gs_this_unit[0, :])
        ax_title.axis('off')
        ax_title.set_title(f'cluster {unit}', fontweight='heavy')

    height_scale = len(y_ticks)/4
    plt.gcf().set_size_inches((width * (len(trial_types)+1) * 3, height * height_scale))
    fig_format = 'png'
    fig_name = f'{fig_title}.{fig_format}'
    plt.savefig(f'/results/{fig_name}', format=fig_format)
    print(f'{fig_name} saved')