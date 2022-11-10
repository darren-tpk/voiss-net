# Import dependencies
import glob
from obspy import UTCDateTime, read, Stream, Trace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.transforms import Bbox
from scipy.signal import spectrogram
import colorcet as cc

def load_data(network,station,channel,location,starttime,endtime,pad=None,local=None,data_dir=None,client=None):

    """
    Load data from local miniseed repository or query data from server
    :param network (str or list): SEED network code(s)
    :param station (str or list): SEED station code(s)
    :param channel (str or list): SEED channel code(s)
    :param location (str or list): SEED location code(s)
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): start time for desired data
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): end time for desired data
    :param pad (float): Number of seconds to add on to the start and end times for desired data [s]. This should be greater than or equal to the taper length if tapering is needed.
    :param local (bool): If `True`, pull local data from data_dir. If `False`, query client for data.
    :param data_dir (str): Full file path for data directory
    :param client (str): Name of desired FDSN client (e.g., IRIS)
    :return: Stream (:class:`~obspy.core.stream.Stream`) data object
    """

    # Sanity check to see if data needs to be pulled from local directory of queried from client
    if (local==True and data_dir!=None and client==None) or (local==False and data_dir==None and client!=None):
        pass
    else:
        raise ValueError('Either set local=True and provide data_dir, or set local=False and provide client.')

    # Edit starttime and endtime if padding is required
    if pad is not None:
        starttime = starttime - pad
        endtime = endtime + pad

    # If we are pulling data from local
    if local:

        # Floor the start and end times, construct a list of UTCDateTimes marking the start of each day
        starttime_floor = UTCDateTime(starttime.date)
        endtime_floor = UTCDateTime(endtime.date)
        num_days = int((endtime_floor - starttime_floor) / 86400)
        time_list = [starttime_floor + i*86400 for i in range(num_days+1)]

        # Initialize list for all station channel data filenames
        data_filenames = []

        # If only one station-channel input is given,
        if type(network) == str and type(station) == str and type(channel) == str and type(location) == str:

            # Loop over time list, craft file string and append
            for time in time_list:
                data_filename = data_dir + station + '.' + channel + '.' + str(time.year) + ':' + f'{time.julday:03}' + ':*'
                data_filenames.append(data_filename)

        # If multiple station-channel inputs are given,
        elif type(network) == list and type(station) == list and type(channel) == list and type(location) == list and len(network) == len(station) == len(channel) == len(location):

            # Loop over station-channel inputs
            for i in range(len(network)):

                # Loop over time list, craft file string and append
                for time in time_list:
                    data_filename = data_dir + station[i] + '.' + channel[i] + '.' + str(time.year) + ':' + f'{time.julday:03}' + ':*'
                    data_filenames.append(data_filename)

        # If the checks fail,
        else:

            # Raise error
            raise ValueError('Inputs are in the wrong format. Either input a string (one station) or a lists of the same length (multiple stations) for network/station/channel/location.')

        # Now permutate the list of filenames and times to glob and load data
        stream = Stream()
        for data_filename in data_filenames:
            matching_filenames = (glob.glob(data_filename))
            for matching_filename in matching_filenames:
                try:
                    stream_contribution = read(matching_filename)
                    stream = stream + stream_contribution
                except:
                    continue

        # Trim and merge stream
        stream = stream.trim(starttime=starttime,endtime=endtime)
        stream = stream.merge()

        return stream

    # If user provides a client,
    elif not local and client is not None:

        # Prepare client
        from obspy.clients.fdsn import Client
        client = Client(client)

        # Initialize stream
        stream = Stream()

        # If only one station-channel input is given,
        if type(network) == str and type(station) == str and type(channel) == str and type(location) == str:

            # Grab stream
            stream_contribution = client.get_waveforms(network, station, location, channel, starttime, endtime, attach_response=True)
            stream = stream + stream_contribution

        # If multiple station-channel inputs are given,
        elif type(network) == list and type(station) == list and type(channel) == list and type(
                location) == list and len(network) == len(station) == len(channel) == len(location):

            # Loop over station-channel inputs
            for i in range(len(network)):

                # Grab stream
                stream_contribution = client.get_waveforms(network[i], station[i], location[i], channel[i], starttime, endtime, attach_response=True)
                stream = stream + stream_contribution

        # If the checks fail,
        else:

            # Raise error
            raise ValueError('Inputs are in the wrong format. Either input a string (one station) or a lists of the same length (multiple stations) for network/station/channel/location.')

        # Trim and merge stream
        stream = stream.trim(starttime=starttime,endtime=endtime)
        stream = stream.merge()

        return stream

    # If not no client is provided
    else:

        # Raise error
        raise ValueError('Please provide an input client (e.g. "IRIS").')

def process_waveform(stream,remove_response=True,detrend=True,taper_length=None,taper_percentage=None,filter_band=(1,10),verbose=True):

    """
    Process waveform by removing response, detrending, tapering and filtering (in that order)
    :param stream (:class:`~obspy.core.stream.Stream`): Input data
    :param remove_response (bool): If `True`, remove response using metadata. If `False`, response is not removed
    :param detrend (bool): If `True`, demean input data. If `False`, data is not demeaned
    :param taper_length (float): Taper length in seconds [s]. This is usually set to be similar as the pad length when loading data. If both taper_length and taper_percentage are `None`, data is not tapered
    :param taper_percentage (float): Taper length in percentage [%], if desired. If both taper_length and taper_percentage are `None`, data is not tapered
    :param filter_band (tuple): Tuple of length 2 storing minimum frequency and maximum frequency for bandpass filter ([Hz],[Hz])
    :param verbose (bool): If `True`, print out a declaration for each processing step
    :return: Stream (:class:`~obspy.core.stream.Stream`) data object
    """

    if verbose:
        print('Processing trace/stream...')
    if remove_response:
        for tr in stream:
            fs_resp = tr.stats.sampling_rate
            pre_filt = [0.005, 0.01, fs_resp/2-2, fs_resp/2]
            tr.remove_response(pre_filt=pre_filt, output='VEL', water_level=60, plot=False)
        if verbose:
            print('Response removed.')
    if detrend:
        #stream.detrend(type='linear')
        stream.detrend('demean')
        if verbose:
            print('Waveform demeaned.')
    if taper_length is not None and taper_percentage is None:
        #stream.taper(max_percentage=.02)
        stream.taper(max_percentage=None, max_length=taper_length/2)
        if verbose:
            print('Waveform tapered by pad length.')
    elif taper_length is None and taper_percentage is not None:
        if taper_percentage > 1:
            taper_percentage = taper_percentage/100
        stream.taper(max_percentage=taper_percentage, max_length=None)
        if verbose:
            print('Waveform tapered by pad percentage.')
    elif taper_length is None and taper_percentage is None:
        pass
    else:
        raise ValueError('Either provide taper_length OR taper_percentage if tapering.')
    if filter_band is not None:
        stream.filter('bandpass', freqmin=filter_band[0], freqmax=filter_band[1], zerophase=True)
        if verbose:
            print('Waveform bandpass filtered from %.2f Hz to %.2f Hz.' % (filter_band[0],filter_band[1]))
    return stream

def plot_spectrogram(stream,starttime,endtime,window_duration,freq_lims,log=False,demean=False,v_percent_lims=(20,100),cmap=cc.cm.rainbow,figsize=(32,9),export_path=None):

    """
    Plot all traces in a stream and their corresponding spectrograms in separate plots using matplotlib
    :param stream (:class:`~obspy.core.stream.Stream`): Input data
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for desired plot
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for desired plot
    :param window_duration (float): Duration of spectrogram window [s]
    :param freq_lims (tuple): Tuple of length 2 storing minimum frequency and maximum frequency for the spectrogram plot ([Hz],[Hz])
    :param log (bool): If `True`, plot spectrogram with logarithmic y-axis
    :param v_percent_lims (tuple): Tuple of length 2 storing the percentile values in the spectrogram matrix used as colorbar limits. Default is (20,100).
    :param cmap (str or :class:`matplotlib.colors.LinearSegmentedColormap`): Colormap for spectrogram plot. Default is colorcet.cm.rainbow.
    :param figsize (tuple): Tuple of length 2 storing output figure size. Default is (32,9).
    :param export_path (str or `None`): If str, export plotted figures as '.png' files, named by the trace id and time. If `None`, show figure in interactive python.
    """

    # Be nice and accept traces as well
    if type(stream) == Trace:
        stream = Stream() + stream

    # Loop over each trace in the stream
    for trace in stream:

        # Check if input trace is infrasound (SEED channel name ends with 'DF'). Else, assume seismic.
        if trace.stats.channel[1:] == 'DF':
            # If infrasound, define corresponding axis labels and reference values
            y_axis_label = 'Pressure (Pa)'
            REFERENCE_VALUE = 20 * 10**-6  # Pressure, in Pa
            colorbar_label = f'Power (dB rel. [{REFERENCE_VALUE * 1e6:g} µPa]$^2$ Hz$^{{-1}}$)'
            rescale_factor = 1  # Plot waveform in Pa
        else:
            # If seismic, define corresponding axis labels and reference values
            y_axis_label = 'Velocity (µm s$^{-1}$)'
            REFERENCE_VALUE = 1  # Velocity, in m/s
            colorbar_label = f'Power (dB rel. {REFERENCE_VALUE:g} [m s$^{{-1}}$]$^2$ Hz$^{{-1}}$)'
            rescale_factor = 10**-6  # Plot waveform in micrometer/s

        # Extract trace information for FFT
        sampling_rate = trace.stats.sampling_rate
        samples_per_segment = int(window_duration * sampling_rate)

        # Compute spectrogram (Note that overlap is 90% of samples_per_segment)
        sample_frequencies, segment_times, spec = spectrogram(trace.data, sampling_rate, window='hann', scaling='density', nperseg=samples_per_segment, noverlap=samples_per_segment*.9)

        # Convert spectrogram matrix to decibels for plotting
        spec_db = 10 * np.log10(abs(spec) / (REFERENCE_VALUE**2))

        # If demeaning is desired, remove temporal mean from spectrogram
        if demean:
            # column_ht = np.shape(spec_db)[0]
            # ratio_vec = np.linspace(0,0.5,column_ht).reshape(column_ht,1)
            spec_db = spec_db - np.mean(spec_db,1)[:, np.newaxis]*ratio_vec

        # Convert trace times to matplotlib dates
        trace_time_matplotlib = trace.stats.starttime.matplotlib_date + (segment_times / dates.SEC_PER_DAY)

        # Prepare time ticks for spectrogram and waveform figure (Note that we divide the duration into 10 equal increments and 11 uniform ticks)
        if ((endtime-starttime)%1800) == 0:
            time_tick_list = np.arange(starttime,endtime+1,(endtime-starttime)/12)
        else:
            time_tick_list = np.arange(starttime,endtime+1,(endtime-starttime)/10)
        time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
        time_tick_labels = [time.strftime('%H:%M') for time in time_tick_list]

        # Craft figure
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=figsize,constrained_layout=True)
        if freq_lims is not None:
            freq_min = freq_lims[0]
            freq_max = freq_lims[1]
            spec_db_plot = spec_db[np.where((sample_frequencies>freq_min) & (sample_frequencies<freq_max)),:]
            c = ax1.imshow(spec_db, extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1],sample_frequencies[0], sample_frequencies[-1]],
                           vmin=np.percentile(spec_db_plot, v_percent_lims[0]), vmax=np.percentile(spec_db_plot, v_percent_lims[1]),
                           origin='lower', aspect='auto', interpolation=None, cmap=cmap)
            # c = ax1.pcolormesh(trace_time_matplotlib, sample_frequencies, spec_db, vmin=np.percentile(spec_db_plot,v_percent_lims[0]), vmax=np.percentile(spec_db_plot,v_percent_lims[1]), cmap=cmap, shading='nearest', rasterized=True)
        else:
            c = ax1.imshow(spec_db, extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1], sample_frequencies[0],sample_frequencies[-1]],
                           vmin=np.percentile(spec_db, v_percent_lims[0]),  vmax=np.percentile(spec_db, v_percent_lims[1]),
                           origin='lower', aspect='auto', interpolation=None, cmap=cmap)
            # c = ax1.pcolormesh(trace_time_matplotlib, sample_frequencies, spec_db, vmin=np.percentile(spec_db,v_percent_lims[0]),vmax=np.percentile(spec_db,v_percent_lims[1]), cmap=cmap, shading='nearest', rasterized=True)
        ax1.set_ylabel('Frequency (Hz)',fontsize=22)
        if log:
            ax1.set_yscale('log')
        if freq_lims is not None:
            ax1.set_ylim([freq_min,freq_max])
        ax1.tick_params(axis='y',labelsize=18)
        ax1.set_xlim([starttime.matplotlib_date,endtime.matplotlib_date])
        ax1.set_xticks(time_tick_list_mpl)
        ax1.set_xticklabels(time_tick_labels,fontsize=18,rotation=30)
        ax1.set_title(trace.id, fontsize=24, fontweight='bold')
        cbar = fig.colorbar(c, aspect=10, pad=0.005, ax=ax1, location='right')
        cbar.set_label(colorbar_label, fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        ax2.plot(trace.times('matplotlib'), trace.data * rescale_factor,'k-',linewidth=1)
        ax2.set_ylabel(y_axis_label, fontsize=22)
        ax2.tick_params(axis='y',labelsize=18)
        ax2.yaxis.offsetText.set_fontsize(18)
        ax2.set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
        ax2.set_xticks(time_tick_list_mpl)
        ax2.set_xticklabels(time_tick_labels,fontsize=18,rotation=30)
        ax2.set_xlabel('UTC Time on ' + starttime.date.strftime('%b %d, %Y'), fontsize=22)
        ax2.grid()
        if export_path is None:
            fig.show()
        else:
            file_label = starttime.strftime('%Y%m%d_%H%M_') + trace.id.replace('.','_')
            fig.savefig(export_path + file_label + '.png',bbox_inches='tight')
            extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(export_path + file_label + 'spec.png',bbox_inches=extent)
            plt.close()

def plot_spectrogram_multi(stream,starttime,endtime,window_duration,freq_lims,log=False,demean=False,v_percent_lims=(20,100),cmap=cc.cm.rainbow,export_path=None):

    """
    Plot all traces in a stream and their corresponding spectrograms in separate plots using matplotlib
    :param stream (:class:`~obspy.core.stream.Stream`): Input data
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for desired plot
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for desired plot
    :param window_duration (float): Duration of spectrogram window [s]
    :param freq_lims (tuple): Tuple of length 2 storing minimum frequency and maximum frequency for the spectrogram plot ([Hz],[Hz])
    :param log (bool): If `True`, plot spectrogram with logarithmic y-axis
    :param v_percent_lims (tuple): Tuple of length 2 storing the percentile values in the spectrogram matrix used as colorbar limits. Default is (20,100).
    :param cmap (str or :class:`matplotlib.colors.LinearSegmentedColormap`): Colormap for spectrogram plot. Default is colorcet.cm.rainbow.
    :param export_path (str or `None`): If str, export plotted figures as '.png' files, named by the trace id and time. If `None`, show figure in interactive python.
    """

    # Initialize figure based on length of input stream
    figsize = (32, 4 * len(stream))
    fig, axs = plt.subplots(len(stream), 1, figsize=figsize)
    fig.subplots_adjust(hspace=0)

    # Loop over each trace in the stream
    for axs_index, trace in enumerate(stream):

        # Check if input trace is infrasound (SEED channel name ends with 'DF'). Else, assume seismic.
        if trace.stats.channel[1:] == 'DF':
            # If infrasound, define corresponding reference values
            REFERENCE_VALUE = 20 * 10 ** -6  # Pressure, in Pa
            rescale_factor = 1  # Plot waveform in Pa
        else:
            # If seismic, define corresponding reference values
            REFERENCE_VALUE = 1  # Velocity, in m/s
            rescale_factor = 10 ** -6  # Plot waveform in micrometer/s

        # Extract trace information for FFT
        sampling_rate = trace.stats.sampling_rate
        samples_per_segment = int(window_duration * sampling_rate)

        # Compute spectrogram (Note that overlap is 90% of samples_per_segment)
        sample_frequencies, segment_times, spec = spectrogram(trace.data, sampling_rate, window='hann',
                                                              scaling='density', nperseg=samples_per_segment,
                                                              noverlap=samples_per_segment * .9)

        # Convert spectrogram matrix to decibels for plotting
        spec_db = 10 * np.log10(abs(spec) / (REFERENCE_VALUE ** 2))

        # If demeaning is desired, remove temporal mean from spectrogram
        if demean:
            # column_ht = np.shape(spec_db)[0]
            # ratio_vec = np.linspace(0,0.5,column_ht).reshape(column_ht,1)
            spec_db = spec_db - np.mean(spec_db,1)[:, np.newaxis]*ratio_vec

        # Convert trace times to matplotlib dates
        trace_time_matplotlib = trace.stats.starttime.matplotlib_date + (segment_times / dates.SEC_PER_DAY)

        # Craft figure
        if freq_lims is not None:
            freq_min = freq_lims[0]
            freq_max = freq_lims[1]
            spec_db_plot = spec_db[np.where((sample_frequencies > freq_min) & (sample_frequencies < freq_max)), :]
            c = axs[axs_index].imshow(spec_db,
                                      extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1],
                                              sample_frequencies[0],
                                              sample_frequencies[-1]],
                                      vmin=np.percentile(spec_db_plot, v_percent_lims[0]),
                                      vmax=np.percentile(spec_db_plot, v_percent_lims[1]),
                                      origin='lower', aspect='auto', interpolation=None, cmap=cmap)
            # c = ax1.pcolormesh(trace_time_matplotlib, sample_frequencies, spec_db, vmin=np.percentile(spec_db_plot,v_percent_lims[0]), vmax=np.percentile(spec_db_plot,v_percent_lims[1]), cmap=cmap, shading='nearest', rasterized=True)
        else:
            c = axs[axs_index].imshow(spec_db,
                                      extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1],
                                              sample_frequencies[0],
                                              sample_frequencies[-1]],
                                      vmin=np.percentile(spec_db, v_percent_lims[0]),
                                      vmax=np.percentile(spec_db, v_percent_lims[1]),
                                      origin='lower', aspect='auto', interpolation=None, cmap=cmap)
            # c = ax1.pcolormesh(trace_time_matplotlib, sample_frequencies, spec_db, vmin=np.percentile(spec_db,v_percent_lims[0]),vmax=np.percentile(spec_db,v_percent_lims[1]), cmap=cmap, shading='nearest', rasterized=True)
        if log:
            axs[axs_index].set_yscale('log')
        if freq_lims is not None:
            axs[axs_index].set_ylim([freq_min, freq_max])
            axs[axs_index].set_yticks(range(2, freq_max + 1, 2))
        axs[axs_index].set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
        axs[axs_index].tick_params(axis='y', labelsize=18)
        axs[axs_index].set_ylabel(trace.id, fontsize=22, fontweight='bold')
    if ((endtime - starttime) % 1800) == 0:
        time_tick_list = np.arange(starttime, endtime + 1, (endtime - starttime) / 12)
    else:
        time_tick_list = np.arange(starttime, endtime + 1, (endtime - starttime) / 10)
    time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
    time_tick_labels = [time.strftime('%H:%M') for time in time_tick_list]
    axs[-1].set_xticks(time_tick_list_mpl)
    axs[-1].set_xticklabels(time_tick_labels, fontsize=22, rotation=30)
    axs[-1].set_xlabel('UTC Time on ' + starttime.date.strftime('%b %d, %Y'), fontsize=25)
    if export_path is None:
        fig.show()
    else:
        file_label = starttime.strftime('%Y%m%d_%H%M') + '__' + endtime.strftime('%Y%m%d_%H%M') + '_' + '_'.join([tr.id.split('.')[1] for tr in stream])
        fig.savefig(export_path + file_label + '.png', bbox_inches='tight')
        extent1 = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        extent2 = axs[-1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        extent = Bbox([extent2._points[0], extent1._points[1]])
        fig.savefig(export_path + file_label + '_spec.png', bbox_inches=extent)
        plt.close()