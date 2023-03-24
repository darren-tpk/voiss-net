# Import dependencies
import os
import glob
import colorcet as cc
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read, Stream, Trace
from matplotlib import dates
from matplotlib.transforms import Bbox
from scipy.signal import spectrogram, find_peaks
from scipy.fft import rfft, rfftfreq
from ordpy import complexity_entropy
from geopy.distance import geodesic as GD
from matplotlib.colors import ListedColormap
from DataGenerator import DataGenerator
from keras.models import load_model
from waveform_collection import gather_waveforms
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

def process_waveform(stream,remove_response=True,rr_output='VEL',detrend=True,taper_length=None,taper_percentage=None,filter_band=None,verbose=True):

    """
    Process waveform by removing response, detrending, tapering and filtering (in that order)
    :param stream (:class:`~obspy.core.stream.Stream`): Input data
    :param remove_response (bool): If `True`, remove response using metadata. If `False`, response is not removed
    :param rr_output (str): Set to 'DISP', 'VEL', 'ACC', or 'DEF'. See obspy documentation for more details.
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
            tr.remove_response(pre_filt=pre_filt, output=rr_output, water_level=60, plot=False)
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

def plot_spectrogram(stream,starttime,endtime,window_duration,freq_lims,log=False,demean=False,v_percent_lims=(20,100),cmap=cc.cm.rainbow,earthquake_times=None,db_hist=False,figsize=(32,9),export_path=None):

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
    :param earthquake_times (list of :class:`~obspy.core.utcdatetime.UTCDateTime`): List of UTCDateTimes storing earthquake origin times to be plotted as vertical black dashed lines.
    :param db_hist (bool): If `True`, plot a db histogram (spanning the plotted timespan) on the right side of the spectrogram across the sample frequencies
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
            # spec_db = spec_db - np.mean(spec_db,1)[:, np.newaxis]*ratio_vec
            spec_db = spec_db - np.mean(spec_db, 1)[:, np.newaxis]

        # Convert trace times to matplotlib dates
        trace_time_matplotlib = trace.stats.starttime.matplotlib_date + (segment_times / dates.SEC_PER_DAY)

        # If earthquake times are provided, convert them from UTCDateTime to matplotlib dates
        if earthquake_times:
            earthquake_times_matplotlib = [eq.matplotlib_date for eq in earthquake_times]
        else:
            earthquake_times_matplotlib = []

        # Prepare time ticks for spectrogram and waveform figure (Note that we divide the duration into 10 equal increments and 11 uniform ticks)
        if ((endtime-starttime)%1800) == 0:
            denominator = 12
            time_tick_list = np.arange(starttime,endtime+1,(endtime-starttime)/denominator)
        else:
            denominator = 10
            time_tick_list = np.arange(starttime,endtime+1,(endtime-starttime)/denominator)
        time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
        time_tick_labels = [time.strftime('%H:%M') for time in time_tick_list]

        # Craft figure
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=figsize,constrained_layout=True)

        # If frequency limits are defined
        if freq_lims is not None:
            # Determine frequency limits and trim spec_db
            freq_min = freq_lims[0]
            freq_max = freq_lims[1]
            spec_db_plot = spec_db[np.flatnonzero((sample_frequencies>freq_min) & (sample_frequencies<freq_max)),:]
            c = ax1.imshow(spec_db, extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1],sample_frequencies[0], sample_frequencies[-1]],
                           vmin=np.percentile(spec_db_plot, v_percent_lims[0]), vmax=np.percentile(spec_db_plot, v_percent_lims[1]),
                           origin='lower', aspect='auto', interpolation=None, cmap=cmap)
            # If we want a db spectrogram across the plotted time span, compute and plot on the right side of the figure
            if db_hist:
                spec_db_hist = np.sum(spec_db_plot,axis=1)
                spec_db_hist = (spec_db_hist-np.min(spec_db_hist)) / (np.max(spec_db_hist)-np.min(spec_db_hist))
                hist_plotting_range = (1/denominator) * (trace_time_matplotlib[-1] - trace_time_matplotlib[0])
                hist_plotting_points = trace_time_matplotlib[-1] - (spec_db_hist * hist_plotting_range)
                sample_frequencies_plot = sample_frequencies[np.flatnonzero((sample_frequencies>freq_min) & (sample_frequencies<freq_max))]
                ax1.plot(hist_plotting_points, sample_frequencies_plot, 'k-', linewidth=10, alpha=0.6)
                ax1.plot([trace_time_matplotlib[-1], trace_time_matplotlib[-1] - hist_plotting_range],
                         [sample_frequencies_plot[-1], sample_frequencies_plot[-1]], 'k-', linewidth=10, alpha=0.6)
                ax1.plot(trace_time_matplotlib[-1] - hist_plotting_range, sample_frequencies_plot[-1], 'k<', markersize=50)
        # If no frequencies limits are given
        else:
            # We go straight into plotting spec_db
            c = ax1.imshow(spec_db, extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1], sample_frequencies[0],sample_frequencies[-1]],
                           vmin=np.percentile(spec_db, v_percent_lims[0]),  vmax=np.percentile(spec_db, v_percent_lims[1]),
                           origin='lower', aspect='auto', interpolation=None, cmap=cmap)
            # If we want a db spectrogram across the plotted time span, compute and plot on the right side of the figure
            if db_hist:
                spec_db_hist = np.sum(spec_db, axis=1)
                spec_db_hist = (spec_db_hist - np.min(spec_db_hist)) / (np.max(spec_db_hist) - np.min(spec_db_hist))
                hist_plotting_range = (1/denominator) * (trace_time_matplotlib[-1] - trace_time_matplotlib[0])
                hist_plotting_points = trace_time_matplotlib[-1] - (spec_db_hist * hist_plotting_range)
                ax1.plot(hist_plotting_points, sample_frequencies, 'k-', linewidth=10, alpha=0.6)
                ax1.plot([trace_time_matplotlib[-1],trace_time_matplotlib[-1]-hist_plotting_range],
                         [sample_frequencies[-1],sample_frequencies[-1]], 'k-', linewidth=10, alpha=0.6)
                ax1.plot(trace_time_matplotlib[-1]-hist_plotting_range, sample_frequencies[-1], 'k<', markersize=50)
        for earthquake_time_matplotlib in earthquake_times_matplotlib:
            ax1.axvline(x=earthquake_time_matplotlib, linestyle='--', color='k', linewidth=3, alpha=0.7)
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

def plot_spectrogram_multi(stream,starttime,endtime,window_duration,freq_lims,log=False,demean=False,v_percent_lims=(20,100),cmap=cc.cm.rainbow,earthquake_times=None,explosion_times=None,db_hist=False,export_path=None,export_spec=True):

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
    :param earthquake_times (list of :class:`~obspy.core.utcdatetime.UTCDateTime`): List of UTCDateTimes storing earthquake origin times to be plotted as vertical black dashed lines.
    :param explosion_times (list of :class:`~obspy.core.utcdatetime.UTCDateTime`): List of UTCDateTimes storing explosion detection times to be plotted as vertical dark red dashed lines.
    :param db_hist (bool): If `True`, plot a db histogram (spanning the plotted timespan) on the right side of the spectrograms across the sample frequencies
    :param export_path (str or `None`): If str, export plotted figures as '.png' files, named by the trace id and time. If `None`, show figure in interactive python.
    :param export_spec (bool): If `True`, export spectrogram with trimmed-off axis labels (useful for labeling)
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
            # spec_db = spec_db - np.mean(spec_db,1)[:, np.newaxis]*ratio_vec
            spec_db = spec_db - np.mean(spec_db, 1)[:, np.newaxis]

        # Convert trace times to matplotlib dates
        trace_time_matplotlib = trace.stats.starttime.matplotlib_date + (segment_times / dates.SEC_PER_DAY)

        # Determine number of ticks smartly
        if ((endtime - starttime) % 1800) == 0:
            denominator = 12
        else:
            denominator = 10

        # If earthquake times are provided, convert them from UTCDateTime to matplotlib dates
        if earthquake_times:
            earthquake_times_matplotlib = [eq.matplotlib_date for eq in earthquake_times]
        else:
            earthquake_times_matplotlib = []

        # If explosion times are provided, convert them from UTCDateTime to matplotlib dates
        if explosion_times:
            explosion_times_matplotlib = [ex.matplotlib_date for ex in explosion_times]
        else:
            explosion_times_matplotlib = []

        # Craft figure
        # If frequency limits are defined
        if freq_lims is not None:
            # Determine frequency limits and trim spec_db
            freq_min = freq_lims[0]
            freq_max = freq_lims[1]
            spec_db_plot = spec_db[np.flatnonzero((sample_frequencies > freq_min) & (sample_frequencies < freq_max)), :]
            axs[axs_index].imshow(spec_db,
                                      extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1],
                                              sample_frequencies[0],
                                              sample_frequencies[-1]],
                                      vmin=np.percentile(spec_db_plot, v_percent_lims[0]),
                                      vmax=np.percentile(spec_db_plot, v_percent_lims[1]),
                                      origin='lower', aspect='auto', interpolation=None, cmap=cmap)
            # If we want a db spectrogram across the plotted time span, compute and plot on the right side of the figure
            if db_hist:
                spec_db_hist = np.sum(spec_db_plot, axis=1)
                spec_db_hist = (spec_db_hist - np.min(spec_db_hist)) / (np.max(spec_db_hist) - np.min(spec_db_hist))
                hist_plotting_range = (1/denominator) * (trace_time_matplotlib[-1] - trace_time_matplotlib[0])
                hist_plotting_points = trace_time_matplotlib[-1] - (spec_db_hist * hist_plotting_range)
                sample_frequencies_plot = sample_frequencies[
                    np.flatnonzero((sample_frequencies > freq_min) & (sample_frequencies < freq_max))]
                axs[axs_index].plot(hist_plotting_points, sample_frequencies_plot, 'k-', linewidth=8, alpha=0.6)
                axs[axs_index].plot([trace_time_matplotlib[-1], trace_time_matplotlib[-1] - hist_plotting_range],
                         [sample_frequencies_plot[-1], sample_frequencies_plot[-1]], 'k-', linewidth=8, alpha=0.6)
                axs[axs_index].plot(trace_time_matplotlib[-1] - hist_plotting_range, sample_frequencies_plot[-1], 'k<', markersize=30)
        # If no frequencies limits are given
        else:
            # We go straight into plotting spec_db
            axs[axs_index].imshow(spec_db,
                                      extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1],
                                              sample_frequencies[0],
                                              sample_frequencies[-1]],
                                      vmin=np.percentile(spec_db, v_percent_lims[0]),
                                      vmax=np.percentile(spec_db, v_percent_lims[1]),
                                      origin='lower', aspect='auto', interpolation=None, cmap=cmap)
            # If we want a db spectrogram across the plotted time span, compute and plot on the right side of the figure
            if db_hist:
                spec_db_hist = np.sum(spec_db, axis=1)
                spec_db_hist = (spec_db_hist - np.min(spec_db_hist)) / (np.max(spec_db_hist) - np.min(spec_db_hist))
                hist_plotting_range = (1 / denominator) * (trace_time_matplotlib[-1] - trace_time_matplotlib[0])
                hist_plotting_points = trace_time_matplotlib[-1] - (spec_db_hist * hist_plotting_range)
                axs[axs_index].plot(hist_plotting_points, sample_frequencies, 'k-', linewidth=8, alpha=0.6)
                axs[axs_index].plot([trace_time_matplotlib[-1], trace_time_matplotlib[-1] - hist_plotting_range],
                         [sample_frequencies[-1], sample_frequencies[-1]], 'k-', linewidth=8, alpha=0.6)
                axs[axs_index].plot(trace_time_matplotlib[-1] - hist_plotting_range, sample_frequencies[-1], 'k<', markersize=30)
        for earthquake_time_matplotlib in earthquake_times_matplotlib:
            axs[axs_index].axvline(x=earthquake_time_matplotlib, linestyle='--', color='k', linewidth=3, alpha=0.7)
        for explosion_time_matplotlib in explosion_times_matplotlib:
            axs[axs_index].axvline(x=explosion_time_matplotlib, linestyle='--', color='darkred', linewidth=3, alpha=0.7)
        if log:
            axs[axs_index].set_yscale('log')
        if freq_lims is not None:
            axs[axs_index].set_ylim([freq_min, freq_max])
            axs[axs_index].set_yticks(range(2, freq_max + 1, 2))
        axs[axs_index].set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
        axs[axs_index].tick_params(axis='y', labelsize=18)
        axs[axs_index].set_ylabel(trace.id, fontsize=22, fontweight='bold')
    time_tick_list = np.arange(starttime, endtime + 1, (endtime - starttime) / denominator)
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
        if export_spec:
            extent1 = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent2 = axs[-1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent = Bbox([extent2._points[0], extent1._points[1]])
            fig.savefig(export_path + file_label + '_spec.png', bbox_inches=extent)
            plt.close()

def calculate_spectrogram(trace,starttime,endtime,window_duration,freq_lims,demean=False):

    """
    Calculates and returns a 2D matrix storing the spectrogram values (of the input trace) in decibels
    :param trace (:class:`~obspy.core.stream.Trace`): Input data
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for desired plot
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for desired plot
    :param window_duration (float): Duration of spectrogram window [s]
    :param freq_lims (tuple): Tuple of length 2 storing minimum frequency and maximum frequency for the spectrogram plot ([Hz],[Hz])
    :return: numpy.ndarray: 2D spectrogram matrix storing power ratio values in decibels
    :return: numpy.ndarray: 1D array storing segment times in (:class:`~obspy.core.utcdatetime.UTCDateTime`) format
    """

    # Check if input trace is infrasound (SEED channel name ends with 'DF'). Else, assume seismic.
    if trace.stats.channel[1:] == 'DF':
        # If infrasound, define corresponding reference values
        REFERENCE_VALUE = 20 * 10 ** -6  # Pressure, in Pa
    else:
        # If seismic, define corresponding reference values
        REFERENCE_VALUE = 1  # Velocity, in m/s

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
        spec_db = spec_db - np.mean(spec_db, 1)[:, np.newaxis]

    # If frequency limits are defined
    if freq_lims is not None:
        # Determine frequency limits and trim spec_db
        freq_min = freq_lims[0]
        freq_max = freq_lims[1]
        spec_db = spec_db[np.flatnonzero((sample_frequencies > freq_min) & (sample_frequencies < freq_max)), :]

    # Calculate UTC times corresponding to all segments
    trace_time_matplotlib = trace.stats.starttime.matplotlib_date + (segment_times / dates.SEC_PER_DAY)
    utc_times = np.array([UTCDateTime(dates.num2date(mpl_time)) for mpl_time in trace_time_matplotlib])

    return spec_db, utc_times
def check_timeline(source,network,station,channel,location,starttime,endtime,model_path,npy_dir=None,export_path=None):

    """
    Pulls data, then loads a trained model to predict the timeline of classes
    :param source (str): Which source to gather waveforms from (e.g. IRIS)
    :param network (str): SEED network code [wildcards (``*``, ``?``) accepted]
    :param station (str): SEED station code or comma separated station codes [wildcards NOT accepted]
    :param channel (str): SEED location code [wildcards (``*``, ``?``) accepted]
    :param location (str): SEED channel code [wildcards (``*``, ``?``) accepted]
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for
            data request
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for
            data request
    :param model_path (str): Path to model .h5 file
    :param npy_dir (str): Path to directory of numpy files
    :param export_path (str): (str or `None`): If str, export plotted figures as '.png' files, named by the trace id and time. If `None`, show figure in interactive python.
    """

    # Define fixed values
    TIME_STEP = 4 * 60
    PAD = 60
    WINDOW_DURATION = 10
    FREQ_LIMS = (0.5, 10)
    V_PERCENT_LIMS = (20, 97.5)

    # Determine if infrasound
    infrasound = True if channel[-1] == 'F' else False
    REFERENCE_VALUE = 20 * 10 ** -6 if infrasound else 1  # Pa for infrasound, m/s for seismic

    # Load data, remove response, and re-order
    stream = gather_waveforms(source=source, network=network, station=station, location=location, channel=channel,
                              starttime=starttime - PAD, endtime=endtime + PAD, verbose=False)
    stream = process_waveform(stream, remove_response=True, detrend=False, taper_length=PAD, verbose=False)
    stream_default_order = [tr.stats.station for tr in stream]
    desired_index_order = [stream_default_order.index(stn) for stn in station.split(',') if stn in stream_default_order]
    stream = Stream([stream[i] for i in desired_index_order])

    # If no existing npy file directory exists, create a temporary directory and populate it
    if not npy_dir:

        # Create a temporary directory if it does not exist
        if not os.path.exists('./npy_temporary'):
            os.mkdir('npy_temporary')

        # Loop over stations that have data
        stream_stations = [tr.stats.station for tr in stream]
        for j, stream_station in enumerate(stream_stations):

            # Choose trace corresponding to station
            trace = stream[j]

            # Calculate spectrogram power matrix
            spec_db, utc_times = calculate_spectrogram(trace, starttime, endtime, window_duration=WINDOW_DURATION, freq_lims=FREQ_LIMS)

            # Define array of time steps for spectrogram slicing
            step_bounds = np.arange(starttime, endtime + TIME_STEP, TIME_STEP)

            # Loop over time steps
            for k in range(len(step_bounds) - 1):

                # Slice spectrogram
                sb1 = step_bounds[k]
                sb2 = step_bounds[k + 1]
                spec_slice_indices = np.flatnonzero([sb1 < t < sb2 for t in utc_times])
                spec_slice = spec_db[:, spec_slice_indices]

                # Enforce shape
                if np.shape(spec_slice) != (94, TIME_STEP):
                    # Try inclusive slicing time span (<= sb2)
                    spec_slice_indices = np.flatnonzero([sb1 < t <= sb2 for t in utc_times])
                    spec_slice = spec_db[:, spec_slice_indices]
                    # If it still doesn't fit our shape, raise error
                    if np.shape(spec_slice) != (94, TIME_STEP):
                        raise ValueError('Spectrogram slicing produced an erroneous shape.')

                # Skip matrices that have a spectrogram data gap
                if np.sum(spec_slice.flatten() < -220) > 50:
                    continue

                # Craft filename and save
                file_name = stream_station + '_' + sb1.strftime('%Y%m%d%H%M') + '_' + \
                            sb2.strftime('%Y%m%d%H%M') + '.npy'
                np.save('./npy_temporary/' + file_name, spec_slice)

        # Get list of npy file paths
        spec_paths_full = glob.glob('./npy_temporary/*.npy')

    # If a pre-existing npy directory exists, get list of npy file paths
    else:
        spec_paths_full = glob.glob(npy_dir + '/*.npy')

    # Filter spec paths by time
    spec_paths = []
    for spec_path in spec_paths_full:
        utc = UTCDateTime(spec_path.split('/')[-1].split('_')[1])
        spec_station = spec_path.split('/')[-1].split('_')[0]
        if starttime <= utc < endtime and spec_station in station:
            spec_paths.append(spec_path)

    # Load model
    saved_model = load_model(model_path)
    nclasses = saved_model.layers[-1].get_config()['units']
    nsubrows = len(station.split(','))

    # Create data generator for input spec paths
    params = {
        "dim": (94, TIME_STEP),
        "batch_size": 60,
        "n_classes": nclasses,
        "shuffle": True,
    }
    spec_placeholder_labels = [0 for i in spec_paths]
    spec_label_dict = dict(zip(spec_paths, spec_placeholder_labels))
    spec_gen = DataGenerator(spec_paths, spec_label_dict, **params)

    # Make predictions
    spec_predictions = saved_model.predict(spec_gen)
    predicted_labels = np.argmax(spec_predictions, axis=1)
    indicators = []
    for i, filepath in enumerate(spec_gen.list_ids):
        filename = filepath.split('/')[-1]
        chunks = filename.split('_')
        indicators.append([chunks[0], UTCDateTime(chunks[1]), predicted_labels[i]])

    # Craft unlabeled matrix
    matrix_length = int(np.ceil((endtime - starttime) / TIME_STEP))
    matrix_height = nsubrows
    matrix_plot = np.ones((matrix_height, matrix_length)) * nclasses
    for indicator in indicators:
        row_index = station.split(',').index(indicator[0])
        col_index = int((indicator[1] - starttime) / TIME_STEP)
        matrix_plot[row_index, col_index] = indicator[2]

    # Add voting row
    na_label = nclasses  # this index is one higher than the number of classes
    new_row = np.ones((1, np.shape(matrix_plot)[1])) * na_label
    matrix_plot = np.concatenate((matrix_plot, new_row))
    # If dealing with seismic, use seismic voting scheme
    if not infrasound:
        # Loop over columns and vote
        for j in range(np.shape(matrix_plot)[1]):
            sub_col = matrix_plot[:, j]
            labels_seen, label_counts = np.unique(sub_col, return_counts=True)
            if len(labels_seen) == 1 and na_label in labels_seen:
                new_row[0, j] = na_label
            elif len(labels_seen) == 1:
                new_row[0, j] = labels_seen[0]
            else:
                if na_label in labels_seen:
                    label_counts = np.delete(label_counts, labels_seen == na_label)
                    labels_seen = np.delete(labels_seen, labels_seen == na_label)
                selected_label_index = np.argwhere(label_counts == np.amax(label_counts))[-1][0]
                new_row[0, j] = labels_seen[selected_label_index]
        # Craft corresponding rgb values
        rgb_values = np.array([
            [193, 39, 45],
            [0, 129, 118],
            [0, 0, 167],
            [238, 204, 22],
            [164, 98, 0],
            [40, 40, 40],
            [255, 255, 255]])
    # Otherwise, use infrasound voting scheme
    else:
        # Loop over columns and vote
        for j in range(np.shape(matrix_plot)[1]):
            sub_col = matrix_plot[:, j]
            labels_seen, label_counts = np.unique(sub_col, return_counts=True)
            if len(labels_seen) == 1 and na_label in labels_seen:
                new_row[0, j] = na_label
            elif len(labels_seen) == 1:
                new_row[0, j] = labels_seen[0]
            elif 1 in labels_seen and label_counts[np.argwhere(labels_seen == 1)[0]] > 1:
                new_row[0, j] = 1
            elif 0 in labels_seen and label_counts[np.argwhere(labels_seen == 0)[0]] > 1:
                new_row[0, j] = 0
            else:
                if na_label in labels_seen:
                    label_counts = np.delete(label_counts, labels_seen == na_label)
                    labels_seen = np.delete(labels_seen, labels_seen == na_label)
                selected_label_index = np.argwhere(label_counts == np.amax(label_counts))[-1][0]
                new_row[0, j] = labels_seen[selected_label_index]
        # Craft corresponding rgb values
        rgb_values = np.array([
            [103, 52, 235],
            [235, 152, 52],
            [40, 40, 40],
            [15, 37, 60],
            [100, 100, 100],
            [255, 255, 255]])
    # Concatenate voting row to plotting matrix
    matrix_plot = np.concatenate((matrix_plot, new_row))
    # Craft color map
    rgb_ratios = rgb_values / 255
    colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0], 1))), axis=1)
    cmap = ListedColormap(colors)

    # Define colorbar keywords for plotting
    real_cbar_tick_interval = 2 * nclasses/(2*np.shape(rgb_values)[0])
    real_cbar_ticks = np.arange(real_cbar_tick_interval/2, nclasses, real_cbar_tick_interval)
    cbar_kws = {'ticks': real_cbar_ticks,
                'drawedges': True,
                'location': 'top',
                'fraction': 0.15,
                'aspect': 40}

    # Initialize figure and craft axes
    figsize = (32, 4 * len(stream) + 5)
    fig = plt.figure(figsize=figsize)
    gs_top = plt.GridSpec(len(stream)+1, 1, top=0.89)
    gs_base = plt.GridSpec(len(stream)+1, 1, hspace=0)
    ax1 = fig.add_subplot(gs_top[0, :])
    ax2 = fig.add_subplot(gs_base[1, :])
    other_axes = [fig.add_subplot(gs_base[i, :], sharex=ax2) for i in range(2, len(stream)+1)]
    axs = [ax2] + other_axes
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # Plot prediction heatmap in top axis
    sns.heatmap(matrix_plot, cmap=cmap, cbar=False, cbar_kws=cbar_kws, alpha=0.8, vmin=0, vmax=nclasses, ax=ax1)
    # for i in range(1,nhours):
    #     ax1.axvline(x=(i * (3600 / time_step)), linestyle='--', linewidth=3, color='k')
    ax1.set_xticks([])
    # ax1.set_xticks(np.arange(0, matrix_length, 3600 / (time_step)))
    # ax1.set_xticklabels([(starttime + 3600 * i).strftime('%y/%m/%d %H:%M') for i in range(nhours)], rotation=0)
    ax1.axhline(nsubrows, color='black')
    ax1.axhline(nsubrows + 1, color='black')
    ax1.axhspan(nsubrows, nsubrows + 1, facecolor='lightgray')
    yticks = np.concatenate((np.arange(0.5, nsubrows, 1), np.array([nsubrows + 1.5])))
    ax1.set_yticks(yticks)
    yticklabels = station.split(',').copy()
    yticklabels.append('VOTE')
    ax1.set_yticklabels(yticklabels, rotation=0, fontsize=24)
    ax1.patch.set_edgecolor('black')
    ax1.patch.set_linewidth(2)
    ax1.set_title('Station-based Voting', fontsize=30)

    # Loop over each trace in the stream and plot spectrograms on lower axes
    for axs_index, trace in enumerate(stream):

        # Extract trace information for FFT
        sampling_rate = trace.stats.sampling_rate
        samples_per_segment = int(WINDOW_DURATION * sampling_rate)

        # Compute spectrogram (Note that overlap is 90% of samples_per_segment)
        sample_frequencies, segment_times, spec = spectrogram(trace.data, sampling_rate, window='hann',
                                                              scaling='density', nperseg=samples_per_segment,
                                                              noverlap=samples_per_segment * .9)

        # Convert spectrogram matrix to decibels for plotting
        spec_db = 10 * np.log10(abs(spec) / (REFERENCE_VALUE ** 2))

        # Convert trace times to matplotlib dates
        trace_time_matplotlib = trace.stats.starttime.matplotlib_date + (segment_times / dates.SEC_PER_DAY)

        # Determine number of ticks smartly
        if ((endtime - starttime) % 1800) == 0:
            denominator = 12
        else:
            denominator = 10

        # Determine frequency limits and trim spec_db
        spec_db_plot = spec_db[np.flatnonzero((sample_frequencies > FREQ_LIMS[0]) & (sample_frequencies < FREQ_LIMS[1])), :]
        axs[axs_index].imshow(spec_db,
                                  extent=[trace_time_matplotlib[0], trace_time_matplotlib[-1],
                                          sample_frequencies[0],
                                          sample_frequencies[-1]],
                                  vmin=np.percentile(spec_db_plot, V_PERCENT_LIMS[0]),
                                  vmax=np.percentile(spec_db_plot, V_PERCENT_LIMS[1]),
                                  origin='lower', aspect='auto', interpolation=None, cmap=cc.cm.rainbow)
        axs[axs_index].set_ylim([FREQ_LIMS[0], FREQ_LIMS[1]])
        axs[axs_index].set_yticks(range(2, FREQ_LIMS[1] + 1, 2))
        axs[axs_index].set_xlim([starttime.matplotlib_date, endtime.matplotlib_date])
        axs[axs_index].tick_params(axis='y', labelsize=22)
        axs[axs_index].set_ylabel(trace.id, fontsize=24, fontweight='bold')
    time_tick_list = np.arange(starttime, endtime + 1, (endtime - starttime) / denominator)
    time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
    time_tick_labels = [time.strftime('%H:%M') for time in time_tick_list]
    axs[-1].set_xticks(time_tick_list_mpl)
    axs[-1].set_xticklabels(time_tick_labels, fontsize=24, rotation=30)
    axs[-1].set_xlabel('UTC Time on ' + starttime.date.strftime('%b %d, %Y'), fontsize=30)
    if export_path is None:
        fig.show()
    else:
        file_label = starttime.strftime('%Y%m%d_%H%M') + '__' + endtime.strftime('%Y%m%d_%H%M') + '_' + model_path.split('/')[-1].split('.')[0]
        fig.savefig(export_path + file_label + '.png', bbox_inches='tight')

def compute_metrics(stream_unprocessed, process_taper=None, metric_taper=None, filter_band=None, window_length=240, overlap=0, vlatlon=(55.4173, -161.8937)):

    """
    :param stream_unprocessed (:class:`~obspy.core.stream.Stream`): Input data (unprocessed -- response is removed within)
    :param padded_length (float): length for which the trace is padded [s]
    :param filter_band (tuple): Tuple of length 2 storing minimum frequency and maximum frequency for bandpass filter ([Hz],[Hz])
    :param window_length (float): window length for each metric to be computed, default is 240 [s]
    :param vlatlon (tuple): Tuple of length 2 storing the latitude and longitude of the target volcano for reduced displacement computation
    :param overlap (float): overlap for time stepping as each metric is computed. Ranges from 0 to 1. If set to 0, time step is equal to window_length.
    :return: numpy.ndarray: 2D array of matplotlib dates, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: 2D array of RSAM, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: 2D array of Reduced Displacement, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: 2D array of Central Frequency, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: 2D array of Dominant Frequency, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: 2D array of Standard Deviation of Top 30 Frequency Peaks, rows corresponding to traces in stream and columns corresponding to values.
    """

    # Remove response to obtain stream in displacement and velocity values
    stream_disp = process_waveform(stream_unprocessed.copy(), remove_response=True, rr_output='DISP', detrend=False,
                                   taper_length=process_taper, taper_percentage=None, filter_band=filter_band, verbose=False)
    stream_vel = process_waveform(stream_unprocessed.copy(), remove_response=True, rr_output='VEL', detrend=False,
                                  taper_length=process_taper, taper_percentage=None, filter_band=filter_band, verbose=False)

    # Get actual desired starttime and endtime by considering pad length
    starttime = stream_unprocessed[0].stats.starttime
    endtime = stream_unprocessed[0].stats.endtime
    if metric_taper:
        starttime += metric_taper + window_length/2
        endtime -= metric_taper + window_length/2
    elif process_taper:
        starttime += process_taper + window_length/2
        endtime -= process_taper + window_length/2
    else:
        starttime += window_length/2
        endtime -= window_length/2

    # Determine time step
    time_step = window_length * (1-overlap)

    # Initialize all metrics
    window_centers = np.arange(starttime, endtime+time_step, time_step)
    metric_length = len(window_centers)
    tmpl = np.ones((len(stream_unprocessed), metric_length)) * np.nan
    rsam = np.ones((len(stream_unprocessed), metric_length)) * np.nan
    dr = np.ones((len(stream_unprocessed), metric_length)) * np.nan
    pe = np.ones((len(stream_unprocessed), metric_length)) * np.nan
    fc = np.ones((len(stream_unprocessed), metric_length)) * np.nan
    fd = np.ones((len(stream_unprocessed), metric_length)) * np.nan
    fsd = np.ones((len(stream_unprocessed), metric_length)) * np.nan

    # Loop over time windows to calculate metrics
    for i, window_center in enumerate(window_centers):

        # Compute corresponding window end and slice stream
        window_start = window_center - window_length/2
        window_end = window_center + window_length/2
        stream_disp_segment = stream_disp.copy().trim(window_start, window_end)
        stream_vel_segment = stream_vel.copy().trim(window_start, window_end)

        # Loop over each trace and do computation
        for j in range(len(stream_disp_segment)):

            # Get trace data
            trace_disp_segment = stream_disp_segment[j].data
            trace_vel_segment = stream_vel_segment[j].data

            # Store matplotlib date for plotting
            tmpl[j,i] = window_center.matplotlib_date

            # Compute RSAM
            rsam[j,i] = np.mean(np.abs(trace_disp_segment))

            # Compute DR
            rms_disp = np.sqrt(np.mean(np.square(trace_disp_segment)))
            station_dist = GD((stream_disp_segment[j].stats.latitude, stream_disp_segment[j].stats.longitude), vlatlon).m
            wavenumber = 1500 / 2  # assume seisvel = 1500 m/s, dominant frequency = 2 Hz
            dr[j, i] = rms_disp * np.sqrt(station_dist) * np.sqrt(wavenumber) * 100 * 100  # cm^2

            # Compute permutation entropy
            pe[j,i] = complexity_entropy(trace_vel_segment, dx=5)[0]

            # Now execute FFT and trim
            fsamp = rfftfreq(len(trace_vel_segment), 1 / stream_vel_segment[j].stats.sampling_rate)
            fspec = np.abs(rfft(trace_vel_segment))[np.flatnonzero(fsamp>1)]
            fsamp = fsamp[np.flatnonzero(fsamp>1)]

            # Compute central frequency
            fc[j,i] = np.sum(fspec * fsamp) / np.sum(fspec)

            # Compute dominant frequency
            fd[j,i] = fsamp[np.argmax(fspec)]

            # Compute standard deviation of top 30 frequency peaks
            fpeaks_index, _ = find_peaks(fspec)
            fpeaks_top30 = np.sort(fspec[fpeaks_index])[-30:]
            fsd[j,i] = np.std(fpeaks_top30)

    return tmpl, rsam, dr, pe, fc, fd, fsd

def compute_pavlof_rsam(stream_unprocessed):
    """
    Pavlof rsam calculation function, written by Matt Haney and adapted by Darren Tan
    :param stream_unprocessed (:class:`~obspy.core.stream.Stream`): Input data (unprocessed -- response is removed within)
    :return: dr (list): List of reduced displacement values,
    """
    # Import geopy
    from geopy.distance import geodesic as GD
    # Define constants
    R = 6372.7976  # km
    drm = 3  # cm^2
    seisvel = 1500  # m/s
    dfrq = 2  # Hz
    vlatlon = (55.4173,-161.8937)
    # Initialize lists
    disteqv = []
    sensf = []
    rmssta = []
    # Compute
    for i, tr in enumerate(stream_unprocessed):
        slatlon = (tr.stats.latitude,tr.stats.longitude)
        disteqv.append(GD((tr.stats.latitude,tr.stats.longitude),vlatlon).km)
        sensf.append(tr.stats.response.instrument_sensitivity.value)
        rmssta.append(drm / (np.sqrt(disteqv[i]*1000) * np.sqrt(seisvel/dfrq)*100*100))  # in m
    rmsstav = np.array(rmssta)*2*np.pi*dfrq
    levels_count = rmsstav * sensf
    q_effect = np.exp(-(np.pi*dfrq*np.array(disteqv)*1000)/(seisvel*200))
    dr = levels_count * q_effect
    return dr

def rotate_NE_to_RT(stream, source_coord):

    # Import necessary packages
    import pyproj
    from math import atan2
    from obspy import Stream

    # Calculate incidence angle from source
    geodesic = pyproj.Geod(ellps='WGS84')
    source_azi, _, _ = geodesic.inv(stream[0].stats.longitude, stream[0].stats.latitude, source_coord[1], source_coord[0])

    # Extract components observed in stream
    all_comps = [tr.stats.channel[-1] for tr in stream]
    if len(set(all_comps)) not in [2, 3]:
        raise ValueError('The input stream does have a valid number of unique components. Either input a 2 (N,E) or 3 (N,E,Z) component stream.')
    trace_E = stream[all_comps.index('E')]
    trace_N = stream[all_comps.index('N')]
    if len(all_comps) == 3:
        trace_Z = stream[all_comps.index('Z')]

    # Get amplitude and azimuth of horizontal motion
    horiz_amp = np.linalg.norm(np.row_stack([trace_E.data, trace_N.data]), axis=0)
    azi_diff = (-1*np.arctan2(trace_N.data, trace_E.data)) + np.pi/2 - (source_azi * np.pi/180)

    # Now resolve to radial and tangential motion
    data_T = horiz_amp * np.sin(azi_diff)
    data_R = horiz_amp * np.cos(azi_diff) * -1  # take outward as positive

    # Create traces
    trace_T = trace_N.copy()
    trace_T.stats.channel = trace_N.stats.channel[:-1] + 'T'
    trace_T.data = data_T
    trace_R = trace_E.copy()
    trace_R.stats.channel = trace_E.stats.channel[:-1] + 'R'
    trace_R.data = data_R

    # Export stream
    if len(all_comps) == 3:
        return Stream([trace_Z, trace_T, trace_R])
    else:
        return Stream([trace_T, trace_R])


