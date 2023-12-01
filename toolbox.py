# Import dependencies
import time
import os
import glob
import pickle
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
from matplotlib import rcParams

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

def process_waveform(stream,remove_response=True,rr_output='VEL',detrend=False,taper_length=None,taper_percentage=None,filter_band=None,verbose=True):

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

    # If the input is a trace and not a stream, convert it to a stream and activate flag to return a trace later
    return_trace = False
    if type(stream) == Trace:
        stream = Stream([stream])
        return_trace = True

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
        stream.filter('bandpass', freqmin=filter_band[0], freqmax=filter_band[1], zerophase=False)
        if verbose:
            print('Waveform bandpass filtered from %.2f Hz to %.2f Hz.' % (filter_band[0],filter_band[1]))

    if return_trace:
        return stream[0]
    else:
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
            SPEC_THRESH = 0  # Power value indicative of gap
            colorbar_label = f'Power (dB rel. [{REFERENCE_VALUE * 1e6:g} µPa]$^2$ Hz$^{{-1}}$)'
            rescale_factor = 1  # Plot waveform in Pa
        else:
            # If seismic, define corresponding axis labels and reference values
            y_axis_label = 'Velocity (µm s$^{-1}$)'
            REFERENCE_VALUE = 1  # Velocity, in m/s
            SPEC_THRESH = -220  # Power value indicative of gap
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
                           vmin=np.percentile(spec_db_plot[spec_db_plot>SPEC_THRESH], v_percent_lims[0]),
                           vmax=np.percentile(spec_db_plot[spec_db_plot>SPEC_THRESH], v_percent_lims[1]),
                           origin='lower', aspect='auto', interpolation='None', cmap=cmap)
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
                           vmin=np.percentile(spec_db[spec_db>SPEC_THRESH], v_percent_lims[0]),
                           vmax=np.percentile(spec_db[spec_db>SPEC_THRESH], v_percent_lims[1]),
                           origin='lower', aspect='auto', interpolation='None', cmap=cmap)
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
            SPEC_THRESH = 0  # Power value indicative of gap
            rescale_factor = 1  # Plot waveform in Pa
        else:
            # If seismic, define corresponding reference values
            REFERENCE_VALUE = 1  # Velocity, in m/s
            SPEC_THRESH = -220  # Power value indicative of gap
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
                                      vmin=np.percentile(spec_db_plot[spec_db_plot>SPEC_THRESH], v_percent_lims[0]),
                                      vmax=np.percentile(spec_db_plot[spec_db_plot>SPEC_THRESH], v_percent_lims[1]),
                                      origin='lower', aspect='auto', interpolation='None', cmap=cmap)
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
                                      vmin=np.percentile(spec_db[spec_db>SPEC_THRESH], v_percent_lims[0]),
                                      vmax=np.percentile(spec_db[spec_db>SPEC_THRESH], v_percent_lims[1]),
                                      origin='lower', aspect='auto', interpolation='None', cmap=cmap)
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

def calculate_spectrogram(trace,starttime,endtime,window_duration,freq_lims,overlap=0.9,demean=False):

    """
    Calculates and returns a 2D matrix storing the spectrogram values (of the input trace) in decibels
    :param trace (:class:`~obspy.core.stream.Trace`): Input data
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for desired spectrogram
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for desired spectrogram
    :param window_duration (float): Duration of spectrogram window [s]
    :param freq_lims (tuple): Tuple of length 2 storing minimum frequency and maximum frequency for the spectrogram plot ([Hz],[Hz])
    :param overlap (float): Ratio of window to overlap when computing spectrogram (0 to 1)
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
                                                          noverlap=samples_per_segment * overlap)

    # Calculate UTC times corresponding to all segments
    trace_time_matplotlib = trace.stats.starttime.matplotlib_date + (segment_times / dates.SEC_PER_DAY)
    utc_times = np.array([UTCDateTime(dates.num2date(mpl_time)) for mpl_time in trace_time_matplotlib])

    # Find desired indices and trim spectrogram output
    desired_indices = np.flatnonzero([starttime <= t < endtime for t in utc_times])
    spec = spec[:,desired_indices]
    utc_times = utc_times[desired_indices]

    # Convert spectrogram matrix to decibels
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

    return spec_db, utc_times

def check_timeline(source,network,station,channel,location,starttime,endtime,model_path,meanvar_path,overlap,generate_fig=True,fig_width=32,fig_height=None,font_s=22,spec_kwargs=None,dr_kwargs=None,export_path=None,transparent=False):

    """
    Pulls data, then loads a trained model to predict the timeline of classes
    :param source (str): Which source to gather waveforms from (e.g. IRIS)
    :param network (str): SEED network code [wildcards (``*``, ``?``) accepted]
    :param station (str): SEED station code or comma separated station codes [wildcards NOT accepted]
    :param channel (str): SEED location code [wildcards (``*``, ``?``) accepted]
    :param location (str): SEED channel code [wildcards (``*``, ``?``) accepted]
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for data request
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for data request
    :param model_path (str): Path to model .h5 file
    :param meanvar_path (str): path to model's meanvar .npy file
    :param overlap (float): Percentage/ratio of overlap for successive spectrogram slices
    :param generate_fig (bool): If `True`, produce timeline figure, if `False`, return outputs without plots
    :param fig_width (float): Figure width [in]
    :param fig_height (float): Figure height [in] (if `None`, figure height = figure width * 0.75)
    :param font_s (float): Font size [points]
    :param spec_kwargs (dict): Dictionary of spectrogram plotting parameters (pad, window_duration, freq_lims, v_percent_lims)
    :param dr_kwargs (dict): Dictionary of reduced displacement plotting parameters (reference_station, filter_band, window_length, overlap, volc_lat, volc_lon, seis_vel, dominant_freq)
    :param export_path (str): (str or `None`): If str, export plotted figures as '.png' files, named by the trace id and time. If `None`, show figure in interactive python.
    :param transparent (bool): If `True`, export with transparent background
    :return: numpy.ndarray: 2D matrix storing all predicted classes (only returns if generate_fig==False or export_path==None)
    :return: numpy.ndarray: 2D matrix storing all predicted probabilities (only returns if generate_fig==False or export_path==None)
    """

    rcParams['font.size'] = font_s

    # Load model
    saved_model = load_model(model_path)
    nclasses = saved_model.layers[-1].get_config()['units']
    nsubrows = len(station.split(','))

    # Extract mean and variance from training
    saved_meanvar = np.load(meanvar_path)
    running_x_mean = saved_meanvar[0]
    running_x_var = saved_meanvar[1]

    # Define fixed values
    spec_height = saved_model.input.shape.as_list()[1]
    interval = saved_model.input.shape.as_list()[2]
    time_step = int(np.round(interval*(1-overlap)))
    spec_kwargs = {} if spec_kwargs is None else spec_kwargs
    pad = spec_kwargs['pad'] if 'pad' in spec_kwargs else 360
    window_duration = spec_kwargs['window_duration'] if 'window_duration' in\
        spec_kwargs else 10
    freq_lims = spec_kwargs['freq_lims'] if 'freq_lims' in spec_kwargs else \
        (0.5, 10)
    v_percent_lims = spec_kwargs['v_percent_lims'] if 'v_percent_lims' in\
        spec_kwargs else (20, 97.5)

    # Determine if infrasound
    infrasound = True if channel[-1] == 'F' else False
    reference_value = 20 * 10 ** -6 if infrasound else 1  # Pa for infrasound, m/s for seismic
    spec_thresh = 0 if infrasound else -220  # Power value indicative of gap

    # Enforce the duration to be a multiple of the model's time step
    if (endtime - starttime - interval) % time_step != 0:
        print('The desired analysis duration is not a multiple of the inbuilt time step.')
        endtime = endtime + (time_step - (endtime - starttime - interval) % time_step)
        print('Rounding up endtime to %s.' % str(endtime))

    # Load data, remove response, and re-order
    successfully_loaded = False
    load_starttime = time.time()
    while not successfully_loaded:
        try:
            stream_raw = gather_waveforms(source=source, network=network, station=station,
                                      location=location, channel=channel,
                                      starttime=starttime - pad, endtime=endtime + pad,
                                      verbose=False)
            stream = process_waveform(stream_raw.copy(), remove_response=True, detrend=False,
                                      taper_length=pad, verbose=False)
            successfully_loaded = True
        except:
            print('Data pull failed, trying again in 10 seconds...')
            time.sleep(10)
            load_currenttime = time.time()
            if load_currenttime-load_starttime < 60:
                pass
            else:
                raise Exception('Data pull timeout for starttime=%s, endtime=%s' % (str(starttime),str(endtime)))

    # reorder both raw and processed streams
    stream_default_order = [tr.stats.station for tr in stream]
    desired_index_order = [stream_default_order.index(stn) for stn in
                           station.split(',') if stn in stream_default_order]
    stream = Stream([stream[i] for i in desired_index_order])
    stream_raw = Stream([stream_raw[i] for i in desired_index_order])

    # If stream sampling rate is not an integer, fix
    for tr in stream:
        if tr.stats.sampling_rate != np.round(tr.stats.sampling_rate):
            tr.stats.sampling_rate = np.round(tr.stats.sampling_rate)

    # Initialize spectrogram slice and id stack
    spec_stack = []
    spec_ids = []

    # Loop over stations that have data
    stream_stations = [tr.stats.station for tr in stream]
    for j, stream_station in enumerate(stream_stations):

        # Choose trace corresponding to station
        trace = stream[j]

        # Calculate spectrogram power matrix
        spec_db, utc_times = calculate_spectrogram(trace, starttime, endtime,
                                                   window_duration=window_duration,
                                                   freq_lims=freq_lims)

        # Reshape spectrogram into stack of desired spectrogram slices
        spec_slices = [spec_db[:, t:(t + interval)] for t in
                       range(0, spec_db.shape[-1] - interval + 1, time_step)]
        spec_tags = [stream_station + '_' + step_bound.strftime('%Y%m%d%H%M%S') + '_' + \
                     (step_bound + interval).strftime('%Y%m%d%H%M%S') \
                     for step_bound in np.arange(starttime,endtime-interval+1,time_step)]
        spec_stack += spec_slices
        spec_ids += spec_tags

    # Convert spectrogram slices to an array
    spec_stack = np.array(spec_stack)
    spec_ids = np.array(spec_ids)

    # If there are spectrogram slices
    if len(spec_stack) != 0:
        # Remove spectrograms with data gap
        keep_index = np.where(np.sum(spec_stack<spec_thresh, axis=(1,2)) < (0.2 * interval))
        spec_stack = spec_stack[keep_index]
        spec_ids = spec_ids[keep_index]

        # Standardize and min-max scale
        spec_stack = (spec_stack - running_x_mean) / np.sqrt(running_x_var + 1e-5)
        spec_stack = (spec_stack - np.min(spec_stack, axis=(1, 2))[:, np.newaxis, np.newaxis]) / \
                     (np.max(spec_stack, axis=(1, 2)) - np.min(spec_stack, axis=(1, 2)))[:, np.newaxis, np.newaxis]

    # Otherwise, return empty class and probability matrix or raise Exception
    else:
        if not generate_fig:
            matrix_length = int(np.ceil((endtime - starttime - interval) / time_step)) + 1
            class_mat = np.ones((nsubrows + 1, matrix_length)) * 6
            prob_mat = np.vstack((np.zeros((nsubrows, matrix_length)), np.ones((1, matrix_length)) * np.nan))
            return class_mat, prob_mat
        else:
            raise Exception('No data available for desired timeline!')

    # Make predictions
    spec_predictions = saved_model.predict(spec_stack)
    predicted_labels = np.argmax(spec_predictions, axis=1)
    indicators = []
    for i, spec_id in enumerate(spec_ids):
        chunks = spec_id.split('_')
        indicators.append([chunks[0], UTCDateTime(chunks[1]) + np.round(interval / 2),
                           predicted_labels[i], spec_predictions[i, :]])

    # Craft plotting matrix and probability matrix
    matrix_length = int(np.ceil((endtime - starttime - interval) / time_step)) + 1
    matrix_height = nsubrows
    matrix_plot = np.ones((matrix_height, matrix_length)) * nclasses
    matrix_probs = np.zeros((matrix_height, matrix_length, np.shape(spec_predictions)[1]))
    for indicator in indicators:
        row_index = station.split(',').index(indicator[0])
        col_index = int((indicator[1] - np.round(interval / 2) - starttime) / time_step)
        matrix_plot[row_index, col_index] = indicator[2]
        matrix_probs[row_index, col_index, :] = indicator[3]

    # Add voting row
    np.seterr(divide='ignore', invalid='ignore')  # Mute division by zero error
    na_label = nclasses  # this index is one higher than the number of classes
    new_row = np.ones((1, np.shape(matrix_plot)[1])) * na_label
    matrix_plot = np.concatenate((matrix_plot, new_row))

    # Sum different station probabilities and derive network max
    matrix_probs_sum = np.sum(matrix_probs, axis=0)
    matrix_contributing_station_count = np.sum(np.sum(matrix_probs, axis=2) != 0, axis=0)
    voted_labels = np.argmax(matrix_probs_sum, axis=1)
    voted_labels[matrix_contributing_station_count==0] = na_label
    voted_probabilities = np.max(matrix_probs_sum, axis=1) / matrix_contributing_station_count  # normalize by number of stations
    matrix_plot = np.concatenate((matrix_plot, np.reshape(voted_labels, (1, np.shape(matrix_plot)[1]))))

    # Return class and probability outputs if figure plotting is not desired
    if not generate_fig:
        class_mat = np.vstack((matrix_plot[:-2,:],matrix_plot[-1:,:]))
        prob_mat = np.vstack((np.max(matrix_probs, axis=2),voted_probabilities))
        return class_mat, prob_mat

    # If dealing with seismic, use seismic voting scheme
    if not infrasound:
        # Craft corresponding rgb values
        if nclasses == 6:
            rgb_values = np.array([
                [193, 39, 45],
                [0, 129, 118],
                [0, 0, 167],
                [238, 204, 22],
                [164, 98, 0],
                [40, 40, 40],
                [255, 255, 255]])
            rgb_keys = ['Broadband\nTremor',
                        'Harmonic\nTremor',
                        'Monochromatic\nTremor',
                        'Earthquake',
                        'Explosion',
                        'Noise',
                        'N/A']
        elif nclasses == 7:
            rgb_values = np.array([
                [193, 39, 45],
                [0, 129, 118],
                [0, 0, 167],
                [238, 204, 22],
                [103, 72, 132],
                [164, 98, 0],
                [40, 40, 40],
                [255, 255, 255]])
            rgb_keys = ['Broadband\nTremor',
                        'Harmonic\nTremor',
                        'Monochromatic\nTremor',
                        'Earthquake',
                        'Long\nPeriod',
                        'Explosion',
                        'Noise',
                        'N/A']
    else:
        # Craft corresponding rgb values
        rgb_values = np.array([
            [103, 52, 235],
            [235, 152, 52],
            [40, 40, 40],
            [15, 37, 60],
            [255, 255, 255]])
        rgb_keys = ['Infrasonic\nTremor',
                    'Explosion',
                    'Wind\nNoise',
                    'Electronic\nNoise',
                    'N/A']

    # Craft color map
    rgb_ratios = rgb_values / 255
    colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0], 1))),
                            axis=1)
    cmap = ListedColormap(colors)

    # Define colorbar keywords for plotting
    real_cbar_tick_interval = 2 * nclasses / (2 * np.shape(rgb_values)[0])
    real_cbar_ticks = np.arange(real_cbar_tick_interval / 2, nclasses,
                                real_cbar_tick_interval)
    cbar_kws = {'ticks': real_cbar_ticks,
                'drawedges': True,
                'location': 'top',
                'fraction': 0.15,
                'aspect': 40}

    # Define plotting parameters based on reduced displacement kwargs
    header_panels = 3 if dr_kwargs is not None else 2
    top_space = 0.905 if dr_kwargs is not None else 0.89

    LW = 0.75
    LW_LABEL = 2

    # Configure shared x-axis ticks and labels
    if (endtime - starttime) >= (6 * 86400):
        denominator = (endtime - starttime) / 86400
        fmt = '%m/%d %H:%M'
    elif (2 * 86400) <= (endtime - starttime) < (6 * 86400):
        denominator = 2 * (endtime - starttime) / 86400
        fmt = '%m/%d %H:%M'
    elif (endtime - starttime) < (2 * 86400) and endtime.date != starttime.date:
        fmt = '%m/%d %H:%M'
        denominator = 12 if ((endtime - starttime) % 1800 == 0) else 10
    else:
        fmt = '%H:%M'
        denominator = 12 if ((endtime - starttime) % 1800 == 0) else 10

    # Initialize figure and craft axes
    fig_height = fig_height if fig_height else (fig_width * .75)
    fig = plt.figure(figsize=(fig_width, fig_height))
    height_ratios = np.ones(len(station.split(',')) + header_panels)
    height_ratios[1:header_panels] = 0.5
    gs_top = plt.GridSpec(len(station.split(',')) + header_panels, 2, top=top_space,
                          height_ratios=height_ratios, width_ratios=[35, 1],
                          wspace=0.05)
    gs_base = plt.GridSpec(len(station.split(',')) + header_panels, 2, hspace=0,
                           height_ratios=height_ratios, width_ratios=[35, 1],
                           wspace=0.05)
    cbar_ax = fig.add_subplot(gs_top[:, 1])
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax2 = fig.add_subplot(gs_top[1, 0])
    ax2b = fig.add_subplot(gs_top[2, 0]) if dr_kwargs is not None else None
    ax3 = fig.add_subplot(gs_base[header_panels, 0])
    other_axes = [fig.add_subplot(gs_base[i, 0], sharex=ax3) for i in
                  range(header_panels + 1, len(station.split(',')) + header_panels)]
    axs = [ax3] + other_axes
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # Plot prediction heatmap in top axis
    sns.heatmap(matrix_plot, cmap=cmap, cbar=False, cbar_kws=cbar_kws,
                alpha=0.8, vmin=0, vmax=nclasses, ax=ax1)
    ax1.set_xticks([])
    ax1.axhline(nsubrows, color='black')
    ax1.axhline(nsubrows + 1, color='black')
    ax1.axhspan(nsubrows, nsubrows + 1, facecolor='white')
    yticks = np.concatenate((np.arange(0.5, nsubrows, 1),
                             np.array([nsubrows + 1.5])))
    ax1.set_yticks(yticks)
    yticklabels = station.split(',').copy()
    yticklabels.append('VOTE')
    ax1.set_yticklabels(yticklabels, rotation=0, fontsize=font_s)
    ax1.set_ylim([len(station.split(',')) + 2, 0])
    ax1.patch.set_edgecolor('black')
    ax1.patch.set_linewidth(LW)
    ax1.set_title('Station-based Voting', fontsize=font_s + 2)

    # Plot probabilities in middle axis
    prob_xvec = np.arange(0.5, len(voted_probabilities) + 0.5, 1)
    ax2.plot(prob_xvec, voted_probabilities, color='k', linewidth=LW)
    ax2.fill_between(prob_xvec, voted_probabilities, where=voted_probabilities >= 0,
                     interpolate=True, color='gray', alpha=0.5)
    ax2.set_xlim([0, len(voted_probabilities)])
    ax2.set_xticks(np.linspace(0, len(voted_probabilities), int(denominator+1)))
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', labelsize=font_s)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_ylabel('$P_{norm}$', fontsize=font_s)

    # If dr_kwargs is not None, plot reduced displacement in middle axis
    if dr_kwargs is not None:
        # Calculate reduce displacement
        print(f"Calculating DR for {stream_raw[stream_stations.index(dr_kwargs['reference_station'])].id}")
        tr_disp = process_waveform(stream_raw[stream_stations.index(dr_kwargs['reference_station'])].copy(),
                                   remove_response=True, rr_output='DISP', detrend=False, taper_length=60,
                                   taper_percentage=None, filter_band=dr_kwargs['filter_band'], verbose=False)
        tr_disp_trimmed = tr_disp.trim(starttime=starttime - dr_kwargs['window_length'] / 2,
                                       endtime=endtime + dr_kwargs['window_length'] / 2)
        window_samples = int(dr_kwargs['window_length'] * tr_disp.stats.sampling_rate)
        tr_disp_segments = [tr_disp_trimmed.data[i:i + window_samples] for i in
                            range(0, len(tr_disp_trimmed.data) - window_samples + 1,
                                  int(window_samples * dr_kwargs['overlap']))]
        rms_disp = np.array([np.sqrt(np.mean(np.square(tr_disp_segment))) for tr_disp_segment in tr_disp_segments])
        station_dist = GD((tr_disp.stats.latitude, tr_disp.stats.longitude),
                          (dr_kwargs['volc_lat'], dr_kwargs['volc_lon'])).m
        wavenumber = dr_kwargs['seis_vel'] / dr_kwargs['dominant_freq']
        dr = rms_disp * np.sqrt(station_dist) * np.sqrt(wavenumber) * 100 * 100  # cm^2

        # Plot reduced displacement
        dr_tvec = np.arange(0.5, len(dr) + 0.5, 1)
        ax2b.plot(dr_tvec, dr, color='k', linewidth=LW)
        ax2b.set_xlim(0, len(dr))
        ax2b.set_xticks(np.linspace(0, len(dr), int(denominator+1)))
        plt.setp(ax2b.get_xticklabels(), visible=False)
        ax2b.set_ylim([0, np.ceil(np.max(dr))])
        ax2b.tick_params(axis='y', labelsize=font_s)
        ax2b.set_yticks(np.linspace(0, np.ceil(np.max(dr)), 3))
        ax2b.set_ylabel('$D_R (cm^2)$\n' + dr_kwargs['reference_station'], fontsize=font_s)

    # Loop over input stations and plot spectrograms on lower axes
    for axs_index, stn in enumerate(station.split(',')):

        # If corresponding trace exists, plot spectrogram
        if stn in stream_stations:

            # Extract trace information for FFT
            trace = stream[stream_stations.index(stn)]
            sampling_rate = trace.stats.sampling_rate
            samples_per_segment = int(window_duration * sampling_rate)

            # Compute spectrogram (Note that overlap is 90% of samples_per_segment)
            sample_frequencies, segment_times, spec = spectrogram(trace.data,
                                                                  sampling_rate,
                                                                  window='hann',
                                                                  scaling='density',
                                                                  nperseg=samples_per_segment,
                                                                  noverlap=samples_per_segment * .9)

            # Convert spectrogram matrix to decibels for plotting
            spec_db = 10 * np.log10(abs(spec) / (reference_value ** 2))

            # Convert trace times to matplotlib dates
            trace_time_matplotlib = trace.stats.starttime.matplotlib_date + \
                                    (segment_times / dates.SEC_PER_DAY)

            # Determine frequency limits and trim spec_db
            spec_db_plot = spec_db[
                           np.flatnonzero((sample_frequencies > freq_lims[0]) & \
                                          (sample_frequencies < freq_lims[1])), :]
            axs[axs_index].imshow(spec_db,
                                  extent=[trace_time_matplotlib[0],
                                          trace_time_matplotlib[-1],
                                          sample_frequencies[0],
                                          sample_frequencies[-1]],
                                  vmin=np.percentile(spec_db_plot[spec_db_plot > spec_thresh], v_percent_lims[0]),
                                  vmax=np.percentile(spec_db_plot[spec_db_plot > spec_thresh], v_percent_lims[1]),
                                  origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)

            # Label y-axis with trace information
            axs[axs_index].set_ylabel(stream[stream_stations.index(stn)].id, fontsize=font_s, fontweight='bold')

        else:
            # If corresponding trace does not exist, label station as 'No Data'
            axs[axs_index].set_ylabel(stn + ' (No Data)', fontsize=font_s, fontweight='bold')

        # Tidy up axes
        axs[axs_index].set_ylim([freq_lims[0], freq_lims[1]])
        axs[axs_index].set_yticks(range(2, freq_lims[1] + 1, int(freq_lims[1] / 5)))
        axs[axs_index].set_xlim([starttime.matplotlib_date,
                                 endtime.matplotlib_date])
        axs[axs_index].tick_params(axis='y', labelsize=font_s)

    # Format time ticks
    time_tick_list = np.arange(starttime, endtime + 1, (endtime - starttime) \
                               / denominator)
    time_tick_list_mpl = [t.matplotlib_date for t in time_tick_list]
    time_tick_labels = [time.strftime(fmt) for time in time_tick_list]
    axs[-1].set_xticks(time_tick_list_mpl)
    axs[-1].set_xticklabels(time_tick_labels, fontsize=font_s, rotation=30)
    if endtime.date == starttime.date:
        axs[-1].set_xlabel('UTC Time on ' + starttime.date.strftime('%b %d, %Y'), \
                           fontsize=font_s)
    elif (endtime - starttime) < (2 * 86400):
        axs[-1].set_xlabel('UTC Time starting from ' +
                           starttime.date.strftime('%b %d, %Y'),
                           fontsize=font_s)
    else:
        axs[-1].set_xlabel('UTC Time', fontsize=font_s)

    # Plot colorbar
    for i, rgb_ratio in enumerate(rgb_ratios):
        cbar_ax.axhspan(i, i + 1, color=rgb_ratio)
    cbar_ax.set_yticks(np.arange(0.5, len(rgb_ratios) + 0.5, 1))
    cbar_ax.set_yticklabels(rgb_keys, fontsize=font_s)
    cbar_ax.yaxis.tick_right()
    cbar_ax.set_ylim([0, len(rgb_ratios)])
    cbar_ax.invert_yaxis()
    cbar_ax.set_xticks([])

    # Show figure or export
    class_mat = np.vstack((matrix_plot[:-2, :], matrix_plot[-1:, :]))
    prob_mat = np.vstack((np.max(matrix_probs, axis=2), voted_probabilities))
    if export_path is None:
        fig.show()
        print('Done!')
    else:
        file_label = starttime.strftime('%Y%m%d_%H%M') + '__' +\
            endtime.strftime('%Y%m%d_%H%M') + '_' +\
                model_path.split('/')[-1].split('.')[0]
        fig.savefig(export_path + file_label + '.png', bbox_inches='tight',
                    transparent=transparent)
        print('Done!')
    return class_mat, prob_mat

def generate_timeline_indicators(source,network,station,channel,location,starttime,endtime,model_path,meanvar_path,overlap,spec_kwargs=None,export_path=None):

    """
    Pulls data or leverage npy directory to generate list of timeline indicators
    :param source (str): Which source to gather waveforms from (e.g. IRIS)
    :param network (str): SEED network code [wildcards (``*``, ``?``) accepted]
    :param station (str): SEED station code or comma separated station codes [wildcards NOT accepted]
    :param channel (str): SEED location code [wildcards (``*``, ``?``) accepted]
    :param location (str): SEED channel code [wildcards (``*``, ``?``) accepted]
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for data request
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for data request
    :param model_path (str): Path to model .h5 file
    :param meanvar_path (str): path to model's meanvar .npy file
    :param overlap (float): Percentage/ratio of overlap for successive spectrogram slices
    :param spec_kwargs (dict): Dictionary of spectrogram plotting parameters (pad, window_duration, freq_lims, v_percent_lims)
    :param export_path (str): (str or `None`): If str, export indicators in a .pkl with the full filepath export_path + 'indicators.pkl'
    """

    # Enforce the duration to be a multiple of 14400s (4 hours) as we process timelines by 4 hour time steps
    if (endtime - starttime) % 14400 != 0:
        print('The desired analysis duration (endtime - starttime) is not a multiple of 14400s (4 hours).')
        endtime = endtime + (14400 - (endtime - starttime) % 14400)
        print('Rounding up endtime to %s.' % str(endtime))

    # Load model
    saved_model = load_model(model_path)
    spec_height = saved_model.input.shape.as_list()[1]
    interval = saved_model.input.shape.as_list()[2]
    nclasses = saved_model.layers[-1].get_config()['units']
    nsubrows = len(station.split(','))

    # Extract mean and variance from training
    saved_meanvar = np.load(meanvar_path)
    running_x_mean = saved_meanvar[0]
    running_x_var = saved_meanvar[1]

    # Define fixed values
    time_step = int(np.round(interval*(1-overlap)))
    spec_kwargs = {} if spec_kwargs is None else spec_kwargs
    pad = spec_kwargs['pad'] if 'pad' in spec_kwargs else 360
    window_duration = spec_kwargs['window_duration'] if 'window_duration' in spec_kwargs else 10
    freq_lims = spec_kwargs['freq_lims'] if 'freq_lims' in spec_kwargs else (0.5, 10)

    # Determine if infrasound
    infrasound = True if channel[-1] == 'F' else False
    spec_thresh = 0 if infrasound else -220  # Power value indicative of gap

    # Split analysis duration into 4h-long chunks and start timer
    num4h = (endtime - starttime) / 14400
    process_tstart = time.time()

    # Initialize master indicator list
    indicators = []

    # Loop over days and run checker
    for n in range(int(num4h)):

        # Determine start and end time of current step
        t1 = starttime + n * 14400 - 2 * 60
        if n == (num4h-1):
            t2 = starttime + (n + 1) * 14400 + 2 * 60
        else:
            t2 = starttime + (n + 1) * 14400 + 2 * 60 - time_step
        print('Now at %s, time elapsed: %.2f hours' %
              ((t1+2*60).strftime('%Y-%m-%dT%H:%M:%S'),(time.time()-process_tstart)/3600))

        # Load data, remove response, and re-order
        successfully_loaded = False
        load_starttime = time.time()
        while not successfully_loaded:
            try:
                stream = gather_waveforms(source=source, network=network, station=station,
                                          location=location, channel=channel,
                                          starttime=t1 - pad, endtime=t2 + pad,
                                          verbose=False)
                stream = process_waveform(stream, remove_response=True, detrend=False,
                                          taper_length=pad, verbose=False)
                successfully_loaded = True
            except:
                print('Data pull failed, trying again in 10 seconds...')
                time.sleep(10)
                load_currenttime = time.time()
                if load_currenttime-load_starttime < 180:
                    pass
                else:
                    raise Exception('Data pull timeout for t1=%s, t2=%s' % (str(t1),str(t2)))
        stream_default_order = [tr.stats.station for tr in stream]
        desired_index_order = [stream_default_order.index(stn) for stn in
                               station.split(',') if stn in stream_default_order]
        stream = Stream([stream[i] for i in desired_index_order])

        # If stream sampling rate is not an integer, fix
        for tr in stream:
            if tr.stats.sampling_rate != np.round(tr.stats.sampling_rate):
                tr.stats.sampling_rate = np.round(tr.stats.sampling_rate)

        # Initialize spectrogram slice and id stack
        spec_stack = []
        spec_ids = []

        # Loop over stations that have data
        stream_stations = [tr.stats.station for tr in stream]
        for j, stream_station in enumerate(stream_stations):

            # Choose trace corresponding to station
            trace = stream[j]

            # Calculate spectrogram power matrix
            spec_db, utc_times = calculate_spectrogram(trace, t1, t2,
                                                       window_duration=window_duration,
                                                       freq_lims=freq_lims)

            # Reshape spectrogram into stack of desired spectrogram slices
            spec_slices = [spec_db[:, t:(t + interval)] for t in
                           range(0, spec_db.shape[-1] - interval + 1, time_step)]
            spec_tags = [stream_station + '_' + step_bound.strftime('%Y%m%d%H%M%S') + '_' + \
                         (step_bound + interval).strftime('%Y%m%d%H%M%S') \
                         for step_bound in np.arange(t1, t2 - interval + 1, time_step)]
            spec_stack += spec_slices
            spec_ids += spec_tags

        # Convert spectrogram slices to an array
        spec_stack = np.array(spec_stack)
        spec_ids = np.array(spec_ids)

        # If there are spectrogram slices
        if len(spec_stack) != 0:

            # Remove spectrograms with data gap
            keep_index = np.where(np.sum(spec_stack<spec_thresh, axis=(1,2)) < (0.2 * interval))
            spec_stack = spec_stack[keep_index]
            spec_ids = spec_ids[keep_index]

            # If no spectrograms pass the gap check, continue to next time step
            if len(spec_ids) == 0:
                continue

            # Standardize and min-max scale
            spec_stack = (spec_stack - running_x_mean) / np.sqrt(running_x_var + 1e-5)
            spec_stack = (spec_stack - np.min(spec_stack, axis=(1, 2))[:, np.newaxis, np.newaxis]) / \
                         (np.max(spec_stack, axis=(1, 2)) - np.min(spec_stack, axis=(1, 2)))[:, np.newaxis, np.newaxis]

            # Make predictions
            spec_predictions = saved_model.predict(spec_stack)
            predicted_labels = np.argmax(spec_predictions, axis=1)
            for i, spec_id in enumerate(spec_ids):
                chunks = spec_id.split('_')
                indicators.append([chunks[0], UTCDateTime(chunks[1]) + int(np.round(interval / 2)),
                                   predicted_labels[i], spec_predictions[i, :]])

            # Save as pickle for each step
            if export_path is None:
                with open('./indicators.pkl', 'wb') as f:
                    pickle.dump(indicators, f)
            else:
                if export_path[-4:] != '.pkl':
                    with open(export_path + '.pkl', 'wb') as f:
                        pickle.dump(indicators, f)
                else:
                    with open(export_path, 'wb') as f:
                        pickle.dump(indicators, f)

        # If not, continue to next time step
        else:
            continue

def plot_timeline(starttime, endtime, time_step, type, model_path, indicators_path, plot_title,
                  export_path=None, transparent=False, fig_width=None, fig_height=None, font_s=18,
                  plot_stations=None, plot_labels=False, labels_kwargs=None):
    """
    Plot timeline figure showing station-specific and probability-sum voting by monthly rows
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for timeline
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for timeline
    :param time_step (float): time step used to divide plot into columns
    :param type (str): defined as either 'seismic' or 'infrasound' to determine color palette
    :param model_path (str): path to model .h5 file
    :param indicators_path (str): path to pkl file that stores timeline indicators for plotting
    :param plot_title (str): plot title to use
    :param export_path (str): filepath to export figures (condensed plot will tag "_condensed" to filename)
    :param transparent (bool): if `True`, export figures with transparent background
    :param fig_width (float): Figure width [in]
    :param fig_height (float): Figure height [in]
    :param font_s (int): font size for figure
    :param plot_stations (str): comma-delimited string of stations to include in plot, in that order
    :param plot_labels (bool): if `True`, plot manual timeframe with manual labels. Requires `labels_kwargs`.
    :param labels_kwargs (dict): dictionary with the keys `start_date` (UTCDateTime), `end_date` (UTCDateTime) and `labels_dir` (str)
    :return: None
    """

    print('Plotting timeline...')

    # Load indicators
    with open(indicators_path, 'rb') as f:  # Unpickling
        indicators = pickle.load(f)

    # Determine number of stations and filter indicator list if needed
    if plot_stations:
        stations = plot_stations.split(',')
        indicators = [indicator for indicator in indicators if indicator[0] in stations]
    else:
        stations = list(np.unique([i[0] for i in indicators]))
    nsubrows = len(stations)

    # Filter indicator list by time
    indicators = [indicator for indicator in indicators if (starttime<=indicator[1]<=endtime)]

    # Determine number of months
    month_list = []
    month_utcdate = UTCDateTime(starttime.year,starttime.month,1)
    while month_utcdate <= endtime:
        month_list.append(month_utcdate.strftime('%b \'%y'))
        if month_utcdate.month+1 <= 12:
            month_utcdate += UTCDateTime(month_utcdate.year,month_utcdate.month+1,1) - month_utcdate
        else:
            month_utcdate += UTCDateTime(month_utcdate.year+1, 1, 1) - month_utcdate
    nmonths = len(month_list)

    # Load model to determine number of classes
    saved_model = load_model(model_path)
    nclasses = saved_model.layers[-1].get_config()['units']
    na_label = nclasses

    # Define fixed params
    TICK_DAYS = [0, 7, 14, 28]

    # Craft unlabeled matrix and store probabilities
    matrix_length = int(31 * (86400 / time_step))
    matrix_height = nmonths * nsubrows
    matrix_plot = np.ones((matrix_height, matrix_length)) * na_label
    matrix_probs = np.zeros((matrix_height, matrix_length, nclasses))
    for indicator in indicators:
        utc = indicator[1]
        row_index = nsubrows * month_list.index(utc.strftime('%b \'%y')) + stations.index(indicator[0])
        col_index = int((indicator[1] - UTCDateTime(utc.year, utc.month, 1)) / time_step)
        matrix_plot[row_index, col_index] = indicator[2]
        matrix_probs[row_index, col_index, :] = indicator[3]

    # Craft labeled matrix
    if plot_labels:
        labeled_start_index = int((labels_kwargs['start_date'] - UTCDateTime(labels_kwargs['start_date'].year,
                                                                             labels_kwargs['start_date'].month,
                                                                             1)) / time_step)
        labeled_end_index = int((labels_kwargs['end_date'] - UTCDateTime(labels_kwargs['end_date'].year,
                                                                         labels_kwargs['end_date'].month,
                                                                         1)) / time_step)
        labeled_start_row = int(np.floor((labels_kwargs['start_date'] - starttime) / (31 * 86400)))
        labeled_end_row = int(np.floor((labels_kwargs['end_date'] - starttime) / (31 * 86400)))
        labeled_matrix_plot = np.ones((matrix_height, matrix_length)) * na_label
        labeled_spec_paths = glob.glob(labels_kwargs['labels_dir'] + '*.npy')
        labeled_indicators = []
        for i, filepath in enumerate(labeled_spec_paths):
            filename = filepath.split('/')[-1]
            chunks = filename.split('_')
            labeled_indicators.append([chunks[0], UTCDateTime(chunks[1]), int(chunks[3][0])])
        for labeled_indicator in labeled_indicators:
            utc = labeled_indicator[1]
            row_index = nsubrows * month_list.index(utc.strftime('%b \'%y')) + stations.index(labeled_indicator[0])
            col_index = int((labeled_indicator[1] - UTCDateTime(utc.year, utc.month, 1)) / time_step)
            labeled_matrix_plot[row_index, col_index:col_index + int(240 / time_step)] = labeled_indicator[2]

    # Choose color palette depending on data type and nclasses
    if type == 'seismic':
        # Craft corresponding rgb values
        if nclasses == 6:
            rgb_values = np.array([
                [193, 39, 45],
                [0, 129, 118],
                [0, 0, 167],
                [238, 204, 22],
                [164, 98, 0],
                [40, 40, 40],
                [255, 255, 255]])
            rgb_keys = ['Broadband\nTremor',
                        'Harmonic\nTremor',
                        'Monochromatic\nTremor',
                        'Earthquake',
                        'Explosion',
                        'Noise',
                        'N/A']
        elif nclasses == 7:
            rgb_values = np.array([
                [193, 39, 45],
                [0, 129, 118],
                [0, 0, 167],
                [238, 204, 22],
                [103, 72, 132],
                [164, 98, 0],
                [40, 40, 40],
                [255, 255, 255]])
            rgb_keys = ['Broadband\nTremor',
                        'Harmonic\nTremor',
                        'Monochromatic\nTremor',
                        'Earthquake',
                        'Long\nPeriod',
                        'Explosion',
                        'Noise',
                        'N/A']
    else:
        # Craft corresponding rgb values
        rgb_values = np.array([
            [103, 52, 235],
            [235, 152, 52],
            [40, 40, 40],
            [15, 37, 60],
            [255, 255, 255]])
        rgb_keys = ['Infrasonic\nTremor',
                    'Explosion',
                    'Wind\nNoise',
                    'Electronic\nNoise',
                    'N/A']
    rgb_ratios = rgb_values / 255
    colors = np.concatenate((rgb_ratios, np.ones((np.shape(rgb_values)[0], 1))), axis=1)
    cmap = ListedColormap(colors)
    if plot_labels:
        colors[-1][-1] = 0
        labeled_cmap = ListedColormap(colors)

    # Colorbar keywords
    real_cbar_tick_interval = 2 * (len(np.unique(matrix_plot)) - 1) / (2 * np.shape(rgb_values)[0])
    real_cbar_ticks = np.arange(real_cbar_tick_interval / 2, len(np.unique(matrix_plot)) - 1, real_cbar_tick_interval)
    cbar_kws = {'ticks': real_cbar_ticks,
                'drawedges': True,
                'aspect': 30}

    # Define linewidths
    LW = 0.75
    LW_LABEL = 2

    # Craft timeline figure
    fig_width = fig_width if fig_width else 20
    fig_height = fig_height if fig_height else (nmonths * nsubrows / 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(matrix_plot, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.8, vmin=0, vmax=nclasses)
    if plot_labels:
        sns.heatmap(labeled_matrix_plot, cmap=labeled_cmap, cbar=False)
    cbar = ax.collections[0].colorbar
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(LW)
    cbar.ax.set_yticklabels(rgb_keys, fontsize=font_s)
    cbar.ax.invert_yaxis()
    for y in range(0, matrix_height, nsubrows):
        ax.axhline(y=y, color='black')
        for i, station in enumerate(stations):
            ax.text(100, y + i + 1, station, fontsize=font_s-7)
    if plot_labels:
        ax.plot([labeled_start_index, matrix_length], [labeled_start_row * nsubrows, labeled_start_row * nsubrows],
                'k-', linewidth=LW_LABEL)
        ax.plot([labeled_start_index, matrix_length],
                [(labeled_start_row + 1) * nsubrows, (labeled_start_row + 1) * nsubrows], 'k-', linewidth=LW_LABEL)
        for r in range(labeled_start_row, labeled_end_row):
            ax.plot([0, matrix_length], [(r + 1) * nsubrows, (r + 1) * nsubrows], 'k-', linewidth=LW_LABEL)
        ax.plot([0, labeled_end_index], [labeled_end_row * nsubrows, labeled_end_row * nsubrows], 'k-', linewidth=LW_LABEL)
        ax.plot([0, labeled_end_index], [(labeled_end_row + 1) * nsubrows, (labeled_end_row + 1) * nsubrows], 'k-',
                linewidth=LW_LABEL)
    ax.set_yticks(np.arange(nsubrows / 2, matrix_height, nsubrows))
    ax.set_yticklabels(month_list, rotation=0, fontsize=font_s)
    ax.set_xticks(np.array([0, 7, 14, 21, 28]) * (86400 / time_step))
    ax.set_xticklabels([0, 7, 14, 21, 28], rotation=0, fontsize=font_s)
    ax.set_xlabel('Date', fontsize=font_s-2)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(LW)
    ax.set_title(plot_title, fontsize=font_s+7)
    if export_path:
        plt.savefig(export_path, bbox_inches='tight', transparent=transparent)
    else:
        fig.show()

    # Condense timeline figure using probability sum and majority voting
    matrix_condensed = np.ones((nmonths, matrix_length)) * na_label
    matrix_pnorm = np.zeros((nmonths, matrix_length))
    if plot_labels:
        labeled_matrix_condensed = np.ones((nmonths, matrix_length)) * na_label
    for i in range(nmonths):

        # Sum probabilities to find best class and store pnorm
        sub_probs = matrix_probs[nsubrows * i:nsubrows * i + nsubrows, :, :]
        sub_probs_sum = np.sum(sub_probs, axis=0)
        sub_probs_contributing_station_count = np.sum(np.sum(sub_probs, axis=2) != 0, axis=0)
        matrix_condensed[i, :] = np.argmax(sub_probs_sum, axis=1)
        matrix_condensed[i, :][sub_probs_contributing_station_count == 0] = na_label
        matrix_pnorm[i, :] = np.max(sub_probs_sum, axis=1) / sub_probs_contributing_station_count

        # Use majority voting to condense manual labels
        if plot_labels:
            for j in range(matrix_length):
                sub_col = labeled_matrix_plot[nsubrows * i:nsubrows * i + nsubrows, j]
                labels_seen, label_counts = np.unique(sub_col, return_counts=True)
                if len(labels_seen) == 1 and na_label in labels_seen:
                    labeled_matrix_condensed[i, j] = na_label
                elif len(labels_seen) == 1:
                    labeled_matrix_condensed[i, j] = labels_seen[0]
                else:
                    if na_label in labels_seen:
                        label_counts = np.delete(label_counts, labels_seen == na_label)
                        labels_seen = np.delete(labels_seen, labels_seen == na_label)
                    selected_label_index = np.argwhere(label_counts == np.amax(label_counts))[-1][0]
                    labeled_matrix_condensed[i, j] = labels_seen[selected_label_index]

    cbar_kws = {'ticks': real_cbar_ticks,
                'drawedges': True,
                'aspect': 30}

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(matrix_condensed, cmap=cmap, cbar=True, cbar_kws=cbar_kws, alpha=0.8)
    if plot_labels:
        sns.heatmap(labeled_matrix_condensed, cmap=labeled_cmap, cbar=False)
    cbar = ax.collections[0].colorbar
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(LW)
    cbar.ax.set_yticklabels(rgb_keys, fontsize=font_s)
    cbar.ax.invert_yaxis()
    for y in range(0, nmonths):
        ax.axhline(y=y, color='black')
    if plot_labels:
        ax.plot([labeled_start_index, matrix_length], [labeled_start_row, labeled_start_row], 'k-', linewidth=5.5)
        ax.plot([labeled_start_index, matrix_length], [labeled_start_row + 1, labeled_start_row + 1], 'k-',
                linewidth=5.5)
        for r in range(labeled_start_row, labeled_end_row):
            ax.plot([0, matrix_length], [(r + 1), (r + 1)], 'k-', linewidth=5.5)
        ax.plot([0, labeled_end_index], [labeled_end_row, labeled_end_row], 'k-', linewidth=5.5)
        ax.plot([0, labeled_end_index], [labeled_end_row + 1, labeled_end_row + 1], 'k-', linewidth=5.5)
    ax.set_yticks(np.arange(0.5, nmonths))
    ax.set_yticklabels(month_list, rotation=0, fontsize=font_s)
    ax.set_xticks(np.array([0, 7, 14, 21, 28]) * (86400 / time_step))
    ax.set_xticklabels([0, 7, 14, 21, 28], rotation=0, fontsize=font_s)
    ax.set_xlabel('Date', fontsize=font_s)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(LW)
    ax.set_title(plot_title, fontsize=font_s+7)
    if export_path:
        plt.savefig(export_path[:-4] + '_condensed' + export_path[-4:], bbox_inches='tight', transparent=transparent)
    else:
        fig.show()
    print('Done!')

def indicators_to_voted_dataframe(starttime, endtime, time_step, indicators_path, class_order=None, export_path=None):
    """
    Convert indicators pickle file to dataframe of voted classes and pnorm
    :param starttime (:class:`~obspy.core.utcdatetime.UTCDateTime`): Start time for dataframe
    :param endtime (:class:`~obspy.core.utcdatetime.UTCDateTime`): End time for dataframe
    :param time_step (float): time step used for dataframe
    :param indicators_path (str): path to pkl file that stores timeline indicators
    :param class_order (list): list of class names in order of class index
    :param export_path (str): filepath to export dataframe
    :return: Pandas dataframe of time, voted class and pnorm
    """

    # Read indicators pickle file
    with open(indicators_path, 'rb') as f:  # Unpickling
        indicators = pickle.load(f)

    # Derive matrix dimensions and initialize
    matrix_length = int((endtime - starttime) / time_step + 1)
    stations = list(np.sort(list(set([i[0] for i in indicators]))))
    matrix_height = len(stations)
    nclasses = len(indicators[0][-1])
    matrix_probs = np.zeros((matrix_height, matrix_length, nclasses))

    # Fill in matrix
    for indicator in indicators:
        row_index = stations.index(indicator[0])
        col_index = int((indicator[1] - starttime) / 60)
        matrix_probs[row_index, col_index, :] = indicator[3]

    # Sum probabilities to find best class and store pnorm
    probs_sum = np.sum(matrix_probs, axis=0)
    probs_contributing_station_count = np.sum(np.sum(matrix_probs, axis=2) != 0, axis=0)
    matrix_condensed = np.argmax(probs_sum, axis=1)
    matrix_condensed[probs_contributing_station_count == 0] = nclasses
    matrix_pnorm = np.max(probs_sum, axis=1) / probs_contributing_station_count

    # Convert to dataframe
    time_vec = [str(t) for t in np.arange(starttime, endtime + 60, 60)]
    if class_order:
        class_vec = [class_order[i] for i in matrix_condensed]
        df = pd.DataFrame(list(zip(time_vec, class_vec, list(matrix_pnorm))), columns=['time', 'class', 'pnorm'])
    else:
        df = pd.DataFrame(list(zip(time_vec, list(matrix_condensed), list(matrix_pnorm))), columns=['time', 'class', 'pnorm'])

    # Export if desired
    if export_path == None:
        return df
    else:
        df.to_csv(export_path, index=False)

def compute_metrics(stream_unprocessed, process_taper=None, metric_taper=None, filter_band=None, window_length=240, overlap=0, vlatlon=(55.4173, -161.8937)):

    """
    :param stream_unprocessed (:class:`~obspy.core.stream.Stream`): Input data (unprocessed -- response is removed within)
    :param padded_length (float): length for which the trace is padded [s]
    :param filter_band (tuple): Tuple of length 2 storing minimum frequency and maximum frequency for bandpass filter ([Hz],[Hz]). Note that reduced displacement uses its own filter band.
    :param window_length (float): window length for each metric to be computed, default is 240 [s]
    :param vlatlon (tuple): Tuple of length 2 storing the latitude and longitude of the target volcano for reduced displacement computation
    :param overlap (float): overlap for time stepping as each metric is computed. Ranges from 0 to 1. If set to 0, time step is equal to window_length.
    :return: numpy.ndarray: 2D array of matplotlib dates, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: If the input stream IS NOT infrasound, 2D array of Reduced Displacement, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: If the input stream IS infrasound, 2D array of Root-Mean-Square Pressure, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: 2D array of Central Frequency, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: 2D array of Dominant Frequency, rows corresponding to traces in stream and columns corresponding to values.
    :return: numpy.ndarray: 2D array of Normalized Standard Deviation of the Top 30 Frequency Peaks, rows corresponding to traces in stream and columns corresponding to values.
    """

    # Check for empty stream
    if len(stream_unprocessed) == 0:
        print('Input stream object is empty, returning empty arrays.')
        tmpl = dr = pe = fc = fd = fsd = np.array([])
        return tmpl, dr, pe, fc, fd, fsd

    # Check if infrasound or seismic
    all_comp = [tr.stats.channel[-1] for tr in stream_unprocessed]
    if all_comp.count('F') == len(all_comp):
        infrasound = True
    elif all_comp.count('F') == 0:
        infrasound = False
    else:
        raise ValueError('The input stream contains a mix of infrasound and non-infrasound traces.')

    # Remove response to obtain stream in pressure or displacement and velocity values
    if infrasound:
        stream_processed = process_waveform(stream_unprocessed.copy(), remove_response=True, rr_output='DEF', detrend=False,
                                   taper_length=process_taper, taper_percentage=None, filter_band=filter_band, verbose=False)
    else:
        stream_disp = process_waveform(stream_unprocessed.copy(), remove_response=True, rr_output='DISP', detrend=False,
                                       taper_length=process_taper, taper_percentage=None, filter_band=(1, 5), verbose=False)
        stream_processed = process_waveform(stream_unprocessed.copy(), remove_response=True, rr_output='VEL', detrend=False,
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
    # rsam = np.ones((len(stream_unprocessed), metric_length)) * np.nan
    if infrasound:
        rmsp = np.ones((len(stream_unprocessed), metric_length)) * np.nan
    else:
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
        stream_processed_segment = stream_processed.copy().trim(window_start, window_end)
        if not infrasound:
            stream_disp_segment = stream_disp.copy().trim(window_start, window_end)

        # Loop over each trace and do computation
        for j in range(len(stream_processed_segment)):

            # Get trace data
            trace_processed_segment = stream_processed_segment[j].data
            if not infrasound:
                trace_disp_segment = stream_disp_segment[j].data

            # Store matplotlib date for plotting
            tmpl[j,i] = window_center.matplotlib_date

            # # Compute RSAM
            # rsam[j,i] = np.mean(np.abs(trace_disp_segment))

            # If infrasound, compute RMSP
            if infrasound:
                rmsp[j,i] = np.sqrt(np.mean(np.square(trace_processed_segment)))
            # If seismic, compute DR
            else:
                rms_disp = np.sqrt(np.mean(np.square(trace_disp_segment)))
                station_dist = GD((stream_disp_segment[j].stats.latitude, stream_disp_segment[j].stats.longitude), vlatlon).m
                wavenumber = 1500 / 2  # assume seisvel = 1500 m/s, dominant frequency = 2 Hz
                dr[j, i] = rms_disp * np.sqrt(station_dist) * np.sqrt(wavenumber) * 100 * 100  # cm^2

            # Compute permutation entropy
            pe[j,i] = complexity_entropy(trace_processed_segment, dx=5)[0]

            # Now execute FFT and trim
            fsamp = rfftfreq(len(trace_processed_segment), 1 / stream_processed_segment[j].stats.sampling_rate)
            fspec = np.abs(rfft(trace_processed_segment))[np.flatnonzero(fsamp>1)]
            fsamp = fsamp[np.flatnonzero(fsamp>1)]

            # Compute central frequency
            fc[j,i] = np.sum(fspec * fsamp) / np.sum(fspec)

            # Compute dominant frequency
            fd[j,i] = fsamp[np.argmax(fspec)]

            # Compute normalized standard deviation of top 30 frequency peaks
            fpeaks_index, _ = find_peaks(fspec)
            fpeaks_top30 = np.sort(fspec[fpeaks_index])[-30:]
            fsd[j,i] = np.std(fpeaks_top30) / np.mean(fspec)

    if infrasound:
        return tmpl, rmsp, pe, fc, fd, fsd
    else:
        return tmpl, dr, pe, fc, fd, fsd

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

def set_universal_seed(seed_value):
    """
    Reset seed for all applicable randomizers
    :param seed_value  (int): desired randomization seed number
    :return: None
    """

    # Import dependencies
    import os
    import random
    import numpy as np
    import tensorflow as tf
    from keras import backend as K

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

def split_labeled_dataset(npy_dir,testval_ratio,stratified,max_train_samples=None):
    """
    Split labeled filepaths using the given test/validation set ratio by class proportions
    :param npy_dir (str): directory to retrieve raw labeled files
    :param testval_ratio (float): ratio of file counts set aside for the test set and validation set, each
    :param stratified (bool): if `True`, testval_ratio will be applied to each class independently, otherwise the sparse-est class will be the reference
    :param max_train_samples (int): if not `None`, impose a maximum sample count for the training set per class (to toss out excessive samples))
    :return: list: list of training set filepaths
    :return: list: list of validation set filepaths
    :return: list: list of test set filepaths
    """

    # Import all dependencies
    import glob
    import random
    import numpy as np

    # Count the number of samples of each class
    nclasses = len(np.unique([filepath[-5] for filepath in glob.glob(npy_dir + '*.npy')]))
    class_paths = [glob.glob(npy_dir + '*_' + str(c) + '.npy') for c in range(nclasses)]
    class_counts = np.array([len(paths) for paths in class_paths])
    print('\nInitial class counts:')
    print(''.join(['%d: %d\n' % (c, class_counts[c]) for c in range(nclasses)]))

    # Determine number of samples set aside for validation set and test set (each)
    print('Train-val-test split (%.1f-%.1f-%.1f) with stratified=%s and max_train_samples=%s:' % ((1-testval_ratio*2)*100,testval_ratio*100,testval_ratio*100,stratified,str(max_train_samples)))
    if stratified:
        testval_numbers = [int(n) for n in (np.floor(class_counts * testval_ratio))]
    else:
        testval_number = int(np.floor(np.min(class_counts)/(1/testval_ratio)))

    # Return random sampled list
    train_list = []
    val_list = []
    test_list = []
    for c in range(nclasses):
        if stratified:
            testval_number = testval_numbers[c]
        test_list = test_list + list(np.random.choice(class_paths[c], testval_number, replace=False))
        leftover_list = [filepath for filepath in class_paths[c] if filepath not in test_list]
        val_list = val_list + list(np.random.choice(leftover_list, testval_number, replace=False))
        leftover_list = [filepath for filepath in leftover_list if filepath not in val_list]
        if max_train_samples and len(leftover_list)>max_train_samples:
            train_list = train_list + list(np.random.choice(leftover_list, max_train_samples, replace=False))
            train_number = max_train_samples
        else:
            train_list = train_list + leftover_list
            train_number = len(leftover_list)
        print('%d: %d train, %d val, %d test' % (c, train_number, testval_number, testval_number))
    print('Total: %d train, %d val, %d test' % (len(train_list),len(val_list),len(test_list)))

    # Shuffle before returning for good measure
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)

    return train_list, val_list, test_list

def augment_labeled_dataset(npy_dir,omit_index,noise_index,testval_ratio,noise_ratio,plot_example=False):

    """
    Use noise-adding augmentation strategy to generate lists of balanced train, validation and testfile paths.
    :param npy_dir (str): directory to retrieve raw labeled files and create nested augmented file directory
    :param omit_index (list of int): class index to omit when calculating augmented number (extras will be discarded)
    :param noise_index (int): class index pointing to the noise class. Files with this class index will be randomly sampled for augmentation.
    :param testval_ratio (float): ratio of file counts set aside for the test set and validation set, each (note this will be calculated on the sparsest class count)
    :param noise_ratio (float): ratio of noise used to generated augmented samples [augmented_image = (1-noise_ratio) * augment_image) + (noise_ratio * noise_image)]
    :param plot_example (bool): if set to `True`, generate a plot showing examples of augmented images
    :return: list: list of training set filepaths (combining both raw and augmented filepaths)
    :return: list: list of validation set filepaths
    :return: list: list of test set filepaths
    """

    # Import all dependencies
    import os
    import glob
    import shutil
    import random
    import numpy as np

    # Count the number of samples of each class
    nclasses = len(np.unique([filepath[-5] for filepath in glob.glob(npy_dir + '*.npy')]))
    class_paths = [glob.glob(npy_dir + '*_' + str(c) + '.npy') for c in range(nclasses)]
    class_counts = np.array([len(paths) for paths in class_paths])
    print('\nInitial class counts:')
    print(''.join(['%d: %d\n' % (c,class_counts[c]) for c in range(nclasses)]))

    # Determine number of samples set aside for validation set and test set (each)
    testval_number = int(np.floor(np.min(class_counts)/(1/testval_ratio)))
    print('Setting aside samples for validation and test set (%.1f%% of sparsest class count each)' % (testval_ratio*100))
    print('%d samples kept for validation set (%d per class)' % (nclasses*testval_number,testval_number))
    print('%d samples kept for test set (%d per class)' % (nclasses*testval_number,testval_number))

    # Calculate augmented number
    print('\nCalculating augmented number...')
    leftover_counts = class_counts - 2*testval_number
    augmented_number = np.mean(leftover_counts[[i for i in range(nclasses) if i not in omit_index]])
    augmented_number = int(np.min([augmented_number] + list(leftover_counts[omit_index])))
    print('Class index %s are omitted and class index %d will be used as noise samples...' % (str(','.join([str(i) for i in omit_index])),noise_index))
    print('%d samples will be gathered for training set (%d per class)' % (nclasses*augmented_number,augmented_number))

    # Determine test and validation sample list
    test_list = []
    val_list = []
    keep_list = []
    for c in range(nclasses):
        test_list = test_list + list(np.random.choice(class_paths[c], testval_number, replace=False))
        leftover_list = [filepath for filepath in class_paths[c] if filepath not in test_list]
        val_list = val_list + list(np.random.choice(leftover_list, testval_number, replace=False))
        leftover_list = [filepath for filepath in leftover_list if filepath not in val_list]
        if c in omit_index:
            keep_list = keep_list + list(np.random.choice(leftover_list, augmented_number, replace=False))
        elif c == noise_index:
            keep_list = keep_list + list(np.random.choice(leftover_list, augmented_number, replace=False))
            noise_list = [filepath for filepath in leftover_list if filepath not in keep_list]
        elif len(leftover_list) <= augmented_number:
            keep_list = keep_list + leftover_list
        else:
            raise ValueError('Class index %d has more samples than the augment number. Check class counts!' % c)

    # Commence augmentation
    print('\nCreating nested augmented directory and commencing augmentation...')

    # Create a temporary directory if it does not exist
    if not os.path.exists(npy_dir + 'augmented/'):
        os.mkdir(npy_dir + 'augmented/')

    # Clear all existing files in augmented subfolder if any
    for f in glob.glob(npy_dir + 'augmented/*.npy'):
        os.remove(f)

    # Randomly sample based on count difference
    aug_list = []
    for c in range(nclasses):
        if c in omit_index or c == noise_index:
            continue
        else:
            keep_sublist = [f for f in keep_list if int(f.split('_')[-1][0]) == c]
            count_difference = augmented_number - len(keep_sublist)
            aug_list = aug_list + list(np.random.choice(keep_sublist, count_difference, replace=True))

    # Check if augment list and noise list have the same length
    if len(aug_list) == len(noise_list):
        print('Augmentation list and noise list match in length. Proceeding...\n')
    else:
        print('Augmentation list and noise list do NOT match in length. Noise list will be trimmed.')
        noise_list = list(np.random.choice(noise_list, len(aug_list), replace=False))

    # Shuffle and add noise to augment samples
    print('Shuffling and adding noise samples to augment samples...')
    random.shuffle(aug_list)
    random.shuffle(noise_list)
    for augment_sample, noise_sample in zip(aug_list,noise_list):

        # Load both images and sum them using noise ratio
        augment_image = np.load(augment_sample)
        noise_image = np.load(noise_sample)
        augmented_image = ((1-noise_ratio) * augment_image) + (noise_ratio * noise_image)

        # Determine filepath by checking for uniqueness
        n = 0
        augmented_filepath = npy_dir + 'augmented/' + augment_sample.split('/')[-1][:-4] + 'aug' + str(n) + '.npy'
        while os.path.isfile(augmented_filepath):
            n += 1
            augmented_filepath = npy_dir + 'augmented/' + augment_sample.split('/')[-1][:-4] + 'aug' + str(n) + '.npy'
        # Save augmented image as a unique file
        np.save(augmented_filepath, augmented_image)

    # Compile train list
    train_list = glob.glob(npy_dir + 'augmented/*.npy') + keep_list
    print('Done!')

    # Plot examples if desired
    if plot_example:
        import matplotlib.pyplot as plt
        import colorcet as cc
        indices = np.random.choice(range(len(aug_list)), 5)
        fig, ax = plt.subplots(5, 3, figsize=(4.2, 10))
        for i, n in enumerate(indices):
            augment_image = np.load(aug_list[n])
            noise_image = np.load(noise_list[n])
            augmented_image = ((1 - noise_ratio) * augment_image) + (noise_ratio * noise_image)
            ax[i,0].imshow(augment_image, vmin=np.percentile(augment_image, 20), vmax=np.percentile(augment_image, 97.5),
                         origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
            ax[i,0].set_xticks([])
            ax[i,0].set_yticks([])
            ax[i,0].set_title('Class ' + str(aug_list[n].split('_')[-1][0]) + '', fontsize=10)
            ax[i,1].imshow(noise_image, vmin=np.percentile(noise_image, 20), vmax=np.percentile(noise_image, 97.5),
                         origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
            ax[i,1].set_xticks([])
            ax[i,1].set_yticks([])
            ax[i,1].set_title('Noise ' + str('%.2f' % noise_ratio), fontsize=10)
            ax[i,2].imshow(augmented_image, vmin=np.percentile(augmented_image, 20),
                         vmax=np.percentile(augmented_image, 97.5),
                         origin='lower', aspect='auto', interpolation='None', cmap=cc.cm.rainbow)
            ax[i,2].set_xticks([])
            ax[i,2].set_yticks([])
            ax[i,2].set_title('Augmented ', fontsize=10)
        fig.show()

    return train_list, val_list, test_list

# def compute_pavlof_rsam(stream_unprocessed):
#     """
#     Pavlof rsam calculation function, written by Matt Haney and adapted by Darren Tan
#     :param stream_unprocessed (:class:`~obspy.core.stream.Stream`): Input data (unprocessed -- response is removed within)
#     :return: dr (list): List of reduced displacement values,
#     """
#     # Import geopy
#     from geopy.distance import geodesic as GD
#     # Define constants
#     R = 6372.7976  # km
#     drm = 3  # cm^2
#     seisvel = 1500  # m/s
#     dfrq = 2  # Hz
#     vlatlon = (55.4173,-161.8937)
#     # Initialize lists
#     disteqv = []
#     sensf = []
#     rmssta = []
#     # Compute
#     for i, tr in enumerate(stream_unprocessed):
#         slatlon = (tr.stats.latitude,tr.stats.longitude)
#         disteqv.append(GD((tr.stats.latitude,tr.stats.longitude),vlatlon).km)
#         sensf.append(tr.stats.response.instrument_sensitivity.value)
#         rmssta.append(drm / (np.sqrt(disteqv[i]*1000) * np.sqrt(seisvel/dfrq)*100*100))  # in m
#     rmsstav = np.array(rmssta)*2*np.pi*dfrq
#     levels_count = rmsstav * sensf
#     q_effect = np.exp(-(np.pi*dfrq*np.array(disteqv)*1000)/(seisvel*200))
#     dr = levels_count * q_effect
#     return dr
