# -*- coding: utf-8 -*-
"""

Author: jianpu
Email: xianpuji@hhu.edu.cn
"""

"""
Spectral Analysis of Atmospheric Data
====================================
This script performs spectral analysis on atmospheric data (OLR) to identify
wave patterns in the tropical atmosphere.

Author: jianpu
Email: xianpuji@hhu.edu.cn
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.signal as signal
import math
import time
import os
from scipy import fft
import cmaps  # Specialized colormaps

# ================ CONFIGURATION PARAMETERS ===================
class Config:
    """Configuration parameters for the analysis"""
    # Data parameters
    DEFAULT_DATA_PATH = r"I://olr.day.mean.nc"
    DEFAULT_VARIABLE = "olr"
    DEFAULT_LAT_RANGE = [-15, 15]
    DEFAULT_TIME_RANGE = ('1997', '2014')
    
    # Analysis parameters
    WINDOW_SIZE_DAYS = 96
    WINDOW_SKIP_DAYS = 30
    SAMPLES_PER_DAY = 1
    
    # Filtering parameters
    FREQ_CUTOFF = 1.0 / WINDOW_SIZE_DAYS
    
    # Plotting parameters
    CONTOUR_LEVELS = np.array([.3, .4, .5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.4, 1.7, 2., 2.4, 2.8, 4])
    COLORMAP = cmaps.amwg_blueyellowred  # Specialized colormap for meteorological data
    WAVENUMBER_LIMIT = 15  # For plot x-axis limits

# ================ UTILITY FUNCTIONS ===================
def decompose_to_symmetric_antisymmetric(data_array):
    """
    Decompose data into symmetric and antisymmetric components with respect to the equator.
    
    Parameters:
    -----------
    data_array : xarray.DataArray
        Input data with a 'lat' dimension
        
    Returns:
    --------
    xarray.DataArray
        Data with symmetric and antisymmetric components
    """
    lat_dim = data_array.dims.index('lat')
    nlat = data_array.shape[lat_dim]
    
    # Calculate symmetric and antisymmetric components
    symmetric = 0.5 * (data_array.values - np.flip(data_array.values, axis=lat_dim))
    antisymmetric = 0.5 * (data_array.values + np.flip(data_array.values, axis=lat_dim))
    
    # Convert back to DataArrays
    symmetric = xr.DataArray(symmetric, dims=data_array.dims, coords=data_array.coords)
    antisymmetric = xr.DataArray(antisymmetric, dims=data_array.dims, coords=data_array.coords)
    
    # Combine components
    result = data_array.copy()
    half = nlat // 2
    
    if nlat % 2 == 0:
        # Even number of latitudes
        result.isel(lat=slice(0, half))[:] = symmetric.isel(lat=slice(0, half))
        result.isel(lat=slice(half, None))[:] = antisymmetric.isel(lat=slice(half, None))
    else:
        # Odd number of latitudes (equator is in the middle)
        result.isel(lat=slice(0, half))[:] = symmetric.isel(lat=slice(0, half))
        result.isel(lat=slice(half+1, None))[:] = antisymmetric.isel(lat=slice(half+1, None))
        result.isel(lat=half)[:] = symmetric.isel(lat=half)
        
    return result

def remove_annual_cycle(data, samples_per_day, freq_cutoff):
    """
    Remove the annual cycle (low-frequency component) from the data.
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input data with a 'time' dimension
    samples_per_day : float
        Number of samples per day
    freq_cutoff : float
        Frequency cutoff for the low-pass filter
        
    Returns:
    --------
    xarray.DataArray
        Data with annual cycle removed
    """
    n_time, _, _ = data.shape
    
    # Remove linear trend
    detrended_data = signal.detrend(data, axis=0)
    
    # Apply FFT
    fourier_transform = fft.rfft(detrended_data, axis=0)
    frequencies = fft.rfftfreq(n_time, d=1. / float(samples_per_day))
    
    # Apply low-frequency filter
    cutoff_index = np.argwhere(frequencies <= freq_cutoff).max()
    if cutoff_index > 1:
        fourier_transform[1:cutoff_index + 1, ...] = 0.0
    
    # Inverse FFT
    filtered_data = fft.irfft(fourier_transform, axis=0, n=n_time)
    
    return xr.DataArray(filtered_data, dims=data.dims, coords=data.coords)

def smooth_121(array):
    """
    Apply a 1-2-1 smoothing filter to the input array.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array to be smoothed
        
    Returns:
    --------
    numpy.ndarray
        Smoothed array
    """
    weight = np.array([1., 2., 1.]) / 4.0
    return np.convolve(np.r_[array[0], array, array[-1]], weight, 'valid')

# ================ MAIN ANALYSIS CLASS ===================
class SpectralAnalysis:
    """
    Class for performing spectral analysis on atmospheric data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the spectral analysis with configuration.
        
        Parameters:
        -----------
        config : Config, optional
            Configuration object with analysis parameters
        """
        self.config = config or Config()
        self.NA = np.newaxis
        self.pi = math.pi
        
        # Initialize data structures
        self.raw_data = None
        self.processed_data = None
        self.power_symmetric = None
        self.power_antisymmetric = None
        self.background = None
        self.frequency = None
        self.wavenumber = None
        
    def load_data(self, data_path=None, variable=None, lat_range=None, time_range=None):
        """
        Load data from a netCDF file.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the netCDF file
        variable : str, optional
            Name of the variable to load
        lat_range : list, optional
            Latitude range [min, max]
        time_range : tuple, optional
            Time range (start_year, end_year)
            
        Returns:
        --------
        self
        """
        # Use default values if not provided
        data_path = data_path or self.config.DEFAULT_DATA_PATH
        variable = variable or self.config.DEFAULT_VARIABLE
        lat_range = lat_range or self.config.DEFAULT_LAT_RANGE
        time_range = time_range or self.config.DEFAULT_TIME_RANGE
        
        print(f"Loading data from {data_path}")
        
        # Load data
        try:
            ds = xr.open_dataset(data_path).sortby('lat')
            self.raw_data = ds[variable].sel(
                time=slice(*time_range), 
                lat=slice(*lat_range)
            ).sortby('lat').transpose('time', 'lat', 'lon')
            
            print(f"Data loaded successfully: {self.raw_data.shape} (time, lat, lon)")
            return self
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self):
        """
        Preprocess the data: detrend, remove annual cycle, and decompose into symmetric and antisymmetric components.
        
        Returns:
        --------
        self
        """
        print("Preprocessing data...")
        start_time = time.time()
        
        # Detrend the data
        mean_value = self.raw_data.mean(dim='time')
        detrended = signal.detrend(self.raw_data, axis=0, type='linear')
        detrended = xr.DataArray(detrended, dims=self.raw_data.dims, coords=self.raw_data.coords) + mean_value
        
        # Remove annual cycle
        filtered = remove_annual_cycle(
            detrended, 
            self.config.SAMPLES_PER_DAY, 
            self.config.FREQ_CUTOFF
        )
        
        # Decompose into symmetric and antisymmetric components
        self.processed_data = decompose_to_symmetric_antisymmetric(filtered)
        
        print(f"Preprocessing completed in {time.time() - start_time:.1f} seconds.")
        return self
        
    def compute_spectra(self):
        """
        Compute the power spectra using windowed FFT.
        
        Returns:
        --------
        self
        """
        print("Computing power spectra...")
        start_time = time.time()
        
        # Extract dimensions
        ntim, nlat, nlon = self.processed_data.shape
        
        # Calculate window parameters
        spd = self.config.SAMPLES_PER_DAY
        nDayWin = self.config.WINDOW_SIZE_DAYS
        nDaySkip = self.config.WINDOW_SKIP_DAYS
        nDayTot = ntim / spd
        nSampWin = nDayWin * spd
        nSampSkip = nDaySkip * spd
        nWindow = int((nDayTot * spd - nSampWin) / (nSampSkip + nSampWin)) + 1
        
        print(f"Analysis parameters: Window size: {nDayWin} days, Skip: {nDaySkip} days")
        print(f"Total windows: {nWindow}, Window samples: {nSampWin}")
        
        # Initialize power accumulator
        sumpower = np.zeros((nSampWin, nlat, nlon))
        
        # Process each window
        ntStrt, ntLast = 0, nSampWin
        for nw in range(int(nWindow)):
            # Extract window data
            data = self.processed_data[ntStrt:ntLast, :, :]
            data = signal.detrend(data, axis=0)
            
            # Apply taper window
            window = signal.windows.tukey(nSampWin, 0.1, True)
            data *= window[:, self.NA, self.NA]
            
            # Compute FFT
            power = fft.fft2(data, axes=(0, 2)) / (nlon * nSampWin)
            sumpower += np.abs(power) ** 2
            
            # Move to next window
            ntStrt = ntLast + nSampSkip
            ntLast = ntStrt + nSampWin
            
            if nw % 10 == 0:
                print(f"Processed window {nw+1}/{nWindow}")
        
        # Normalize by number of windows
        sumpower /= nWindow
        
        # Setup frequency and wavenumber arrays
        if nlon % 2 == 0:
            self.wavenumber = fft.fftshift(fft.fftfreq(nlon) * nlon)[1:]
            sumpower = fft.fftshift(sumpower, axes=2)[:, :, nlon:0:-1]
        else:
            self.wavenumber = fft.fftshift(fft.fftfreq(nlon) * nlon)
            sumpower = fft.fftshift(sumpower, axes=2)[:, :, ::-1]
        
        self.frequency = fft.fftshift(fft.fftfreq(nSampWin, d=1. / float(spd)))[nSampWin // 2:]
        sumpower = fft.fftshift(sumpower, axes=0)[nSampWin // 2:, :, :]
        
        # Compute symmetric and antisymmetric power
        self.power_symmetric     = 2.0 * sumpower[:, nlat // 2:, :].sum(axis=1)
        self.power_antisymmetric = 2.0 * sumpower[:, :nlat // 2, :].sum(axis=1)
        
        # Convert to DataArrays
        self.power_symmetric = xr.DataArray(
            self.power_symmetric,
            dims=("frequency", "wavenumber"),
            coords={"wavenumber": self.wavenumber, "frequency": self.frequency}
        )
        
        self.power_antisymmetric = xr.DataArray(
            self.power_antisymmetric,
            dims=("frequency", "wavenumber"),
            coords={"wavenumber": self.wavenumber, "frequency": self.frequency}
        )
        
        # Mask zero frequency
        self.power_symmetric[0, :] = np.ma.masked
        self.power_antisymmetric[0, :] = np.ma.masked
        
        # Compute background spectrum
        self.background = sumpower.sum(axis=1)
        self.background[0, :] = np.ma.masked
        
        print(f"Spectrum computation completed in {time.time() - start_time:.1f} seconds.")
        return self
    
    def smooth_background(self):
        """
        Smooth the background spectrum.
        
        Returns:
        --------
        self
        """
        print("Smoothing background spectrum...")
        start_time = time.time()
        
        # Select wavenumbers to smooth
        wave_smooth_indices = np.where(np.abs(self.wavenumber) <= 27)[0]
        
        # Smooth each frequency
        for idx, freq in enumerate(self.frequency):
            # Determine smoothing iterations based on frequency
            if freq < 0.1:
                smooth_iterations = 5
            elif freq >= 0.1 and freq < 0.2:
                smooth_iterations = 10
            elif freq >= 0.2 and freq < 0.3:
                smooth_iterations = 20
            else:  # freq >= 0.3
                smooth_iterations = 40
                
            # Apply smoothing
            for _ in range(smooth_iterations):
                self.background[idx, wave_smooth_indices] = smooth_121(
                    self.background[idx, wave_smooth_indices]
                )
        
        # Smooth across frequencies
        for wave_idx in wave_smooth_indices:
            for _ in range(10):
                self.background[:, wave_idx] = smooth_121(self.background[:, wave_idx])
        
        print(f"Background smoothing completed in {time.time() - start_time:.1f} seconds.")
        return self
    
    def plot_spectra(self, output_path=None):
        """
        Plot the normalized power spectra.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the plot. If None, the plot will be displayed.
            
        Returns:
        --------
        self
        """
        print("Plotting results...")
        
        # Create figure
        cmap=cmaps.amwg_blueyellowred
        

        fig, axes = plt.subplots(1, 2, figsize=(12, 7), dpi=200)
        im1 = (self.power_symmetric/self.background).plot.contourf(ax = axes[0], cmap=self.config.COLORMAP,add_colorbar=False,
                                                                   extend='neither',
                                                                   levels=self.config.CONTOUR_LEVELS)
        im2 = (self.power_antisymmetric/self.background).plot.contourf(ax = axes[1], cmap=self.config.COLORMAP,add_colorbar=False,
                                                                       extend='neither',
                                                                       levels=self.config.CONTOUR_LEVELS)
        
        fig.colorbar(im1, ax=[axes[0],axes[1]],orientation='horizontal',shrink=.45,aspect=30,pad=0.1)
        axes[0].set_title("Symmetric Component Spectrum")
        axes[1].set_title("Antisymmetric Component Spectrum")
        for ax in axes: 
            ax.axvline(0, linestyle='dashed', color='lightgray')
            ax.set_xlim([-15,15])
            ax.set_ylim([0,0.5])
        plt.show()


# SpectralAnalysis
start_time = time.time()
# 创建分析对象
analysis = SpectralAnalysis()

# 加载OLR数据
analysis.load_data(data_path="I://olr.day.mean.nc")

# 预处理数据
analysis.preprocess_data()

# 计算功率谱
analysis.compute_spectra()

# 平滑背景谱
analysis.smooth_background()

# 绘制结果
analysis.plot_spectra(output_path="spectral_plot.png")
print(f"Background smoothing completed in {time.time() - start_time:.1f} seconds.")