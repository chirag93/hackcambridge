import sys
import os
sys.path.append('/Users/adam2392/Documents/tvb/')
# sys.path.append('/Users/adam2392/Documents/tvb/_tvbdata/')
# sys.path.append('/Users/adam2392/Documents/tvb/_tvblibrary/')
# from tvb.simulator.lab import *
import numpy as np
import pandas as pd
import scipy
from sklearn import cluster

from sklearn.preprocessing import MinMaxScaler
import tvbsim

'''
Module for preprocessing and organizing data computed by FFT on iEEG data
into a huge compressed data structure

Data will be stored as npz compressed file

matrix will be H x W x F x T (height x width x frequency band x time window),
T x H x W x F 
where height and width define a grid where power is projected.

It will depend on the number of parcellated regions, so for 84, it will be a 
12x7 image x4 frequency bands (alpha, beta, gamma, high freq)
'''

# get factors of the number of regions
def get_factors(x):
   # This function takes a number and prints the factors

    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            factors.append(i)
    return factors

class PreProcess():
    def __init__(self,datafiles=[],freqbands=None):

        if not freqbands:
            # establish frequency bands
            freqbands = {
                'lowfreq': [0, 16],
                'midfreq': [16, 33],
                'gamma': [33, 90],
                'highgamma': [90, 501],
            }
        self.datafiles = datafiles
        self.freqbands = freqbands

    def _computefreqindices(self, freqs, freqbands):
        freqbandindices = {}
        for band in freqbands:
            lowerband = freqbands[band][0]
            upperband = freqbands[band][1]
            
            # get indices where the freq bands are put in
            freqbandindices[band] = np.where((freqs >= lowerband) & (freqs < upperband))
            freqbandindices[band] = [freqbandindices[band][0][0], freqbandindices[band][0][-1]]
        return freqbandindices

    def compresspowermat(self,datapath):
        print(os.path.join(datapath))
        freqbands = self.freqbands

        # load data from this datafile and run
        powerbands = {}
        data = np.load(os.path.join(datapath), encoding='bytes')
        
        # extract data from the numpy file
        power = data['power']
        freqs = data['freqs']
        timepoints = data['timepoints']
        metadata = data['metadata'].item()
        metadata[b'freqbands'] = freqbands
      
        # compute the freq indices for each band
        freqbandindices = self._computefreqindices(freqs,freqbands)

        # compress data using frequency bands
        for idx, band in enumerate(freqbandindices):
            indices = freqbandindices[band]
            # average between these two indices
            powerbands[band] = np.mean(power[:,indices[0]:indices[1]+1,:], axis=1) #[np.newaxis,:,:]

        return powerbands, timepoints

    def getseiztimes(self,datafile):
        '''
        Create an accompanying list of datafiles to save, so that it is a metadata
        list of all the data that is compressed into the final data structure
        '''
        data = np.load(datafile, encoding='bytes')
        metadata = data['metadata'].item()
        metadata = {k.decode("utf-8"): (v.decode("utf-8") if isinstance(v, bytes) else v) for k,v in metadata.items()}

        onsettimes = metadata['onsettimes']
        offsettimes = metadata['offsettimes']

        if len(onsettimes) > 0:
            if onsettimes[0] > offsettimes[0]:
                onsettimes = metadata['offsettimes']
                offsettimes = metadata['onsettimes']
        return onsettimes, offsettimes

    def loadmetadata(self,datafile):
        data = np.load(datafile, encoding='bytes')
        # extract data from the numpy file
        metadata = data['metadata'].item()
        metadata = {k.decode("utf-8"): (v.decode("utf-8") if isinstance(v, bytes) else v) for k,v in metadata.items()}

        return metadata

    def getimdims(self,metadata):
        regions = metadata['regions']

        # reshape the regions of 84 into a parcellated rectangular "image"
        # height = np.ce
        factors = get_factors(len(regions))
        height = factors[int(len(factors)/2)]
        width = int(len(regions) / height)

        return height, width

    def projectpower_gain(self,powerbands,metadata,verts,areas,regmap):
        seeg_xyz = metadata['seeg_xyz']
        seeg_labels = metadata['seeg_contacts']

        # extract the seeg_xyz coords and the region centers
        region_centers = metadata['region_centers']
        regions = metadata['regions']

        height, width = self.getimdims(metadata)

        # check seeg_xyz moved correctly - In early simulation data results, was not correct
        buff = seeg_xyz - region_centers[:, np.newaxis]
        buff = np.sqrt((buff**2).sum(axis=-1))
        test = np.where(buff==0)
        indice = test[1]

        modgain = tvbsim.util.gain_matrix_inv_square(verts, areas,
                            regmap, len(regions), seeg_xyz)
        modgain = modgain.T

        # map seeg activity -> epileptor source and create data structure
        for idx,band in enumerate(powerbands):
            mapped_power_band = np.tensordot(modgain, powerbands[band], axes=([1],[0]))
                
            if idx==0:
                mapped_power_bands = mapped_power_band.reshape(height, width, mapped_power_band.shape[1], 
                                                             order='C')[np.newaxis,:,:,:]
            else:
                mapped_power_bands = np.append(mapped_power_bands, 
                                        mapped_power_band.reshape(height, 
                                            width, 
                                            mapped_power_band.shape[1], 
                                            order='C') [np.newaxis,:,:,:], axis=0)
            
        # new condensed data structure is H x W x F x T, to concatenate more, add to T dimension
        mapped_power_bands = mapped_power_bands.swapaxes(0,3).swapaxes(1,2)
   
        return mapped_power_bands

    def projectpower_knn(self,powerbands,metadata):
        # extract the seeg_xyz coords and the region centers
        region_centers = metadata['region_centers']
        regions = metadata['regions']
        seeg_xyz = metadata['seeg_xyz']
        seeg_labels = metadata['seeg_contacts']

        numfreqbands = len(self.freqbands)

        height, width = self.getimdims(metadata)    

        # map seeg_xyz to 3 closest region_centers
        tree = scipy.spatial.KDTree(region_centers)
        seeg_near_indices = []

        seeg_counter = np.zeros(len(regions))

        for ichan in range(0, len(seeg_labels)):
            near_regions = tree.query(seeg_xyz[ichan,:].squeeze(), k=3)
            near_indices = near_regions[1]
            
            # go through each frequency band and map activity onto those near indices
            for idx,band in enumerate(powerbands):
                # initialize the final result data structure
                if ichan==0 and idx==0:
                    mapped_power_bands = np.zeros((len(regions), 
                                        numfreqbands,
                                        powerbands[band].shape[1]), dtype='float64')

                chanpower = powerbands[band][ichan,:]
                mapped_power_bands[near_indices,idx,:] += chanpower.astype('float64')
            
            seeg_counter[near_indices] += 1
            # seeg_near_indices.append(near_indices)

        # get the average based on how many contributions of the seeg power was to this region
        mapped_power_bands = np.divide(mapped_power_bands, seeg_counter[:,np.newaxis,np.newaxis], 
                                out=np.zeros_like(mapped_power_bands), where=seeg_counter[:,np.newaxis,np.newaxis]!=0)
        
        # reshape for the correct output
        mapped_power_bands = mapped_power_bands.reshape(height, 
                                                        width, 
                                                        numfreqbands, 
                                                        powerbands[band].shape[1], 
                                                    order='C')
        mapped_power_bands = mapped_power_bands.swapaxes(0,3).swapaxes(2,3)

        return mapped_power_bands

    def projectpower_invsquare(self,powerbands,metadata):
        # extract the seeg_xyz coords and the region centers
        region_centers = metadata['region_centers']
        regions = metadata['regions']
        seeg_xyz = metadata['seeg_xyz']
        seeg_labels = metadata['seeg_contacts']

        height, width = self.getimdims(metadata)   

        # map seeg_xyz to the rest of the regions from a factor of 
        dr = region_centers - seeg_xyz[:,np.newaxis] # computes distance along each axis
        ndr = np.sqrt((dr**2).sum(axis=-1)) # computes euclidean distance
        Vr = 1/(ndr**2) # fall off as a function of r^2

        inf_indices = np.where(np.isinf(Vr))
        small_indices = np.where(ndr <= 1)

        # can either set to 1, or the max that there currently is + some small epsilon
        # the problem with setting to 1 is that the signal might drown out everything else
        Vr[small_indices] = np.nanmax(np.ma.masked_invalid(Vr[:])) + np.nanmin(Vr[:])

        # normalize Vr with minmax
        scaler = MinMaxScaler(feature_range=(0, 1))
        Vr = scaler.fit_transform(Vr).T

        # map seeg activity -> epileptor source and create data structure
        for idx,band in enumerate(powerbands):
            mapped_power_band = np.tensordot(Vr, powerbands[band], axes=([1],[0]))
            
            # store the formatted power bands
            if idx==0:
                mapped_power_bands = mapped_power_band.reshape(height, 
                                                                width,
                                                                mapped_power_band.shape[1], 
                                                             order='C')[np.newaxis,:,:,:]
            else:
                mapped_power_bands = np.append(mapped_power_bands, 
                                                        mapped_power_band.reshape(height, 
                                                            width, 
                                                            mapped_power_band.shape[1], 
                                                        order='C')[np.newaxis,:,:,:], axis=0)
            
        # new condensed data structure is H x W x F x T, to concatenate more, add to T dimension
        mapped_power_bands = mapped_power_bands.swapaxes(0,3).swapaxes(1,2)

        return mapped_power_bands

