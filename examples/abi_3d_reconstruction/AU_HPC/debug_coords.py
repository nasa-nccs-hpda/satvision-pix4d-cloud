import numpy as np
import pyhdf
from pyhdf.SD import SD, SDC
from pyhdf.HDF import *
from pyhdf.VS import *
import glob
import os
import netCDF4 as nc
import sys

# Minimal settings to match the main script
LATLONDATA = "/home/al8425b-hpc/NASA/cropTest/testData/ABI_EAST_GEO_TOPO_LOMSK.nc"
CLOUDSATPATH = '/home/al8425b-hpc/NASA/cropTest/testData/cloudsat/'
ROOT_DIR = '/home/al8425b-hpc/NASA/cropTest/testData/cloudsat/2B-CLDCLASS-LIDAR'

def read_2b_cldclass_lidar(cs_file, latbin=None):
    sdc_2bcldclass_lidar = HDF(cs_file, SDC.READ)
    vs_2bcldclass_lidar = sdc_2bcldclass_lidar.vstart()
    Latitude = np.squeeze(vs_2bcldclass_lidar.attach('Latitude')[:])
    Longitude = np.squeeze(vs_2bcldclass_lidar.attach('Longitude')[:])
    
    print(f"Raw Latitude shape: {Latitude.shape}")
    
    ilat = np.squeeze(np.argwhere(
        (Latitude >= latbin[0]) & (Latitude <= latbin[1])))
    
    Latitude = Latitude[ilat]
    Longitude = Longitude[ilat]
    
    print(f"Binned Latitude shape: {Latitude.shape}")
    return Latitude, Longitude

# Test with a specific file found earlier
year = "2020"
day = "001"
orbit = "72861"

cs_file = glob.glob(os.path.join(
    CLOUDSATPATH, '2B-CLDCLASS-LIDAR', year, day,
    year + day + '*' + orbit + '*2B-CLDCLASS-LIDAR*' + 'P1_R05*.hdf'
))

if cs_file:
    # Get lat bounds from ABI data
    f = nc.Dataset(LATLONDATA)
    abiLat = np.array(f['Latitude'])
    latMin, latMax = abiLat.min(), abiLat.max()
    
    Latitude, Longitude = read_2b_cldclass_lidar(cs_file[0], latbin=(latMin, latMax))
    
    # Check a sample range like the script does
    i = 13670 # The index from the filename we inspected
    dRange = np.arange(i + 46, i - 45, -1)
    
    print(f"Index range: {dRange[0]} to {dRange[-1]}")
    try:
        sample_lats = Latitude[dRange]
        print(f"Sample Latitudes (first 5): {sample_lats[:5]}")
        print(f"Unique values in slice: {len(np.unique(sample_lats))}")
    except Exception as e:
        print(f"Error slicing: {e}")
        print(f"Max index available: {len(Latitude)}")
else:
    print("Could not find test file.")
