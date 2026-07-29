import netCDF4 as nc
import glob
import numpy as np

print("Band | kappa0_valid | planck_fk1_valid")
for band in range(1, 17):
    pattern = f'/css/geostationary/NonOptimized/L1/GOES-16-ABI-L1B-FULLD/2019/100/02/OR_ABI-L1b-RadF-M6C{band:02d}_*.nc'
    files = glob.glob(pattern)
    if files:
        ds = nc.Dataset(files[0])
        kappa_val = ds.variables['kappa0'][:]
        planck_val = ds.variables['planck_fk1'][:]
        
        # Check if they are masked arrays or valid floats
        kappa_valid = not (np.ma.is_masked(kappa_val) or np.isnan(kappa_val))
        planck_valid = not (np.ma.is_masked(planck_val) or np.isnan(planck_val))
        print(f" {band:02d}  | {str(kappa_valid):<12} | {str(planck_valid):<16}")
