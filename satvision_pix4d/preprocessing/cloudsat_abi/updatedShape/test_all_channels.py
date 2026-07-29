import netCDF4 as nc
import glob

print("Band | Has kappa0 | Has planck_fk1")
for band in range(1, 17):
    pattern = f'/css/geostationary/NonOptimized/L1/GOES-16-ABI-L1B-FULLD/2019/100/02/OR_ABI-L1b-RadF-M6C{band:02d}_*.nc'
    files = glob.glob(pattern)
    if files:
        ds = nc.Dataset(files[0])
        has_kappa = 'kappa0' in ds.variables
        has_planck = 'planck_fk1' in ds.variables
        
        kappa_val = ds.variables['kappa0'][:] if has_kappa else "--"
        planck_val = ds.variables['planck_fk1'][:] if has_planck else "--"
        print(f" {band:02d}  | {str(has_kappa):<10} | {str(has_planck):<10}")
