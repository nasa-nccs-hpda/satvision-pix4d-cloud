import netCDF4 as nc
import glob

# Find an ABI file
files = glob.glob('/css/geostationary/NonOptimized/L1/GOES-16-ABI-L1B-FULLD/2019/100/*/*.nc')
if files:
    f = files[0]
    print(f"File: {f}")
    ds = nc.Dataset(f)
    print(ds.variables.keys())
    
    if 'kappa0' in ds.variables:
        print("kappa0:", ds.variables['kappa0'][:])
    
    for var in ['planck_fk1', 'planck_fk2', 'planck_bc1', 'planck_bc2']:
        if var in ds.variables:
            print(f"{var}:", ds.variables[var][:])
    
    # Let's check a bit of Rad
    rad = ds.variables['Rad']
    print("Rad min/max:", rad[:].min(), rad[:].max())
else:
    print("No files found")
