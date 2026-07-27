import numpy as np
import pyhdf
from pyhdf.SD import SD, SDC
from pyhdf.HDF import *
from pyhdf.VS import *
import glob
import os
import netCDF4 as nc
import sys

'''
CONFIGURATION PARAMETERS
'''

LATLONDATA = "/home/al8425b-hpc/NASA/cropTest/testData/ABI_EAST_GEO_TOPO_LOMSK.nc"
CLOUDSATPATH = '/home/al8425b-hpc/NASA/cropTest/testData/cloudsat/'
ROOT_DIR = '/home/al8425b-hpc/NASA/cropTest/testData/cloudsat/2B-CLDCLASS-LIDAR'
ABIDATA = "/home/al8425b-hpc/NASA/cropTest/testData/GOES_Data/" 
# SAVEDIR = '/home/al8425b-hpc/NASA/satvision-pix4d/examples/originalChipTest/output/'
SAVEDIR = '/home/al8425b-hpc/NASA/satvision-pix4d/examples/abi_3d_reconstruction/originalChipTest/output'

# Multi-timestep configuration
OFFSETS_MINUTES = [-40, -20, 0, 20, 40]   # every 20 min centered on CloudSat time
CHIP_HALF_SIZE = 64                        # 128x128 chips
ALLOW_MISSING_TIMESTEPS = False            # set True if you want to allow partial time series

# ===============================================


def read_2b_cldclass_lidar(cs_file, latbin=None):
    f_2b_cldclass_lidar = SD(cs_file, SDC.READ)
    sds_obj = f_2b_cldclass_lidar.select('CloudLayerBase')
    cs_clb = sds_obj.get()
    sds_obj = f_2b_cldclass_lidar.select('CloudLayerTop')
    cs_clt = sds_obj.get()
    sds_obj = f_2b_cldclass_lidar.select('CloudLayerType')
    cs_cltype = sds_obj.get()

    sdc_2bcldclass_lidar = HDF(cs_file, SDC.READ)
    vs_2bcldclass_lidar = sdc_2bcldclass_lidar.vstart()
    cs_QC = np.squeeze(vs_2bcldclass_lidar.attach('Data_quality')[:])
    Latitude = np.squeeze(vs_2bcldclass_lidar.attach('Latitude')[:])
    Longitude = np.squeeze(vs_2bcldclass_lidar.attach('Longitude')[:])

    ilat = np.squeeze(np.argwhere(
        (Latitude >= latbin[0]) & (Latitude <= latbin[1])))
    cs_clb = cs_clb[ilat, :]
    cs_clt = cs_clt[ilat, :]
    cs_QC = cs_QC[ilat]
    cs_cltype = cs_cltype[ilat, :]
    Latitude = Latitude[ilat]
    Longitude = Longitude[ilat]

    return (cs_clb, cs_clt, cs_cltype, cs_QC, Latitude, Longitude)

# ===============================================


def read_cs_ecmwf(aux_file, latbin=None):
    f_ecmwf = SD(aux_file, SDC.READ)
    sds_obj = f_ecmwf.select('Pressure')
    Pressure = sds_obj.get()
    sds_obj = f_ecmwf.select('Temperature')
    Temperature = sds_obj.get()
    sds_obj = f_ecmwf.select('Specific_humidity')
    Specific_humidity = sds_obj.get()

    sdc_ecmwf = HDF(aux_file, SDC.READ)
    vs_ecmwf = sdc_ecmwf.vstart()
    EC_height = np.squeeze(vs_ecmwf.attach('EC_height')[:])
    Profile_time = np.squeeze(vs_ecmwf.attach('Profile_time')[:])
    UTC_start = np.squeeze(vs_ecmwf.attach('UTC_start')[:])
    Latitude = np.squeeze(vs_ecmwf.attach('Latitude')[:])
    Longitude = np.squeeze(vs_ecmwf.attach('Longitude')[:])
    DEM_elevation = np.squeeze(vs_ecmwf.attach('DEM_elevation')[:])
    Skin_temperature = np.squeeze(vs_ecmwf.attach('Skin_temperature')[:])
    Surface_pressure = np.squeeze(vs_ecmwf.attach('Surface_pressure')[:])
    Temperature_2m = np.squeeze(vs_ecmwf.attach('Temperature_2m')[:])
    U10_velocity = np.squeeze(vs_ecmwf.attach('U10_velocity')[:])
    V10_velocity = np.squeeze(vs_ecmwf.attach('V10_velocity')[:])

    UTC_Time = UTC_start + Profile_time
    UTC_Time = UTC_Time / 60. / 60.

    ilat = np.squeeze(np.argwhere(
        (Latitude >= latbin[0]) & (Latitude <= latbin[1])))
    Pressure = Pressure[ilat, :]
    Temperature = Temperature[ilat, :]
    Specific_humidity = Specific_humidity[ilat, :]
    DEM_elevation = DEM_elevation[ilat]
    Skin_temperature = Skin_temperature[ilat]
    Temperature_2m = Temperature_2m[ilat]
    U10_velocity = U10_velocity[ilat]
    V10_velocity = V10_velocity[ilat]
    UTC_Time = UTC_Time[ilat]

    return (Pressure, DEM_elevation, Temperature, Specific_humidity, EC_height,
            Temperature_2m, U10_velocity, V10_velocity, UTC_Time)


def find_files(directory, prefix):
    matching_files = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            matching_files.append(os.path.join(directory, filename))
    return matching_files


def gather_files(YYYY, DDD, HH, ROOT):
    ABI_ = {
        "ROOT_PATH": None,
        "YYYY": None,
        "DDD": None,
        "HH": None,
        "00": [],
        "10": [],
        "15": [],
        "20": [],
        "30": [],
        "40": [],
        "45": [],
        "50": [],
        "L200": None,
        "L210": None,
        "L220": None,
        "L230": None,
        "L240": None,
        "L250": None,
        "everyten": False,
    }

    # _ABI_PATH_ = ROOT + YYYY + "/" + DDD + "/" + HH
    _ABI_PATH_ = os.path.join(ROOT, YYYY, DDD, HH)

    if not os.path.isdir(_ABI_PATH_):
        raise FileNotFoundError(f"ABI hour path does not exist: {_ABI_PATH_}")

    for filename in os.listdir(_ABI_PATH_):
        if ABI_["ROOT_PATH"] is None:
            ABI_["ROOT_PATH"] = _ABI_PATH_
            ABI_["YYYY"] = filename[27:31]
            ABI_["DDD"] = filename[31:34]
            ABI_["HH"] = filename[34:36]

        MM = filename[36:38]
        if MM == "10":
            ABI_["everyten"] = True

        if MM in ABI_:
            ABI_[MM].append(filename)

    return ABI_


def get_L1B_L2(abipaths, l2path, YYYY, DDD, HH, ROOT):
    if len(abipaths) != 16:
        raise ImportError(f"This timestep is bad: expected 16 channels, got {len(abipaths)} for {YYYY}/{DDD}/{HH}")

    CHANNELS = []

    for file in abipaths:
        L1B = np.asarray(nc.Dataset(
            os.path.join(ROOT, YYYY, DDD, HH, file), 'r')["Rad"])
        CHANNEL = int(file[19:21])
        CHANNELS.append((L1B, CHANNEL))

    CHANNELS.sort(key=lambda x: x[1])
    CHANNELS = [C[0] for C in CHANNELS]

    T = []
    for C in CHANNELS:
        S = C.shape[0] // 5424
        if S == 1:
            C = np.repeat(C, 2, axis=0)
            C = np.repeat(C, 2, axis=1)
        if S == 4:
            C = C[::2, ::2]
        T.append(C)

    CHANNELS = T
    ABI = np.stack(CHANNELS, axis=2).astype(np.float32)

    return ABI


BOUND_SIZE = 1600
LENGTH = 10848

f = nc.Dataset(LATLONDATA)
abiLong = np.array(f['Longitude'])
abiLat = np.array(f['Latitude'])

abiLongB = abiLong[BOUND_SIZE:LENGTH-BOUND_SIZE, BOUND_SIZE:LENGTH-BOUND_SIZE]
abiLatB = abiLat[BOUND_SIZE:LENGTH-BOUND_SIZE, BOUND_SIZE:LENGTH-BOUND_SIZE]
abiLongB[abiLongB == -999] = 10
abiLatB[abiLatB == -999] = 10
abiLongB[abiLongB < 0] += 360

longMin = abiLongB.min()
longMax = abiLongB.max()
latMin = abiLatB.min()
latMax = abiLatB.max()

GATHERED_ABI_FILES = {}
COLLECTED_ABI_DATA = {}

latSlice = abiLat[:, 5424]
latSlice = latSlice[18:-18]
longSlice = abiLong[5424, :]
longSlice = longSlice[18:-18]
latSlice = latSlice[::-1]


def interpArray(Temperature, EC_height):
    z_grid = np.arange(40) * 0.5
    N = Temperature.shape[0]
    Temperature_grid = np.zeros((N, len(z_grid)), dtype=np.float32)

    for i in range(N):
        Temperature_tmp = np.squeeze(Temperature[i, :])
        ivalid = np.squeeze(np.where(Temperature_tmp > 0))

        Temperature_grid[i, :] = np.interp(
            z_grid,
            np.flip(EC_height[ivalid] / 1000.),
            np.flip(Temperature_tmp[ivalid])
        )

    return Temperature_grid


def is_leap_year(year):
    year = int(year)
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def increment_day(year, day, step):
    year = int(year)
    day = int(day)

    while step != 0:
        days_in_year = 366 if is_leap_year(year) else 365

        if step > 0:
            day += 1
            if day > days_in_year:
                year += 1
                day = 1
            step -= 1
        else:
            day -= 1
            if day < 1:
                year -= 1
                day = 366 if is_leap_year(year) else 365
            step += 1

    return str(year), f"{day:03d}"


def shift_time(yy, ddn, t_hours, delta_minutes):
    total_minutes = t_hours * 60.0 + delta_minutes
    day_shift = 0

    while total_minutes < 0:
        total_minutes += 24 * 60
        day_shift -= 1

    while total_minutes >= 24 * 60:
        total_minutes -= 24 * 60
        day_shift += 1

    new_t_hours = total_minutes / 60.0
    new_yy, new_ddn = increment_day(yy, ddn, day_shift)

    return new_yy, new_ddn, new_t_hours


def find_abi_coords(lat, lon):
    if lat < latMin or lat > latMax or lon < longMin or lon > longMax:
        raise ValueError("Latitude and Longitude are not correctly bounded")

    AREA_SIZE = 1000

    lati = len(latSlice) - np.searchsorted(latSlice, lat) + 17
    loni = np.searchsorted(longSlice, lon) + 18

    distances = (
        np.abs(abiLat[lati-AREA_SIZE:lati+AREA_SIZE, loni-AREA_SIZE:loni+AREA_SIZE] - lat) +
        np.abs(abiLong[lati-AREA_SIZE:lati+AREA_SIZE, loni-AREA_SIZE:loni+AREA_SIZE] - lon)
    )

    coords = np.array(np.unravel_index(distances.argmin(), distances.shape))

    if coords[0] == 0 or coords[1] == 0 or coords[1] == 2 * AREA_SIZE - 1 or coords[0] == 2 * AREA_SIZE - 1:
        distances = np.abs(abiLat - lat) + np.abs(abiLong - lon)
        coords = np.unravel_index(distances.argmin(), distances.shape)
        coords = np.array(coords)
    else:
        coords[0] += lati - AREA_SIZE
        coords[1] += loni - AREA_SIZE

    if coords[0] < BOUND_SIZE or coords[1] < BOUND_SIZE or coords[1] > LENGTH - BOUND_SIZE or coords[0] > LENGTH - BOUND_SIZE:
        raise ValueError("Bad latitude and longitude")

    return coords


def get_time_key_info(t, yy, ddn, ABI_ROOT):
    if np.floor(t) < 0 or np.floor(t) >= 24:
        raise ValueError("outside 0-23 bound")

    hour = int(np.floor(t))
    hour_str = f"{hour:02d}"

    data = GATHERED_ABI_FILES.get(f'{yy}-{ddn}-{hour_str}')
    if data is None:
        data = gather_files(str(yy), str(ddn), hour_str, ABI_ROOT)
        GATHERED_ABI_FILES[f'{yy}-{ddn}-{hour_str}'] = data

    if data["everyten"]:
        minutes = np.round((t - np.floor(t)) * 6).astype(int) * 10
    else:
        minutes = np.round((t - np.floor(t)) * 4).astype(int) * 15

    if minutes == 60:
        if hour != 23:
            hour += 1
            minutes = 0
            hour_str = f"{hour:02d}"

            data = GATHERED_ABI_FILES.get(f'{yy}-{ddn}-{hour_str}')
            if data is None:
                data = gather_files(str(yy), str(ddn), hour_str, ABI_ROOT)
                GATHERED_ABI_FILES[f'{yy}-{ddn}-{hour_str}'] = data
        else:
            minutes = 50 if data["everyten"] else 45

    minute_str = f"{minutes:02d}"

    if minute_str not in data or len(data[minute_str]) == 0:
        raise FileNotFoundError(f"No ABI files for {yy}-{ddn} {hour_str}:{minute_str}")

    return data, hour_str, minute_str


def get_abi_chip_at_time(t, yy, ddn, coords, ABI_ROOT):
    data, hour_str, minute_str = get_time_key_info(t, yy, ddn, ABI_ROOT)

    abi_key = f'{yy}-{ddn}-{hour_str}-{minute_str}'
    abi = COLLECTED_ABI_DATA.get(abi_key)

    if abi is None:
        abi = get_L1B_L2(
            data[minute_str],
            data["L200"],
            data["YYYY"],
            data["DDD"],
            data["HH"],
            ABI_ROOT
        )
        COLLECTED_ABI_DATA[abi_key] = abi

    chip = abi[
        coords[0] - CHIP_HALF_SIZE:coords[0] + CHIP_HALF_SIZE,
        coords[1] - CHIP_HALF_SIZE:coords[1] + CHIP_HALF_SIZE,
        :
    ]

    if chip.shape != (2 * CHIP_HALF_SIZE, 2 * CHIP_HALF_SIZE, 16):
        raise ValueError(f"Bad chip shape: {chip.shape}")

    return chip, hour_str, minute_str


def processTimeSeries(t, yy, ddn, lat, lon, ABI_ROOT, offsets_minutes=None,
                      allow_missing=False):

    if offsets_minutes is None:
        offsets_minutes = OFFSETS_MINUTES

    coords = find_abi_coords(lat, lon)
    # print(coords)


    chips = []
    time_meta = []
    valid_mask = []

    for dt_min in offsets_minutes:
        try:
            yy_t, ddn_t, t_t = shift_time(yy, ddn, t, dt_min)
            chip, hour_str, minute_str = get_abi_chip_at_time(
                t_t, yy_t, ddn_t, coords, ABI_ROOT
            )
            valid = 1
        except Exception as e:
            print("Exceptions inside offsets_minutes", e)
            if not allow_missing:
                raise e

            chip = np.full((2 * CHIP_HALF_SIZE, 2 * CHIP_HALF_SIZE, 16), np.nan, dtype=np.float32)
            yy_t, ddn_t, t_t = None, None, None
            hour_str, minute_str = None, None
            valid = 0

        print(chip.shape)
        chips.append(chip)
        valid_mask.append(valid)
        time_meta.append({
            "year": yy_t,
            "doy": ddn_t,
            "utc_hour_decimal": t_t,
            "hour": hour_str,
            "minute": minute_str,
            "offset_minutes": dt_min,
        })

    chips = np.stack(chips, axis=0).astype(np.float32)  # (T, H, W, C)
    valid_mask = np.array(valid_mask, dtype=np.int8)

    return chips, coords, time_meta, valid_mask


translation = [1, 2, 0, 4, 5, 6, 3, 8, 9, 10, 11, 13, 14, 15]


def processFile(yy, ddn, orbit, latb):

    #cs_file = glob.glob(
    #    CLOUDSATPATH + '2B-CLDCLASS-LIDAR/' + yy + '/' +
    #    ddn + '/' + yy + ddn + '*' + orbit + '*2B-CLDCLASS-LIDAR*' + 'P1_R05*.hdf'
    #)
    #aux_file = glob.glob(
    #    CLOUDSATPATH + 'ECMWF-AUX/' + yy + '/' +
    #    ddn + '/' + yy + ddn + '*' + orbit + '*ECMWF-AUX*' + 'P1_R05*.hdf'
    #)


    cs_file = glob.glob(os.path.join(
        CLOUDSATPATH,
        '2B-CLDCLASS-LIDAR',
        yy,
        ddn,
        yy + ddn + '*' + orbit + '*2B-CLDCLASS-LIDAR*' + 'P1_R05*.hdf'
    ))
    aux_file = glob.glob(os.path.join(
        CLOUDSATPATH,
        'ECMWF-AUX',
        yy,
        ddn,
        yy + ddn + '*' + orbit + '*ECMWF-AUX*' + 'P1_R05*.hdf'
    ))


    print(cs_file)
    print(aux_file)


    if len(cs_file) == 0 or len(aux_file) == 0:
        print("Did not find any cs_file or aux_file.")
        return

    [
        Pressure,
        DEM_elevation,
        Temperature,
        Specific_humidity,
        EC_height,
        Temperature_2m,
        U10_velocity,
        V10_velocity,
        UTC_Time
    ] = read_cs_ecmwf(aux_file[0], latbin=latb)

    [
        cs_clb,
        cs_clt,
        cs_cltype,
        cs_QC,
        Latitude,
        Longitude
    ] = read_2b_cldclass_lidar(cs_file[0], latbin=latb)

    Longitude[Longitude < 0] += 360

    Pressure = interpArray(Pressure, EC_height)
    Temperature = interpArray(Temperature, EC_height)
    Specific_humidity = interpArray(Specific_humidity, EC_height)

    N = len(cs_clb)

    cs_clb_2 = np.floor(2 * cs_clb).astype(int)
    cs_clt_2 = np.floor(2 * cs_clt).astype(int)

    cloud_class_mask_40_level = np.zeros((N, 40), dtype=np.int8)

    for i in range(N):
        for j in range(10):
            if cs_clb[i, j] < 0:
                break
            start_idx = cs_clb_2[i, j]
            end_idx = cs_clt_2[i, j] + 1
            cloud_type = cs_cltype[i, j]
            cloud_class_mask_40_level[i, start_idx:end_idx] = cloud_type

    i = 0

    while i < N:
        try:
            print(
                UTC_Time[i],
                yy,
                ddn,
                Latitude[i],
                Longitude[i],
                ABIDATA,
                OFFSETS_MINUTES,
                ALLOW_MISSING_TIMESTEPS
            )
            chip_stack, coords, chip_times, abi_valid_mask = processTimeSeries(
                UTC_Time[i],
                yy,
                ddn,
                Latitude[i],
                Longitude[i],
                ABIDATA,
                offsets_minutes=OFFSETS_MINUTES,
                allow_missing=ALLOW_MISSING_TIMESTEPS
            )
        except ValueError as e:
            print(e)
            i += 20
            continue
        except ImportError as e:
            print(e)
            i += 90
            continue
        except FileNotFoundError as e:
            print(e)
            i += 20
            continue
        except Exception as e:
            print(e)
            i += 20
            continue

        if i + 46 >= N:
            break

        dRange = np.arange(i + 46, i - 45, -1)

        print("Generating dictionary")
        aux_data = {
            "Pressure": Pressure[dRange],
            "DEM_elevation": DEM_elevation[dRange],
            "Temperature": Temperature[dRange],
            "Specific_humidity": Specific_humidity[dRange],
            "Temperature_2m": Temperature_2m[dRange],
            "U10_velocity": U10_velocity[dRange],
            "V10_velocity": V10_velocity[dRange],
            "UTC_Time": UTC_Time[dRange],
            "Latitude": Latitude[dRange],
            "Longitude": Longitude[dRange],
            "Cloud_mask": cloud_class_mask_40_level[dRange],
            "Cloud_mask_binary": (cloud_class_mask_40_level[dRange] != 0).astype(np.int8),
            "Cloud_class": cs_cltype[dRange],
            "ABI_offsets_minutes": np.array(OFFSETS_MINUTES, dtype=np.int32),
            "ABI_valid_mask": abi_valid_mask,
            "ABI_time_year": np.array([x["year"] if x["year"] is not None else "" for x in chip_times], dtype=object),
            "ABI_time_doy": np.array([x["doy"] if x["doy"] is not None else "" for x in chip_times], dtype=object),
            "ABI_time_utc_hour_decimal": np.array(
                [np.nan if x["utc_hour_decimal"] is None else x["utc_hour_decimal"] for x in chip_times],
                dtype=np.float32
            ),
            "ABI_time_hour": np.array([x["hour"] if x["hour"] is not None else "" for x in chip_times], dtype=object),
            "ABI_time_minute": np.array([x["minute"] if x["minute"] is not None else "" for x in chip_times], dtype=object),
        }

        fileName = f'{yy}-{ddn}-{orbit}_{coords[0]}-{coords[1]}-{i}'

        # optional band reorder if needed later:
        # chip_stack = chip_stack[..., translation]

        np.savez(
            os.path.join(SAVEDIR, fileName),
            chip=chip_stack,   # (T, 128, 128, 16)
            data=aux_data
        )

        print(fileName, UTC_Time[i], chip_stack.shape, flush=True)
        i += 45


# =================================
#    MAIN
# =================================

year = 0
dayskip = 0

try:
    year = sys.argv[1]
    dayskip = sys.argv[2]
    print("year", year, "dayskip", dayskip)
except Exception:
    print("Pass in year and dayskip as arguments")
    sys.exit(1)

print(os.listdir(os.path.join(ROOT_DIR, year)))


for day in os.listdir(ROOT_DIR + '/' + year):

    # if int(day) < int(dayskip):
    #    continue
    if int(day) > 1:
        continue

    print("Starting", day)
    print(os.path.join(ROOT_DIR, year, day))

    print(os.listdir(os.path.join(ROOT_DIR, year, day)))


    for file in os.listdir(os.path.join(ROOT_DIR, year, day)):
        if file.endswith('hdf'):
            orbit = file.split('_')[1]
            print(f"PROCESSING FILE {year} {day} {orbit}", flush=True)
            processFile(year, day, orbit, (latMin, latMax))

    GATHERED_ABI_FILES = {}
    COLLECTED_ABI_DATA = {}
