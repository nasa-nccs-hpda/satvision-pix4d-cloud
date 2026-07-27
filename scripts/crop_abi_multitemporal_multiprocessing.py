"""
Multitemporal CloudSat–ABI Collocation Pipeline

This script generates spatiotemporal training samples by collocating CloudSat
2B-CLDCLASS-LIDAR profiles with GOES-16 ABI L1B radiances.

For each CloudSat profile:
    • The nearest ABI pixel location is identified using a precomputed lat/lon grid.
    • Multiple ABI timesteps are extracted around the CloudSat observation time
      (e.g., every 20 minutes using configurable offsets).
    • A spatial chip (default: 128x128 pixels, 16 channels) is extracted per timestep.
    • Corresponding CloudSat + ECMWF auxiliary profiles are aligned and included.

Outputs:
    • .npz files containing:
        - chip: (T, H, W, C) multitemporal ABI data
        - data: dictionary with atmospheric profiles, cloud masks, and metadata

Key Features:
    • Supports configurable temporal offsets and chip sizes
    • Handles ABI 10-min and 15-min cadence automatically
    • Parallel processing over CloudSat orbits (ProcessPoolExecutor)
    • Robust handling of missing timesteps (optional)

Typical Use Case:
    Dataset generation for deep learning models (e.g., cloud classification,
    vertical structure prediction, or multimodal Earth observation models).

Notes:
    • Requires pyhdf (HDF4) and netCDF4 libraries
    • ABI data must be organized as: ABI_ROOT/YYYY/DDD/HH/
    • CloudSat data must follow standard mission directory structure
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import netCDF4 as nc
import numpy as np
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF
import pyhdf.VS
from pyhdf.VS import VS

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


def read_2b_cldclass_lidar(cs_file, latbin):
    f_2b = SD(cs_file, SDC.READ)
    cs_clb = f_2b.select("CloudLayerBase").get()
    cs_clt = f_2b.select("CloudLayerTop").get()
    cs_cltype = f_2b.select("CloudLayerType").get()

    hdf_2b = HDF(cs_file, SDC.READ)
    vs = hdf_2b.vstart()
    cs_qc = np.squeeze(vs.attach("Data_quality")[:])
    latitude = np.squeeze(vs.attach("Latitude")[:])
    longitude = np.squeeze(vs.attach("Longitude")[:])

    ilat = np.squeeze(np.argwhere((latitude >= latbin[0]) & (latitude <= latbin[1])))

    cs_clb = cs_clb[ilat, :]
    cs_clt = cs_clt[ilat, :]
    cs_qc = cs_qc[ilat]
    cs_cltype = cs_cltype[ilat, :]
    latitude = latitude[ilat]
    longitude = longitude[ilat]

    return cs_clb, cs_clt, cs_cltype, cs_qc, latitude, longitude


def read_cs_ecmwf(aux_file, latbin):
    f_ecmwf = SD(aux_file, SDC.READ)
    pressure = f_ecmwf.select("Pressure").get()
    temperature = f_ecmwf.select("Temperature").get()
    specific_humidity = f_ecmwf.select("Specific_humidity").get()

    hdf_ecmwf = HDF(aux_file, SDC.READ)
    vs = hdf_ecmwf.vstart()
    ec_height = np.squeeze(vs.attach("EC_height")[:])
    profile_time = np.squeeze(vs.attach("Profile_time")[:])
    utc_start = np.squeeze(vs.attach("UTC_start")[:])
    latitude = np.squeeze(vs.attach("Latitude")[:])
    longitude = np.squeeze(vs.attach("Longitude")[:])
    dem_elevation = np.squeeze(vs.attach("DEM_elevation")[:])
    temperature_2m = np.squeeze(vs.attach("Temperature_2m")[:])
    u10_velocity = np.squeeze(vs.attach("U10_velocity")[:])
    v10_velocity = np.squeeze(vs.attach("V10_velocity")[:])

    utc_time = (utc_start + profile_time) / 3600.0
    ilat = np.squeeze(np.argwhere((latitude >= latbin[0]) & (latitude <= latbin[1])))

    pressure = pressure[ilat, :]
    temperature = temperature[ilat, :]
    specific_humidity = specific_humidity[ilat, :]
    dem_elevation = dem_elevation[ilat]
    temperature_2m = temperature_2m[ilat]
    u10_velocity = u10_velocity[ilat]
    v10_velocity = v10_velocity[ilat]
    utc_time = utc_time[ilat]

    return (
        pressure,
        dem_elevation,
        temperature,
        specific_humidity,
        ec_height,
        temperature_2m,
        u10_velocity,
        v10_velocity,
        utc_time,
    )


def interp_array(arr, ec_height):
    z_grid = np.arange(40) * 0.5
    n = arr.shape[0]
    out = np.zeros((n, len(z_grid)), dtype=np.float32)

    for i in range(n):
        arr_tmp = np.squeeze(arr[i, :])
        ivalid = np.squeeze(np.where(arr_tmp > 0))
        out[i, :] = np.interp(
            z_grid,
            np.flip(ec_height[ivalid] / 1000.0),
            np.flip(arr_tmp[ivalid]),
        )

    return out


def build_latlon_context(latlondata, bound_size, length):
    with nc.Dataset(latlondata) as f:
        abi_long = np.array(f["Longitude"])
        abi_lat = np.array(f["Latitude"])

    abi_long_b = abi_long[bound_size:length - bound_size, bound_size:length - bound_size].copy()
    abi_lat_b = abi_lat[bound_size:length - bound_size, bound_size:length - bound_size].copy()

    abi_long_b[abi_long_b == -999] = 10
    abi_lat_b[abi_lat_b == -999] = 10
    abi_long_b[abi_long_b < 0] += 360

    long_min = abi_long_b.min()
    long_max = abi_long_b.max()
    lat_min = abi_lat_b.min()
    lat_max = abi_lat_b.max()

    lat_slice = abi_lat[:, 5424]
    lat_slice = lat_slice[18:-18][::-1]

    long_slice = abi_long[5424, :]
    long_slice = long_slice[18:-18]

    return {
        "abi_long": abi_long,
        "abi_lat": abi_lat,
        "lat_slice": lat_slice,
        "long_slice": long_slice,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "long_min": long_min,
        "long_max": long_max,
        "bound_size": bound_size,
        "length": length,
    }


def find_abi_coords(lat, lon, ctx):
    abi_lat = ctx["abi_lat"]
    abi_long = ctx["abi_long"]
    lat_slice = ctx["lat_slice"]
    long_slice = ctx["long_slice"]
    bound_size = ctx["bound_size"]
    length = ctx["length"]

    if lat < ctx["lat_min"] or lat > ctx["lat_max"] or lon < ctx["long_min"] or lon > ctx["long_max"]:
        raise ValueError("Latitude and Longitude are not correctly bounded")

    area_size = 1000
    lati = len(lat_slice) - np.searchsorted(lat_slice, lat) + 17
    loni = np.searchsorted(long_slice, lon) + 18

    distances = (
        np.abs(abi_lat[lati - area_size:lati + area_size, loni - area_size:loni + area_size] - lat)
        + np.abs(abi_long[lati - area_size:lati + area_size, loni - area_size:loni + area_size] - lon)
    )
    coords = np.array(np.unravel_index(distances.argmin(), distances.shape))

    if coords[0] == 0 or coords[1] == 0 or coords[0] == 2 * area_size - 1 or coords[1] == 2 * area_size - 1:
        distances = np.abs(abi_lat - lat) + np.abs(abi_long - lon)
        coords = np.array(np.unravel_index(distances.argmin(), distances.shape))
    else:
        coords[0] += lati - area_size
        coords[1] += loni - area_size

    if (
        coords[0] < bound_size
        or coords[1] < bound_size
        or coords[0] > length - bound_size
        or coords[1] > length - bound_size
    ):
        raise ValueError("Bad latitude and longitude")

    return coords


def gather_files(year, day, hour, abi_root, abi_file_cache):
    key = f"{year}-{day}-{hour}"
    if key in abi_file_cache:
        return abi_file_cache[key]

    abi_info = {
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
        "everyten": False,
    }

    hour_path = os.path.join(abi_root, year, day, hour)
    if not os.path.isdir(hour_path):
        raise FileNotFoundError(f"ABI hour path does not exist: {hour_path}")

    for filename in os.listdir(hour_path):
        if abi_info["ROOT_PATH"] is None:
            abi_info["ROOT_PATH"] = hour_path
            abi_info["YYYY"] = filename[27:31]
            abi_info["DDD"] = filename[31:34]
            abi_info["HH"] = filename[34:36]

        mm = filename[36:38]
        if mm == "10":
            abi_info["everyten"] = True
        if mm in abi_info:
            abi_info[mm].append(filename)

    abi_file_cache[key] = abi_info
    return abi_info


def get_l1b_stack(abipaths, year, day, hour, abi_root, abi_data_cache):
    cache_key = f"{year}-{day}-{hour}-{'|'.join(sorted(abipaths))}"
    if cache_key in abi_data_cache:
        return abi_data_cache[cache_key]

    if len(abipaths) != 16:
        raise ImportError(
            f"Bad timestep: expected 16 channels, got {len(abipaths)} for {year}/{day}/{hour}"
        )

    channels = []
    for file in abipaths:
        path = os.path.join(abi_root, year, day, hour, file)
        with nc.Dataset(path, "r") as ds:
            l1b = np.asarray(ds["Rad"])
        channel = int(file[19:21])
        channels.append((l1b, channel))

    channels.sort(key=lambda x: x[1])
    channels = [c[0] for c in channels]

    resized = []
    for c in channels:
        s = c.shape[0] // 5424
        if s == 1:
            c = np.repeat(c, 2, axis=0)
            c = np.repeat(c, 2, axis=1)
        elif s == 4:
            c = c[::2, ::2]
        resized.append(c)

    abi = np.stack(resized, axis=2).astype(np.float32)
    abi_data_cache[cache_key] = abi
    return abi


def get_time_key_info(t, year, day, abi_root, abi_file_cache):
    if np.floor(t) < 0 or np.floor(t) >= 24:
        raise ValueError("outside 0-23 bound")

    hour = int(np.floor(t))
    hour_str = f"{hour:02d}"
    data = gather_files(year, day, hour_str, abi_root, abi_file_cache)

    if data["everyten"]:
        minutes = np.round((t - np.floor(t)) * 6).astype(int) * 10
    else:
        minutes = np.round((t - np.floor(t)) * 4).astype(int) * 15

    if minutes == 60:
        if hour != 23:
            hour += 1
            minutes = 0
            hour_str = f"{hour:02d}"
            data = gather_files(year, day, hour_str, abi_root, abi_file_cache)
        else:
            minutes = 50 if data["everyten"] else 45

    minute_str = f"{minutes:02d}"

    if minute_str not in data or len(data[minute_str]) == 0:
        raise FileNotFoundError(f"No ABI files for {year}-{day} {hour_str}:{minute_str}")

    return data, hour_str, minute_str


def get_abi_chip_at_time(
    t,
    year,
    day,
    coords,
    abi_root,
    chip_half_size,
    abi_file_cache,
    abi_data_cache,
):
    data, hour_str, minute_str = get_time_key_info(t, year, day, abi_root, abi_file_cache)
    abi = get_l1b_stack(data[minute_str], data["YYYY"], data["DDD"], data["HH"], abi_root, abi_data_cache)

    chip = abi[
        coords[0] - chip_half_size:coords[0] + chip_half_size,
        coords[1] - chip_half_size:coords[1] + chip_half_size,
        :
    ]

    expected_shape = (2 * chip_half_size, 2 * chip_half_size, 16)
    if chip.shape != expected_shape:
        raise ValueError(f"Bad chip shape: got {chip.shape}, expected {expected_shape}")

    return chip, hour_str, minute_str


def process_time_series(
    t,
    year,
    day,
    lat,
    lon,
    abi_root,
    offsets,
    chip_half_size,
    allow_missing_timesteps,
    ctx,
    abi_file_cache,
    abi_data_cache,
):
    coords = find_abi_coords(lat, lon, ctx)
    chips = []
    time_meta = []
    valid_mask = []

    for dt_min in offsets:
        try:
            yy_t, ddn_t, t_t = shift_time(year, day, t, dt_min)
            chip, hour_str, minute_str = get_abi_chip_at_time(
                t_t,
                yy_t,
                ddn_t,
                coords,
                abi_root,
                chip_half_size,
                abi_file_cache,
                abi_data_cache,
            )
            valid = 1
        except Exception:
            if not allow_missing_timesteps:
                raise
            chip = np.full(
                (2 * chip_half_size, 2 * chip_half_size, 16),
                np.nan,
                dtype=np.float32
            )
            yy_t, ddn_t, t_t = None, None, None
            hour_str, minute_str = None, None
            valid = 0

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

    chips = np.stack(chips, axis=0).astype(np.float32)
    valid_mask = np.array(valid_mask, dtype=np.int8)
    return chips, coords, time_meta, valid_mask


def process_orbit(job):
    year, day, orbit, config = job

    abi_file_cache = {}
    abi_data_cache = {}
    ctx = build_latlon_context(
        config["latlondata"],
        config["bound_size"],
        config["length"],
    )

    cs_glob = os.path.join(
        config["cloudsatpath"],
        "2B-CLDCLASS-LIDAR",
        year,
        day,
        year + day + "*" + orbit + "*2B-CLDCLASS-LIDAR*" + "P1_R05*.hdf",
    )
    aux_glob = os.path.join(
        config["cloudsatpath"],
        "ECMWF-AUX",
        year,
        day,
        year + day + "*" + orbit + "*ECMWF-AUX*" + "P1_R05*.hdf",
    )

    cs_files = glob.glob(cs_glob)
    aux_files = glob.glob(aux_glob)

    if len(cs_files) == 0 or len(aux_files) == 0:
        return {"job": (year, day, orbit), "status": "missing_cloudsat_or_aux", "saved": 0}

    cs_file = cs_files[0]
    aux_file = aux_files[0]

    try:
        (
            pressure,
            dem_elevation,
            temperature,
            specific_humidity,
            ec_height,
            temperature_2m,
            u10_velocity,
            v10_velocity,
            utc_time,
        ) = read_cs_ecmwf(aux_file, latbin=(ctx["lat_min"], ctx["lat_max"]))

        (
            cs_clb,
            cs_clt,
            cs_cltype,
            cs_qc,
            latitude,
            longitude,
        ) = read_2b_cldclass_lidar(cs_file, latbin=(ctx["lat_min"], ctx["lat_max"]))

        longitude[longitude < 0] += 360

        pressure = interp_array(pressure, ec_height)
        temperature = interp_array(temperature, ec_height)
        specific_humidity = interp_array(specific_humidity, ec_height)

        n = len(cs_clb)
        cs_clb_2 = np.floor(2 * cs_clb).astype(int)
        cs_clt_2 = np.floor(2 * cs_clt).astype(int)
        cloud_class_mask_40_level = np.zeros((n, 40), dtype=np.int8)

        for i in range(n):
            for j in range(10):
                if cs_clb[i, j] < 0:
                    break
                start_idx = cs_clb_2[i, j]
                end_idx = cs_clt_2[i, j] + 1
                cloud_type = cs_cltype[i, j]
                cloud_class_mask_40_level[i, start_idx:end_idx] = cloud_type

        saved = 0
        i = 0

        while i < n:
            try:
                if i + 46 >= n:
                    break

                d_range = np.arange(i + 46, i - 45, -1)

                chip_stack, coords, chip_times, abi_valid_mask = process_time_series(
                    utc_time[i],
                    year,
                    day,
                    latitude[i],
                    longitude[i],
                    config["abi_root"],
                    config["offsets"],
                    config["chip_half_size"],
                    config["allow_missing_timesteps"],
                    ctx,
                    abi_file_cache,
                    abi_data_cache,
                )

                file_name = f"{year}-{day}-{orbit}_{coords[0]}-{coords[1]}-{i}.npz"
                out_path = os.path.join(config["save_dir"], file_name)

                if (not config["overwrite"]) and os.path.exists(out_path):
                    i += 45
                    continue

                aux_data = {
                    "Pressure": pressure[d_range],
                    "DEM_elevation": dem_elevation[d_range],
                    "Temperature": temperature[d_range],
                    "Specific_humidity": specific_humidity[d_range],
                    "Temperature_2m": temperature_2m[d_range],
                    "U10_velocity": u10_velocity[d_range],
                    "V10_velocity": v10_velocity[d_range],
                    "UTC_Time": utc_time[d_range],
                    "Latitude": latitude[d_range],
                    "Longitude": longitude[d_range],
                    "Cloud_mask": cloud_class_mask_40_level[d_range],
                    "Cloud_mask_binary": (cloud_class_mask_40_level[d_range] != 0).astype(np.int8),
                    "Cloud_class": cs_cltype[d_range],
                    "CloudSat_QC": cs_qc[d_range],
                    "ABI_offsets_minutes": np.array(config["offsets"], dtype=np.int32),
                    "ABI_valid_mask": abi_valid_mask,
                    "ABI_time_year": np.array([x["year"] if x["year"] is not None else "" for x in chip_times], dtype=object),
                    "ABI_time_doy": np.array([x["doy"] if x["doy"] is not None else "" for x in chip_times], dtype=object),
                    "ABI_time_utc_hour_decimal": np.array(
                        [np.nan if x["utc_hour_decimal"] is None else x["utc_hour_decimal"] for x in chip_times],
                        dtype=np.float32,
                    ),
                    "ABI_time_hour": np.array([x["hour"] if x["hour"] is not None else "" for x in chip_times], dtype=object),
                    "ABI_time_minute": np.array([x["minute"] if x["minute"] is not None else "" for x in chip_times], dtype=object),
                }

                np.savez(out_path, chip=chip_stack, data=aux_data)
                saved += 1
                i += 45

            except ValueError:
                i += 20
            except ImportError:
                i += 90
            except FileNotFoundError:
                i += 20
            except Exception as e:
                return {
                    "job": (year, day, orbit),
                    "status": "failed",
                    "saved": saved,
                    "error": repr(e),
                }

        return {"job": (year, day, orbit), "status": "ok", "saved": saved}

    except Exception as e:
        return {"job": (year, day, orbit), "status": "failed", "saved": 0, "error": repr(e)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clip multitemporal ABI chips matched to CloudSat profiles."
    )
    parser.add_argument("--latlondata", type=str, required=True)
    parser.add_argument("--cloudsatpath", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--abi-root", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--dayskip", type=int, default=0)
    parser.add_argument("--day-end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--offsets", type=int, nargs="+", default=[-40, -20, 0, 20, 40])
    parser.add_argument("--chip-half-size", type=int, default=64)
    parser.add_argument("--allow-missing-timesteps", action="store_true")
    parser.add_argument("--bound-size", type=int, default=1600)
    parser.add_argument("--length", type=int, default=10848)
    parser.add_argument("--orbit", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_jobs(root_dir, year, dayskip=0, day_end=None, orbit=None):
    year_dir = os.path.join(root_dir, year)
    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"Year directory not found: {year_dir}")

    jobs = []

    for day in sorted(os.listdir(year_dir)):
        try:
            day_int = int(day)
        except ValueError:
            continue

        if day_int < dayskip:
            continue
        if day_end is not None and day_int > day_end:
            continue

        day_dir = os.path.join(year_dir, day)
        if not os.path.isdir(day_dir):
            continue

        for file in sorted(os.listdir(day_dir)):
            if not file.endswith(".hdf"):
                continue

            this_orbit = file.split("_")[1]
            if orbit is not None and this_orbit != orbit:
                continue

            jobs.append((year, day, this_orbit))

    return sorted(set(jobs))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    config = {
        "latlondata": args.latlondata,
        "cloudsatpath": args.cloudsatpath,
        "root_dir": args.root_dir,
        "abi_root": args.abi_root,
        "save_dir": args.save_dir,
        "offsets": args.offsets,
        "chip_half_size": args.chip_half_size,
        "allow_missing_timesteps": args.allow_missing_timesteps,
        "bound_size": args.bound_size,
        "length": args.length,
        "overwrite": args.overwrite,
    }

    try:
        jobs0 = discover_jobs(
            args.root_dir,
            args.year,
            dayskip=args.dayskip,
            day_end=args.day_end,
            orbit=args.orbit,
        )
    except Exception as e:
        print(f"Failed discovering jobs: {e}", file=sys.stderr)
        sys.exit(1)

    if len(jobs0) == 0:
        print("No jobs found.", flush=True)
        return

    jobs = [(year, day, orbit, config) for year, day, orbit in jobs0]
    print(f"Discovered {len(jobs)} jobs", flush=True)

    results = []

    if args.workers == 1:
        for job in jobs:
            result = process_orbit(job)
            print(result, flush=True)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_map = {ex.submit(process_orbit, job): job for job in jobs}
            for fut in as_completed(future_map):
                job = future_map[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "job": job[:3],
                        "status": "failed",
                        "saved": 0,
                        "error": repr(e),
                    }
                print(result, flush=True)
                results.append(result)

    n_ok = sum(r["status"] == "ok" for r in results)
    n_fail = sum(r["status"] == "failed" for r in results)
    total_saved = sum(r.get("saved", 0) for r in results)

    print(
        {
            "total_jobs": len(results),
            "ok": n_ok,
            "failed": n_fail,
            "total_saved": total_saved,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()