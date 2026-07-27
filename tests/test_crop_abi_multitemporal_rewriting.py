from datetime import datetime, timezone
import json

import numpy as np
import pytest

from satvision_pix4d.preprocessing.cloudsat_abi.abi import (
    ABIArchive,
    ABIFileInfo,
    ABIGeometry,
)
from satvision_pix4d.preprocessing.cloudsat_abi.cloudsat import (
    CloudSatAuxiliaryTransect,
    CloudSatOrbitFile,
    CloudSatTransect,
)
from satvision_pix4d.preprocessing.cloudsat_abi.config import (
    CropConfig,
    get_satellite,
)
from satvision_pix4d.preprocessing.cloudsat_abi.pipeline import (
    CloudSatABICollocationPipeline,
    CloudSatLabelError,
    run_parallel,
)
from satvision_pix4d.preprocessing.cloudsat_abi.merra2 import MERRA2Reader
from satvision_pix4d.readers.abi_l1b_common_grid import (
    common_to_native_indices,
    normalize_abi_l1b_for_model,
)
from satvision_pix4d.preprocessing.cloudsat_abi.writer import NPZChipWriter
from satvision_pix4d.view.cloudsat_abi_cropping_cli import (
    build_parser,
    config_from_args,
)


def abi_name(channel, platform="G16", stamp="2019336235959"):
    return (
        f"OR_ABI-L1b-RadF-M6C{channel:02d}_{platform}_s{stamp}_"
        "e2019337000959_c.nc"
    )


def make_config(tmp_path, **overrides):
    values = {
        "abi_root": tmp_path / "abi",
        "cloudsat_root": tmp_path / "cloudsat",
        "cloudsat_index_root": tmp_path / "cloudsat" / "2B-CLDCLASS-LIDAR",
        "latlon_path": tmp_path / "grid.nc",
        "output_dir": tmp_path / "output",
        "year": 2019,
        "satellite": get_satellite("goes18"),
        "transect": (-30.0, 30.0),
        "offsets": (-20, 0, 20),
        "chip_size": 8,
        "profile_stride": 2,
        "profiles_per_chip": 5,
        "metadata": frozenset({"cloudsat"}),
    }
    values.update(overrides)
    return CropConfig(**values)


def make_transect(tmp_path, profiles=9):
    base = np.full((profiles, 3), -1.0)
    top = np.full((profiles, 3), -1.0)
    cloud_type = np.zeros((profiles, 3), dtype=np.int8)
    base[:, 0], top[:, 0], cloud_type[:, 0] = 2.0, 3.0, 4
    return CloudSatTransect(
        source=tmp_path / "orbit.hdf",
        latitude=np.linspace(-4, 4, profiles),
        longitude=np.linspace(-80, -72, profiles),
        utc_hour=np.full(profiles, 12.0),
        quality=np.zeros(profiles, dtype=np.int8),
        cloud_layer_base=base,
        cloud_layer_top=top,
        cloud_layer_type=cloud_type,
        cloud_layer_count=np.ones(profiles, dtype=np.int8),
    )


def make_auxiliary_transect(tmp_path, profiles=9):
    levels = 40
    profile_values = np.arange(profiles * levels, dtype=np.float32).reshape(
        profiles, levels
    )
    return CloudSatAuxiliaryTransect(
        source=tmp_path / "aux.hdf",
        pressure=profile_values + 100.0,
        dem_elevation=np.arange(profiles, dtype=np.float32),
        temperature=profile_values + 200.0,
        specific_humidity=profile_values + 300.0,
        ec_height=np.broadcast_to(
            np.arange(levels, dtype=np.float32) * 500.0,
            (profiles, levels),
        ),
        temperature_2m=np.arange(profiles, dtype=np.float32) + 280.0,
        skin_temperature=np.arange(profiles, dtype=np.float32) + 285.0,
        surface_pressure=np.arange(profiles, dtype=np.float32) + 900.0,
        u10_velocity=np.arange(profiles, dtype=np.float32) + 1.0,
        v10_velocity=np.arange(profiles, dtype=np.float32) + 2.0,
        utc_time=np.arange(profiles, dtype=np.float32) / 10.0,
        latitude=np.linspace(-4, 4, profiles),
        longitude=np.linspace(-80, -72, profiles),
    )


def test_abi_file_info_reads_platform_channel_and_year_rollover():
    info = ABIFileInfo.from_path(abi_name(7))

    assert info is not None
    assert info.timestamp == datetime(
        2019, 12, 2, 23, 59, 59, tzinfo=timezone.utc
    )
    assert info.channel == 7
    assert info.platform == "G16"


def test_archive_selects_complete_requested_platform_scan_across_hour(tmp_path):
    directory = tmp_path / "2019" / "336" / "23"
    directory.mkdir(parents=True)
    for channel in range(1, 17):
        (directory / abi_name(channel)).touch()
        (directory / abi_name(channel, platform="G17")).touch()

    geometry = type("GeometryStub", (), {"latitude": np.zeros((8, 8))})()
    archive = ABIArchive(
        tmp_path, geometry, get_satellite("goes16"), max_delta_minutes=2
    )
    scan_time, paths = archive.nearest_scan(
        datetime(2019, 12, 3, 0, 0, 30, tzinfo=timezone.utc)
    )

    assert scan_time == datetime(
        2019, 12, 2, 23, 59, 59, tzinfo=timezone.utc
    )
    assert set(paths) == set(range(1, 17))
    assert all("_G16_" in path.name for path in paths.values())


def test_geometry_nearest_uses_local_refinement():
    geometry = ABIGeometry.__new__(ABIGeometry)
    rows, columns = np.mgrid[0:100, 0:100]
    geometry.latitude = rows.astype(np.float32)
    geometry.longitude = columns.astype(np.float32)
    geometry.valid = np.ones((100, 100), dtype=bool)
    geometry.use_360 = False
    geometry.lat_min = 0.0
    geometry.lat_max = 99.0
    geometry.coarse_target_size = 256

    assert geometry.nearest(52.1, 37.2) == (52, 37)


def test_geometry_retains_original_inner_disk_and_rejects_limb_chip():
    geometry = ABIGeometry.__new__(ABIGeometry)
    latitude = np.arange(100, dtype=np.float32).reshape(10, 10)
    geometry.latitude = latitude
    geometry.longitude = latitude + 100
    geometry.valid = np.ones((10, 10), dtype=bool)
    geometry.valid[3:7, 5:7] = False

    assert geometry.inside_inner_disk(2, 2, 2)
    assert geometry.inside_inner_disk(8, 8, 2)
    assert not geometry.inside_inner_disk(9, 8, 2)
    assert geometry.valid_fraction(5, 5, 4) == 0.5
    chip_latitude, chip_longitude = geometry.crop_latlon(5, 5, 4)
    assert chip_latitude.shape == (4, 4)
    assert chip_latitude[0, 0] == 33
    assert chip_longitude[-1, -1] == 166


def test_native_indices_match_original_one_km_resampling():
    assert ABIArchive._native_indices(3, 7, 2.0).tolist() == [6, 8, 10, 12]
    assert ABIArchive._native_indices(3, 7, 1.0).tolist() == [3, 4, 5, 6]
    assert ABIArchive._native_indices(3, 7, 0.5).tolist() == [1, 2, 2, 3]
    assert common_to_native_indices(3, 7, 0.5).tolist() == [1, 2, 2, 3]


def test_shared_abi_model_normalization_is_explicit():
    chip = np.asarray([1.0, 3.0], dtype=np.float32)
    normalized = normalize_abi_l1b_for_model(chip, mean=1.0, std=2.0)

    assert normalized.tolist() == [0.0, 1.0]


def test_merra2_reader_samples_chip_pixels_and_normalizes_variables(tmp_path):
    reader = MERRA2Reader(tmp_path, variables=("T", "QV", "T2m"))

    assert reader.outputs == ("Temperature", "WV", "T2m")

    class FakeVariable:
        dimensions = ("time", "lev", "lat", "lon")

        def __init__(self):
            self.data = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)

        def __getitem__(self, item):
            return self.data[item]

    lat_index = np.asarray([[0, 1], [2, 1]])
    lon_index = np.asarray([[1, 2], [3, 0]])

    sampled = reader._sample_variable(FakeVariable(), 0, lat_index, lon_index)

    assert sampled.shape == (2, 2, 2)
    assert sampled[0, 0].tolist() == [1, 13]
    assert sampled[1, 0].tolist() == [11, 23]


def test_cloudsat_transect_owns_window_and_cloud_mask_logic(tmp_path):
    transect = make_transect(tmp_path)

    arrays = transect.metadata_arrays(center=4, count=5)

    assert arrays["cloudsat_latitude"].shape == (5,)
    assert arrays["cloudsat_cloud_class"].shape == (5, 40)
    assert arrays["cloudsat_cloud_class"][0, 4:7].tolist() == [4, 4, 4]
    assert arrays["cloudsat_cloud_binary_mask"][0, 4:7].tolist() == [1, 1, 1]
    with pytest.raises(IndexError):
        transect.profile_window(center=1, count=5)


def test_cloudsat_mask_distinguishes_clear_from_missing_profiles(tmp_path):
    transect = make_transect(tmp_path, profiles=3)
    transect.cloud_layer_count[:] = [1, 0, -9]
    transect.cloud_layer_base[1:] = -99
    transect.cloud_layer_top[1:] = -99
    transect.cloud_layer_type[1:] = -9

    mask = transect.cloud_mask()

    assert np.any(mask[0] > 0)
    assert np.all(mask[1] == 0)
    assert np.all(mask[2] == -1)
    assert transect.profile_validity().tolist() == [True, True, False]


def test_crop_config_validates_merra_and_normalizes_transect(tmp_path):
    config = make_config(tmp_path, transect=(30.0, -30.0))
    assert config.transect == (-30.0, 30.0)

    with pytest.raises(ValueError, match="merra2_root"):
        make_config(tmp_path, metadata=frozenset({"merra2"}))

    with pytest.raises(ValueError, match="requires CloudSat"):
        make_config(
            tmp_path, metadata=frozenset(), profile_selection="chip"
        )

    with pytest.raises(ValueError, match="cloudsat_aux"):
        make_config(tmp_path, metadata=frozenset({"cloudsat_aux"}))


def test_cli_builds_typed_satellite_configuration(tmp_path):
    args = build_parser().parse_args([
        "--abi-root", str(tmp_path / "abi"),
        "--cloudsat-root", str(tmp_path / "cloudsat"),
        "--latlon-path", str(tmp_path / "west.nc"),
        "--output-dir", str(tmp_path / "output"),
        "--year", "2019",
        "--transect", "-30", "30",
        "--satellite", "goes18",
        "--metadata",
    ])

    config = config_from_args(args)

    assert config.satellite.name == "GOES-18"
    assert config.satellite.region == "west"
    assert config.metadata == frozenset()


def test_cli_accepts_cloudsat_aux_metadata_group(tmp_path):
    aux_root = tmp_path / "ECMWF-AUX"
    args = build_parser().parse_args([
        "--abi-root", str(tmp_path / "abi"),
        "--cloudsat-root", str(tmp_path / "cloudsat"),
        "--latlon-path", str(tmp_path / "west.nc"),
        "--output-dir", str(tmp_path / "output"),
        "--year", "2019",
        "--satellite", "goes18",
        "--metadata", "cloudsat", "cloudsat_aux",
        "--cloudsat-aux-root", str(aux_root),
    ])

    config = config_from_args(args)

    assert config.metadata == frozenset({"cloudsat", "cloudsat_aux"})
    assert config.cloudsat_aux_root == aux_root


def test_cloudsat_auxiliary_arrays_are_saved_with_selected_profiles(tmp_path):
    transect = make_transect(tmp_path)
    auxiliary_transect = make_auxiliary_transect(tmp_path)
    orbit_file = CloudSatOrbitFile(336, "72433", transect.source)

    class FakeGeometry:
        @staticmethod
        def nearest(latitude, longitude):
            return 10, 20

        @staticmethod
        def crop_latlon(row, column, size):
            return (
                np.zeros((size, size), dtype=np.float32),
                np.zeros((size, size), dtype=np.float32),
            )

        @staticmethod
        def solar_zenith_angle(latitudes, longitudes, timestamp):
            return np.full(latitudes.shape, 80.0, dtype=np.float32)

        @staticmethod
        def view_zenith_angle(latitudes, longitudes, satellite_longitude):
            return np.full(latitudes.shape, 20.0, dtype=np.float32)

    class FakeABI:
        geometry = FakeGeometry()

        @staticmethod
        def crop(requested, row, column, size):
            return np.zeros((size, size, 16), dtype=np.float32), requested

    config = make_config(
        tmp_path,
        metadata=frozenset({"cloudsat", "cloudsat_aux"}),
    )
    pipeline = CloudSatABICollocationPipeline(
        config,
        abi_archive=FakeABI(),
        cloudsat_reader=object(),
        writer=NPZChipWriter(config.output_dir),
    )

    sample = pipeline.build_sample(
        orbit_file, transect, center=4, auxiliary_transect=auxiliary_transect
    )
    arrays = sample.auxiliary_arrays

    assert arrays["cloudsat_pressure"].shape == (5, 40)
    assert arrays["cloudsat_temperature"].shape == (5, 40)
    assert arrays["cloudsat_specific_humidity"].shape == (5, 40)
    assert arrays["cloudsat_ec_height"].shape == (5, 40)
    assert arrays["cloudsat_dem_elevation"].tolist() == [2, 3, 4, 5, 6]
    assert arrays["cloudsat_temperature_2m"].tolist() == [
        282, 283, 284, 285, 286
    ]
    assert arrays["cloudsat_skin_temperature"].tolist() == [
        287, 288, 289, 290, 291
    ]
    assert arrays["cloudsat_surface_pressure"].tolist() == [
        902, 903, 904, 905, 906
    ]
    assert sample.metadata["cloudsat_aux_source"] == str(
        auxiliary_transect.source
    )


@pytest.mark.parametrize(
    ("satellite", "filename"),
    [
        ("goes16", "ABI_EAST_GEO_TOPO_LOMSK.nc"),
        ("goes18", "ABI_WEST_GEO_TOPO_LOMSK.nc"),
        ("goes19", "ABI_EAST_GEO_TOPO_LOMSK.nc"),
    ],
)
def test_cli_infers_geometry_file_from_satellite(tmp_path, satellite, filename):
    parser = build_parser()
    args = parser.parse_args([
        "--abi-root", str(tmp_path / "abi"),
        "--cloudsat-root", str(tmp_path / "cloudsat"),
        "--latlon-dir", str(tmp_path / "geometry"),
        "--output-dir", str(tmp_path / "output"),
        "--year", "2019",
        "--transect", "-30", "30",
        "--satellite", satellite,
    ])

    config = config_from_args(args)

    assert config.latlon_path == tmp_path / "geometry" / filename


def test_explicit_geometry_file_overrides_directory_inference(tmp_path):
    parser = build_parser()
    override = tmp_path / "custom-grid.nc"
    args = parser.parse_args([
        "--abi-root", str(tmp_path / "abi"),
        "--cloudsat-root", str(tmp_path / "cloudsat"),
        "--latlon-dir", str(tmp_path / "geometry"),
        "--latlon-path", str(override),
        "--output-dir", str(tmp_path / "output"),
        "--year", "2019",
        "--transect", "-30", "30",
        "--satellite", "goes18",
    ])

    assert config_from_args(args).latlon_path == override


def test_pipeline_components_are_injectable_and_preserve_output_schema(tmp_path):
    transect = make_transect(tmp_path)
    orbit_file = CloudSatOrbitFile(336, "72433", transect.source)

    class FakeGeometry:
        @staticmethod
        def nearest(latitude, longitude):
            return 10, 20

        @staticmethod
        def crop_latlon(row, column, size):
            return (
                np.zeros((size, size), dtype=np.float32),
                np.zeros((size, size), dtype=np.float32),
            )

        @staticmethod
        def solar_zenith_angle(latitudes, longitudes, timestamp):
            return np.full(latitudes.shape, 80.0, dtype=np.float32)

        @staticmethod
        def view_zenith_angle(latitudes, longitudes, satellite_longitude):
            return np.full(latitudes.shape, 20.0, dtype=np.float32)

    class FakeABI:
        geometry = FakeGeometry()

        @staticmethod
        def crop(requested, row, column, size):
            return np.zeros((size, size, 16), dtype=np.float32), requested

    class FakeCloudSatReader:
        @staticmethod
        def discover_orbits(year, day_start, day_end, orbit):
            return [orbit_file]

        @staticmethod
        def read(path, latitude_bounds):
            return transect

    config = make_config(tmp_path, max_chips=1)
    pipeline = CloudSatABICollocationPipeline(
        config,
        abi_archive=FakeABI(),
        cloudsat_reader=FakeCloudSatReader(),
        writer=NPZChipWriter(config.output_dir),
    )

    assert pipeline.run() == 1
    outputs = list(config.output_dir.glob("*.npz"))
    assert len(outputs) == 1
    assert outputs[0].name.startswith("GOES18_west_")
    with np.load(outputs[0]) as data:
        assert data["ABI/chip"].shape == (3, 8, 8, 16)
        assert data["ABI/offsets_minutes"].tolist() == [-20, 0, 20]
        assert data["ABI/solar_zenith_angle"].shape == (3, 8, 8)
        assert data["ABI/view_zenith_angle"].shape == (3, 8, 8)
        assert data["ABI/solar_zenith_angle"][0, 0, 0] == 80.0
        assert data["ABI/view_zenith_angle"][0, 0, 0] == 20.0
        assert data["CloudSat/cloud_class"].shape == (5, 40)
        assert data["CloudSat/cloud_binary_mask"].shape == (5, 40)
        assert data["CloudSat/abi_row"].shape == (5,)
        assert "chip" not in data.files
        assert "cloudsat_cloud_class" not in data.files
        metadata = str(data["metadata_json"])
        assert '"satellite":"GOES-18"' in metadata
        assert '"satellite_region":"west"' in metadata
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["cloudsat_cloud_pixel_fraction"] == pytest.approx(0.075)
        assert metadata["cloudsat_cloud_pixel_percentage"] == pytest.approx(7.5)
        assert metadata["cloudsat_cloud_percentage"] == pytest.approx(7.5)
        assert metadata["cloudsat_cloudy_profile_fraction"] == pytest.approx(1.0)
        assert metadata["cloudsat_cloudy_profile_percentage"] == pytest.approx(
            100.0
        )


def test_chip_profile_selection_covers_top_to_bottom_in_abi_row_order(tmp_path):
    transect = make_transect(tmp_path)
    orbit_file = CloudSatOrbitFile(336, "72433", transect.source)

    class TrackGeometry:
        @staticmethod
        def nearest(latitude, longitude):
            return int(round(50 + latitude)), 50

        @staticmethod
        def crop_latlon(row, column, size):
            return (
                np.zeros((size, size), dtype=np.float32),
                np.zeros((size, size), dtype=np.float32),
            )

        @staticmethod
        def solar_zenith_angle(latitudes, longitudes, timestamp):
            return np.full(latitudes.shape, 80.0, dtype=np.float32)

        @staticmethod
        def view_zenith_angle(latitudes, longitudes, satellite_longitude):
            return np.full(latitudes.shape, 20.0, dtype=np.float32)

    class FakeABI:
        geometry = TrackGeometry()

        @staticmethod
        def crop(requested, row, column, size):
            return np.zeros((size, size, 16), dtype=np.float32), requested

    config = make_config(tmp_path, profile_selection="chip")
    pipeline = CloudSatABICollocationPipeline(
        config,
        abi_archive=FakeABI(),
        cloudsat_reader=object(),
        writer=NPZChipWriter(config.output_dir),
    )

    sample = pipeline.build_sample(orbit_file, transect, center=4)

    assert sample.auxiliary_arrays["cloudsat_abi_row"].tolist() == list(
        range(46, 54)
    )
    assert sample.auxiliary_arrays["cloudsat_profile_index"].tolist() == list(
        range(8)
    )


def test_missing_cloudsat_center_is_rejected_before_abi_io(tmp_path):
    transect = make_transect(tmp_path)
    transect.cloud_layer_count[:] = -9
    orbit_file = CloudSatOrbitFile(336, "72433", transect.source)

    class FailIfUsedGeometry:
        @staticmethod
        def nearest(latitude, longitude):
            raise AssertionError("ABI geometry should not be used")

    class FailIfUsedABI:
        geometry = FailIfUsedGeometry()

        @staticmethod
        def crop(requested, row, column, size):
            raise AssertionError("ABI data should not be read")

    config = make_config(tmp_path)
    pipeline = CloudSatABICollocationPipeline(
        config,
        abi_archive=FailIfUsedABI(),
        cloudsat_reader=object(),
        writer=NPZChipWriter(config.output_dir),
    )

    with pytest.raises(CloudSatLabelError, match="no valid retrieval"):
        pipeline.build_sample(orbit_file, transect, center=4)


def test_parallel_runner_validates_worker_count_before_loading_data(tmp_path):
    config = make_config(tmp_path)

    with pytest.raises(ValueError, match="workers must be at least 1"):
        run_parallel(config, workers=0)

    limited = make_config(tmp_path, max_chips=2)
    with pytest.raises(ValueError, match="max_chips requires workers=1"):
        run_parallel(limited, workers=2)
