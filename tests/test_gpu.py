"""Tests for reading the GPU's clock floor and temperature.

The kernel's files are recreated in a temporary directory with the values
the device reported on 2026-11-17: a floor of 114.75 MHz under a top of
510.
"""

from hazard_zone.gpu import clock_floor_warning, temperature


def devfreq(tmp_path, low, high):
    directory = tmp_path / "devfreq" / "17000000.gpu"
    directory.mkdir(parents=True)
    (directory / "min_freq").write_text(f"{low}\n")
    (directory / "max_freq").write_text(f"{high}\n")
    return str(tmp_path / "devfreq" / "17000000.*")


def zone(tmp_path, index, kind, millidegrees):
    directory = tmp_path / "thermal" / f"thermal_zone{index}"
    directory.mkdir(parents=True)
    (directory / "type").write_text(f"{kind}\n")
    (directory / "temp").write_text(f"{millidegrees}\n")
    return str(tmp_path / "thermal" / "thermal_zone*")


# -- the clock floor ---------------------------------------------------------


def test_a_floor_below_the_top_is_warned_with_both_numbers(tmp_path):
    warning = clock_floor_warning(devfreq(tmp_path, 114750000, 510000000))
    assert warning is not None
    assert "115" in warning and "510" in warning


def test_the_warning_says_how_to_fix_it(tmp_path):
    """The path to write is in the message: the fix needs no other document."""
    warning = clock_floor_warning(devfreq(tmp_path, 114750000, 510000000))
    assert warning is not None
    assert "min_freq" in warning and "17000000.gpu" in warning


def test_a_floor_at_the_top_gives_no_warning(tmp_path):
    assert clock_floor_warning(devfreq(tmp_path, 510000000, 510000000)) is None


def test_no_gpu_directory_gives_no_warning(tmp_path):
    """A desktop, or another board: the system runs the same."""
    assert clock_floor_warning(str(tmp_path / "nothing" / "*")) is None


def test_an_unreadable_value_gives_no_warning(tmp_path):
    assert clock_floor_warning(devfreq(tmp_path, "busy", 510000000)) is None


# -- the temperature ---------------------------------------------------------


def test_the_gpu_sensor_is_found_among_the_others(tmp_path):
    zone(tmp_path, 0, "CPU-therm", 45000)
    pattern = zone(tmp_path, 1, "GPU-therm", 47500)
    zone(tmp_path, 2, "AUX-therm", 44000)
    assert temperature(pattern) == 47.5


def test_no_gpu_sensor_gives_none(tmp_path):
    pattern = zone(tmp_path, 0, "CPU-therm", 45000)
    assert temperature(pattern) is None


def test_no_sensors_at_all_give_none(tmp_path):
    assert temperature(str(tmp_path / "nothing" / "*")) is None


def test_an_unreadable_temperature_gives_none(tmp_path):
    pattern = zone(tmp_path, 0, "GPU-therm", "")
    assert temperature(pattern) is None
