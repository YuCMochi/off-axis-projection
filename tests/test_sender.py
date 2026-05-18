import struct
import pytest


def test_pack_opentrack_length_and_values():
    from sender import pack_opentrack
    data = pack_opentrack(1.0, 2.0, 3.0, 10.0, 20.0, 30.0)
    assert len(data) == 48
    assert struct.unpack("<6d", data) == (1.0, 2.0, 3.0, 10.0, 20.0, 30.0)


def test_pack_freed_length():
    from sender import pack_freed
    data = pack_freed(0, 0, 0, 0, 0, 0)
    assert len(data) == 29


def test_pack_freed_message_type_and_camera_id():
    from sender import pack_freed
    data = pack_freed(0, 0, 0, 0, 0, 0)
    assert data[0] == 0xD1
    assert data[1] == 0x00


def test_pack_freed_zero_checksum_value():
    from sender import pack_freed
    data = pack_freed(0, 0, 0, 0, 0, 0)
    # first 28 bytes sum: 0xD1=209, rest zero -> checksum = (256-209)%256 = 47
    assert data[28] == 47


def test_pack_freed_checksum_makes_total_zero_mod256():
    from sender import pack_freed
    data = pack_freed(10.0, 20.0, 150.0, 45.0, -10.0, 5.0)
    assert sum(data) % 256 == 0


def test_pack_freed_checksum_nonzero_rotation():
    from sender import pack_freed
    data = pack_freed(0, 0, 0, 90.0, 0, 0)
    assert sum(data) % 256 == 0


def test_pack_freed_clamps_overflow_without_raising():
    from sender import pack_freed
    data = pack_freed(99999, 99999, 99999, 999, 999, 999)
    assert len(data) == 29
    assert sum(data) % 256 == 0


def test_pack_freed_clamps_underflow_without_raising():
    from sender import pack_freed
    data = pack_freed(-99999, -99999, -99999, -999, -999, -999)
    assert len(data) == 29
    assert sum(data) % 256 == 0
