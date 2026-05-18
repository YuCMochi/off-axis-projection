"""sender.py — UDP packet builders and send helper."""
from __future__ import annotations

import socket
import struct


def pack_opentrack(x, y, z, yaw, pitch, roll) -> bytes:
    return struct.pack("<6d", x, y, z, yaw, pitch, roll)


def _to_24bit(value: float) -> bytes:
    val_int = round(value)
    val_int = max(-8388608, min(8388607, val_int))
    return val_int.to_bytes(3, byteorder="big", signed=True)


def pack_freed(x_cm: float, y_cm: float, z_cm: float,
               yaw: float, pitch: float, roll: float) -> bytes:
    packet = bytearray()
    packet.append(0xD1)
    packet.append(0x00)
    packet.extend(_to_24bit(yaw   * 32768))
    packet.extend(_to_24bit(pitch * 32768))
    packet.extend(_to_24bit(roll  * 32768))
    packet.extend(_to_24bit(x_cm * 10 * 64))
    packet.extend(_to_24bit(y_cm * 10 * 64))
    packet.extend(_to_24bit(z_cm * 10 * 64))
    packet.extend(_to_24bit(0))
    packet.extend(_to_24bit(0))
    packet.extend(bytes([0x00, 0x00]))
    checksum = (256 - (sum(packet) % 256)) % 256
    packet.append(checksum)
    return bytes(packet)


def send(sock: socket.socket, host: str, port: int, data: bytes) -> None:
    sock.sendto(data, (host, port))
