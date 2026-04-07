"""
Serial Sensor Reader for PTSD Trigger Detection System

Reads real physiological data from hardware sensors via serial port.
Hardware: MAX30102 (HR + temperature) + GSR module + ESP32 microcontroller

Supports HYBRID mode: some sensors real, some dummy/neutral.
Controlled by SENSOR_SOURCES in config.py.

Expected serial data format from ESP32 (JSON):
    {"heart_rate": 75.2, "gsr": 3.5, "skin_temp": 34.1}
    (HRV is calculated automatically from HR intervals)
"""

import sys
import os
import time
import json
import random
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import SERIAL_PORT, SERIAL_BAUD_RATE, SENSOR_SOURCES, NEUTRAL_VALUES
from src.utils.logger import setup_logger

logger = setup_logger("SerialReader")


class SerialSensorStream:
    """
    Reads physiological sensor data from ESP32 via serial port.
    Supports hybrid mode: mix real hardware + dummy/neutral per sensor.
    Drop-in replacement for DummySensorStream.
    """

    def __init__(self):
        """Initialize serial connection to ESP32."""
        self.port = SERIAL_PORT
        self.baud_rate = SERIAL_BAUD_RATE
        self.serial_conn = None
        self.current_state = "unknown"

        # For HRV calculation from HR intervals
        self._last_hr_time = None
        self._hr_intervals = []

        # For dummy fallback on non-serial sensors
        self._dummy_stream = None

        # Connect serial if any sensor is SERIAL
        has_serial = any(v == "SERIAL" for v in SENSOR_SOURCES.values())
        if has_serial:
            self._connect_serial()

        # Set up dummy stream for DUMMY sensors
        has_dummy = any(v == "DUMMY" for v in SENSOR_SOURCES.values())
        if has_dummy:
            from src.stress.dummy_data import DummySensorStream
            self._dummy_stream = DummySensorStream()

        logger.info(f"HybridSensorStream initialized")
        logger.info(f"  Sources: {SENSOR_SOURCES}")

    def _connect_serial(self):
        """Try connecting to the ESP32 serial port."""
        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=2)
            logger.info(f"Connected to {self.port} at {self.baud_rate} baud")
            time.sleep(2)  # Wait for ESP32 to initialize
        except ImportError:
            logger.error("pyserial not installed! Run: pip install pyserial")
        except Exception as e:
            logger.error(f"Cannot connect to {self.port}: {e}")
            logger.info("Serial sensors will use neutral fallback values.")

    def _read_serial(self) -> dict:
        """Read one JSON line from serial port."""
        if self.serial_conn is None or not self.serial_conn.is_open:
            return {}
        try:
            line = self.serial_conn.readline().decode("utf-8").strip()
            if line:
                return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug(f"Serial parse error: {e}")
        except Exception as e:
            logger.error(f"Serial read error: {e}")
        return {}

    def _calculate_hrv(self, heart_rate: float) -> float:
        """
        Calculate HRV (heart rate variability) from consecutive HR readings.
        HRV = standard deviation of R-R intervals in ms.
        """
        now = time.time()
        if self._last_hr_time is not None and heart_rate > 0:
            # R-R interval in ms
            rr_interval = 60000.0 / heart_rate
            self._hr_intervals.append(rr_interval)
            # Keep last 10 intervals
            if len(self._hr_intervals) > 10:
                self._hr_intervals = self._hr_intervals[-10:]

        self._last_hr_time = now

        if len(self._hr_intervals) >= 3:
            # SDNN (standard deviation of NN intervals)
            mean_rr = sum(self._hr_intervals) / len(self._hr_intervals)
            variance = sum((x - mean_rr) ** 2 for x in self._hr_intervals) / len(self._hr_intervals)
            hrv = math.sqrt(variance)
            return round(hrv, 1)

        return NEUTRAL_VALUES["hrv"]  # Not enough data yet

    def get_reading(self) -> dict:
        """
        Get one sensor reading using the per-sensor source config.
        Each sensor value comes from its configured source:
          SERIAL  → from ESP32 hardware
          DUMMY   → from dummy simulator
          AUTO    → calculated from other sensors
          NEUTRAL → fixed neutral value
        """
        # Get serial data (if any sensor uses it)
        serial_data = {}
        if self.serial_conn and self.serial_conn.is_open:
            serial_data = self._read_serial()

        # Get dummy data (if any sensor uses it)
        dummy_data = {}
        if self._dummy_stream:
            dummy_data = self._dummy_stream.get_reading()

        # Build reading per sensor
        reading = {}

        # ---- Heart Rate ----
        src = SENSOR_SOURCES.get("heart_rate", "DUMMY")
        if src == "SERIAL":
            reading["heart_rate"] = serial_data.get("heart_rate", NEUTRAL_VALUES["heart_rate"])
        elif src == "DUMMY":
            reading["heart_rate"] = dummy_data.get("heart_rate", NEUTRAL_VALUES["heart_rate"])
        else:  # NEUTRAL
            reading["heart_rate"] = NEUTRAL_VALUES["heart_rate"]

        # ---- GSR ----
        src = SENSOR_SOURCES.get("gsr", "DUMMY")
        if src == "SERIAL":
            reading["gsr"] = serial_data.get("gsr", NEUTRAL_VALUES["gsr"])
        elif src == "DUMMY":
            reading["gsr"] = dummy_data.get("gsr", NEUTRAL_VALUES["gsr"])
        else:
            reading["gsr"] = NEUTRAL_VALUES["gsr"]

        # ---- HRV ----
        src = SENSOR_SOURCES.get("hrv", "AUTO")
        if src == "AUTO":
            reading["hrv"] = self._calculate_hrv(reading["heart_rate"])
        elif src == "DUMMY":
            reading["hrv"] = dummy_data.get("hrv", NEUTRAL_VALUES["hrv"])
        else:
            reading["hrv"] = NEUTRAL_VALUES["hrv"]

        # ---- Skin Temperature ----
        src = SENSOR_SOURCES.get("skin_temp", "AUTO")
        if src == "AUTO" and SENSOR_SOURCES.get("heart_rate") == "SERIAL":
            # MAX30102 has a built-in temperature sensor
            reading["skin_temp"] = serial_data.get("skin_temp", NEUTRAL_VALUES["skin_temp"])
        elif src == "DUMMY":
            reading["skin_temp"] = dummy_data.get("skin_temp", NEUTRAL_VALUES["skin_temp"])
        else:
            reading["skin_temp"] = NEUTRAL_VALUES["skin_temp"]

        reading["timestamp"] = time.time()
        reading["state"] = self.current_state
        reading["sources"] = dict(SENSOR_SOURCES)  # Include source info for dashboard

        return reading

    def set_state(self, state: str):
        """Change the dummy sensor state (for testing)."""
        self.current_state = state
        if self._dummy_stream:
            self._dummy_stream.set_state(state)

    def close(self):
        """Close serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Serial connection closed")
