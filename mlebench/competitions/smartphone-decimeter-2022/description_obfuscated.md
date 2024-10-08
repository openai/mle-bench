# Task

Compute smartphones location based on raw location measurements from Android smartphones collected in opensky and light urban roads.

# Metric

Submissions are scored on the mean of the 50th and 95th percentile distance errors. For every `phone` and once per second, the horizontal distance (in meters) is computed between the predicted latitude/longitude and the ground truth latitude/longitude. These distance errors form a distribution from which the 50th and 95th percentile errors are calculated (i.e. the 95th percentile error is the value, in meters, for which 95% of the distance errors are smaller). The 50th and 95th percentile errors are then averaged for each phone. Lastly, the mean of these averaged values is calculated across all phones in the test set.

# Submission Format

For each `phone` and `UnixTimeMillis` in the sample submission, you must predict the latitude and longitude. The sample submission typically requires a prediction once per second but may include larger gaps if there were too few valid GNSS signals. The submission file should contain a header and have the following format:

```
phone,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
2020-05-15-US-MTV-1_Pixel4,1273608785432,37.904611315634504,-86.48107806249548
2020-05-15-US-MTV-1_Pixel4,1273608786432,37.904611315634504,-86.48107806249548
2020-05-15-US-MTV-1_Pixel4,1273608787432,37.904611315634504,-86.48107806249548
```

# Dataset

**[train/test]/[drive_id]/[phone_name]/supplemental/[phone_name][.20o/.21o/.22o/.nmea]** - Equivalent data to the gnss logs in other formats used by the GPS community.

**train/[drive_id]/[phone_name]/ground_truth.csv** - Reference locations at expected timestamps.

- `MessageType` - "Fix", the prefix of sentence.

- `Provider` - "GT", short for ground truth.

- `[Latitude/Longitude]Degrees` - The [WGS84](https://en.wikipedia.org/w/index.php?title=World_Geodetic_System&oldid=1013033380) latitude, longitude (in decimal degrees) estimated by the reference GNSS receiver (NovAtel SPAN). When extracting from the NMEA file, linear interpolation has been applied to align the location to the expected non-integer timestamps.

- `AltitudeMeters` - The height above the WGS84 ellipsoid (in meters) estimated by the reference GNSS receiver.

- `SpeedMps`* - The speed over ground in meters per second.

- `AccuracyMeters` - The estimated horizontal accuracy radius in meters of this location at the 68th percentile confidence level. This means that there is a 68% chance that the true location of the device is within a distance of this uncertainty of the reported location.

- `BearingDegrees` - Bearing is measured in degrees clockwise from north. It ranges from 0 to 359.999 degrees.

- `UnixTimeMillis` - An integer number of milliseconds since the GPS epoch (1970/1/1 midnight UTC). Converted from [GnssClock](https://developer.android.com/reference/android/location/GnssClock).

**[train/test]/[drive_id]/[phone_name]/device_gnss.csv** - Each row contains raw GNSS measurements, derived values, and a baseline estimated location.. This baseline was computed using correctedPrM and the satellite positions, using a standard Weighted Least Squares (WLS) solver, with the phone's position (x, y, z), clock bias (t), and isrbM for each unique signal type as states for each epoch. Some of the raw measurement fields are not included in this file because they are deprecated or are not populated in the original gnss_log.txt.

- `MessageType` - "Raw", the prefix of sentence.

- `utcTimeMillis` - Milliseconds since UTC epoch (1970/1/1), converted from GnssClock.

- `TimeNanos` - The GNSS receiver internal hardware clock value in nanoseconds.

- `LeapSecond` - The leap second associated with the clock's time.

- `FullBiasNanos` - The difference between hardware clock (getTimeNanos()) inside GPS receiver and the true GPS time since 0000Z, January 6, 1980, in nanoseconds.

- `BiasNanos` - The clock's sub-nanosecond bias.

- `BiasUncertaintyNanos` - The clock's bias uncertainty (1-sigma) in nanoseconds.

- `DriftNanosPerSecond` - The clock's drift in nanoseconds per second.

- `DriftUncertaintyNanosPerSecond` - The clock's drift uncertainty (1-sigma) in nanoseconds per second.

- `HardwareClockDiscontinuityCount` - Count of hardware clock discontinuities.

- `Svid` - The satellite ID.

- `TimeOffsetNanos` - The time offset at which the measurement was taken in nanoseconds.

- `State` - Integer signifying sync state of the satellite. Each bit in the integer attributes to a particular state information of the measurement. See the **metadata/raw_state_bit_map.json** file for the mapping between bits and states.

- `ReceivedSvTimeNanos` - The received GNSS satellite time, at the measurement time, in nanoseconds.

- `ReceivedSvTimeUncertaintyNanos` - The error estimate (1-sigma) for the received GNSS time, in nanoseconds.

- `Cn0DbHz` - The carrier-to-noise density in dB-Hz.

- `PseudorangeRateMetersPerSecond` - The pseudorange rate at the timestamp in m/s.

- `PseudorangeRateUncertaintyMetersPerSecond` - The pseudorange's rate uncertainty (1-sigma) in m/s.

- `AccumulatedDeltaRangeState` - This indicates the state of the 'Accumulated Delta Range' measurement. Each bit in the integer attributes to state of the measurement. See the **metadata/accumulated_delta_range_state_bit_map.json** file for the mapping between bits and states.

- `AccumulatedDeltaRangeMeters` - The accumulated delta range since the last channel reset, in meters.

- `AccumulatedDeltaRangeUncertaintyMeters` - The accumulated delta range's uncertainty (1-sigma) in meters.

- `CarrierFrequencyHz` - The carrier frequency of the tracked signal.

- `MultipathIndicator` - A value indicating the 'multipath' state of the event.

- `ConstellationType` - GNSS constellation type. The mapping to human readable values is provided in the **metadata/constellation_type_mapping.csv** file.

- `CodeType` - The GNSS measurement's code type. Only available in recent logs.

- `ChipsetElapsedRealtimeNanos` - The elapsed real-time of this clock since system boot, in nanoseconds. Only available in recent logs.

- `ArrivalTimeNanosSinceGpsEpoch` - An integer number of nanoseconds since the GPS epoch (1980/1/6 midnight UTC). Its value equals round((Raw::TimeNanos - Raw::FullBiasNanos), for each unique epoch described in the Raw sentences.

- `RawPseudorangeMeters` - Raw pseudorange in meters. It is the product between the speed of light and the time difference from the signal transmission time (receivedSvTimeInGpsNanos) to the signal arrival time (Raw::TimeNanos - Raw::FullBiasNanos - Raw;;BiasNanos). Its uncertainty can be approximated by the product between the speed of light and the ReceivedSvTimeUncertaintyNanos.

- `SignalType` - The GNSS signal type is a combination of the constellation name and the frequency band. Common signal types measured by smartphones include GPS_L1, GPS_L5, GAL_E1, GAL_E5A, GLO_G1, BDS_B1I, BDS_B1C, BDS_B2A, QZS_J1, and QZS_J5.

- `ReceivedSvTimeNanosSinceGpsEpoch` - The signal transmission time received by the chipset, in the numbers of nanoseconds since the GPS epoch. Converted from ReceivedSvTimeNanos, this derived value is in a unified time scale for all constellations, while ReceivedSvTimeNanos refers to the time of day for GLONASS and the time of week for non-GLONASS constellations.

- `SvPosition[X/Y/Z]EcefMeters` - The satellite position (meters) in an ECEF coordinate frame at best estimate of "true signal transmission time" defined as ttx = receivedSvTimeInGpsNanos - satClkBiasNanos (defined below). They are computed with the satellite broadcast ephemeris, and have ~1-meter error with respect to the true satellite position.

- `Sv[Elevation/Azimuth]Degrees` - The elevation and azimuth in degrees of the satellite. They are computed using the WLS estimated user position.

- `SvVelocity[X/Y/Z]EcefMetersPerSecond` - The satellite velocity (meters per second) in an ECEF coordinate frame at best estimate of "true signal transmission time" ttx. They are computed with the satellite broadcast ephemeris, with this algorithm.

- `SvClockBiasMeters` - The satellite time correction combined with the satellite hardware delay in meters at the signal transmission time (receivedSvTimeInGpsNanos). Its time equivalent is termed as satClkBiasNanos. satClkBiasNanos equals the satelliteTimeCorrection minus the satelliteHardwareDelay. As defined in IS-GPS-200H Section 20.3.3.3.3.1, satelliteTimeCorrection is calculated from ∆tsv = af0 + af1(t - toc) + af2(t - toc)2 + ∆tr, while satelliteHardwareDelay is defined in Section 20.3.3.3.3.2. Parameters in the equations above are provided on the satellite broadcast ephemeris.

- `SvClockDriftMetersPerSecond` - The satellite clock drift in meters per second at the signal transmission time (receivedSvTimeInGpsNanos). It equals the difference of the satellite clock biases at t+0.5s and t-0.5s.

- `IsrbMeters` - The Inter-Signal Range Bias (ISRB) in meters from a non-GPS-L1 signal to GPS-L1 signals. For example, when the isrbM of GPS L5 is 1000m, it implies that a GPS L5 pseudorange is 1000m longer than the GPS L1 pseudorange transmitted by the same GPS satellite. It's zero for GPS-L1 signals. ISRB is introduced in the GPS chipset level and estimated as a state in the Weighted Least Squares engine.

- `IonosphericDelayMeters` - The ionospheric delay in meters, estimated with the Klobuchar model.

- `TroposphericDelayMeters` - The tropospheric delay in meters, estimated with the EGNOS model by Nigel Penna, Alan Dodson and W. Chen (2001).

- `WlsPositionXEcefMeters` - WlsPositionYEcefMeters,WlsPositionZEcefMeters: User positions in ECEF estimated by a Weighted-Least-Square (WLS) solver.

**[train/test]/[drive_id]/[phone_name]/device_imu.csv** - Readings the phone's accelerometer, gyroscope, and magnetometer.

- `MessageType` - which of the three instruments the row's data is from.

- `utcTimeMillis` - The sum of `elapsedRealtimeNanos` below and the estimated device boot time at UTC, after a recent NTP (Network Time Protocol) sync.

- `Measurement[X/Y/Z]` - [x/y/z]_uncalib without bias compensation.

- `Bias[X/Y/Z]MicroT` - Estimated [x/y/z]_bias. Null in datasets collected in earlier dates.

**[train/test]/[drive_id]/[phone_name]/supplemental/rinex.o** A text file of GNSS measurements on , collected from Android APIs (same as the "Raw" messages above), then converted to the [RINEX v3.03 format](http://rtcm.info/RINEX_3.04.IGS.RTCM_Final.pdf). refers to the last two digits of the year. During the conversion, the following treatments are taken to comply with the RINEX format. Essentially, this file contains a subset of information in the _GnssLog.txt.

1. The epoch time (in GPS time scale as per RINEX standard) is computed from [TimeNanos - (FullBiasNanos + BiasNanos)](https://developer.android.com/reference/android/location/GnssClock#getFullBiasNanos()), and then floored to the 100-nanosecond level to meet the precision requirement of the RINEX epoch time. The sub-100-nanosecond part has been wiped out by subtracting the raw pseudorange by the distance equivalent, and by subtracting the carrier phase range by the product of pseudorange rate and the sub-100-nanosecond part.
2. An epoch of measurements will not be converted to RINEX, if any of the following conditions occurs: a) the BiasUncertaintyNanos is larger or equal than 1E6, b) GnssClock values are invalid, e.g. FullBiasNanos is not a meaningful number.
3. Pseudorange value will not be converted if any of the following conditions occurs:

- [STATE_CODE_LOCK](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_CODE_LOCK) (for non-GAL E1) or [STATE_GAL_E1BC_CODE_LOCK](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_GAL_E1BC_CODE_LOCK) (for GAL E1) is not set,
- [STATE_TOW_DECODED](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_TOW_DECODED) and [STATE_TOW_KNOWN](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_TOW_KNOWN) are not set for non-GLO signals,
- [STATE_GLO_TOD_DECODED](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_GLO_TOD_DECODED) and [STATE_GLO_TOD_KNOWN](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_GLO_TOD_KNOWN) are not set for GLO signals,
- CN0 is less than 20 dB-Hz,

1. ReceivedSvTimeUncertaintyNanos is larger than 500 nanoseconds, Carrier frequency is out of nominal range of each band. Loss of lock indicator (LLI) is set to 1 if [ADR_STATE_CYCLE_SLIP](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_CYCLE_SLIP) is set, or 2 if [ADR_STATE_HALF_CYCLE_REPORTED](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_HALF_CYCLE_REPORTED) is set and [ADR_STATE_HALF_CYCLE_RESOLVED](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_HALF_CYCLE_RESOLVED) is not set, or blank if [ADR_STATE_VALID](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_VALID) is not set or [ADR_STATE_RESET](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_RESET) is set, or 0 otherwise.

**[train/test]/[drive_id]/[phone_name]/supplemental/gnss_log.txt** - The phone's logs as generated by the [GnssLogger App](https://play.google.com/store/apps/details?id=com.google.android.apps.location.gps.gnsslogger&hl=en_US&gl=US). [This notebook](https://www.kaggle.com/sohier/loading-gnss-logs/) demonstrates how to parse the logs. Each gnss file contains several sub-datasets, each of which is detailed below:

Raw - The raw GNSS measurements of one GNSS signal (each satellite may have 1-2 signals for L5-enabled smartphones), collected from the Android API [GnssMeasurement](https://developer.android.com/reference/android/location/GnssMeasurement).

- `utcTimeMillis` - Milliseconds since UTC epoch (1970/1/1), converted from [GnssClock](https://developer.android.com/reference/android/location/GnssClock)

- [`TimeNanos`](https://developer.android.com/reference/android/location/GnssClock#getTimeNanos()) - The GNSS receiver internal hardware clock value in nanoseconds.

- [`LeapSecond`](https://developer.android.com/reference/android/location/GnssClock#getLeapSecond()) - The leap second associated with the clock's time.

- [`TimeUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssClock#getTimeUncertaintyNanos()) - The clock's time uncertainty (1-sigma) in nanoseconds.

- [`FullBiasNanos`](https://developer.android.com/reference/android/location/GnssClock#getFullBiasNanos()) - The difference between hardware clock [getTimeNanos()](https://developer.android.com/reference/android/location/GnssClock#getTimeNanos()) inside GPS receiver and the true GPS time since 0000Z, January 6, 1980, in nanoseconds.

- [`BiasNanos`](https://developer.android.com/reference/android/location/GnssClock#getBiasNanos()) - The clock's sub-nanosecond bias.

- [`BiasUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssClock#getBiasUncertaintyNanos()) - The clock's bias uncertainty (1-sigma) in nanoseconds.

- [`DriftNanosPerSecond`](https://developer.android.com/reference/android/location/GnssClock#getDriftNanosPerSecond()) - The clock's drift in nanoseconds per second.

- [`DriftUncertaintyNanosPerSecond`](https://developer.android.com/reference/android/location/GnssClock#getDriftUncertaintyNanosPerSecond()) - The clock's drift uncertainty (1-sigma) in nanoseconds per second.

- [`HardwareClockDiscontinuityCount`](https://developer.android.com/reference/android/location/GnssClock#getHardwareClockDiscontinuityCount()) - Count of hardware clock discontinuities.

- [`Svid`](https://developer.android.com/reference/android/location/GnssMeasurement#getSvid()) - The satellite ID. More info can be found [here](https://developer.android.com/reference/android/location/GnssMeasurement#getSvid()).

- [`TimeOffsetNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getTimeOffsetNanos()) - The time offset at which the measurement was taken in nanoseconds.

- [`State`](https://developer.android.com/reference/android/location/GnssMeasurement#getState()) - Integer signifying sync state of the satellite. Each bit in the integer attributes to a particular state information of the measurement. See the **metadata/raw_state_bit_map.json** file for the mapping between bits and states.

- [`ReceivedSvTimeNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getReceivedSvTimeNanos()) - The received GNSS satellite time, at the measurement time, in nanoseconds.

- [`ReceivedSvTimeUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getReceivedSvTimeUncertaintyNanos()) - The error estimate (1-sigma) for the received GNSS time, in nanoseconds.

- [`Cn0DbHz`](https://developer.android.com/reference/android/location/GnssMeasurement#getCn0DbHz()) - The carrier-to-noise density in dB-Hz.

- [`PseudorangeRateMetersPerSecond`](https://developer.android.com/reference/android/location/GnssMeasurement#getPseudorangeRateMetersPerSecond()) - The pseudorange rate at the timestamp in m/s.

- [`PseudorangeRateUncertaintyMetersPerSecond`](https://developer.android.com/reference/android/location/GnssMeasurement#getPseudorangeRateUncertaintyMetersPerSecond()) - The pseudorange's rate uncertainty (1-sigma) in m/s.

- [`AccumulatedDeltaRangeState`](https://developer.android.com/reference/android/location/GnssMeasurement#getAccumulatedDeltaRangeState()) - This indicates the state of the 'Accumulated Delta Range' measurement. Each bit in the integer attributes to state of the measurement. See the **metadata/accumulated_delta_range_state_bit_map.json** file for the mapping between bits and states.

- [`AccumulatedDeltaRangeMeters`](https://developer.android.com/reference/android/location/GnssMeasurement#getAccumulatedDeltaRangeMeters()) - The accumulated delta range since the last channel reset, in meters.

- [`AccumulatedDeltaRangeUncertaintyMeters`](https://developer.android.com/reference/android/location/GnssMeasurement#getAccumulatedDeltaRangeUncertaintyMeters()) - The accumulated delta range's uncertainty (1-sigma) in meters.

- [`CarrierFrequencyHz`](https://developer.android.com/reference/android/location/GnssMeasurement#getCarrierFrequencyHz()) - The carrier frequency of the tracked signal.

- [`CarrierCycles`](https://developer.android.com/reference/android/location/GnssMeasurement#getCarrierCycles()) - The number of full carrier cycles between the satellite and the receiver. Null in these datasets.

- [`CarrierPhase`](https://developer.android.com/reference/android/location/GnssMeasurement#getCarrierPhase()) - The RF phase detected by the receiver. Null in these datasets.

- [`CarrierPhaseUncertainty`](https://developer.android.com/reference/android/location/GnssMeasurement#getCarrierPhaseUncertainty()) - The carrier-phase's uncertainty (1-sigma). Null in these datasets.

- [`MultipathIndicator`](https://developer.android.com/reference/android/location/GnssMeasurement#getMultipathIndicator()) - A value indicating the 'multipath' state of the event.

- [`SnrInDb`](https://developer.android.com/reference/android/location/GnssMeasurement#getSnrInDb()) - The (post-correlation & integration) Signal-to-Noise ratio (SNR) in dB.

- [`ConstellationType`](https://developer.android.com/reference/android/location/GnssMeasurement#getConstellationType()) - GNSS constellation type. It's an integer number, whose mapping to string value is provided in the constellation_type_mapping.csv file.

- [`AgcDb`](https://developer.android.com/reference/android/location/GnssMeasurement#getAutomaticGainControlLevelDb()) - The Automatic Gain Control level in dB.

- [`BasebandCn0DbHz`](https://developer.android.com/reference/android/location/GnssMeasurement#getBasebandCn0DbHz()) - The baseband carrier-to-noise density in dB-Hz. Only available in Android 11.

- [`FullInterSignalBiasNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getFullInterSignalBiasNanos()) - The GNSS measurement's inter-signal bias in nanoseconds with sub-nanosecond accuracy. Only available in Pixel 5 logs in 2021. Only available in Android 11.

- [`FullInterSignalBiasUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getFullInterSignalBiasUncertaintyNanos()) - The GNSS measurement's inter-signal bias uncertainty (1 sigma) in nanoseconds with sub-nanosecond accuracy. Only available in Android 11.

- [`SatelliteInterSignalBiasNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getSatelliteInterSignalBiasNanos()) - The GNSS measurement's satellite inter-signal bias in nanoseconds with sub-nanosecond accuracy. Only available in Android 11.

- [`SatelliteInterSignalBiasUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getSatelliteInterSignalBiasUncertaintyNanos()) - The GNSS measurement's satellite inter-signal bias uncertainty (1 sigma) in nanoseconds with sub-nanosecond accuracy. Only available in Android 11.

- [`CodeType`](https://developer.android.com/reference/android/location/GnssMeasurement#getCodeType()) - The GNSS measurement's code type. Only available in recent logs.

- [`ChipsetElapsedRealtimeNanos`](https://developer.android.com/reference/android/location/GnssClock#getElapsedRealtimeNanos()) - The elapsed real-time of this clock since system boot, in nanoseconds. Only available in recent logs.

Status - The status of a GNSS signal, as collected from the Android API [GnssStatus](https://developer.android.com/reference/android/location/GnssStatus.Callback).

- `UnixTimeMillis` - Milliseconds since UTC epoch (1970/1/1), reported from the last location changed by [GPS](https://developer.android.com/reference/android/location/LocationManager#GPS_PROVIDER) provider.

- [`SignalCount`](https://developer.android.com/reference/android/location/GnssStatus#getSatelliteCount()) - The total number of satellites in the satellite list.

- `SignalIndex` - The index of current signal.

- [`ConstellationType`](https://developer.android.com/reference/android/location/GnssStatus#getConstellationType(int)): The constellation type of the satellite at the specified index.

- [`Svid`](https://developer.android.com/reference/android/location/GnssStatus#getSvid(int)): The satellite ID.

- [`CarrierFrequencyHz`](https://developer.android.com/reference/android/location/GnssStatus#getCarrierFrequencyHz(int)): The carrier frequency of the signal tracked.

- [`Cn0DbHz`](https://developer.android.com/reference/android/location/GnssStatus#getCn0DbHz(int)): The carrier-to-noise density at the antenna of the satellite at the specified index in dB-Hz.

- [`AzimuthDegrees`](https://developer.android.com/reference/android/location/GnssStatus#getAzimuthDegrees(int)): The azimuth the satellite at the specified index.

- [`ElevationDegrees`](https://developer.android.com/reference/android/location/GnssStatus#getElevationDegrees(int)): The elevation of the satellite at the specified index.

- [`UsedInFix`](https://developer.android.com/reference/android/location/GnssStatus#usedInFix(int)): Whether the satellite at the specified index was used in the calculation of the most recent position fix.

- [`HasAlmanacData`](https://developer.android.com/reference/android/location/GnssStatus#hasAlmanacData(int)): Whether the satellite at the specified index has almanac data.

- [`HasEphemerisData`](https://developer.android.com/reference/android/location/GnssStatus#hasEphemerisData(int)): Whether the satellite at the specified index has ephemeris data.

- [`BasebandCn0DbHz`](https://developer.android.com/reference/android/location/GnssStatus#getBasebandCn0DbHz(int)): The baseband carrier-to-noise density of the satellite at the specified index in dB-Hz.

OrientationDeg - Each row represents an estimated device orientation, collected from Android API [SensorManager#getOrientation](https://developer.android.com/reference/android/hardware/SensorManager#getOrientation(float%5B%5D,%20float%5B%5D)).This message is only available in logs collected since March 2021.

- `utcTimeMillis` - The sum of `elapsedRealtimeNanos` below and the estimated device boot time at UTC, after a recent NTP (Network Time Protocol) sync.

- [`elapsedRealtimeNanos`](https://developer.android.com/reference/android/hardware/SensorEvent#timestamp) - The time in nanoseconds at which the event happened.

- `yawDeg` - If the screen is in portrait mode, this value equals the Azimuth degree (modulus to 0°~360°). If the screen is in landscape mode, it equals the sum (modulus to 0°~360°) of the screen rotation angle (either 90° or 270°) and the Azimuth degree. *Azimuth*, refers to the angle of rotation about the -z axis. This value represents the angle between the device's y axis and the magnetic north pole.

- `rollDeg` - *Roll*, angle of rotation about the y axis. This value represents the angle between a plane perpendicular to the device's screen and a plane perpendicular to the ground.

- `pitchDeg` - *Pitch*, angle of rotation about the x axis. This value represents the angle between a plane parallel to the device's screen and a plane parallel to the ground.
