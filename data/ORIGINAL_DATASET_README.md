---
license: cc-by-4.0
---
This readme file was generated on [2025-08-01] by [Sizhe Ma, Dr. Katherine Flanigan, Dr. Mario Berges]
[Formats template from: Cornell University's Research Data Management Service Group]

## GENERAL INFORMATION

- Title of Dataset: A Multimodal Railway Vibration and Vision Dataset (Rail-VIVID)

- Author/Principal Investigator Information
  Name: Katherine Flanigan
  ORCID: 0000-0002-2454-5713
  Institution: Carnegie Mellon University
  Address: 5000 Forbes Ave
  Email: kflaniga@andrew.cmu.edu

- Author/Associate or Co-investigator Information
  Name: Mario Berges
  ORCID: 0000-0003-2948-9236
  Institution: Carnegie Mellon University
  Address: 5000 Forbes Ave
  Email: marioberges@cmu.edu

- Author/Alternate Contact Information
  Name: Sizhe Ma
  ORCID: 0000-0002-2532-5915
  Institution: Carnegie Mellon University
  Address: 5000 Forbes Ave
  Email: sizhem@andrew.cmu.edu

- Date and type of data collection: 2025-04-11 --- Real World Data;
                                    2025-04-18 --- Real World Data;


- Geographic location of data collection: 2025-04-11 --- Railroad Development Corporation (RDC), Orbisonia, PA --- from Point A [40.23951169072744, -77.89729802853955] to Point B [40.23184123709821, -77.87387005111827];
                                          2025-04-18 --- Railroad Development Corporation (RDC), Orbisonia, PA --- from Point A [40.23951169072744, -77.89729802853955] to Point B [40.23184123709821, -77.87387005111827];

- Information about funding sources that supported the collection of the data: This work was funded by the Federal Railroad Administration under Contract No. 693JJ623C000020 and the Pennsylvania Infrastructure Technology Alliance. The authors gratefully acknowledge Pop-Up Metro, LLC for providing access to the testbed and supporting operational staff.

## SHARING/ACCESS INFORMATION

- Licenses/restrictions placed on the data: No competing interests declared.

- **Citation for dataset: Ma, S., Flanigan, K. A., Bergés, M., & Brooks, J. D. (2026). A Multimodal Railway Vibration and Vision Dataset (Rail-VIVID). Scientific Data. 10.1038/s41597-026-07555-y**

- Links to other publicly accessible locations of the data: None.

- Links/relationships to ancillary data sets: Weather data from a station located 40.23 km from the site.

- Was data derived from another source? No.
  If yes, list source(s): 

- Recommended citation for this dataset: 

## DATA & FILE OVERVIEW

### File List [Name of the folder >>> Description of the folder]:

Anomaly >>> Contains 3D point cloud scans and physical characteristics of the nine documented ground-truth track anomalies. 
AtoB_[Speed]_[Index] >>> Data folders for runs from Point A to Point B (uphill), categorized by speed (20 km/hr, 22.5 km/hr). 
BtoA_[Speed]_[Index] >>> Data folders for runs from Point B to Point A (downhill), categorized by speed (22.5 km/hr, 25 km/hr, 27.5 km/hr). 
Script >>> Contains Python data acquisition and computational processing scripts used for syncing and calibrating the data.

- Relationship between files, if important: Each run folder contains a .CSV file with synchronized vibration, environmental, and GPS data, and a subfolder of .JPG images. Both components are linked by shared Unix timestamps in their filenames.

- Additional related data collected that was not included in the current data package: None.

- Are there multiple versions of the dataset? See "Release Notes" at the bottom.


## METHODOLOGICAL INFORMATION

- Description of methods used for collection/generation of data: Data were collected using six accelerometers and a global shutter camera mounted on a Pop-Up Metro Class 230 train.
-- Guillermo Montero, Jeremy Yin, Katherine A. Flanigan, Mario Bergés, James D. Brooks, "Anomaly identification algorithms for indirect structural health monitoring using a laboratory-scale railroad track system," Proc. SPIE 12488, Health Monitoring of Structural and Biological Systems XVII, 124881L (25 April 2023); https://doi.org/10.1117/12.2658463
-- Jeremy Yin, Guillermo Montero, Katherine A. Flanigan, Mario Bergés, James D. Brooks, "Open-source hardware and software for a laboratory-scale track and moving vehicle actuation system used for indirect broken rail detection," Proc. SPIE 12486, Sensors and Smart Structures Technologies for Civil, Mechanical, and Aerospace Systems 2023, 1248609 (18 April 2023); https://doi.org/10.1117/12.2658438

- **Known issues:** (From previous FPS fluctuation) Image-to-location alignment was anchored at the start and end of each run; between these two anchor points, frame timestamps reflect when each frame began saving rather than its capture instant. Under variable-FPS conditions this introduces a non-constant drift that can displace an image by up to ~15 m from its true location. Location-dependent analyses should therefore validate image positions against in-scene references (e.g., the surveyed cones at Point A, Point B, and each anomaly) rather than relying on timestamps alone.

- Methods for processing the data: Raw accelerometer data was digitized at 2 kHz, converted to physical units (g), and passed through a 500 Hz low-pass Butterworth filter.

- Instrument-specific information needed to interpret the data: None. The vibration system uses Silicon Designs Model 2012 accelerometers. The vision system uses a See3CAM 50CUGM industrial camera.

- Software-specific information needed to interpret the data: The data can be interpreted using Python (version 3.8 or higher). Minimum required packages include numpy, pandas, scipy, and matplotlib.

- Standards and calibration information, if appropriate: Accelerometer data were calibrated using standard procedures before each run.

- Environmental/experimental conditions: Data were collected under varying weather conditions typical in Orbisonia, PA, including sunny and overcast days.

- Describe any quality-assurance procedures performed on the data: Quality checks were performed to ensure data integrity, including verifying sensor calibration, checking for missing data, and ensuring proper alignment of GPS and accelerometer data. Data anomalies were flagged and manually reviewed.

- People involved with sample collection, processing, analysis and/or submission: Sizhe Ma (data collection, data processing and quality assurance), Katherine Flanigan (supervised the project and provided critical guidance and suggestions throughout the study), Mario Berges (supervised the project and provided critical guidance and suggestions throughout the study), Guillermo Montero (experimental design), Jeremy Yin (experimental design).


## DATA-SPECIFIC INFORMATION:

- Number of variables: 12

- Number of cases/rows: Ranging from 347034 to 593028 rows, with a total of 9401255 rows.

- Variable List [Variable name >>> description (units)]: 
-- Timestamp >>> The time at which the data was recorded (MM/DD/YYYY HH:MM:SS AM/PM).
-- Channel_1 >>> Accelerometer reading from sensor 1 (m/s^2).
-- Channel_2 >>> Accelerometer reading from sensor 2 (m/s^2).
-- Channel_3 >>> Accelerometer reading from sensor 3 (m/s^2).
-- Channel_4 >>> Accelerometer reading from sensor 4 (m/s^2).
-- Channel_5 >>> Accelerometer reading from sensor 5 (m/s^2).
-- Channel_6 >>> Accelerometer reading from sensor 6 (m/s^2).
-- Temperature >>> Interpolated ambient temperature upsampled from hourly weather readings at 15 minutes intervals (°F).
-- Humidity >>> Interpolated relative humidity upsampled from weather readings at 15 minutes intervals (0 to 1).
-- Latitude >>> GPS latitude coordinate at the time of data recording (degrees).
-- Longitude >>> GPS longitude coordinate at the time of data recording (degrees).

- Missing data codes: no known missing data.

- Specialized formats or other abbreviations used: None (no known missing data)

## RELEASE NOTES:

**Version:** 1.0.2
**Release Date:** June 11th, 2026

### Summary
No changes to the underlying data files from v1.0.1; Added DOI.

### Key Features
- Processed accelerometer, environmental, and GPS data.
- Data segmented by direction and speed.

### Known Issues
- (From previous FPS fluctuation) Image-to-location alignment was anchored at the start and end of each run; between these two anchor points, frame timestamps reflect when each frame began saving rather than its capture instant. Under variable-FPS conditions this introduces a non-constant drift that can displace an image by up to ~15 m from its true location. Location-dependent analyses should therefore validate image positions against in-scene references (e.g., the surveyed cones at Point A, Point B, and each anomaly) rather than relying on timestamps alone.

### Future Plans
- Update timestamps for vision data.