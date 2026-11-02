# modules for user_thread
import threading

# modules for video_thread
import numpy as np
import cv2
from datetime import datetime

# modules for DAQ_thread
import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
import os

# modules for GPS_thread
import serial
import time

###########################################################
###################### Constants ##########################
###########################################################
# global running variable
running = True

current_time = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
print("System is started at {}. Files associated with this run will include this time stamp".format(current_time))

os.mkdir("C:\\{}".format(current_time))

print("Creating new directory for run")
run_directory = "C:\\{}".format(current_time)

print("Creating Video directory")
os.mkdir(run_directory + "\\Video_Data\\")

print("Creating Acceleration directory")
os.mkdir(run_directory + "\\Acceleration_Data\\")

print("Creating GPS directory")
os.mkdir(run_directory + "\\GPS_Data\\")

# paths
VIDEO_PATH = "C:\\{}\\Video_Data\\".format(current_time)
ACCEL_PATH = "C:\\{}\\Acceleration_Data\\".format(current_time)
GPS_PATH = "C:\\{}\\GPS_Data\\".format(current_time)

# GPS COM Port
GPS_PORT = "COM3"

# define some constants for the daq
sampling_rate = 2000
buffer_size_per_channel = 4000
channels = 6
data = np.zeros((channels, buffer_size_per_channel))

# get channel information for module 1 and 2
system = nidaqmx.system.System.local()
DAQ_device1 = system.devices['cDAQ9185-213ABA2Mod1']
DAQ_device2 = system.devices['cDAQ9185-213ABA2Mod2']

channel_names = [ai.name for ai in DAQ_device1.ai_physical_chans] + [ai.name for ai in DAQ_device2.ai_physical_chans]

print("Successfully found channels: ", channel_names)

if channels > len(channel_names):
    print("Specified number of channels {} is less than the number of channels {} found!".format(channels, len(channel_names)))
    running = False
    exit()

##########################################################
############# user_thread function #######################
##########################################################
'''
This is the global variable that tells everything to stop once running = False
The input keeps running from being false but once the user enters anything, running will be false
'''

def Ask_User():
    global running
    input("Enter anything to stop")
    running = False

#########################################################
################# video_thread function #################
#########################################################

'''
This is the function that operates the vision system to record video data.
There are two main variables, "cap" which is the video capture variable and
"out" which is the video writer variable.
'''

def Video():
    global running
    
    # define the video capture variable

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # set resolution and FPS
    cap.set(cv2.CAP_PROP_FPS,100) 
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    
    # main loop which reads frames and saves the info
    print("Video Capture Starting...")
    print(cap.isOpened())
    while cap.isOpened() and running:
        # read from the video capture variable
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        display_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
        
        # display the resized frame in a window named 'Live Feed'
        cv2.imshow('Live Feed', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
        
        # write the video frame to the video file
        cv2.imwrite(VIDEO_PATH + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")) + ".jpg", frame)

    # release everything if job is finished
    print("Video releasing and closing")
    cap.release()
    # close any open windows that cv2 generated (i.e. the video stream)
    cv2.destroyAllWindows()


##########################################################################
############## DAQ_thread function/thread and helper function ############
##########################################################################

# initialize the read task for all of the channels
def initialize_read_task():
    global channel_names

    # initalize read task
    read_task = nidaqmx.Task()

    # initialize channels sequentially
    for channel_number in range(channels):
        read_task.ai_channels.add_ai_voltage_chan(channel_names[channel_number])

    # set channel timings
    read_task.timing.cfg_samp_clk_timing(rate=sampling_rate, 
                                     sample_mode=AcquisitionType.CONTINUOUS,
                                     samps_per_chan=buffer_size_per_channel)
    
    # set up the reader for the read task
    reader = AnalogMultiChannelReader(read_task.in_stream)

    return read_task, reader

def DAQ():
    global running

    # initalize the read task and reader and start
    read_task, reader = initialize_read_task()
    read_task.start()

    while running:
        # create the file path
        FULL_FILE_NAME = os.path.join(ACCEL_PATH, str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))+".csv")

        # read the samples
        reader.read_many_sample(data=data, number_of_samples_per_channel=buffer_size_per_channel)

        # save the data to file and add timestamp as header
        np.savetxt(FULL_FILE_NAME, data, fmt="%f", delimiter=",", newline="\n",header=
                   'time:' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")) +
                   ',sampling_rate:'+ str(sampling_rate) +
                   ',channels:'+ str(channels) +
                   ',buffer_size:' + str(buffer_size_per_channel))

    # pause the read task
    read_task.stop()

    # close the read task and free up the allocated memory (the buffer)
    read_task.close()


##############################################################
################## GPS_thread function #######################
##############################################################

def GPS():
    global running

    # initialize the GPS serial port
    ser = serial.Serial(GPS_PORT, baudrate=57600, timeout= 2)

    # if something crashes, the port may not be closed from the last session
    ser.close()

    # open the port
    ser.open()

    ser.flushInput()
    ser.flushOutput()

    GPS_FILE_NAME_BASE = GPS_PATH + current_time
    with open(GPS_FILE_NAME_BASE+'.nmea', 'wb') as f_data, open(GPS_FILE_NAME_BASE + '.csv', 'w', newline='\n') as f_timestamp:
        while running:
            nmea_call = ser.readline()
            f_timestamp.write(str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")) + '\n') 
            f_data.write(nmea_call)
            time.sleep(0.001)

    
    # close the files and serial port
    ser.close()
    f_data.close()
    f_timestamp.close()


##############################################################
################ Thread setup ################################
##############################################################

# create the threads
video_thread = threading.Thread(target=Video)
DAQ_thread = threading.Thread(target=DAQ)
GPS_thread = threading.Thread(target=GPS)
user_thread = threading.Thread(target=Ask_User)

# start the threads
print("Starting Video Thread...")
video_thread.start()

print("Starting DAQ thread...")
DAQ_thread.start()

print("Starting GPS thread...")
GPS_thread.start()

print("Starting User thread...")
user_thread.start()

user_thread.join()
print("User thread closed")

video_thread.join()
print("Video thread closed")

DAQ_thread.join()
print("DAQ thead closed")

GPS_thread.join()
print("GPS thread closed")
print("All threads closed")
