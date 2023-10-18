#
#
#
################################
# SETUP HERE
#

license_file = "license_key_00395217_-_DTU_Compute_IS404-100106341154"

# DONT CHANGE BELOW



################################
# Preface here
#
# from psychopy import prefs, visual, core, event, monitors, tools, logging
import numpy as np
import tobii_research as tr
import time
import random
import os
import pylsl as lsl
import sys
import csv

def save_gaze_data_to_csv(gaze_data, filename):
    print(gaze_data[0])
    try:
        with open(filename, mode='w', newline='') as csv_file:
            fieldnames = [
                'device_time_stamp',
                'left_gaze_origin_validity',
                'right_gaze_origin_validity',
                'left_gaze_origin_in_user_x',
                'left_gaze_origin_in_user_y',
                'left_gaze_origin_in_user_z',
                'right_gaze_origin_in_user_x',
                'right_gaze_origin_in_user_y',
                'right_gaze_origin_in_user_z',
                'left_gaze_origin_in_trackbox_x',
                'left_gaze_origin_in_trackbox_y',
                'left_gaze_origin_in_trackbox_z',
                'right_gaze_origin_in_trackbox_x',
                'right_gaze_origin_in_trackbox_y',
                'right_gaze_origin_in_trackbox_z',
                'left_gaze_point_validity',
                'right_gaze_point_validity',
                'left_gaze_point_in_user_x',
                'left_gaze_point_in_user_y',
                'left_gaze_point_in_user_z',
                'right_gaze_point_in_user_x',
                'right_gaze_point_in_user_y',
                'right_gaze_point_in_user_z',
                'left_gaze_point_on_display_area_x',
                'left_gaze_point_on_display_area_y',
                'right_gaze_point_on_display_area_x',
                'right_gaze_point_on_display_area_y',
                'left_pupil_validity',
                'right_pupil_validity',
                'left_pupil_diameter',
                'right_pupil_diameter'
            ]

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for data in gaze_data:
                row = {
                    'device_time_stamp': data[0],
                    'left_gaze_origin_validity': data[1],
                    'right_gaze_origin_validity': data[2],
                    'left_gaze_origin_in_user_x': data[3],
                    'left_gaze_origin_in_user_y': data[4],
                    'left_gaze_origin_in_user_z': data[5],
                    'right_gaze_origin_in_user_x': data[6],
                    'right_gaze_origin_in_user_y': data[7],
                    'right_gaze_origin_in_user_z': data[8],
                    'left_gaze_origin_in_trackbox_x': data[9],
                    'left_gaze_origin_in_trackbox_y': data[10],
                    'left_gaze_origin_in_trackbox_z': data[11],
                    'right_gaze_origin_in_trackbox_x': data[12],
                    'right_gaze_origin_in_trackbox_y': data[13],
                    'right_gaze_origin_in_trackbox_z': data[14],
                    'left_gaze_point_validity': data[15],
                    'right_gaze_point_validity': data[16],
                    'left_gaze_point_in_user_x': data[17],
                    'left_gaze_point_in_user_y': data[18],
                    'left_gaze_point_in_user_z': data[19],
                    'right_gaze_point_in_user_x': data[20],
                    'right_gaze_point_in_user_y': data[21],
                    'right_gaze_point_in_user_z': data[22],
                    'left_gaze_point_on_display_area_x': data[23],
                    'left_gaze_point_on_display_area_y': data[24],
                    'right_gaze_point_on_display_area_x': data[25],
                    'right_gaze_point_on_display_area_y': data[26],
                    'left_pupil_validity': data[27],
                    'right_pupil_validity': data[28],
                    'left_pupil_diameter': data[29],
                    'right_pupil_diameter': data[30]
                }
                writer.writerow(row)

        print(f"Gaze data saved to {filename} successfully.")
    except Exception as e:
        print(f"Error saving gaze data to {filename}: {e}")


# Find Eye Tracker and Apply License (edit to suit actual tracker serial no)
ft = tr.find_all_eyetrackers()
if len(ft) == 0:
    print("No Eye Trackers found!?")
    exit(1)

# Pick first tracker
mt = ft[0]
print("Found Tobii Tracker at '%s'" % (mt.address))


# Apply license
if license_file != "":
    with open(license_file, "rb") as f:
        license = f.read()

        res = mt.apply_licenses(license)
        if len(res) == 0:
            print("Successfully applied license from single key")
        else:
            print("Failed to apply license from single key. Validation result: %s." % (res[0].validation_result))
            exit
else:
    print("No license file installed")

channels = 31 # count of the below channels, incl. those that are 3 or 2 long
gaze_stuff = [
    ('device_time_stamp', 1),

    ('left_gaze_origin_validity',  1),
    ('right_gaze_origin_validity',  1),

    ('left_gaze_origin_in_user_coordinate_system',  3),
    ('right_gaze_origin_in_user_coordinate_system',  3),

    ('left_gaze_origin_in_trackbox_coordinate_system',  3),
    ('right_gaze_origin_in_trackbox_coordinate_system',  3),

    ('left_gaze_point_validity',  1),
    ('right_gaze_point_validity',  1),

    ('left_gaze_point_in_user_coordinate_system',  3),
    ('right_gaze_point_in_user_coordinate_system',  3),

    ('left_gaze_point_on_display_area',  2),
    ('right_gaze_point_on_display_area',  2),

    ('left_pupil_validity',  1),
    ('right_pupil_validity',  1),

    ('left_pupil_diameter',  1),
    ('right_pupil_diameter',  1)
]
    

def unpack_gaze_data(gaze_data):
    x = []
    for s in gaze_stuff:
        d = gaze_data[s[0]]
        if isinstance(d, tuple):
            x = x + list(d)
        else:
            x.append(d)
    return x

last_report = 0
N = 0

gaze_data_list = []

def gaze_data_callback(gaze_data):
    '''send gaze data'''

    '''
    This is what we get from the tracker:

    device_time_stamp

    left_gaze_origin_in_trackbox_coordinate_system (3)
    left_gaze_origin_in_user_coordinate_system (3)
    left_gaze_origin_validity
    left_gaze_point_in_user_coordinate_system (3)
    left_gaze_point_on_display_area (2)
    left_gaze_point_validity
    left_pupil_diameter
    left_pupil_validity

    right_gaze_origin_in_trackbox_coordinate_system (3)
    right_gaze_origin_in_user_coordinate_system (3)
    right_gaze_origin_validity
    right_gaze_point_in_user_coordinate_system (3)
    right_gaze_point_on_display_area (2)
    right_gaze_point_validity
    right_pupil_diameter
    right_pupil_validity

    system_time_stamp
    '''

    #for k in sorted(gaze_data.keys()):
    #    print(' ' + k + ': ' +  str(gaze_data[k]))

    try:
        global last_report
        global outlet
        global N
        global halted

        sts = gaze_data['system_time_stamp'] / 1000000.

        outlet.push_sample(unpack_gaze_data(gaze_data), sts)

        # Append the received gaze data to the list
        gaze_data_list.append(unpack_gaze_data(gaze_data))
        
        if sts > last_report + 5:
            sys.stdout.write("%14.3f: %10d packets\r" % (sts, N))
            last_report = sts
        N += 1

        # print(unpack_gaze_data(gaze_data))
    except:
        print("Error in callback: ")
        print(sys.exc_info())

        halted = True


def start_gaze_tracking():
    mt.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    return True

def end_gaze_tracking():
    print("End gaze")
    if gaze_data_list:
        filename = "gaze_data.csv"
        save_gaze_data_to_csv(gaze_data_list, filename)
    mt.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    
    return True

halted = False


# Set up lsl stream
def setup_lsl():
    global channels
    global gaze_stuff

    info = lsl.StreamInfo('Tobii', 'ET', channels, 90, 'float32', mt.address)
    info.desc().append_child_value("manufacturer", "Tobii")
    channels = info.desc().append_child("channels")
    cnt = 0
    for s in gaze_stuff:
        if s[1]==1:
            cnt += 1
            channels.append_child("channel") \
                    .append_child_value("label", s[0]) \
                    .append_child_value("unit", "device") \
                    .append_child_value("type", 'ET')
        else:
            for i in range(s[1]):
                cnt += 1
                channels.append_child("channel") \
                        .append_child_value("label", "%s_%d" % (s[0], i)) \
                        .append_child_value("unit", "device") \
                        .append_child_value("type", 'ET')

    outlet = lsl.StreamOutlet(info)

    return outlet

outlet = setup_lsl()

# Main loop; run until escape is pressed
print("%14.3f: LSL Running; press CTRL-C repeatedly to stop" % lsl.local_clock())
start_gaze_tracking()
try:
    while not halted:
        time.sleep(1)
        keys = ()  # event.getKeys()
        if len(keys) != 0:
            if keys[0]=='escape':
                halted = True

        if halted:
            break

        # print(lsl.local_clock())

except:
    print("Halting...")

print("terminating tracking now")
end_gaze_tracking()

