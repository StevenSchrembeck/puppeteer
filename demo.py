# The Puppeteer demo has 2 main modes:

# 1) Live mode (you must have an Ultraleap hand tracker to use this)
#   Simply type: python demo.py live
#   Make sure your ultraleap is plugged in and running (you can verify this in the visualizer)
#   live hand tracking data will be sent directly to your buzz

# 2) Gesture training mode
#   Type: 
#   python demo.py okay peace fist chop
#       or:
#   python demo.py thehorns surfsup closepaper farpaper thumbsup thumbsup thumbsup
#   or any other combination of valid gesture names (shown below)
#   A random, static gesture will be chosen from your list and sent to your buzz
#   You have 5 seconds to guess (in your head) which is being sent before the answer is logged
#   The gestures will cycle randomly between your selections forever
#   Pictures of each gesture are included in the gesture_visual_explanation folder

# Important notes:
#   - Puppeteer will only run on Windows machines with a valid bluetooth adapter
#   - Python 3 is required (make sure you install the dependencies in requirements.txt)
#   - Only the left hand buzz is setup for this demo. If you want to record your own tracking data, encoders, or run multiple hands, you'll need to dig into the full_project_archive.zip

gestures = {
    "thumbsup": [113,0,255,92],
    "peace": [169,0,255,77],
    "okay": [193,0,255,93],
    "surfsup": [132,0,255,39],
    "fingergun": [150,0,255,100],
    "fist": [135,0,255,73],
    "farpaper": [110,198,255,0],
    "closepaper": [166,0,137,255],
    "thehorns": [181,0,255,59],
    "chop": [210,0,255,38],
}

import asyncio
import websockets
import json
import pickle
import time
import pdb
import os
import sys
import random
import numpy as np
import asyncio
from bleak import BleakClient
from bleak import discover
from neosensory_python import NeoDevice

live_demo = False
arguments = sys.argv
selected_gestures = []
if (len(arguments) > 1 and arguments[1] == "live"):
    live_demo = True
else:
    if(len(arguments) > 1):
        selected_gestures = arguments[1:]
    else:
        selected_gestures = gestures
        print("Using all gestures")

# demo settings
GESTURE_GUESS_TIME = 8 # seconds
GESTURE_GUESS_TIME = max(GESTURE_GUESS_TIME,3)
TIME_BETWEEN_GUESSES = 5 # seconds
BUZZ_MOTOR_COUNT = 4
BUZZ_MAX_VIBRATION_QUANTIZATION = 255 # look at that fancy word. it just means it has 255 distinct levels of vibration
encoder_path = './pca_encoder.pkl'
last_milli_time = None

def store_object(obj, filename):
    with open(filename, 'wb') as output: # overwrites any existing file
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

current_milli_time = lambda: int(round(time.time() * 1000))


def encode_pca_data(encoder, data, scale=True, reshape=True, to_list=False, to_int=True): # TODO it's an odd design that encode_data isn't bundled with the encoder object. consider switching to a wrapper class and pickling that instance
    if(data is None):
        return None
    if (reshape):
        data = data.reshape(data.shape[0],-1) # smush the non-index dimensions together into a single vector for PCA      
    data = encoder.transform(data)
    if (scale):
        data = scale_data(data)
    data = data.squeeze()
    if (to_int):
        data = data.astype('uint16')
    if (to_list):
        data = data.tolist()
    return data

def scale_data(data):
    # scale the data between 0 and 255.0 to fit inside what buzz can model
    data = np.interp(data, (data.min(), data.max()), (0, BUZZ_MAX_VIBRATION_QUANTIZATION))
    return data

def extract_features(data_list, reshape=False, hand_type="left"):  # pull out only the data we want
    hand_properties = ["palmNormal", "palmPosition", "wrist"]
    finger_properties = ["tipPosition"]
    number_of_human_fingers_per_hand = 5  # hey, you never know

    if type(data_list) is str:  # force conversion to a list for a single frame
        data_list = [data_list]
    data_as_arrays = np.zeros(
        (len(data_list), 8, 3)
    )  # the second position could be dynamic based on num of properties, but who cares?

    for index, frame in enumerate(data_list):
        json_frame = json.loads(frame)
        hands = json_frame.get("hands")
        if(hands[0].get("type") == hand_type):
            hand = hands[0]
        elif(len(hands) > 1 and (hands[1].get("type") == hand_type)):
            hand = hands[1]
        else:
            return None
        hand_id = hand.get("id")
        hand_data = np.array([hand.get(prop) for prop in hand_properties])

        if hand_data.shape[1] != 3 and hand_data.shape[0] != len(
            hand_properties
        ):  # expects each property value is a length 3 vector and that the number of properties match
            1 / 0  # boom

        data_as_arrays[index][0 : len(hand_data)] = hand_data

        for finger_index, finger in enumerate(json_frame.get("pointables")):
            if (
                finger.get("handId") == hand_id
            ):  # don't include any data from a second hand, yet
                finger_data = np.array([finger.get(prop) for prop in finger_properties])

                if finger_data.shape[1] != 3 and finger_data.shape[0] != len(
                    finger_properties
                ):
                    1 / 0  # boom

                finger_data_start_index = len(hand_data) + finger_index
                finger_data_end_index = len(hand_data) + finger_index + len(finger_data)
                data_as_arrays[index][
                    finger_data_start_index:finger_data_end_index
                ] = finger_data  # looks complicated, but i'm just inserting some arrays with configurable size based on the length of finger_properties

    # sample data (assuming there are 8 total properties. originally 3 for a hand and 1 for each finger)
    # data_as_arrays[index]
    # array([[   0.237143,   -0.946325,   -0.21962 ],
    #        [ -82.490501,  202.77829 ,   26.042208],
    #        [ -82.362137,  189.479279,   95.077568],
    #        [ -15.075111,  178.282196,   -9.779669], # finger data starts on this row
    #        [ -55.99604 ,  211.92009 ,  -68.624962],
    #        [ -88.151657,  201.456131,  -77.498871],
    #        [-111.65979 ,  186.951691,  -65.425003],
    #        [-129.974991,  175.578094,  -39.394772]])

    if reshape == True:
        data_as_arrays = data_as_arrays.reshape(len(data_list), -1)  # reshape from separated features like (1000, 8, 3) to a list of 1d vectors like (1000, 24). less readable but neccessary for most encoders

    return data_as_arrays

async def initialize_buzz(blacklist_addresses = None, buzzLabel = ""):
    print("Initialing Buzz: " + buzzLabel)
    buzz_addr = "not yet set"  # e.g. "EB:CA:85:38:19:1D"
    devices = await discover()
    for d in devices:
        if str(d).find("Buzz") > 0:
            print("    Found a Buzz! " + str(d) +
             "\r\nAddress substring: " + str(d)[:17])
            # set the address to a found Buzz
            if(blacklist_addresses is None): # register the first buzz you find if there's no blacklisted addresses
                buzz_addr = str(d)[:17]
                break
            elif(not(str(d)[:17] in blacklist_addresses)): # otherwise the first one you find not in the blacklist
                buzz_addr = str(d)[:17]
                break
            else:
                print("    Ignoring already registered Buzz: " + str(d)[:17])

    if(buzz_addr == "not yet set"):
        print("No buzzes found. Dividing by zero. Get ready to explode")
        exit()

    client = BleakClient(buzz_addr)
    try:
        await client.connect()

        my_buzz = NeoDevice(client)

        await asyncio.sleep(1)

        x = await client.is_connected()
        print("    Connection State: {0}\r\n".format(x))    

        await asyncio.sleep(1)

        await my_buzz.request_developer_authorization()

        await my_buzz.accept_developer_api_terms()

        await my_buzz.pause_device_algorithm()

        await my_buzz.clear_motor_queue()

        async def send_vibration_frame(motor_vibrate_frame):
            try:
                await my_buzz.vibrate_motors(motor_vibrate_frame)                
            except KeyboardInterrupt:
                await my_buzz.resume_device_algorithm()
                exit()

        return send_vibration_frame, buzz_addr
    except Exception as e:
        print(e)
        client.disconnect()
        exit()


print("Starting demo...")
try:
    encoder = load_object(encoder_path)
except:
    print("Unable to find pre-fitted encoder file: '" + encoder_path + "' exiting.")    
    exit()

async def gesture_loop(send_vibration_frame_left):
    print("Get ready!")
    print("\n")
    await asyncio.sleep(TIME_BETWEEN_GUESSES)
    while True:
        gesture_names = list(selected_gestures)
        chosen_gesture = random.choice(gesture_names)
        gesture_vibration = gestures[chosen_gesture]
        # print(gesture_vibration)

        print("Start guessing, you have " + str(GESTURE_GUESS_TIME) + " seconds...")
        print("Its one of: " + str(gesture_names))
        await send_vibration_frame_left(gesture_vibration)
        await asyncio.sleep(GESTURE_GUESS_TIME - 2)
        print("The answer is... \n")
        await asyncio.sleep(2)
        print(chosen_gesture)
        await send_vibration_frame_left([0,0,0,0])
        print("\n")
        await asyncio.sleep(TIME_BETWEEN_GUESSES)

async def live_loop(loop):
    global last_milli_time
    listener_host = "localhost"
    listener_port = "6437"
    listener_path = "/v6.json"
    listener_websocket_resource_url = f"ws://{listener_host}:{listener_port}{listener_path}"

    async with websockets.connect(listener_websocket_resource_url) as websocket:
        last_milli_time = current_milli_time()
        (send_vibration_frame_left, buzz_address) = await initialize_buzz(buzzLabel="Left hand")

        async for message in websocket:
            current_time = current_milli_time()            
            if (current_time - last_milli_time > 100):
                encoded_frame_left = await test_hand_tracking_data(message, encoder)
                print("Sending frame: " + str(encoded_frame_left))

                await send_vibration_frame_left(encoded_frame_left)
                last_milli_time = current_time


async def test_hand_tracking_data(message, encode):
    def get_empty_frame_if_none(encoded_frame):
        if (encoded_frame is None):
            return [0,0,0,0] # when skipping frames with bad hand data, stop vibrating
        else:
            return encoded_frame

    if(message_contains_hand_and_finger_data(message)):
        try:
            frame_data_left = extract_features(message)
            frame_data_left = encode_pca_data(encoder, frame_data_left, to_list=True) # TODO switch to a general encoding function. not pca specific
            frame_data_left = get_empty_frame_if_none(frame_data_left)
            return frame_data_left
        except:
            print("Dropping frame. Failed to encode data")
            return [0,0,0,0]
    else:
        print("Skipping empty or partially empty frame")
        return [0,0,0,0]
    
def message_contains_hand_and_finger_data(message):
    if (message.find("hands\":[]") < 0): # make sure there are hands in frame and the ultraleap is ready and sending data (it only sends some status messages while it first turns on)
        entry = json.loads(message) # TODO this is slow for little reason. find a better "data not ready yet" detector
        if ("hands" in entry and len(entry["hands"]) > 0):
            if("pointables" in entry and len(entry["pointables"]) > 0):
                return True
    return False

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    if (not(live_demo)):        
        (send_vibration_frame_left, buzz_address) = loop.run_until_complete(initialize_buzz())
        loop.run_until_complete(gesture_loop(send_vibration_frame_left))
    else:
        print("Running live hand tracking demo")
        loop.run_until_complete(asyncio.wait([live_loop(loop)]))
    loop.run_forever()