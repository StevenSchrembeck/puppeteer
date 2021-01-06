![logo](https://user-images.githubusercontent.com/5417135/103720247-f345e700-4f98-11eb-9f5b-66956275bf8a.png)

_Synthetic proprioception through external brain computer interface_



## Installation

1. Clone the repo
2. Install the requirements with: `pip3 install -r .\requirements.txt` or `pip install -r .\requirements.txt` if you only have python 3


## Running the demo
The Puppeteer demo has 2 main modes, live and training. First make sure bluetooth is enabled, your ultraleap (if running live mode) is plugged, and your neosensory buzz is turned on.

1) Live mode (you must have an Ultraleap hand tracker to use this)
  Simply type: `python3 demo.py live` or `python demo.py live`
  Make sure your ultraleap is plugged in and running (you can verify this in the visualizer)
  live hand tracking data will be sent directly to your buzz

2) Gesture training mode
  Type: 
  ```python demo.py okay peace fist chop```
      or:
  ```python demo.py thehorns surfsup closepaper farpaper thumbsup thumbsup thumbsup```
  or any other combination of valid gesture names (shown below)
  A random, static gesture will be chosen from your list and sent to your buzz
  You have 8 seconds to guess (in your head) which is being sent before the answer is logged
  The gestures will cycle randomly between your selections forever.
  Pictures of each gesture are included in the gesture_visual_explanation folder

Important notes:
  - Puppeteer will only run on Windows machines with a valid bluetooth adapter
  - Python 3 is required (make sure you install the dependencies in requirements.txt)
  - Only the left hand buzz is setup for this demo. If you want to record your own tracking data, encoders, or run multiple hands, you'll need to dig into the full_project_archive.zip
