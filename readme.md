# autonomic_driving

This project is an implementation of autonomic driving for robomaster ep.  
It was tested in a dark room with 4 IR cameras.  
I used `python 3.8.10` 

### Example output for robot:  
<img src="./resources_for_readme/gif_videos/robot_autonomic_driving.gif" width="480" height="360" />

## prerequisites
1. `Python 3.8` installed (python 3.10 will not with the robomaster sdk).

## Instructions
For running the code to the following:
1. Open any folder you want to clone the project into.
1. right click on an empty area of the folder and click on `Open in Terminal`.
1. In the terminal, write the command `git clone git@github.com:davidcohn1234/autonomic_driving.git` and wait for the command to finish clonning.
1. Open `Pycharm` then click on `File->Open` and choose the folder `drone_and_robot_shapes_detection` that you just clonned.
1. Click on `File->Settings...`. 
1. In the Settings windows click on `Project->Python Interpreter` then click on `Add Interpreter->Add Local Interpreter`.
1. in the window `Add Python Interpreter` make sure you're making a **new** environment and not an existing environemt and that you use base interpreter `python 3.8` (I don't think it will work for python 3.10). Then click on `OK` in all windows.
1. Open terminal in pycharm (you will see bellow the tab `Terminal`. Click on it).
1. Make sure you see `(venv)` at the beginning of the prompt in the terminal. If you don't see it then open a new terminal (the plus sign near the `Local` tab). In the new tab you're supposed to see the `(venv)`.
1. Write the command `pip install -r requirements.txt`
1. Click on the script name above and choose the script `script_autonomic_driving`.
1. Make sure the script is run on the environment you just created (In python Interpreter make sure the python path is in the environment you created).
1. Run the code
1. If you run the code for the first time it will first download the 4 example movies from the 4 cameras and then extract each movie into a folder of images. So it might take time (only at the first run). when you run it again, you'll see the output images immediately.

