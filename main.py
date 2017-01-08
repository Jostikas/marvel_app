#!/usr/bin/python3
"""
*******************************************************************************
    GUI and mainloop logic for the Marvel App
    Copyright (C) 2016  Laur Joost <daremion@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************
"""

import cv2
import numpy as np
from processor import Processor
from settings import Settings
from threading import Timer, Lock

# NOTE: Since OpenCV 3.2.0 waitKey results are automatically truncated to uint8.
# That means that it is impossible to capture F-keys and other special keys,
# except combinations that translate to ASCII special characters (CTRL-D to EOF
# as an example). To get the full keycode, 3.2.0 provides a new command, waitKeyEx
if cv2.__version__ >= '3.2.0':  # May OpenCV never reach versions higher than 9 :)
    myWaitKey = cv2.waitKeyEx
else: # TODO: This branch is untested.
    myWaitKey = cv2.waitKey


class Main(object):
    def __init__(self, settings, cam_id=0):
        """Create a GUI.

        GUI consists of three main parts:
        * Main window - Displays the output image of the program
        * Settings window - Has sliders to determine the skin detection thres-
        hold and target hue
        * Debug window - Has sliders for selecting the dataset to use and
        shows the image histogram along with the boundaries of values that
        are considered skin.

        :param settings: Settings store, an instance of class Settings
        :type settings: Settings
        :param cam_id: ID of the camera to use as input. Default: 0
        :type cam_id: int
        """
        super(Main, self).__init__()  # Call the parent class constructor.
        self.settings = settings  # makes settings available in other methods
        self.proc = Processor(settings)  # Constructs the processor object
        self.cam = cv2.VideoCapture(cam_id)  # Constructs the camera object
        self.processed = None  # These will store output images
        self.hist_img = None
        self._overlay_lock = Lock()  # Used for msg overlay deletion syncro
        # Flags for determining what windows are (or should be) open.
        self.enabled = {'main': True,
                        'settings': False,
                        'debug': False
                        }
        # Create the default windows
        if self.enabled['main']:
            self._create_main_window()
        if self.enabled['settings']:
            self._create_settings_window()
        if self.enabled['debug']:
            self._create_debug_window()

    def start(self):
        """Initialize and run the main loop"""
        self.settings.read()  # Read the settings from the settings file
        self.message('Press F1 for help.', 6)
        self.run()  # Run the main loop
        self.clean()  # If the main loop has exited, clean up

    def run(self):
        """Main loop"""
        while True:
            ret, frame = self.cam.read()  # Read frame from the camera
            if not ret:
                raise IOError('Failed to capture frame from camera.')
            # Send frame for processing
            self.processed, self.hist_img = self.proc.process(frame, self.enabled['debug'])
            self.update()  # Update windows
            key = myWaitKey(1)  # Get input key
            if key == 27:  # If key was ESC, break
                break
            elif key != -1:  # it's -1 if no key was pressed, which in openCV 3.2.0 is 255 suddenly.
                print(key)
                self._process_input(key)  # react to input

    def clean(self):
        self.cam.release()
        self.settings.store()
        cv2.destroyAllWindows()

    def update(self):
        """Update window contents for the GUI"""
        self._update_main()
        if self.enabled['settings']:
            pass  # Don't know if there's any processing needed
        if self.enabled['debug']:
            self._update_debug()

    def toggle_settings(self):
        """Enable or disable showing the settings window."""
        if not self.enabled['settings']:
            self._create_settings_window()
            self.enabled['settings'] = True
        else:
            self._destroy_settings_window()
            self.enabled['settings'] = False

    def toggle_debug(self):
        if not self.enabled['debug']:
            self._create_debug_window()
            self.enabled['debug'] = True
        else:
            self._destroy_debug_window()
            self.enabled['debug'] = False

    def message(self, message, seconds=0):
        """Create an overlay message. A message already showing is replaced.

        :param message: Message to display.
        :param seconds: How long to display. Default 0 - until hide_message is
        called.
        """
        try:
            self._overlay_timer.cancel()
        except AttributeError:  # Raised when no timer has yet been defined.
            pass
        self._overlay_msg = message.splitlines()
        # Create timer that hides the
        if seconds:
            self._overlay_timer = Timer(seconds, self.hide_message)
            self._overlay_timer.start()

    def hide_message(self):
        """Hide the overlay message. Idempotent."""
        try:
            self._overlay_timer.cancel()
        except AttributeError:  # Raised when no timer has yet been defined.
            pass
        with self._overlay_lock:
            self._overlay_msg = None

    def show_help(self):
        """Show help message"""
        help_message = ("F1: Show this text\n"
                        "S: Show settings\n"
                        "W: Set white balance\n"
                        "R: Reset custom histogram\n"
                        "Ctrl-D: Open debug screen\n"
                        "F12: About\n"
                        "ESC: Exit the program\n"
                        "This text will disappear by itself."
                        )
        self.message(help_message, 10)

    def show_about(self):
        """Show about message (required by GNU GPL)"""
        about_message = ("             Marvel App\n"
                         "         Copyright (C) 2016\n"
                         "Alenka Vasilova, Laur Joost, Reka Suga\n\n"
                         "This program comes with ABSOLUTELY\n"
                         "NO WARRANTY. This is free software\n"
                         "and you are welcome to redistribute\n"
                         "it under certain conditions; see the\n"
                         "included 'LICENCE.txt' file for details\n"
                         )
        self.message(about_message, 10)

    def show_wb(self):
        cv2.namedWindow('WB', True)
        cv2.setMouseCallback('WB', self._mouse_cb_wb)
        ret, self._wbframe = self.cam.read()
        cv2.imshow('WB', self._wbframe)
        cv2.waitKey(1)
        cv2.waitKey(0)
        cv2.destroyWindow('WB')


    def _draw_overlay_message(self, img):
        """Draws the overlay message on a hsv image

        :param img: CV_8UC3 image to draw on
        :type img: np.ndarray
        """
        li = 20  # Line interval in pixels for background
        ci = 9  # Character interval in pixels for background
        with self._overlay_lock:
            if self._overlay_msg is not None:  # If there is a message to show
                ## Create a background
                # First determine the text size in lines and chars on longest line
                rows = len(self._overlay_msg)
                cols = max(len(line) for line in self._overlay_msg)
                # Construct a background with enough space
                img[:rows * li, :cols * ci, :] //= 3  # reduce brightness
                # Then add all lines to the picture.
                for n, line in enumerate(self._overlay_msg):
                    x = 4
                    y = (n + 1) * li - 5
                    cv2.putText(img,  # Image to add the text to
                                line,  # text to add
                                (x, y),  # position to add it to
                                cv2.FONT_HERSHEY_PLAIN,  # Font to use
                                1,  # Size of text
                                (255, 255, 255)  # Color of text (white)
                                )

    def _create_main_window(self):
        """Create the main window with results output"""
        cv2.namedWindow('Marvel App', True)  # Second param is auto-resize

    def _create_settings_window(self):
        """Create the settings window with basic settings sliders"""
        cv2.namedWindow('Settings', False)  # Second argument is auto-resize
        cv2.createTrackbar('Skin_thresh',  # Slider name
                           'Settings',  # Window to add it to
                           self.settings['skin_thresh'],  # starting value
                           255,  # How many steps does the slider have
                           self._create_setting_changer('skin_thresh', 1)
                           )
        # TODO: Color picker
        cv2.createTrackbar('Target',  # Slider name
                           'Settings',  # Window to add it to
                           self.settings['target'],  # starting value
                           255,  # How many steps does the slider have
                           self._create_setting_changer('target', 1)
                           )
        cv2.createTrackbar('Data_thresh',  # Slider name
                           'Settings',  # Window to add it to
                           self.settings['data_thresh'],  # starting value
                           255,  # How many steps does the slider have
                           self._create_setting_changer('data_thresh', 1)
                           )
        cv2.createTrackbar('Edges',  # Slider name
                           'Settings',  # Window to add it to
                           self.settings['edges'],  # starting value
                           1,  # How many steps does the slider have
                           self._create_setting_changer('edges', 1)
                           )

    def _destroy_settings_window(self):
        cv2.destroyWindow('Settings')

    def _create_debug_window(self):
        """Create the debug window with histogram and debug sliders."""
        cv2.namedWindow('Debug', True)
        cv2.createTrackbar('SFA',  # Slider name
                           'Debug',  # Window to add it to
                           self.settings['dataset_sfa'],  # starting value
                           1,  # How many steps does the slider have
                           self._create_setting_changer('dataset_sfa', 1)
                           )
        cv2.createTrackbar('SFA_GT',  # Slider name
                           'Debug',  # Window to add it to
                           self.settings['dataset_sfagt'],  # starting value
                           1,  # How many steps does the slider have
                           self._create_setting_changer('dataset_sfagt', 1)
                           )
        cv2.createTrackbar('IBTD',  # Slider name
                           'Debug',  # Window to add it to
                           self.settings['dataset_ibtd'],  # starting value
                           1,  # How many steps does the slider have
                           self._create_setting_changer('dataset_ibtd', 1)
                           )
        cv2.createTrackbar('UCI',  # Slider name
                           'Debug',  # Window to add it to
                           self.settings['dataset_uci'],  # starting value
                           1,  # How many steps does the slider have
                           self._create_setting_changer('dataset_uci', 1)
                           )
        cv2.createTrackbar('Custom',  # Slider name
                           'Debug',  # Window to add it to
                           self.settings['dataset_custom'],  # starting value
                           1,  # How many steps does the slider have
                           self._create_setting_changer('dataset_custom', 1)
                           )
        cv2.createTrackbar('Raw',  # Slider name
                           'Debug',  # Window to add it to
                           self.settings['raw'],  # starting value
                           1,  # How many steps does the slider have
                           self._create_setting_changer('raw', 1)
                           )
        cv2.setMouseCallback('Marvel App', self._mouse_cb_hist)

    def _destroy_debug_window(self):
        cv2.destroyWindow('Debug')

    def _create_setting_changer(self, setting, scale):
        """Creates a function that takes a value, scales and stores it.

        This function is for use in trackbar callbacks.

        :param setting: Key of the setting to change, should correspond to a
        key in self.settings.
        :param scale: value is scaled by this before storing
        """

        def callback(value):
            self.settings[setting] = scale * value
            if setting.startswith('dataset_'):
                self.proc.compose_hist()

        return callback

    def _process_input(self, key):
        """React to input keys

        param key: Key code as output from cv2.waitKey()
        """
        lookup = {ord('s'): self.toggle_settings,
                  ord('S'): self.toggle_settings,
                  ord('w'): self.show_wb,
                  ord('W'): self.show_wb,
                  ord('r'): self.proc.reset_custom_hist,
                  ord('R'): self.proc.reset_custom_hist,
                  4: self.toggle_debug,  # Ctrl-d is ASCII EOT, ie 0x04
                  7340032: self.show_help,  # Code for F1 button.
                  8060928: self.show_about  # Code for F12 button.
                  }
        # Get the function corresponding to the input key from lookup.
        # If key not in lookup, do nothing.
        lookup.get(key, lambda: None)()

    def _update_main(self):
        # Add overlay message (if any)
        self._draw_overlay_message(self.processed)
        # Show the image on screen.
        cv2.imshow('Marvel App', self.processed)

    def _update_debug(self):
        cv2.imshow('Debug', self.hist_img)

    def _mouse_cb_wb(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            swatch = self._wbframe[y-2:y+3, x-2:x+3,:]
            self.proc.set_wb(swatch)

    def _mouse_cb_hist(self, event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEMOVE:
            self.proc.set_mouse_loc(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.proc.change_custom_hist(x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.proc.change_custom_hist(x, y, False)

if __name__ == '__main__':
    settings = Settings('settings.txt')
    # TODO: Clean that up.
    settings.update({'skin_thresh': 75, 'target': 70})
    app = Main(settings)
    app.start()