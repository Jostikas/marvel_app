"""
***************************************************************************
    Persistent settings store based on python's dict and plaintext
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
*************************************************************************
"""
from ast import literal_eval
DEF_SETFILE = 'default_settings.txt'

class Settings(dict):
    """Persistent settings store based on dict and plaintext storage"""

    def __init__(self, settingsfile):
        super(Settings, self).__init__()
        self.settingsfile = settingsfile
        self.read()

    def read(self, settingsfile=None):
        """Read settings from settingsfile.

        :param settingsfile: File to read from. If None, read from the file
        given during instantiation.
        :return: None
        """
        if settingsfile is None:
            settingsfile = self.settingsfile
        try:
            with open(settingsfile, 'r') as f:
                temp = literal_eval(f.read())
                if type(temp) != type(dict()):
                    raise ValueError
        except FileNotFoundError or ValueError:
            print("Settings file ", settingsfile, ' not found or corrupt. Using default.')
            with open(DEF_SETFILE, 'r') as f:
                temp = literal_eval(f.read())
        finally:
            self.clear()
            self.update(temp)


    def store(self, settingsfile=None):
        """Store settings to file

        :param settingsfile: File to store to. If None, use the file
        given during instantiation.
        :return: None
        """
        if settingsfile is None:
            settingsfile = self.settingsfile
        with open(settingsfile, 'w') as f:
            f.write(repr(self))

