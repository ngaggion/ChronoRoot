""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

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
"""

from graph.fileFunc import download_file_from_google_drive
from zipfile import ZipFile
import pathlib
import os

file_id = '1eVVbWqPUjwYCONeUmx-5nq1wyanhXcTh'
destination = os.path.join(pathlib.Path().absolute(),'modelWeights.zip')

print('Downloading model weights from Google Drive')

download_file_from_google_drive(file_id, destination)

print('Extracting model weights')

with ZipFile('modelWeights.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

try:
    os.remove(destination)
except:
    pass
  
print('Extraction finished')
