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

from graph.ChronoRoot import ChronoRootAnalyzer
import argparse

if __name__ == "__main__":
    conf = {}
    file = exec(open('config.conf').read(), conf)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath', type=str, help='Output directory', nargs="?")
    parser.add_argument('--imgpath', type=str, help='Input directory', nargs="?")
    parser.add_argument('--segpath', type=str, help='Output directory', nargs="?")

    args = parser.parse_args()

    if not args.savepath:
        pass
    else:
        conf['Project'] = args.savepath

    if not args.imgpath:
        pass
    else:
        conf['Path'] = args.imgpath

    if not args.segpath:
        pass
    else:
        conf['SegPath'] = args.path

    ChronoRootAnalyzer(conf)
