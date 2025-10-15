import os
import rasterio
from rasterio.errors import RasterioError

print("Current working directory:", os.getcwd())

def test_hdf(path):
    print("Testing file:", path)
    print("Exists?", os.path.exists(path))
    try:
        with rasterio.open(path) as src:
            print("Opened main file; subdatasets:", src.subdatasets)
            if src.subdatasets:
                # try opening first subdataset
                s0 = src.subdatasets[0]
                print("Trying subdataset:", s0)
                with rasterio.open(s0) as sub:
                    arr = sub.read(1)
                    print("Read shape:", arr.shape)
            else:
                print("No subdatasets. Trying to read as single dataset.")
                arr = src.read(1)
                print("Read shape:", arr.shape)
    except RasterioError as e:
        print("RasterioError:", e)
    except Exception as e:
        print("Other error:", e)

if __name__ == "__main__":
    # replace with one of your actual .hdf filenames
    fname = "MOD14A1.A2020209.h08v04.061.2020340220932.hdf"
    test_hdf(fname)

import os

fire_dir = r"C:\Users\JOSE MANUEL\Desktop\COURSE_ENGLISH\ESA Project\modis_data\fire"
print("Files in fire directory:")
print(os.listdir(fire_dir))
test_hdf(os.path.join(fire_dir, fname))

