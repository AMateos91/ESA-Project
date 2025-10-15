import os
import numpy as np
import earthaccess
from earthaccess import Auth, DataCollections, DataGranules, Store
from earthaccess import Auth

username = input("Enter Earthdata username: ")
import getpass
password = getpass.getpass("Enter Earthdata password (hidden): ")

auth = Auth().login(strategy="interactive", persist=True)

if not auth.authenticated:
    raise RuntimeError("Login failed")
else:
    print("Logged in and token saved!")

auth = Auth().login(strategy="interactive")
collections = DataCollections(auth).get()

collections = DataCollections(auth).short_name("MOD14A1").get()

print(f"\nFound {len(collections)} MOD14A1 collections:\n")
for c in collections:
    print(f"Collection ID: {c['meta']['concept-id']}")
    print(f"Short name: {c['umm']['ShortName']}")
    print(f"Version: {c['umm']['Version']}")
    print(f"Provider: {c['meta']['provider-id']}")
    print(f"Title: {c['umm']['EntryTitle']}")
    
    # Try fetching granules for each collection in a known period
    gran_query = DataGranules(auth).concept_id(c['meta']['concept-id']).temporal("2020-08-01", "2020-08-10")
    try:
        hits = gran_query.hits()
        print(f"Granules available (2020-08-01 to 2020-08-10): {hits}\n")
    except Exception as e:
        print(f"Error querying granules: {e}\n")

print(f"Found {len(collections)} collections")
first = collections[0]
print(first)
print(dir(first))

def download_modis_fire_and_imagery(bbox, start_date, end_date, product_fire, product_imagery, output_dir):
    
    os.makedirs(os.path.join(output_dir, "fire"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagery"), exist_ok=True)
    
    # Get collections
    colls_fire = DataCollections(auth).short_name(product_fire).version("061").get()
    colls_img = DataCollections(auth).short_name(product_imagery).version("061").get()
    # Test granule query with NO bounding box — just time range
    if not colls_fire or not colls_img:
        raise ValueError("Product short name not found in DataCollections")

    coll_fire = colls_fire[0]
    coll_img = colls_img[0]
   
    print("Fire collection:", coll_fire)
    print("Imagery collection:", coll_img)
    q_test = DataGranules(auth).concept_id(coll_fire.concept_id()).temporal("2020-08-01", "2020-08-10")
    print("Test hits for MOD14A1 (no bbox):", q_test.hits())

    # Get concept_id
    concept_id_fire = coll_fire.concept_id()
    concept_id_img = coll_img.concept_id()

    # Query fire granules
    q_fire = (DataGranules(auth)
              .concept_id(concept_id_fire)
              .temporal(start_date, end_date)
              .bounding_box(*bbox))
    print("Fire hits:", q_fire.hits())
    fire_grans = q_fire.get()

    # Query imagery granules
    q_img = (DataGranules(auth)
             .concept_id(concept_id_img)
             .temporal(start_date, end_date)
             .bounding_box(*bbox))
    print("Imagery hits:", q_img.hits())
    img_grans = q_img.get()

    print("\n--- DEBUG INFO ---")
    print("Product fire:", product_fire)
    print("Product imagery:", product_imagery)
    print("Start date:", start_date)
    print("End date:", end_date)
    print("Bounding box:", bbox)
    print(f"Fire granules found: {len(fire_grans)}")
    print(f"Imagery granules found: {len(img_grans)}")
    print("------------------\n")
    if not fire_grans and not img_grans:
        raise ValueError(f"No granules found for {product_fire} or {product_imagery} in given region/time")

    store = Store(auth)
    if fire_grans:
        store.get(fire_grans, local_path=os.path.join(output_dir, "fire"))
    if img_grans:
        store.get(img_grans, local_path=os.path.join(output_dir, "imagery"))

    print("Download complete.")

# Example usage
if __name__ == '__main__':
    bbox = (10.0, 45.0, 15.0, 48.0)  # Example bounding box for Northern Italy
    download_modis_fire_and_imagery(
        bbox=bbox,
        start_date = "2020-08-01",
        end_date = "2020-08-10",
        product_fire="MOD14A1",
        product_imagery="MOD09GA",
        output_dir="./modis_data"
    )

import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Example simplified model: a UNet-like segmentation network
class FireUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=[64, 128, 256]):
        super(FireUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Down path
        prev_channels = in_channels
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = feature
        # Up path
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = feature
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for idx, up in enumerate(self.ups):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            skip = skip_connections[-(idx+1)]
            # If needed, crop or pad x to match skip’s size
            if x.shape != skip.shape:
                # handle size mismatch
                diffY = skip.size()[2] - x.size()[2]
                diffX = skip.size()[3] - x.size()[3]
                x = nn.functional.pad(x, [diffX // 2, diffX - diffX//2, diffY //2, diffY - diffY//2])
            x = torch.cat((skip, x), dim=1)
            x = up(x)
        return torch.sigmoid(self.final_conv(x))
    # Add this utility function here
def read_hdf_as_array(hdf_path, subdataset_index=0):
    import rasterio
    with rasterio.open(hdf_path) as src:
          print(f"{hdf_path} has {len(src.subdatasets)} subdatasets")
    if len(src.subdatasets) == 0:
            raise ValueError(f"No subdatasets found in {hdf_path}")
    with rasterio.open(src.subdatasets[subdataset_index]) as sub:
            return sub.read()  # shape: (bands, height, width)

# Dataset class
class MODISFireDataset(Dataset):
    def __init__(self, image_paths, label_shp_paths, tile_size=128, transform=None):
        """
        image_paths: list of filepaths to multi-band images (GeoTIFF etc.)
        label_shp_paths: list of shapefiles or vector fire points / polygons for same dates/tiles
        """
        self.image_paths = image_paths
        self.label_shp_paths = label_shp_paths
        self.tile_size = tile_size
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_shp_paths[idx]
        
        # Load image (multi‐band)
        with rasterio.open(img_path) as src:
            # assume bands are stacked
          img = read_hdf_as_array(img_path).astype(np.float32)  # shape: (bands, H, W)

# For simplicity, define a dummy profile (needed for mask alignment)
          profile = {
    'height': img.shape[1],
    'width': img.shape[2],
    'transform': rasterio.transform.from_origin(0, 0, 1, 1),  # placeholder if needed
}

        # Load labels: vector fire points/polygons
        gdf = gpd.read_file(lbl_path)
        # create a mask of fire: 1 if any fire in pixel, 0 otherwise
        # Rasterize gdf to same grid as image
        mask = np.zeros((profile['height'], profile['width']), dtype=np.uint8)
        # using rasterio.features.rasterize etc.
        from rasterio.features import rasterize
        shapes = [(geom, 1) for geom in gdf.geometry]
        mask = rasterize(shapes, out_shape=(profile['height'], profile['width']),
                         transform=profile['transform'], fill=0, dtype=np.uint8)
        
        # Possibly split into tiles
        # For simplicity, take one random tile
        h, w = mask.shape
        if h < self.tile_size or w < self.tile_size:
            # pad
            pad_h = max(0, self.tile_size - h)
            pad_w = max(0, self.tile_size - w)
            img = np.pad(img, ((0,0), (0,pad_h), (0,pad_w)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0,pad_h), (0,pad_w)), mode='constant', constant_values=0)
        # choose tile
        i = np.random.randint(0, img.shape[1] - self.tile_size + 1)
        j = np.random.randint(0, img.shape[2] - self.tile_size + 1)
        img_tile = img[:, i:i+self.tile_size, j:j+self.tile_size]
        mask_tile = mask[i:i+self.tile_size, j:j+self.tile_size]
        
        # normalization
        # e.g. scale reflectance bands to [0,1] or z‐score
        img_tile = img_tile / (img_tile.max() + 1e-6)
        
        # to torch
        img_tensor = torch.from_numpy(img_tile).float()
        mask_tensor = torch.from_numpy(mask_tile).unsqueeze(0).float()
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            # perhaps also data augmentation
        
        return img_tensor, mask_tensor

def train(image_paths, label_paths):
    # Paths / list setup
    import glob
    from glob import glob

    image_paths = sorted(glob("MOD14A1.A2020209.h08v04.061.2020340220932.hdf"))
    label_paths = sorted(glob("MOD14A1.A2020209.h08v04.061.2020340220932.hdf"))

# Then update your Dataset class to select the correct subdataset band when opening the HDF
    ds = MODISFireDataset(image_paths, label_paths, tile_size=128, transform=None)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FireUNet(in_channels=10, out_channels=1)  # e.g., 10 bands
    model = model.to(device)
    criterion = nn.BCELoss()  # for binary segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dl.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
import h5py

file_path = "MOD14A1.A2020209.h08v04.061.2020340220932.hdf"

try:
    with h5py.File(file_path, 'r') as file:
        print("File opened successfully.")
        print("Top-level keys (datasets/groups):")
        print(list(file.keys()))
        dataset = file['FireMask']
        print(dataset)
except Exception as e:
    print("Failed to open the file.")
    print("Error:", e)
  
if __name__ == "__main__":
    import glob
    import os
    # Option 1 - know the exact filename
    image_paths = sorted(glob.glob(r"C:\Users\JOSE MANUEL\Desktop\COURSE_ENGLISH\ESA Project\modis_data\fire\MOD14A1.A2020209.h08v04.061.2020340220932\*.hdf"))
    label_paths = sorted(glob.glob(r"C:\Users\JOSE MANUEL\Desktop\COURSE_ENGLISH\ESA Project\modis_data\fire\MOD14A1.A2020209.h08v04.061.2020340220932\*.hdf"))
    train(image_paths, label_paths)

    # Option 2 - want to load all files in a directory
    image_dir = r"C:\Users\JOSE MANUEL\Desktop\COURSE_ENGLISH\ESA Project\modis_data\fire"
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.hdf")))
    label_paths = sorted(glob.glob(os.path.join(image_dir, "*.hdf")))
    train(image_paths, label_paths)


    
