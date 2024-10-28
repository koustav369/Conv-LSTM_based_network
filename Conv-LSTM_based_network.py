import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import shapefile
import rasterio
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import os
import matplotlib.animation as animation

# Set the latitude-longitude bounding box
lon_min, lon_max = 65, 100
lat_min, lat_max = 7, 38

# Set the path to the folder containing the NetCDF files
data_folder = 'D:\\2nd objective data\\sm_era5(0.1)\\sm2013'

# Create an empty list to store the clipped NetCDF files
ds_list = []

# Loop through each NetCDF file in the folder
for filename in os.listdir(data_folder):
    if filename.endswith('.nc'):  # Check if the file is a NetCDF file
        # Load the NetCDF file and clip it to the specified latitude-longitude bounding box
        ds = xr.open_dataset(os.path.join(data_folder, filename))
        ds_clip = ds.sel(longitude=slice(lon_min, lon_max),
                         latitude=slice(lat_max, lat_min))
        # Append the clipped NetCDF file to the list
        ds_list.append(ds_clip)

# Concatenate the clipped files along the time dimension
ds = xr.concat(ds_list, dim='time')

# ds['Dew_Point_Temperature_2m_Mean'].plot.pcolormesh(x='lon', y='lat', col='time')
# plt.show()

# Write the concatenated data to a new NetCDF file
ds.to_netcdf(os.path.join(
    'D:\\2nd objective data\\clipped_data', 'stacked_SM_n_data.nc'))


def get_shapefile_extent(shapefile_path):
    sf = shapefile.Reader(shapefile_path)
    shapes = sf.shapes()
    bbox = shapes[0].bbox  # Assuming the first shape contains the entire extent
    return bbox

def regrid_netcdf_file(variable_path, shapefile_path, grid_resolution):
    extent = get_shapefile_extent(shapefile_path)
    lon_min, lat_min, lon_max, lat_max = extent

    # Create a meshgrid within the extended lon and lat range
    target_lon = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
    target_lat = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
    target_lon_mesh, target_lat_mesh = np.meshgrid(target_lon, target_lat, indexing='ij')

    # Open the NetCDF file
    nc_file = netCDF4.Dataset(variable_path)
    variable_names = nc_file.variables.keys()

    # Get the last variable name
    last_variable_name = list(variable_names)[-1]
    variable_data = np.array(nc_file.variables[last_variable_name][:])

    # Check if the longitude range of the target grid falls within the original data
    lon_variable_name = 'lon'
    if 'lon' not in variable_names:
        lon_variable_name = 'longitude'
    original_lon = np.array(nc_file.variables[lon_variable_name][:])

    # Check if the latitude range is within bounds
    lat_variable_name = 'lat'
    if 'lat' not in variable_names:
        lat_variable_name = 'latitude'
    original_lat = np.array(nc_file.variables[lat_variable_name][:])

    # Exclude latitude and longitude values that fall outside the original data range
    target_lon_subset = target_lon[(target_lon >= np.min(original_lon)) & (target_lon <= np.max(original_lon))]
    target_lat_subset = target_lat[(target_lat >= np.min(original_lat)) & (target_lat <= np.max(original_lat))]
    target_lon_mesh_subset, target_lat_mesh_subset = np.meshgrid(target_lon_subset, target_lat_subset, indexing='ij')

    # Perform regridding for each time step in the data
    original_time, original_rows, original_cols = variable_data.shape

    # Create an empty 3D array to store the regridded data
    regridded_data = np.empty((T, len(target_lon_subset), len(target_lat_subset)))

    # Create 3D latitude and longitude matrices
    lat_mesh_3d = np.empty((T, len(target_lon_subset), len(target_lat_subset)))
    lon_mesh_3d = np.empty((T, len(target_lon_subset), len(target_lat_subset)))

    for t in range(T):
        regridder = RegularGridInterpolator((original_lat, original_lon), variable_data[t,:,:])
        target_points = np.column_stack((target_lat_mesh_subset.ravel(), target_lon_mesh_subset.ravel()))
        regridded_data[t, :, :] = regridder(target_points).reshape(target_lat_mesh_subset.shape)
        lat_mesh_3d[t, :, :] = target_lat_mesh_subset
        lon_mesh_3d[t, :, :] = target_lon_mesh_subset

    nc_file.close()

    return regridded_data, lat_mesh_3d, lon_mesh_3d

def shapefile_to_logical_array(shapefile_path, lat_mesh_3d, lon_mesh_3d):
    # Read the shapefile using geopandas
    gdf = gpd.read_file(shapefile_path)
    # Create an empty logical array to hold the result
    logical_array = np.zeros(lat_mesh_3d.shape[1:], dtype=bool)

    # Loop through each polygon and update the logical array
    for geom in gdf['geometry']:
        for i in range(lat_mesh_3d.shape[1]):
            for j in range(lon_mesh_3d.shape[2]):
                point = Point(lon_mesh_3d[0, i, j], lat_mesh_3d[0, i, j])
                if point.within(geom):
                    logical_array[i, j] = True

    return logical_array

# Specify the paths to your stacked .nc files for each variable
variable_paths = [
    "/content/drive/MyDrive/soil_moisture_prediction/stacked_ATEMP_data.nc",
    "/content/drive/MyDrive/soil_moisture_prediction/stacked_DPT_data.nc",
    "/content/drive/MyDrive/soil_moisture_prediction/stacked_PPTF_data.nc",
    "/content/drive/MyDrive/soil_moisture_prediction/stacked_RH_data.nc",
    "/content/drive/MyDrive/soil_moisture_prediction/stacked_SRF_data.nc",
    "/content/drive/MyDrive/soil_moisture_prediction/stacked_STP_data.nc",
    "/content/drive/MyDrive/soil_moisture_prediction/stacked_VP_data.nc",
    "/content/drive/MyDrive/soil_moisture_prediction/stacked_SM_n_data.nc"
]

# Specify the path to your shapefile
#shapefile_path = 'D:\\2nd objective data\\New folder\\Cauveri.shp'

# Define the desired grid resolution
grid_resolution = 0.1 # Example resolution in degrees

# Dictionary to store regridded data
regridded_data_dict = {}

# Specify the variable for which you want to save lat-lon information
variable_index = 0  # Change this to the desired index (0 for the first variable, 1 for the second, and so on)

# Iterate over the variable paths and regrid each file
for i, variable_path in enumerate(variable_paths):
    regridded_data, lat_mesh_3d, lon_mesh_3d = regrid_netcdf_file(variable_path, shapefile_path, grid_resolution)

    # Create variable names dynamically
    variable_name = f"Variable_{i+1}"

    # Store the regridded data in the dictionary
    regridded_data_dict[variable_name] = regridded_data

    # Check if the current iteration corresponds to the desired variable index
    if i == variable_index:
        # Save the lat-lon information for the desired variable in the dictionary
        regridded_data_dict[variable_name + '_lat'] = lat_mesh_3d
        regridded_data_dict[variable_name + '_lon'] = lon_mesh_3d


# Get the logical array from the shapefile
logical_array = shapefile_to_logical_array(shapefile_path, lat_mesh_3d, lon_mesh_3d)

# Clip the regridded data using the logical array
clipped_data_dict = {}

for variable_name, regridded_data in regridded_data_dict.items():
    clipped_data_t = []
    for i in range(regridded_data.shape[0]):
        # Use boolean indexing to clip the regridded data and create a masked array
        clipped_data = np.ma.masked_array(regridded_data[i, :, :], mask=~logical_array)

        # Replace masked values with NaN
        clipped_data = clipped_data.filled(np.nan)

        clipped_data_t.append(clipped_data)

    clipped_data_dict[variable_name] = clipped_data_t




import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Assuming the clipped_data_dict contains data for the desired variables at keys 'Variable_1', 'Variable_2', ..., 'Variable_7', and 'Variable_Y'
X_variable_names = ['Variable_1', 'Variable_2', 'Variable_3', 'Variable_4', 'Variable_5', 'Variable_6', 'Variable_7']
Y_variable_name = 'Variable_8'

X_data = [clipped_data_dict[variable_name] for variable_name in X_variable_names]
Y_data = clipped_data_dict[Y_variable_name]

# Step 1: Stack each variable in X_data and Y_data separately
X_data_stacked = [np.stack(data) for data in X_data]
Y_data_stacked = np.stack(Y_data)

# Step 2: Flatten the 45x36 arrays to make them 2D (45*36 = 1620)
X_data_flattened = [data.reshape(data.shape[0], -1) for data in X_data_stacked]
Y_data_flattened = Y_data_stacked.reshape(Y_data_stacked.shape[0], -1)

# Step 3: Remove NaN columns from X_data_flattened and Y_data_flattened
def remove_nan_columns(data):
    return data[:, ~np.isnan(data).any(axis=0)]

X_data_cleaned = [remove_nan_columns(data) for data in X_data_flattened]
Y_data_cleaned = remove_nan_columns(Y_data_flattened)

# Step 4: Stack the cleaned X_data in the third dimension
X_data_cleaned_stacked = np.stack(X_data_cleaned, axis=2)

# Step 5: Convert back to tensors
X_data_cleaned_tensor = torch.tensor(X_data_cleaned_stacked, dtype=torch.float32)
Y_data_cleaned_tensor = torch.tensor(Y_data_cleaned, dtype=torch.float32)


# saving permanently

np.save('X_data_array.npy', X_data_array)
np.save('Y_data_array.npy', Y_data_array)


### Conv-LSTM Based Network

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dropout
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout

#folder_path = "D:/2nd objective data/x_y_arrays"

X_data_array = np.load('X_data_array.npy')
Y_data_array = np.load('Y_data_array.npy')

# Reshaping X_data_array
X_data_reshaped = np.transpose(X_data_array, (1, 2, 3, 0))
X_data_reshaped = np.expand_dims(X_data_reshaped, axis=0)  # Adding the samples dimension

# Reshaping Y_data_array
Y_data_reshaped = np.expand_dims(Y_data_array, axis=(0, -1))

Y_data_reshaped.shape, X_data_reshaped.shape



# matrix to display
matrix_to_display = X_data_reshaped[0][0][:, :, 0]  # Extracting the first channel (there is only one channel, it is not rgb)

plt.figure(figsize=(10, 8))
plt.imshow(matrix_to_display, cmap='viridis', interpolation='none')
plt.colorbar(label='Value')
plt.title('Matrix Visualization')
plt.show()



# matrix to display
matrix_to_display = Y_data_reshaped[0][0][:, :, 0]  # Extracting the first channel (there is only one channel, it is not rgb)

plt.figure(figsize=(10, 8))
plt.imshow(matrix_to_display, cmap='viridis', interpolation='none')
plt.colorbar(label='Value')
plt.title('Matrix Visualization')
plt.show()

Y_data_reshaped[0][0][20][20]
Y_data_reshaped[0][0][20][0]


# Removing the leading dimension of 1 and reshaping the data (was redundant)
X_data_reshaped_new = X_data_reshaped.squeeze(0)
Y_data_reshaped_new = Y_data_reshaped.squeeze(0)

X_data_reshaped = X_data_reshaped_new
Y_data_reshaped = Y_data_reshaped_new


X_data_reshaped.shape, Y_data_reshaped.shape

#Need to take care of nan values

X_data_reshaped[np.isnan(X_data_reshaped)] = 0
Y_data_reshaped[np.isnan(Y_data_reshaped)] = 0


###Converting to 3 day forecasting tasks


def reshape_data_sliding_window(X_data, Y_data, window_size, pred_size=3):
    X_sequences = []
    Y_sequences = []
    for i in range(X_data.shape[0] - window_size - pred_size + 1):
        X_sequences.append(X_data[i:i+window_size])
        Y_sequences.append(Y_data[i+window_size:i+window_size+pred_size])
    return np.array(X_sequences), np.array(Y_sequences)

WINDOW_SIZE = 10
X_data_sequences, Y_data_next_3_days = reshape_data_sliding_window(X_data_reshaped, Y_data_reshaped, WINDOW_SIZE)


# Define a learning rate schedule
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00001,
    decay_steps=10000,
    decay_rate=0.96
)

# Create an optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)




def build_model(input_shape, output_shape):
    model = Sequential()

    # Add ConvLSTM layer
    #model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=input_shape, activation='relu'))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # Another ConvLSTM layer
    #model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    # Another ConvLSTM layer
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    # Another ConvLSTM layer
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # Final ConvLSTM layer for output
    model.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'))
    model.add(Flatten())
    model.add(Dense(np.prod(output_shape), activation='linear')) # Linear output for regression
    model.add(tf.keras.layers.Reshape(output_shape))
    
    
    # Compile the model
    #optimizer = keras.optimizers.Adam(lr=0.0001)
    #model.compile(optimizer='adam',loss='mse')
    model.compile(optimizer=optimizer,loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.MeanSquaredError()])

    return model



# Build and train the model
model = build_model((WINDOW_SIZE, 45, 36, 7), (3, 45, 36, 1))
model.summary()
history=model.fit(X_data_sequences, Y_data_next_3_days, epochs=45, batch_size=10, validation_split=0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train_loss','Val_loss'])
plt.title('Train_loss Vs Val_loss')
plt.show()
plt.savefig('Train_loss Vs Val_loss.png')

# Save the model in the HDF5 format
model.save("regression_3_LCauveri.h5")  # It will save the model in a single HDF5 file