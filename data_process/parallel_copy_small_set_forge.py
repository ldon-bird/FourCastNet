import os
import numpy as np
import xarray as xr
import h5py

def load_variable_array(src, varname, src_idx=None):
    with xr.open_dataset(src) as ds:
        if varname not in ds:
            raise KeyError(f"Variable '{varname}' not found in {src}")

        da = ds[varname]

        # Reduce on level index if necessary
        if 'pressure_level' in da.dims:
            if src_idx is None:
                raise ValueError(f"src_idx must be provided for variable {varname} with vertical levels")
            # print(da.pressure_level)
            da = da.sel(pressure_level=da.pressure_level[src_idx])

        if 'valid_time' not in da.dims:
            raise ValueError(f"Variable {varname} lacks time dimension")

        # Ensure order (time, lat, lon)
        da = da.transpose('valid_time', 'latitude', 'longitude')
        return da.load()

def collect_variables(variable_specs, shape=(40, 20, 721, 1440), dtype=np.float32):
    """
    variable_specs: list of (src_file, channel_idx, varname, src_idx)
        src_idx is optional (can be None) if the variable has no vertical level
    """
    full_array = np.full(shape, np.nan, dtype=dtype)

    for src, ch_idx, varname, src_idx in variable_specs:
        print(f"Reading {varname} (channel {ch_idx}) from {src}, level index {src_idx}")
        da = load_variable_array(src, varname, src_idx)
        if da.shape[1:] != shape[2:]:
            raise ValueError(f"Unexpected spatial shape for {varname}: {da.shape}")
        if da.shape[0] != shape[0]:
            raise ValueError(f"Unexpected time length for {varname}: {da.shape}")

        full_array[:, ch_idx, :, :] = da.values

    return full_array, da.coords['latitude'].values, da.coords['longitude'].values


def save_to_hdf5(array, lat, lon, variable_names, out_path):
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('fields', data=array, dtype=np.float32)#, compression='gzip')
        # f.create_dataset('lat', data=lat)
        # f.create_dataset('lon', data=lon)
        # f.create_dataset('variable_names', data=np.array(variable_names, dtype='S'))
    print(f"Saved to {out_path} with shape {array.shape}")



# Define your variables
dest = '/home/dlj-debian/fcn-tc-mocha/processed_era5/mocha.h5'

variable_specs = [
    # (src, channel_idx, varname, src_idx)
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/sfc-2023-tc-mocha.nc',  0, 'u10', None),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/sfc-2023-tc-mocha.nc',  1, 'v10', None),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/sfc-2023-tc-mocha.nc',  2, 't2m', None),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/sfc-2023-tc-mocha.nc',  3, 'sp',  None),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/sfc-2023-tc-mocha.nc',  4, 'msl', None),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',   5, 't',   1),  # 850
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',   6, 'u',   0),  # 1000
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',   7, 'v',   0),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',   8, 'z',   0),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',   9, 'u',   1),  # 850
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  10, 'v',   1),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  11, 'z',   1),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  12, 'u',   2),  # 500
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  13, 'v',   2),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  14, 'z',   2),
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  15, 't',   2),  # t500
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  16, 'z',   3),  # z50
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  17, 'r',   2),  # r500
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/pl-2023-tc-mocha.nc',  18, 'r',   1),  # r850
    ('/home/dlj-debian/fcn-tc-mocha/src_era5/sfc-2023-tc-mocha.nc', 19, 'tcwv',None),
]

variable_names = [v[2] for v in variable_specs]

array, lat, lon = collect_variables(variable_specs)
save_to_hdf5(array, lat, lon, variable_names, dest)
