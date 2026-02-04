import h5py

f = h5py.File('/work/vita/datasets/audio/sonicmaster/audios/test_sonicmaster/shard_0000.h5', 'r')
key = 'sample_00000_2033679'
print(f"Accessing key: {key}")
print(f"Type: {type(f[key])}")
print(f"Is dataset: {isinstance(f[key], h5py.Dataset)}")
print(f"Is group: {isinstance(f[key], h5py.Group)}")

if isinstance(f[key], h5py.Group):
    print(f"Group keys: {list(f[key].keys())}")
    for subkey in list(f[key].keys())[:3]:
        print(f"  Subkey '{subkey}' type: {type(f[key][subkey])}")
        if isinstance(f[key][subkey], h5py.Dataset):
            print(f"    Shape: {f[key][subkey].shape}")
elif isinstance(f[key], h5py.Dataset):
    print(f"Dataset shape: {f[key].shape}")
    print(f"Dataset dtype: {f[key].dtype}")

f.close()
