import xarray as xr
from weatherbenchAfrica.data_loaders.base import BaseDataset

class IMERGDataset(BaseDataset):
    def __init__(self, root, years, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.years = years

    def load(self):
        files = [f"{self.root}/imerg_{year}.nc" for year in self.years]
        ds = xr.open_mfdataset(files, combine='by_coords')
        ds = ds.rename({'precipitationCal': 'precip'})
        return ds
