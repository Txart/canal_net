## Required conda installation steps
Create conda environment with Python 3.6 (this is what I have been using to develop and test the code).
And install all packages in one go.
``conda create -n [name] python=3.6 -c conda-forge numpy scipy pandas geopandas rasterio matplotlib tqdm networkx``

Install required packages
``conda install -c conda-forge numpy scipy pandas geopandas rasterio matplotlib tqdm networkx``

NOTE: It may be necessary to set a strict channel priority for conda-forge in the .condarc file.