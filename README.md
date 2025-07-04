### 1. Environment Setup
```
pip install torch==2.4.0+cu121 numpy==2.1.2 PyYAML==6.0.2
```
```
pip install prettytable matplotlib scipy torch-summary tqdm pandas data scikit-learn torch_geometric geopandas pygeohash
```

### Dataset Download
1. [Foursquare NYC 和Foursquare Tokyo](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
2. [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)、[california-counties.geojson](https://github.com/codeforgermany/click_that_hood/blob/main/public/data/california-counties.geojson)

### Model Training
nyc
```
python train-DCDST.py --name DCDST-nyc
```
tky
```
python train-DCDST.py --data-train ./dataset/foursquare/tky/tky_train.csv --data-val ./dataset/foursquare/tky/tky_val.csv --data-adj-matrix  dataset/foursquare/tky/graph_A.csv --data-node-feats dataset/foursquare/tky/graph_X.csv --data-dist-matrix dataset/foursquare/tky/graph_dist.csv --name DCDST-tky
```
