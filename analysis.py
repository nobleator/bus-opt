import re
import shapefile
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


## Data cleaning and preparation
# Download population data from US Census API
# VA = 51, Arlington county = 013, Total population = P0120001
# https://api.census.gov/data/2010/sf1/?get=P0120001&for=block:*&in=state:51%20county:013
# Input data has extra [, ], and " characters, so strip those.
# lines[0] is the header row, so skip it.
data = []
with open('census_api_resp.json', 'r') as fid:
    lines = fid.readlines()
    for line in lines[1:]:
        line = re.sub(r'\[|\]|"','',line).rstrip(',\n')
        pop, state, county, tract, block = line.split(',')
        data.append({'pop': pop,
                     'state': state,
                     'county': county,
                     'tract': tract,
                     'block': block})

# Open TIGER/Line data shapefile for Arlington County, VA
# https://www2.census.gov/geo/tiger/TIGER2010/TABBLOCK/2010/
# Pyshp: https://pypi.python.org/pypi/pyshp used as import shapefile
# Shapefile length matches US Census API results length.
# This leads me to believe that index i in the API results maps to shape i 
# in the shapefile.

# Define a function to calculate the centroid from a numpy array of points.
def calc_centroid(points_arr):
    length = points_arr.shape[0]
    sum_x = np.sum(points_arr[:, 0])
    sum_y = np.sum(points_arr[:, 1])
    return sum_x/length, sum_y/length

# Create an index variable to map back to API data.
# Convert shapefile points to numpy array for faster processing.
sf = shapefile.Reader('tl_2010_51013_tabblock10/tl_2010_51013_tabblock10.shp')
shapes = sf.shapes()
indx = 0
for shape in shapes:
    arr = np.array(shape.points)
    lat, lon = calc_centroid(arr)
    data[indx]['lat'] = lat
    data[indx]['lon'] = lon
    indx += 1

## Create Dataframe
pop_df = pd.DataFrame(data)

## Create visualisation
plt.figure()
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)

## K-means cluster analysis
# Convert lat/lon to numpy array for processing with sklearn.
# num_stops is the number of clusters (number of bus stops).
# Set random_state to ensure repeatable outcome.
num_stops = 10
arr = pop_df.loc[:, 'lat':'lon'].values
km = KMeans(n_clusters=num_stops, random_state=0).fit(arr)
# km.labels_ is a list that gives the index of the closest cluster center 
# for each input point.
# For example, arr[i] maps to cluster km.cluster_centers_[km.labels_[i]]
print(km.labels_, km.cluster_centers_)
plt.plot(38.88, -77.1, '.r-')
#plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1])
plt.show()

# https://catalog.data.gov/dataset/current-business-license-list-c47fc/resource/731119ac-0df7-44b1-b4f3-148e2693121e
# https://geocoding.geo.census.gov/geocoder/

"""
# https://wiki.openstreetmap.org/wiki/Downloading_data
# Arlington bounding box (left, bottom, right, top):
arl_bbox = ['-77.1723251343','38.8272895813','-77.032081604','38.9342803955']
url = 'https://api.openstreetmap.org/api/0.6/map?bbox=' + ','.join(arl_bbox)
# Look for building tags to estimate demand.
# https://wiki.openstreetmap.org/wiki/Map_Features#Building
# Bus stops Arlington County VA catalog.data.gov
"""
