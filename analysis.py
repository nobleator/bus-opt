# TODO: Distance via Haversine
# TODO: Compare results to IRL bus stops/routes/travel times


import re
import random
import shapefile
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self):
        """
        Set random_state to ensure repeatable outcome.
        num_stops is the number of bus stops and number of clusters

        https://towardsdatascience.com/clustering-the-us-population-observation-weighted-k-means-f4d58b370002
        https://catalog.data.gov/dataset/current-business-license-list-c47fc/resource/731119ac-0df7-44b1-b4f3-148e2693121e
        https://geocoding.geo.census.gov/geocoder/
        https://wiki.openstreetmap.org/wiki/Downloading_data
        Arlington bounding box (left, bottom, right, top):
        arl_bbox = ['-77.1723251343','38.8272895813','-77.032081604','38.9342803955']
        url = 'https://api.openstreetmap.org/api/0.6/map?bbox=' + ','.join(arl_bbox)
        Look for building tags to estimate demand.
        https://wiki.openstreetmap.org/wiki/Map_Features#Building
        Bus stops Arlington County VA catalog.data.gov
        """
        random.seed(1)
        self.num_stops = 20

    def calc_centroid(self, points_arr):
        """
        Define a function to calculate the centroid from a numpy array of points.
        """
        length = points_arr.shape[0]
        sum_x = np.sum(points_arr[:, 0])
        sum_y = np.sum(points_arr[:, 1])
        return sum_x/length, sum_y/length

    def gen_df(self):
        """
        Download population data from US Census API
        VA = 51, Arlington county = 013, Total population = P0120001
        https://api.census.gov/data/2010/sf1/?get=P0120001&for=block:*&in=state:51%20county:013
        Input data has extra [, ], and " characters, so strip those.
        lines[0] is the header row, so skip it.
        
        Open TIGER/Line data shapefile for Arlington County, VA
        https://www2.census.gov/geo/tiger/TIGER2010/TABBLOCK/2010/
        Pyshp: https://pypi.python.org/pypi/pyshp used as import shapefile
        Shapefile length matches US Census API results length.
        This leads me to believe that index i in the API results maps to shape i 
        in the shapefile.
        Create an index variable to map back to API data.
        Convert shapefile points to numpy array for processing in calc_centroid.
        """
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
        sf = shapefile.Reader('tl_2010_51013_tabblock10/tl_2010_51013_tabblock10.shp')
        shapes = sf.shapes()
        indx = 0
        for shape in shapes:
            arr = np.array(shape.points)
            lat, lon = self.calc_centroid(arr)
            lon_1, lat_1, lon_2, lat_2 = shape.bbox
            data[indx]['cent_lat'] = lat
            data[indx]['cent_lon'] = lon
            data[indx]['bbox_lat1'] = lat_1
            data[indx]['bbox_lat2'] = lat_2
            data[indx]['bbox_lon1'] = lon_1
            data[indx]['bbox_lon2'] = lon_2
            indx += 1
        self.df = pd.DataFrame(data)
        with pd.HDFStore('store.h5') as store:
            store['df'] = self.df
        self.df.to_pickle('store.pkl')

    def read_df(self):
        with pd.HDFStore('store.h5') as store:
            self.df = store['df']

    def gen_big_arr(self):
        """
        To accurately represent population weights, multiply the location data by the
        population. E.g. (36, -77, 120) becomes 120 copies of (36, -77).
        Each population point will be randomly distributed within the bounding box for that block.            
        """
        big_data = []
        for indx, row in self.df.iterrows():
            for _ in range(int(row['pop'])):
                lat_1, lat_2, lon_1, lon_2  = row['bbox_lat1':'bbox_lon2']
                lat = random.uniform(lat_1, lat_2)
                lon = random.uniform(lon_1, lon_2)
                big_data.append([lat, lon])
        self.big_arr = np.array(big_data)
    
    def kmeans(self, arr):
        """
        K-means cluster analysis
        Convert lat/lon to numpy array for processing with sklearn.
        num_stops is the number of clusters (number of bus stops).
        
        km.labels_ is a list that gives the index of the closest cluster center 
        for each input point.
        For example, arr[i] maps to cluster km.cluster_centers_[km.labels_[i]]
        """
        km = KMeans(n_clusters=self.num_stops, random_state=0).fit(arr)
        cl_cent = km.cluster_centers_
        plt.figure()
        plt.scatter(arr[:,0], arr[:,1], s=0.25, c='blue')
        plt.scatter(cl_cent[:,0], cl_cent[:,1], s=0.75, c='red')
        plt.show()

    def k_db_tsp(self):
        """
        K-means + DBSCAN + TSP
        Use k-means to generate population clusters, which correspond to bus stop locations.
        Use DBSCAN to create clusters of bus stops (using the results from k-means) to generate routes.
        Use traveling salesman problem algorithm to optimize path within routes.
        """
        return None

    def k_mst_tsp(self):
        """
        K-means + altered minimum spanning tree + TSP
        Use k-means to generate population clusters, which correspond to bus stop locations.
        """
        return None

    def k_sa_vrp(self):
        """
        K-means + simulated annealing vehicle routing heuristic
        Use k-means to generate population clusters, which correspond to bus stop locations.
        """
        return None

    def k_ga_vrp(self):
        """
        K-means + genetic algorithm vehicle routing heuristic
        Use k-means to generate population clusters, which correspond to bus stop locations.
        """
        return None

    def graph_shapefile(self, file):
        sf = shapefile.Reader(file)
        plt.figure()
        for shape in sf.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            plt.plot(x,y)
        plt.show()

    def total_travel_time(self, num_stops, num_buses):
        """
        Takes a number of bus stops and a list bus routes (one route per bus)
        and calculcates total time "cost".

        Random average personal drive distance (bounds?)
        Average bus travel distance = total loop distance / 2
        Average wait time = ???
        If distance to stop > X or wait time > Y then pop drives themselves
        Report total travel time, buses + cars

        Assumptions:
        Bus stops are determined via k-means clustering (k can vary)
        80% of pops travel somewhere.
        Pops will walk up to 0.25mi to a bus stop.
        If a pop needs to travel but isn't close enough to a bus stop, they will drive.
        Every pop that drives themselves adds to traffic.
        Traffic includes all buses plus all pops that drive.
        Driving speed is inversely proportional to traffic.
        All vehicles use the same driving speed.
        Travel time = distance / driving speed
        Buses have a fixed total capacity.
        Demand is always met before supply (to avoid satisfying demand with the same pops that started at that node)
        """
        return None


if __name__ == '__main__':
    """
    Explain how optimization is computationally prohibitive
    Function to measure performance (total travel + wait times)
    K-means to generate stops
    DBSCAN to generate routes (modify to include 1+ source and 1+ demand node)
    TSP to optimize pathing within routes (Google ortools)
    Generate random supply and demand and simulate to generate results table
    """
    A = Analyzer()
    A.gen_df()
    #A.read_df()
    #A.gen_big_arr()
    #A.kmeans(A.big_arr)
    #A.graph_shapefile('tl_2010_51013_tabblock10/tl_2010_51013_tabblock10.shp')
