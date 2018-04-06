# TODO: Distance via Haversine
# TODO: Compare results to IRL bus stops/routes/travel times


import re
import random as rnd
import shapefile
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
import networkx as nx


class BetterBus:
    def __init__(self):
        """
        Set random_state to ensure repeatable outcome.

        https://towardsdatascience.com/clustering-the-us-population-observation-weighted-k-means-f4d58b370002
        https://catalog.data.gov/dataset/current-business-license-list-c47fc/resource/731119ac-0df7-44b1-b4f3-148e2693121e
        https://geocoding.geo.census.gov/geocoder/
        https://wiki.openstreetmap.org/wiki/Downloading_data
        Arlington bounding box (left, bottom, right, top):
        arl_bbox = ['-77.1723251343','38.8272895813',
                    '-77.032081604','38.9342803955']
        'https://api.openstreetmap.org/api/0.6/map?bbox=' + ','.join(arl_bbox)
        Look for building tags to estimate demand.
        https://wiki.openstreetmap.org/wiki/Map_Features#Building
        Bus stops Arlington County VA catalog.data.gov
        """
        rnd.seed(1)
        self.sfile = 'tl_2010_51013_tabblock10/tl_2010_51013_tabblock10.shp'

    def calc_centroid(self, points_arr):
        """
        Define a function to calculate the centroid from a numpy array of
        points.
        """
        length = points_arr.shape[0]
        sum_x = np.sum(points_arr[:, 0])
        sum_y = np.sum(points_arr[:, 1])
        return sum_x / length, sum_y / length

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
        This leads me to believe that index i in the API results maps to shape
        i in the shapefile.
        Create an index variable to map back to API data.
        Convert shapefile points to numpy array for processing in
        calc_centroid.
        """
        data = []
        with open('census_api_resp.json', 'r') as fid:
            lines = fid.readlines()
            for line in lines[1:]:
                line = re.sub(r'\[|\]|"', '', line).rstrip(',\n')
                pop, state, county, tract, block = line.split(',')
                data.append({'pop': int(pop),
                             'state': state,
                             'county': county,
                             'tract': tract,
                             'block': block})
        sfile = 'tl_2010_51013_tabblock10/tl_2010_51013_tabblock10.shp'
        sf = shapefile.Reader(sfile)
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
        try:
            with pd.HDFStore('store.h5') as store:
                self.df = store['df']
        except Exception as e:
            print(e)

    def gen_arr(self):
        """
        To accurately represent population weights, multiply the location
        data by the population. E.g. (36, -77, 120) becomes 120 copies
        of (36, -77).
        Each population point will be randomly distributed within the
        bounding box for that block.
        """
        data = []
        for indx, row in self.df.iterrows():
            for _ in range(int(row['pop'])):
                lat = rnd.uniform(row.loc['bbox_lat1'], row.loc['bbox_lat2'])
                lon = rnd.uniform(row.loc['bbox_lon1'], row.loc['bbox_lon2'])
                data.append([lat, lon])
        self.arr = np.array(data)

    def graph_shapefile(self):
        sf = shapefile.Reader(self.sfile)
        plt.figure()
        for shape in sf.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            plt.plot(x, y)
        plt.show()

    def performance(self, num_stops, routes):
        """
        Measures performance of proposed routes compared to existing routes.
        """
        return None

    def get_dist(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def nx_mst(self, n_stops):
        ltic = time.time()
        stops = KMeans(n_clusters=n_stops).fit(self.arr).cluster_centers_
        print('K-means done in {} sec'.format(time.time() - ltic))
        ltic = time.time()
        # NetworkX already has an implementation for MST.
        G = nx.Graph()
        positions = {}
        for i, p1 in enumerate(stops):
            positions[i] = (p1[0], p1[1])
            for j, p2 in enumerate(stops):
                G.add_edge(i, j, weight=self.get_dist(p1, p2))
        print('Graph done in {} sec'.format(time.time() - ltic))
        ltic = time.time()
        mst = nx.algorithms.minimum_spanning_tree(G)
        print('MST done in {} sec'.format(time.time() - ltic))
        ltic = time.time()
        # total_dist = sum([mst.adj[k1][k2]['weight']
        #                   for k1 in mst.adj for k2 in mst.adj[k1]])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.arr[:, 0], self.arr[:, 1], s=0.01, c='#5d8eec')
        nx.draw_networkx(mst, positions, node_color='#F95151', node_size=5,
                         with_labels=False)
        plt.savefig('map_nx_{}.png'.format(n_stops), dpi=300)
        # plt.show()
        print('Plotting done in {} sec'.format(time.time() - ltic))


if __name__ == '__main__':
    """
    Do k-means to determine depot locations.
    Split each depot region into bus regions with k-means.
    Christofides TSP heuristic for each bus region.
    County > k-means > Depots > k-means > Bus stops >
    MST > Trees > Christofides > TSP routes > 2-opt > TSP routes
    Given bus stop locations, create MSTs, one for each set of stops.

    """
    tic = time.time()
    BB = BetterBus()
    # BB.gen_df()
    # print('gen_df done in {} sec'.format(time.time() - tic))
    BB.read_df()
    print('read_df done in {} sec'.format(time.time() - tic))
    toc = time.time()
    BB.gen_arr()
    print('gen_arr done in {} sec'.format(time.time() - toc))

    # Population point color: #5d8eec
    # Stop location color: #F95151
    toc = time.time()
    for n in (50, 75, 100):
        print('Beginning route {}'.format(n))
        BB.nx_mst(n)
        print('Route {0} done in {1} sec'.format(n, time.time() - toc))
        toc = time.time()
    print('Total time elapsed {} sec'.format(time.time() - tic))
