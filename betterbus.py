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

    def nx_mst(self, n_stops, points):
        ltic = time.time()
        stops = KMeans(n_clusters=n_stops).fit(points).cluster_centers_
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
        total_dist = sum([mst.adj[k1][k2]['weight']
                          for k1 in mst.adj for k2 in mst.adj[k1]])
        return mst, positions, total_dist

    def christofides(self):
        """
        Algorithm:
        1) Create minimum spanning tree of graph
        2) Find all nodes with odd degree (odd number of arcs)
        3) Create minimum weight perfect matching graph of nodes from 2)
        4) Combine graphs from 1) and 3) (all nodes should have even degree)
        5) Skip repeated nodes???
        """
        return None

    def nearest_neighbor(self):
        """
        Algorithm:
        1) Start at a random city
        2) Find the nearest unvisited city and add to graph
        3) Repeat 2) until all cities are visited
        4) Repeat 1)-3) for a different random starting city
        5) Select starting city (and route) with shortest route
        """
        return None


if __name__ == '__main__':
    """
    Variations:
    Split county into various levels (buses, depots, etc) before routing.
    Different TSP heuristics (Christofides, nearest neighbor, branch/bound).
    Different clustering algorithms (k-means, k-median, DBSCAN).
    Varying numbers of clusters/stops/buses/depots.

    Add businesses as locations.
    To incentivize routing between the two, artificially reduce the
    weight on arcs between businesses and population centers.
    Pop to pop arc = distance
    Business to business arc = distance
    Pop to business arc = 0.75 * distance

    Methodology:
    Split county into depot regions with k-means.
    Split each depot region into bus regions with k-means.
    TSP heuristic for each bus region.
    County >k-means> Depots >k-means> Bus stops >TSP heuristic>
    TSP routes >2-opt> TSP routes

    How to connect routes with each other? For transfers, long-trips, etc.
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
    toc = time.time()
    n_depots = 2
    n_buses = 20
    n_stops = 200
    n_stops_per_depot = n_stops // n_depots
    n_buses_per_depot = n_buses // n_depots
    n_stops_per_bus = n_stops_per_depot // n_buses_per_depot
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set(adjustable='box', aspect=0.6)
    ax.tick_params(axis='both', which='both', bottom='off',
                   top='off', labelbottom='off', right='off',
                   left='off', labelleft='off')
    msg = '{0} n_depots, {1} n_buses_per_depot, {2} n_stops_per_bus'
    ax.set_title(msg.format(n_depots, n_buses_per_depot, n_stops_per_bus),
                 size=8)
    ax.scatter(BB.arr[:, 0], BB.arr[:, 1], s=0.01, c='#5d8eec')
    km_depots = KMeans(n_clusters=n_depots).fit(BB.arr)
    for clust1 in range(n_depots):
        ax.scatter(km_depots.cluster_centers_[:, 0],
                   km_depots.cluster_centers_[:, 1], s=8, c='#00ff00')
        depot_arr = BB.arr[np.where(km_depots.labels_ == clust1)]
        km_routes = KMeans(n_clusters=n_buses_per_depot).fit(depot_arr)
        for clust2 in range(n_buses_per_depot):
            bus_arr = depot_arr[np.where(km_routes.labels_ == clust2)]
            mst, positions, dist = BB.nx_mst(n_stops_per_bus, bus_arr)
            nx.draw_networkx(mst, positions, node_color='#f95151',
                             node_size=5, with_labels=False, ax=ax)
    plt.savefig('map_single.png', dpi=400)
    """

    # Loop to compare different input values
    n_depots = [1, 2, 3]
    n_stops_per_depot = [25, 50]  # , 75, 100]
    for depots in n_depots:
        km = KMeans(n_clusters=depots).fit(BB.arr)
        print('Depot cluster done in {} sec'.format(time.time() - toc))
        toc = time.time()
        fig, axes = plt.subplots(round(len(n_stops_per_depot) / 2), 2)
        for cluster in range(depots):
            # Assign each point to new array corresponding to clusters
            new_arr = BB.arr[np.where(km.labels_ == cluster)]
            for i, n in enumerate(n_stops_per_depot):
                print('Beginning depot {0} route {1}'.format(depots, n))
                mst, positions, dist = BB.nx_mst(n, new_arr)
                ax = axes[i]
                ax.set(adjustable='box', aspect=0.6)
                ax.tick_params(axis='both', which='both', bottom='off',
                               top='off', labelbottom='off', right='off',
                               left='off', labelleft='off')
                ax.set_title('{} stops per depot'.format(n), size=8)
                if cluster == 0:
                    # Population dots, blue (#5d8eec)
                    ax.scatter(BB.arr[:, 0], BB.arr[:, 1], s=0.01, c='#5d8eec')
                # Bus stops, red (#f95151), and routes, black
                nx.draw_networkx(mst, positions, node_color='#f95151',
                                 node_size=5, with_labels=False, ax=ax)
                msg = 'Depot {0} route {1} done in {2} sec'
                print(msg.format(depots, n, time.time() - toc))
                toc = time.time()
        plt.savefig('map_{}depots.png'.format(depots), dpi=300)
        # plt.show()
        print('Plotting depot {0} done in {1} sec'.format(depots,
                                                          time.time() - toc))
    print('Total time elapsed {} sec'.format(time.time() - tic))
    """
