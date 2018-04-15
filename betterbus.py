import re
import random as rnd
import shapefile
import numpy as np
import pandas as pd
import sklearn.cluster as sk_cl
import matplotlib.pyplot as plt
import networkx as nx
import time


class BetterBus:
    # TODO: Random state not working?
    def __init__(self, n):
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
        ts = time.time()
        rnd.seed(1)
        plt.style.use('bmh')
        self.sfile = 'tl_2010_51013_tabblock10/tl_2010_51013_tabblock10.shp'
        self.n = n
        try:
            self.read_df()
        except FileNotFoundError:
            self.gen_df()
        self.gen_arr()
        te = time.time()
        print('__init__() complete in {0} sec'.format(te - ts))

    def calc_centroid(self, points_arr):
        """
        Define a function to calculate the centroid from a numpy array of
        points.
        """
        ts = time.time()
        length = points_arr.shape[0]
        sum_x = np.sum(points_arr[:, 0])
        sum_y = np.sum(points_arr[:, 1])
        return sum_x / length, sum_y / length
        te = time.time()
        print('calc_centroid() complete in {0} sec'.format(te - ts))

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
        ts = time.time()
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
        te = time.time()
        print('gen_df() complete in {0} sec'.format(te - ts))

    def read_df(self):
        ts = time.time()
        try:
            with pd.HDFStore('store.h5') as store:
                self.df = store['df']
        except Exception as e:
            print(e)
        te = time.time()
        print('read_df() complete in {0} sec'.format(te - ts))

    def gen_arr(self):
        """
        To accurately represent population weights, multiply the location
        data by the population. E.g. (36, -77, 120) becomes 120 copies
        of (36, -77).
        Each population point will be randomly distributed within the
        bounding box for that block.
        """
        ts = time.time()
        data = []
        for indx, row in self.df.iterrows():
            for _ in range(int(row['pop'])):
                lat = rnd.uniform(row.loc['bbox_lat1'], row.loc['bbox_lat2'])
                lon = rnd.uniform(row.loc['bbox_lon1'], row.loc['bbox_lon2'])
                data.append([lat, lon])
        self.arr = np.array(data)
        te = time.time()
        print('gen_arr() complete in {0} sec'.format(te - ts))

    # TODO: Compare results to IRL bus stops/routes/travel times
    def performance(self, G):
        """
        Measures performance of proposed routes compared to existing routes.
        """
        ts = time.time()
        total_dist = sum([G.adj[k1][k2]['weight']
                          for k1 in G.adj for k2 in G.adj[k1]])
        te = time.time()
        print('performance() complete in {0} sec'.format(te - ts))
        return total_dist

    # TODO: Distance via Haversine
    def get_dist(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def christofides(self, points, show_steps=False):
        """
        Algorithm:
        1) Create minimum spanning tree of graph
        2) Find all nodes with odd degree (odd number of edges)
        3) Create minimum weight perfect matching graph of nodes from 2)
        4) Combine graphs from 1) and 3) (all nodes should have even degree)
        5) Remove (skip) repeated nodes
        6) Improve result with 2-opt

        Returns NetworkX Graph object with ~TSP edges.
        """
        ts = time.time()
        pos = {i: (p[1], p[0]) for i, p in enumerate(points)}
        edges = [(i, j, self.get_dist(p1, p2)) for i, p1 in enumerate(points)
                 for j, p2 in enumerate(points)]
        G = nx.Graph()
        G.add_weighted_edges_from(edges)

        # Step 1)
        mst = nx.algorithms.minimum_spanning_tree(G)
        if show_steps:
            self.draw(mst, pos, 'mst')

        # Step 2)
        odds = [n for n in range(mst.number_of_nodes())
                if mst.degree(n) % 2 != 0]
        if show_steps:
            print('odds: {0}'.format(odds))

        # Step 3)
        adj = np.array([[self.get_dist(pos[n1], pos[n2])
                         for n1 in odds] for n2 in odds])
        # Prevent self-selection
        np.fill_diagonal(adj, np.inf)
        # Prevent re-selection of existing edges
        for edge in mst.edges():
            if edge[0] in odds and edge[1] in odds:
                i = odds.index(edge[0])
                j = odds.index(edge[1])
                adj[i, j] = np.inf
                adj[j, i] = np.inf
        new_edges = []
        selected = []
        # Find minimum weight edges
        # np.argmin(adj, axis=1) includes duplicates, which we want to avoid
        for indx, row in enumerate(adj):
            # If a node was selected already, skip it
            if indx in selected:
                continue
            min_indx = np.argmin(adj, axis=1)[indx]
            n1 = odds[indx]
            n2 = odds[min_indx]
            new_edges.append((n1, n2, adj[indx, min_indx]))
            # Prevent double selection
            # Overwrite columns with np.inf and add to selected list
            adj[:, indx] = np.inf
            adj[:, min_indx] = np.inf
            selected.extend((indx, min_indx))
        if show_steps:
            print('new_edges: {0}'.format(new_edges))

        # Step 4)
        mm_mst = mst.copy()
        mm_mst.add_weighted_edges_from(new_edges)
        if show_steps:
            self.draw(mm_mst, pos, 'min matching mst')

        # Step 5)
        # TODO: review this step
        node = 0
        nodelist = []
        stack = [node]
        while len(nodelist) < len(mm_mst.nodes()):
            node = stack.pop(0)
            if node not in nodelist:
                nodelist.append(node)
            for edge in mm_mst.edges(node):
                if edge[1] not in stack and edge[1] not in nodelist:
                    stack.insert(0, edge[1])
        tsp_edges = [(nodelist[i], nodelist[i + 1],
                      self.get_dist(pos[nodelist[i]], pos[nodelist[i + 1]]))
                     for i in range(len(nodelist[:-2]))]

        # Connect first and last nodes -> Is this correct?
        tsp_edges.append((tsp_edges[-1][1], tsp_edges[0][0],
                          self.get_dist(pos[tsp_edges[-1][1]],
                                        pos[tsp_edges[0][0]])))

        tsp = nx.Graph()
        tsp.add_weighted_edges_from(tsp_edges)
        if show_steps:
            self.draw(tsp, pos, 'tsp')
        # TODO: Step 6)
        te = time.time()
        print('christofides() complete in {0} sec'.format(te - ts))
        return tsp, pos

    def nearest_neighbor(self):
        """
        Algorithm:
        1) Start at a random city
        2) Find the nearest unvisited city and add to graph
        3) Repeat 2) until all cities are visited
        4) Repeat 1)-3) for a different random starting city
        5) Select starting city (and route) with shortest route
        """
        ts = time.time()
        te = time.time()
        print('nearest_neighbor() complete in {0} sec'.format(te - ts))
        return None

    def draw(self, G, p, title):
        """
        Takes a NetworkX Graph object, a dictionary of node positions,
        and a title.
        """
        ts = time.time()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='both', bottom='off',
                       top='off', labelbottom='off', right='off',
                       left='off', labelleft='off')
        ax.set_title(title, size=8)
        # Draw shapefile outlines
        sf = shapefile.Reader(self.sfile)
        for shape in sf.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            plt.plot(x, y, linewidth=0.1)
        # Draw population dots
        ax.scatter(self.arr[:, 1], self.arr[:, 0], s=0.01, c='#5d8eec')
        # Draw input route
        nx.draw_networkx(G, p, node_color='#F95151', node_size=5,
                         with_labels=False)
        plt.savefig(title, dpi=300)
        plt.show(block=False)
        te = time.time()
        print('draw() complete in {0} sec'.format(te - ts))


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
    Tie each route back to its depot and add routes between depots.
    Add more depots.
    """
    n = 50
    BB = BetterBus(n)
    stops = sk_cl.KMeans(n_clusters=BB.n).fit(BB.arr).cluster_centers_
    tsp, pos = BB.christofides(stops, show_steps=True)

    # n_depots = 10
    # n_buses = 40
    # n_stops = 200
    # n_stops_per_depot = n_stops // n_depots
    # n_buses_per_depot = n_buses // n_depots
    # n_stops_per_bus = n_stops // n_buses
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.set(adjustable='box', aspect=0.6)
    # ax.tick_params(axis='both', which='both', bottom='off',
    #                top='off', labelbottom='off', right='off',
    #                left='off', labelleft='off')
    # msg = '{0} n_depots, {1} n_buses_per_depot, {2} n_stops_per_bus'
    # ax.set_title(msg.format(n_depots, n_buses_per_depot, n_stops_per_bus),
    #              size=8)
    # # Draw population (blue)
    # ax.scatter(BB.arr[:, 0], BB.arr[:, 1], s=0.01, c='#5d8eec')
    # km_depots = sk_cl.KMeans(n_clusters=n_depots).fit(BB.arr)
    # for clust1 in range(n_depots):
    #     # Draw depot centers (green)
    #     ax.scatter(km_depots.cluster_centers_[:, 0],
    #                km_depots.cluster_centers_[:, 1], s=8, c='#00ff00')
    #     depot_arr = BB.arr[np.where(km_depots.labels_ == clust1)]
    #     km_routes = sk_cl.KMeans(n_clusters=n_buses_per_depot).fit(depot_arr)
    #     for clust2 in range(n_buses_per_depot):
    #         # Add depot to each route?
    #         bus_arr = depot_arr[np.where(km_routes.labels_ == clust2)]
    #         mst, positions, dist = BB.nx_mst(n_stops_per_bus, bus_arr)
    #         # Draw stops (red) and routes (black)
    #         nx.draw_networkx(mst, positions, node_color='#f95151',
    #                          node_size=5, with_labels=False, ax=ax)
    # plt.savefig('map_d{0}b{1}s{2}.png'.format(n_depots,
    #                                           n_buses_per_depot,
    #                                           n_stops_per_bus), dpi=400)
