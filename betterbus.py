import re
import random as rnd
import shapefile
import numpy as np
import pandas as pd
import sklearn.cluster as sk_cl
import matplotlib.pyplot as plt
import networkx as nx
import time
import math


class BetterBus:
    # TODO: Random state not working?
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
        ts = time.time()
        rnd.seed(1)
        plt.style.use('ggplot')
        self.sfile = 'tl_2010_51013_tabblock10/tl_2010_51013_tabblock10.shp'
        try:
            self.read_arr()
        except Exception as e:
            print(e)
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
        np.save('arr_file', self.arr)
        te = time.time()
        print('gen_arr() complete in {0} sec'.format(te - ts))

    def read_arr(self):
        ts = time.time()
        self.arr = np.load('arr_file.npy')
        te = time.time()
        print('read_arr() complete in {0} sec'.format(te - ts))

    def gen_timetable(self, graph, pos, n_buses):
        """
        Create timetable like this:
        time  |  bus  | node |  route
        --------------------------------
        00:00 | bus01 |  A   | A-B-C-D-E
        00:00 | bus03 |  A   | E-D-C-B-A
        00:10 | bus01 |  B   | A-B-C-D-E
        00:15 | bus03 |  E   | E-D-C-B-A
        """
        n_buses = 10
        # Split buses into half, one half in each direction.
        half_buses = n_buses // 2
        # TODO: Loop for an entire day, not just one lap per bus
        # TODO: Find starting stop for each bus.
        # Speed in km/hr
        # 40 km/hr ~= 25 mph
        bus_speed = 40
        trip_length = sum([e[2]['weight']
                           for e in list(graph.edges(data=True))])
        trip_time = trip_length * bus_speed
        start_time_increments = round(trip_time // half_buses)
        route1 = [0]
        while len(route1) < len(graph.nodes()):
            for edge in list(graph.edges(route1[-1])):
                if edge[1] not in route1:
                    route1.append(edge[1])
        route2 = [node for node in reversed(route1)]
        # Shift so they both start at the same location
        route2.insert(0, route2.pop())
        data = []
        start_time = 0
        for bus in range(half_buses + 1):
            ctr = 0
            arrival_time = start_time
            while ctr < len(route1) - 1:
                arrival_node = route1[ctr + 1]
                dist = bb.get_dist(pos[route1[ctr]], pos[arrival_node])
                arrival_time += round(dist * bus_speed)
                data.append({'bus': bus,
                             'time': arrival_time,
                             'route': route1,
                             'node': arrival_node})
                ctr += 1
            start_time += start_time_increments
        start_time = 0
        for bus in range(half_buses + 1, n_buses + 1):
            ctr = 0
            arrival_time = start_time
            while ctr < len(route2) - 1:
                arrival_node = route2[ctr + 1]
                dist = bb.get_dist(pos[route2[ctr]], pos[arrival_node])
                arrival_time += round(dist * bus_speed)
                data.append({'bus': bus,
                             'time': arrival_time,
                             'route': route2,
                             'node': arrival_node})
                ctr += 1
            start_time += start_time_increments
        self.timetable = pd.DataFrame(data)

    def performance(self, graph, pos, n_buses, reps=1):
        """
        Measures performance of proposed routes compared to existing routes.
        Build timetable from route(s) and simulate travel.
        """
        ts = time.time()
        self.gen_timetable(graph, pos, n_buses)
        bus_speed = 40
        walk_speed = 5
        scores = []
        for _ in range(reps):
            start, end = self.arr[np.random.choice(self.arr.shape[0], 2,
                                                   replace=False), :]
            # Problem with lat/lon ordering
            start = (start[1], start[0])
            end = (end[1], end[0])
            nodelist = list(graph.nodes())
            start_node, walk_dist = min([(n, self.get_dist(start, pos[n]))
                                         for n in nodelist],
                                        key=lambda x: x[1])
            end_node, temp_walk = min([(n, self.get_dist(end, pos[n]))
                                       for n in nodelist],
                                      key=lambda x: x[1])
            walk_dist += temp_walk
            walk_time = walk_dist * walk_speed
            # TODO: Generate t (start time) randomly
            t = 0
            min_time = sum([graph.adj[k1][k2]['weight']
                            for k1 in graph.adj for k2 in graph.adj[k1]])
            for row in self.timetable.loc[(self.timetable['node'] == start_node) &
                                          (self.timetable['time'] > t)].iterrows():
                dist = self.route_dist(start_node, end_node,
                                       row[1]['route'], pos)
                trip_time = dist / bus_speed
                if trip_time < min_time:
                    # selected_bus = row[1]['bus']
                    # min_dist = dist
                    min_time = trip_time
                    wait_time = row[1]['time'] - t
            score = trip_time + wait_time * 1.5 + walk_time * 2
            scores.append(score)
        scores = np.array(scores)
        te = time.time()
        print('performance() complete in {0} sec'.format(te - ts))
        return np.mean(scores), np.std(scores)

    def route_dist(self, n1, n2, route, positions):
        """
        Calculates distance between 2 nodes along a given route.
        """
        dist = 0
        done = False
        curr_indx = route.index(n1)
        while not done:
            next_indx = curr_indx + 1
            if next_indx >= len(route):
                next_indx = 0
            dist += self.get_dist(positions[route[curr_indx]],
                                  positions[route[curr_indx]])
            if next_indx == route.index(n2):
                done = True
            curr_indx += 1
            if curr_indx >= len(route):
                curr_indx = 0
        return dist

    def get_dist(self, p1, p2):
        """
        Returns Haversine distance (in km) between two lat/lon tuples.

        https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
        """
        r = 6384
        # Pi/180
        p = 0.017453292519943295
        a = 0.5 - math.cos((p2[0] - p1[0]) * p) / 2 + math.cos(p1[0] * p) * \
            math.cos(p2[0] * p) * (1 - math.cos((p2[1] - p1[1]) * p)) / 2
        return 2 * r * math.asin(math.sqrt(a))

    def valid_route(self, route):
        """
        Returns False if the route has sub-tours or same-node edges.
        """
        edges = []
        for edge in route:
            if edge[0] == edge[1]:
                return False
            if len(edges) > 0 and edge[0] != edges[-1][1]:
                return False
            edges.append(edge)
        return True

    def swap(self, route, i, j):
        new_route = route[:i]
        new_route.append((route[i][0], route[j][0]))
        for indx in range(j - 1, i, -1):
            edge = route[indx]
            new_route.append((edge[1], edge[0]))
        new_route.append((route[i][1], route[j][1]))
        new_route += route[j + 1:]
        return new_route

    def two_opt(self, graph, pos):
        route = list(graph.edges())
        dist = sum([self.get_dist(pos[e[0]], pos[e[1]]) for e in route])
        i1 = 0
        while i1 < len(route) - 2:
            i2 = i1 + 1
            while i2 < len(route):
                new_route = self.swap(route, i1, i2)
                new_dist = sum([self.get_dist(pos[e[0]], pos[e[1]])
                                for e in new_route])
                if new_dist < dist and self.valid_route(new_route):
                    route = new_route
                    dist = new_dist
                    i1 = 0
                    break
                i2 += 1
            i1 += 1
        edges = [(e[0], e[1], self.get_dist(pos[e[0]], pos[e[1]]))
                 for e in route]
        tsp = nx.Graph()
        tsp.add_weighted_edges_from(edges)
        return tsp

    def christofides(self, points, show_steps=False, show_final=False):
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

        # Step 2)
        odds = [n for n in range(mst.number_of_nodes())
                if mst.degree(n) % 2 != 0]

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

        # Step 4)
        mm_mst = mst.copy()
        mm_mst.add_weighted_edges_from(new_edges)

        # Step 5)
        p_node = 0
        n_node = 0
        nodelist = [n_node]
        edgelist = []
        edgestack = []
        while len(nodelist) < len(mm_mst.nodes()):
            for edge in mm_mst.edges(n_node):
                edgestack.insert(0, edge)
            while n_node in nodelist:
                n_node = edgestack.pop(0)[1]
            nodelist.append(n_node)
            edgelist.append((p_node, n_node))
            p_node = n_node
        tsp_edges = [(e[0], e[1], self.get_dist(pos[e[0]], pos[e[1]]))
                     for e in edgelist]
        wrap = (nodelist[-1], nodelist[0], self.get_dist(pos[nodelist[-1]],
                                                         pos[nodelist[0]]))
        tsp_edges.append(wrap)
        tsp = nx.DiGraph()
        tsp.add_weighted_edges_from(tsp_edges)

        # Step 6)
        old = tsp.copy()
        two_opt_ctr = 0
        while True:
            new = self.two_opt(old, pos)
            if list(new.edges()) == list(old.edges()):
                break
            old = new.copy()
            two_opt_ctr += 1
        print('two_opt() called {0} times'.format(two_opt_ctr))
        improved_tsp = new.copy()

        if show_steps:
            self.draw(mst, pos, 'mst')
            self.draw(mm_mst, pos, 'min matching mst')
            self.draw(tsp, pos, 'tsp')
            self.draw(improved_tsp, pos, 'improved tsp')
        elif show_final:
            self.draw(improved_tsp, pos, 'improved tsp')
        te = time.time()
        print('christofides() complete in {0} sec'.format(te - ts))
        return improved_tsp, pos

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

    # TODO: Modify to allow list of graphs (draw as subplots?)
    def draw(self, graph, pos, title):
        """
        Takes a NetworkX Graph object, a dictionary of node positions,
        and a title.
        """
        ts = time.time()
        fig = plt.figure(figsize=(12, 8), dpi=300)
        ax = fig.add_subplot(111)
        ax.tick_params(axis='both', which='both', bottom='off',
                       top='off', labelbottom='off', right='off',
                       left='off', labelleft='off')
        ax.set(adjustable='box', aspect=0.75)
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
        nx.draw_networkx(graph, pos, node_color='#F95151', node_size=5,
                         with_labels=False)
        # plt.savefig(title, dpi=300)
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

    Methodology option:
    Split county into depot regions with k-means.
    Split each depot region into bus regions with k-means.
    TSP heuristic for each bus region.
    County >k-means> Depots >k-means> Bus stops >TSP heuristic>
    TSP routes >2-opt> TSP routes

    How to connect routes with each other? For transfers, long-trips, etc.
    Tie each route back to its depot and add routes between depots.
    Add more depots.
    """
    tic = time.time()
    # n_stops = 25
    bb = BetterBus()
    results = []
    for n_stops in range(25, 101, 25):
        mbkm = sk_cl.MiniBatchKMeans(n_clusters=n_stops)
        stops = mbkm.fit(bb.arr).cluster_centers_
        tsp, pos = bb.christofides(stops, show_final=False)
        n_buses = n_stops // 2
        # n_buses = 10
        r = 1000
        score_avg, score_std = bb.performance(tsp, pos, n_buses, reps=r)
        results.append({'n_stops': n_stops,
                        'n_buses': n_buses,
                        'reps': r,
                        'avg_score': score_avg,
                        'std_score': score_std})
    res_df = pd.DataFrame(results)
    """
    For each simulation pop:
    Pick start time
    Pick start and end points
    Find closest bus stop to start and end
    Calculate best bus for trip
    Calculate pickup and dropoff times
    Get distance from start to first stop, convert to time, multiply 1.5x
    Get time between arrival at stop and bus arrival, multiply 2x
    Get time traveling, multiply 1x
    Get distance from last stop to end, convert to time, multiply 1.5x
    """
    toc = time.time()
    print('Run complete in {0} sec'.format(toc - tic))
