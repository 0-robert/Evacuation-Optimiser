import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

import random

from shapely.geometry import Point, LineString
import numpy as np
# I need these to start randomly place exits for the genetic algorithm

import geopandas as gpd
import time
from math import ceil

from dotenv import load_dotenv
import os

load_dotenv()  # Automatically looks for .env in the current directory

# --- Person Class ---
class Person:
    def __init__(self, id, speed, start_node, path):
        self.id = id
        self.speed = .5  # meters per tick
        self.current_node = start_node
        self.path = path
        self.position_on_edge = 0.0
        self.on_edge = None
        self.status = "waiting"
        self.time_elapsed = 0

    def __repr__(self):
        return f"<Person {self.id}: {self.status} at {self.current_node or self.on_edge}>"

class EvacuationSimulator:
    def __init__(self):
        self.counter = 1

        # --- Load GeoData from QGIS ---
        self.gdf = gpd.read_file(os.getenv("NODES_GEOJSON"))
        self.walks = gpd.read_file(os.getenv("WALKS_GEOJSON"))


        self.G = nx.Graph()

        self._add_nodes()
        self._add_edges()
        self._add_people()

# --- init: Add Nodes ---
    def _add_nodes(self):
        for _, row in self.gdf.iterrows():
            nodeConstruct = {
                "Type": row["Type"],  # 'classroom', 'path', or 'final'
                "width": 0 + row.get("Width", 0),
                "students": 0 + row.get("ClassN", 0),
                "pos": (row.geometry.bounds[0], row.geometry.bounds[1]),
                "occupants": []  # list of person IDs currently at this node
            }
            self.G.add_node(row["Name"], **nodeConstruct)

# --- init: Add Edges ---
    def _add_edges(self):
        for _, row in self.walks.iterrows():
            start = row["start"]
            end = row["end"]
            length = row["Length"]
            self.G.add_edge(start, end, length=length, occupants=[])

    def _add_single_edge(self, start, end, length):
        self.G.add_edge(start, end, length=length, occupants=[])

# --- Helper: Find Closest Exit ---
    def _find_path_to_exit(self, start_node):
        exits = [n for n, d in self.G.nodes(data=True) if d["Type"] == "Final"]
        if not exits:
            print("❌ No exit nodes found!")
        shortest = None
        min_dist = float("inf")
        for exit_node in exits:
            try:
                path = nx.shortest_path(self.G, start_node, exit_node, weight="length")
                dist = nx.path_weight(self.G, path, weight="length")
                if dist < min_dist:
                    shortest = path
                    min_dist = dist
            except Exception as e:
                print(f"⚠️ Path error from {start_node} to {exit_node}: {e}")
                continue
        if not shortest:
            print(f"⚠️ No path found from {start_node} to any exit.")
        return shortest if shortest else []

# --- init: Add People ---
    def _add_people(self):
        self.people = []
        person_id = 0
        for node, data in self.G.nodes(data=True):
            if data["Type"] == "Classroom":
                data["width"] = max(data["students"], 1)  # Make width = student count
                for _ in range(int(data["students"])):
                    path = self._find_path_to_exit(node)
                    if not path:
                        print(f"❌ Person {person_id} in {node} has no path to exit.")
                    person = Person(id=person_id, speed=1.0, start_node=node, path=path[1:])  # skip current node
                    self.G.nodes[node]["occupants"].append(person_id)
                    self.people.append(person)
                    person_id += 1

                    person = Person(id=person_id, speed=1.0, start_node=node, path=path[1:])  # skip current node
                    self.G.nodes[node]["occupants"].append(person_id)
                    self.people.append(person)
                    person_id += 1
        print(f"✅ Created {len(self.people)} people.")

# --- main func: Just simulate the graph ---

    def simulate(self):
        self._init_visuals()
        TICKS = 2000
        print_interval = 10
        for tick in range(TICKS):
            # if tick % 2 == 0:
            #     self.draw_graph(tick)
            #     pass
            # -- Don't need visuals for now

            exited = self._count_exited()

            if tick % print_interval == 0:
                self._print_status(tick, exited)

            if self._all_exited(exited):
                print(f"All escaped in {tick} ticks")
                return tick

            for person in self.people:
                if person.status == "exited":
                    continue

                person.time_elapsed += 1

                if person.status == "waiting":
                    self._handle_waiting_person(person)
                elif person.status == "moving":
                    self._handle_moving_person(person)

# --- main func: Run the simulation with a new exit and return the amount of ticks it takes for everyone to exit ---

    def simulate_with_new_exit(self, exit_node_to_test):
        # Add exit to graph temporarily
        added_edges = []
        candidates = []

        self._reset_exit_node_simulation()
        
        # Convert exit_node_to_test to Point if it isn't already
        if hasattr(exit_node_to_test, 'item'):
            # It's a numpy 0-d array
            exit_point = exit_node_to_test.item()
        elif hasattr(exit_node_to_test, 'iloc'):
            # It's a GeoSeries
            exit_point = exit_node_to_test.iloc[0]
        elif not isinstance(exit_node_to_test, Point):
            # Try to convert to Point
            exit_point = Point(exit_node_to_test)
        else:
            exit_point = exit_node_to_test
        
        for node, data in self.G.nodes(data=True):
            if data['Type'] != 'Path':
                continue
            
            node_pos = data['pos']
            node_point = Point(node_pos)
            dist = exit_point.distance(node_point)
            
            if dist <= 100:
                norm_dist = 1 - (dist / 100)
                candidates.append({'node': node, 'dist': dist, 'score': norm_dist})
            else:
                print(dist)
        
        if not candidates:
            print(f"⚠️ No Path nodes within 100m")
            return float('inf')
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        exit_name = f"temp_exit_{id(exit_point)}"
        self.G.add_node(exit_name,
                   Type="Final",
                   width=4,
                   students=0,
                   pos=(exit_point.x, exit_point.y),
                   occupants=[])
        
        num_connections = min(2, len(candidates))
        for c in candidates[:num_connections]:
            self._add_single_edge(c['node'], exit_name, length=c['dist']*100000)
            # idk why by 100000, the  scaling just works that way
            added_edges.append((c['node'], exit_name))
        
        # Recompute paths for all people to new exit
        for person in self.people:
            try:
                full_path = nx.shortest_path(self.G, source=person.current_node, target=self._get_closest_exit_name(person), weight='length')
                # Remove current node from path (person is already there)
                person.path = full_path[1:] if len(full_path) > 1 else []
            except nx.NetworkXNoPath:
                person.path = []
        
        # self._init_visuals()
        # -- Don't need visuals for now
        TICKS = 3000
        print_interval = 10
        for tick in range(TICKS):
            if tick == 50:
                self._download_image()
            
            #     pass
            # -- Don't need visuals for now

            exited = self._count_exited()

            # if tick % print_interval == 0:
            #     self._print_status(tick, exited)

            if self._all_exited(exited):
                print(f"All escaped in {tick} ticks")
                # Clean up temporary exit
                for edge in added_edges:
                    self.G.remove_edge(*edge)
                self.G.remove_node(exit_name)
                return tick

            for person in self.people:
                if person.status == "exited":
                    continue

                person.time_elapsed += 1

                if person.status == "waiting":
                    self._handle_waiting_person(person)
                elif person.status == "moving":
                    self._handle_moving_person(person)
        
        # Clean up if didn't finish
        for edge in added_edges:
            self.G.remove_edge(*edge)
        self.G.remove_node(exit_name)
        return TICKS


# --- helper: reset the simulation environment (needed when testing multiple different exits)

    def _reset_exit_node_simulation(self):
        self.G = nx.Graph()
        self._add_nodes()
        self._add_edges()
        self.people = []
        self._add_people()

# --- helper: returns the closest exit

    def _get_closest_exit_name(self, person):
        current_pos = self.G.nodes[person.current_node]['pos']
        current_point = Point(current_pos)
        
        min_dist = float('inf')
        closest_exit = None
        
        for node, data in self.G.nodes(data=True):
            if data['Type'] == 'Final':
                exit_pos = data['pos']
                exit_point = Point(exit_pos)
                dist = current_point.distance(exit_point)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_exit = node
        
        return closest_exit

# --- helper: inits the visuals

    def _init_visuals(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.manager.full_screen_toggle()
        self.pos = {n: d["pos"] for n, d in self.G.nodes(data=True)}

# ---  draws the graph (for the moving simulation, not a static image of the graph)
        
    def draw_graph(self, tick):
        self.ax.clear()

        # --- Node Visuals ---
        node_colors = []
        node_sizes = []
        labels = {}

        for node, data in self.G.nodes(data=True):
            num_occupants = len(data["occupants"])
            node_sizes.append(100 + 20 * num_occupants)

            is_bottleneck = (
                data["Type"] != "Final" and
                any(
                    p.status == "waiting" and p.current_node == node and
                    (p.path and len(self.G.nodes[p.path[0]]["occupants"]) >= self.G.nodes[p.path[0]]["width"])
                    for p in self.people
                )
            )
            node_colors.append('red' if is_bottleneck else 'skyblue')
            labels[node] = f'{node}\n{num_occupants}'

        nx.draw(
            self.G, self.pos, ax=self.ax, with_labels=True, labels=labels,
            node_size=node_sizes, node_color=node_colors,
            edge_color='gray', font_size=8, font_weight='bold'
        )

        # --- People on Edges ---
        for person in self.people:
            if person.status == "moving" and person.on_edge:
                start, end = person.on_edge
                x0, y0 = self.pos[start]
                x1, y1 = self.pos[end]

                # Compute position along the edge (linear interpolation)
                edge_length = self.G.edges[start, end]["length"]
                frac = min(person.position_on_edge / edge_length, 1.0)
                px = x0 + frac * (x1 - x0)
                py = y0 + frac * (y1 - y0)

                # Plot the person as a small dot
                self.ax.plot(px, py, 'o', color='black', markersize=4)

        self.ax.set_title(f"Evacuation Simulation - Tick {tick}")
        plt.pause(0.001)


    def _download_image(self):
            
            fig, ax = plt.subplots(figsize=(8, 6))
            self.pos = {n: d["pos"] for n, d in self.G.nodes(data=True)}

            # --- Node Visuals ---
            node_colors = []
            node_sizes = []
            labels = {}

            now = time.time()

            for node, data in self.G.nodes(data=True):
                num_occupants = len(data["occupants"])
                node_sizes.append(100 + 20 * num_occupants)

                is_bottleneck = (
                    data["Type"] != "Final" and
                    any(
                        p.status == "waiting" and p.current_node == node and
                        (p.path and len(self.G.nodes[p.path[0]]["occupants"]) >= self.G.nodes[p.path[0]]["width"])
                        for p in self.people
                    )
                )
                node_colors.append('red' if is_bottleneck else 'skyblue')
                labels[node] = f'{node}\n{num_occupants}'

            nx.draw(
                self.G, self.pos, ax=ax, with_labels=True, labels=labels,
                node_size=node_sizes, node_color=node_colors,
                edge_color='gray', font_size=8, font_weight='bold'
            )

            # --- People on Edges ---
            for person in self.people:
                if person.status == "moving" and person.on_edge:
                    start, end = person.on_edge
                    x0, y0 = self.pos[start]
                    x1, y1 = self.pos[end]

                    edge_length = self.G.edges[start, end]["length"]
                    frac = min(person.position_on_edge / edge_length, 1.0)
                    px = x0 + frac * (x1 - x0)
                    py = y0 + frac * (y1 - y0)
                    ax.plot(px, py, 'o', color='black', markersize=4)

            ax.set_title(f"Evacuation Optimisation:  Brute Force - Simulation {self.counter}")
            self.counter += 1

            # --- Save figure ---
            os.makedirs("bruteforce", exist_ok=True)
            filename = f"time_{str(now)}.png"  # zero-padded numbers
            plt.savefig(filename)
            plt.close(ax.figure)  # Close to free memory

# ---  draws a static graph of the network (used for the evolution to show which exit was the most successfull)

    def _draw_winning_exit(self, winning_exit, method):

            added_edges = []
            candidates = []

            fig, ax = plt.subplots(figsize=(8, 6))
            self.pos = {n: d["pos"] for n, d in self.G.nodes(data=True)}


            self._reset_exit_node_simulation()

            if hasattr(winning_exit, 'item'):
                # It's a numpy 0-d array
                exit_point = winning_exit.item()
            elif hasattr(winning_exit, 'iloc'):
                # It's a GeoSeries
                exit_point = winning_exit.iloc[0]
            elif not isinstance(winning_exit, Point):
                # Try to convert to Point
                exit_point = Point(winning_exit)
            else:
                exit_point = winning_exit

            
            for node, data in self.G.nodes(data=True):
                if data['Type'] != 'Path':
                    continue
                
                node_pos = data['pos']
                node_point = Point(node_pos)
                dist = exit_point.distance(node_point)
                
                if dist <= 100:
                    norm_dist = 1 - (dist / 100)
                    candidates.append({'node': node, 'dist': dist, 'score': norm_dist})
            
            if not candidates:
                print(f"⚠️ No Path nodes within 100m")
                return float('inf')
            
            candidates.sort(key=lambda x: x['score'], reverse=True)

            exit_name = f"temp_exit_{id(exit_point)}"
            pos_tuple = (exit_point.x, exit_point.y)

            self.G.add_node(
                exit_name,
                Type="Final",
                width=4,
                students=0,
                pos=pos_tuple,
                occupants=[]
            )

            self.pos[exit_name] = pos_tuple


            
            num_connections = min(2, len(candidates))
            for c in candidates[:num_connections]:
                self._add_single_edge(c['node'], exit_name, length=c['dist']*100000)
                # idk why by 100000, the  scaling just works that way
                added_edges.append((c['node'], exit_name))

            node_colors = []
            node_sizes = []
            labels = {}

            for node, data in self.G.nodes(data=True):
                num_occupants = len(data["occupants"])
                node_sizes.append(100 + 20 * num_occupants)

                is_bottleneck = (
                    data["Type"] != "Final" and
                    any(
                        p.status == "waiting" and p.current_node == node and
                        (p.path and len(self.G.nodes[p.path[0]]["occupants"]) >= self.G.nodes[p.path[0]]["width"])
                        for p in self.people
                    )
                )
                node_colors.append('red' if is_bottleneck else 'skyblue')
                labels[node] = f'{node}\n{num_occupants}'
            


            nx.draw(
                self.G, self.pos, ax=ax,
                with_labels=True, labels=labels,
                node_size=node_sizes, node_color=node_colors,
                edge_color='gray', font_size=8, font_weight='bold'
            )

            ax.set_title(f"Winning Exit ({method})")
            plt.show()

    def _count_exited(self):
        '''Count number of people who have exited.'''
        return sum(p.status == "exited" for p in self.people)


    def _print_status(self, tick, exited):
        '''Print current simulation status.'''
        waiting = sum(p.status == "waiting" for p in self.people)
        moving = sum(p.status == "moving" for p in self.people)
        print(f"[Tick {tick}] Exited: {exited}, Waiting: {waiting}, Moving: {moving}")


    def _all_exited(self, exited):
        '''Check if all people have exited.'''
        return len(self.people) == exited


    def _handle_waiting_person(self, person):
        '''Handle a person in waiting status.'''
        if not person.path:
            person.status = "exited"
            # print(f",/ Person {person.id} exited at node {person.current_node} with no more path.")
            return

        next_node = person.path[0]
        
        if self._can_enter_node(next_node):
            self._start_moving(person, next_node)
        # else:
            # Node is full
            # print(f"!! Person {person.id} at {person.current_node} blocked — {next_node} full.")


    def _can_enter_node(self, node):
        '''Check if a node has capacity for another person.'''
        node_capacity = self.G.nodes[node]["width"]
        current_occupants = self.G.nodes[node]["occupants"]
        incoming = sum(
            1 for p in self.people
            if p.status == "moving" and p.on_edge and p.on_edge[1] == node
        )
        future_occupants = len(current_occupants) + incoming
        
        return future_occupants < node_capacity or self.G.nodes[node]["Type"] != "Path"


    def _start_moving(self, person, next_node):
        '''Transition person from waiting to moving status.'''
        if person.current_node == next_node:
            return
        
        person.on_edge = (person.current_node, next_node)
        person.position_on_edge = 0.0
        self.G.nodes[person.current_node]["occupants"].remove(person.id)
        self.G.edges[person.current_node, next_node]["occupants"].append(person.id)
        person.status = "moving"
        # print(f"-> Person {person.id} started moving from {person.current_node} to {next_node}")


    def _handle_moving_person(self, person):
        '''Handle a person in moving status.'''
        start, end = person.on_edge
        edge_length = self.G.edges[start, end]["length"]
        person.position_on_edge += person.speed

        if person.position_on_edge >= edge_length:
            self._arrive_at_node(person, start, end)


    def _arrive_at_node(self, person, start, end):
        '''Handle person arriving at a node.'''
        person.current_node = end
        person.on_edge = None
        person.path.pop(0)
        self.G.edges[start, end]["occupants"].remove(person.id)
        self.G.nodes[end]["occupants"].append(person.id)

        if self.G.nodes[end]["Type"] == "Final":
            person.status = "exited"
            # print(f"🏁 Person {person.id} exited at {end} in {person.time_elapsed} ticks")
        else:
            person.status = "waiting"
            # print(f"🛬 Person {person.id} arrived at {end} and is now waiting")

    def brute_force_exit_placement(self, num_samples=100):
        """Test evenly-spaced points along the boundary line"""
        
        potential_exit_line_gdf = gpd.read_file(os.getenv("POTENTIAL_EXITS_GEOJSON"))
        geom = potential_exit_line_gdf.geometry.iloc[0]
        
        if geom.geom_type == 'MultiLineString':
            from shapely.ops import linemerge
            potential_exit_line = linemerge(geom)
            if potential_exit_line.geom_type == 'MultiLineString':
                potential_exit_line = max(potential_exit_line.geoms, key=lambda line: line.length)
        else:
            potential_exit_line = geom
        
        total_length = potential_exit_line.length
        if hasattr(total_length, 'item'):
            total_length = total_length.item()
        
        print(f"Testing {num_samples} evenly-spaced exit locations...")
        print(f"Boundary length: {total_length:.6f} degrees")
        
        results = []
        
        for i in range(num_samples):
            # Evenly spaced points
            distance = (i / num_samples) * total_length
            exit_point = potential_exit_line.interpolate(distance)
            
            # Test this exit
            evac_time = self.simulate_with_new_exit(exit_point)
            
            point = exit_point.item() if hasattr(exit_point, 'item') else exit_point
            results.append({
                'position': (point.x, point.y),
                'distance': distance,
                'point': exit_point,
                'evac_time': evac_time
            })
            
            if i % 10 == 0:
                print(f"  Tested {i}/{num_samples}: Best so far = {min(r['evac_time'] for r in results):.0f} ticks")
        
        # Find best
        best = min(results, key=lambda x: x['evac_time'])
        
        print("\n" + "="*50)
        print("BRUTE FORCE RESULTS")
        print("="*50)
        print(f"\n🏆 BEST EXIT LOCATION:")
        print(f"   Position: ({best['position'][0]:.6f}, {best['position'][1]:.6f})")
        print(f"   Distance along boundary: {best['distance']:.6f}")
        print(f"   Evacuation Time: {best['evac_time']:.0f} ticks")
        
        # Show top 5
        print(f"\n📊 TOP 5 EXIT LOCATIONS:")
        results.sort(key=lambda x: x['evac_time'])
        for i, r in enumerate(results[:5], 1):
            print(f"   #{i}: ({r['position'][0]:.6f}, {r['position'][1]:.6f}) - {r['evac_time']:.0f} ticks")
        
        print("="*50 + "\n")
        
        self._draw_winning_exit(best['point'], method="bruteforce")

        return best['position'], best['evac_time'], results

    def genetic_algorithm_exit_placement(self, population_size=20, generations=5,
                                         mutation_rate=0.1, mutation_distance=10):
        ''' 
        Args:
            G: Graph
            population_size: Number of exit locations in each generation
            generations: Number of iterations
            mutation_rate: Probability of mutation (0-1)
            mutation_distance: Max distance in meters to mutate along lin'''

        potential_exit_line = gpd.read_file(os.getenv("POTENTIAL_EXITS_GEOJSON"))
        # potential_exit_line = potential_exit_line.to_crs("EPSG:27700")  # British National Grid

        geom = potential_exit_line.geometry.iloc[0]
    
        # Handle MultiLineString
        if geom.geom_type == 'MultiLineString':
            from shapely.ops import linemerge
            potential_exit_line = linemerge(geom)
            if potential_exit_line.geom_type == 'MultiLineString':
                potential_exit_line = max(potential_exit_line.geoms, key=lambda line: line.length)
        else:
            potential_exit_line = geom

        def random_location_on_line():
            """Generate a random point on the boundary line"""
            distance = random.uniform(0, potential_exit_line.length)
            return potential_exit_line.interpolate(distance)
    
        def fitness(exit_point):
            """Lower evacuation time = higher fitness"""
            evac_time = self.simulate_with_new_exit(exit_point)
            return 1 / evac_time if evac_time > 0 else 0
        
        def crossover_arc_interpolation(parent1, parent2):
            """Arc interpolation with proper wrap-around for closed boundaries"""
            # Extract points
            p1 = parent1.item() if hasattr(parent1, 'item') else parent1
            p2 = parent2.item() if hasattr(parent2, 'item') else parent2
            
            dist1 = potential_exit_line.project(p1)
            dist2 = potential_exit_line.project(p2)
            
            # Extract scalars
            if hasattr(dist1, 'item'):
                dist1 = dist1.item()
            if hasattr(dist2, 'item'):
                dist2 = dist2.item()
            
            total_length = potential_exit_line.length
            if hasattr(total_length, 'item'):
                total_length = total_length.item()
            
            # For a closed loop, consider wrap-around
            # Check if line is closed (first point equals last point)
            is_closed = potential_exit_line.coords[0] == potential_exit_line.coords[-1]
            
            if is_closed:
                # Calculate both possible distances
                direct_dist = abs(dist2 - dist1)
                wrap_dist = total_length - direct_dist
                
                if direct_dist <= wrap_dist:
                    # Direct route is shorter
                    child_dist = (dist1 + dist2) / 2
                else:
                    # Wrap-around route is shorter
                    # Add half the wrap distance to the larger position
                    if dist1 > dist2:
                        child_dist = (dist1 + (dist2 + total_length)) / 2
                    else:
                        child_dist = (dist2 + (dist1 + total_length)) / 2
                    child_dist = child_dist % total_length
            else:
                # Open line - just take midpoint
                child_dist = (dist1 + dist2) / 2
            
            return potential_exit_line.interpolate(child_dist)
            
        def mutate(coords):
            """Mutate by moving point slightly along the line"""
            if random.random() > mutation_rate:
                return coords
            
            point = coords.item() if hasattr(coords, 'item') else coords
            
            current_dist = potential_exit_line.project(point)
            if hasattr(current_dist, 'item'):
                current_dist = current_dist.item()
            
            total_length = potential_exit_line.length
            if hasattr(total_length, 'item'):
                total_length = total_length.item()
            
            # Mutation as PERCENTAGE of total length (not absolute distance), 30% mutation
            max_mutation = total_length * 0.3
            
            offset = np.random.normal(max_mutation/3 , max_mutation)
            new_dist = (current_dist + offset) % total_length
            
            return potential_exit_line.interpolate(new_dist)
        
        # def mutate_large_jump(point):
        #     """
        #     Alternative mutation: occasionally make large jumps for exploration.
        #     """
        #     if random.random() > mutation_rate:
        #         return point
            
        #     if random.random() < 0.1:  # 10% chance of large jump
        #         return random_location_on_line()
        #     else:
        #         # Small local mutation
        #         current_dist = potential_exit_line.project(point)
        #         offset = np.random.normal(0, mutation_distance)
        #         new_dist = (current_dist + offset) % potential_exit_line.length
        #         return potential_exit_line.interpolate(new_dist)
    
        def tournament_select(survivors, fitness_dict, tournament_size=3):
            """
            True tournament selection using pre-computed fitness dictionary.
            Call this AFTER creating fitness_dict once per generation.
            """
            candidates = random.sample(survivors, min(tournament_size, len(survivors)))
            
            # Find best candidate by fitness
            best = max(candidates, key=lambda c: fitness_dict.get(id(c), 0))
            return best
        
        crossover = crossover_arc_interpolation

        population = [random_location_on_line() for _ in range(population_size)]
    
        best_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            scores = [(loc, fitness(loc)) for loc in population]
            # Remove 0s
            scores = [(loc, fit) for loc, fit in scores if fit > 0]

            scores.sort(key=lambda x: x[1], reverse=True)

# DEBUG

            print(f"\nGeneration {gen} population positions:")
            for i, (loc, fit) in enumerate(scores):
                point = loc.item() if hasattr(loc, 'item') else loc
                print(f"  Candidate {i}: ({point.x:.6f}, {point.y:.6f}) = {1/fit:.0f} ticks")
            
            # Track best solution
            best_history.append((scores[0][0], 1/scores[0][1]))
# DEBUG
            best_time = 1/scores[0][1]
            print(f"Generation {gen}: Best = {best_time:.2f} ticks")


            fitness_dict = {id(loc): fit for loc, fit in scores}

            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best evacuation time = {1/scores[0][1]:.2f} ticks")
            
            # Selection (keep top 50%)
            survivors = [x[0] for x in scores[:population_size//2]]
            
            # Elitism: always keep the best solution
            elite = survivors[0]
            
            # Crossover + Mutation to create offspring
            offspring = []
            while len(offspring) < population_size - len(survivors):
                # Tournament selection for parents (better than pure random)
                parent1 = tournament_select(survivors, fitness_dict, tournament_size=1)
                parent2 = tournament_select(survivors, fitness_dict, tournament_size=2)
                
                # Create child
                child = crossover(parent1, parent2)
                child = mutate(child)
                offspring.append(child)
            
            # New population
            population = [elite] + survivors[1:] + offspring
        
        # Return best solution
        final_scores = [(loc, fitness(loc)) for loc in population]
        best = max(final_scores, key=lambda x: x[1])


        # 
        #  -- FINAL RESULTS PRINTING
        # 
        
        print("\n" + "="*50)
        print("GENETIC ALGORITHM RESULTS")
        print("="*50)
        print(f"\n🏆 BEST EXIT LOCATION:")
        
        # Extract coordinates
        best_point = best[0]
        self._draw_winning_exit(best_point, method="genetic")



        if hasattr(best_point, 'item'):
            best_point = best_point.item()
        
        best_x = best_point.x if not hasattr(best_point.x, 'item') else best_point.x.item()
        best_y = best_point.y if not hasattr(best_point.y, 'item') else best_point.y.item()
        
        print(f"   Position: ({best_x:.2f}, {best_y:.2f})")
        print(f"   Evacuation Time: {1/best[1]:.2f} ticks")
        
        print(f"\n📊 ALL CANDIDATES (Final Generation):")
        for i, (loc, fit) in enumerate(final_scores, 1):
            evac_time = 1/fit
            
            # Extract point
            point = loc
            if hasattr(point, 'item'):
                point = point.item()
            
            x = point.x if not hasattr(point.x, 'item') else point.x.item()
            y = point.y if not hasattr(point.y, 'item') else point.y.item()
            
            print(f"   #{i}: Position ({x:.2f}, {y:.2f}) - Time: {evac_time:.2f} ticks")
        print("="*50 + "\n")
        
        return best[0], 1/best[1], best_history            



a = EvacuationSimulator()
# a.simulate()

# --- Simulation Loop ---
print("="*60)
print("BRUTE FORCE SEARCH")
print("="*60)
start_time = time.time()     # Record start time
bf_pos, bf_time, bf_results = a.brute_force_exit_placement(num_samples=50)
end_time = time.time()     # Record end time

print("Execution time of brute force:", end_time - start_time, "seconds")

# # Run GA
# print("\n" + "="*60)
# print("GENETIC ALGORITHM SEARCH")  
# print("="*60)

# start_time = time.time() # Record start time
# ga_pos, ga_time, ga_history = a.genetic_algorithm_exit_placement(
#     population_size=20,
#     generations=5,
#     mutation_rate=0.8,
#     mutation_distance=0.3)
# end_time = time.time()     # Record end time

# print("Execution time of brute force:", end_time - start_time, "seconds")

# time.sleep(10000)
