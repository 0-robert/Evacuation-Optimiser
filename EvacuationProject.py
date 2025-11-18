import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import time
from math import ceil

from dotenv import load_dotenv
import os

load_dotenv()  # Automatically looks for .env in the current directory

print(os.getenv("SECRET_KEY"))
print(os.getenv("DEBUG"))

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
        # --- Load GeoData from QGIS ---
        self.gdf = gpd.read_file(os.getenv("NODES_GEOJSON"))
        self.walks = gpd.read_file(os.getenv("WALKS_GEOJSON"))

        self.G = nx.Graph()

        self.add_nodes()
        self.add_edges()

        self.add_people()

# --- init: Add Nodes ---
    def add_nodes(self):
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
    def add_edges(self):
        for _, row in self.walks.iterrows():
            start = row["start"]
            end = row["end"]
            length = row["Length"]
            self.G.add_edge(start, end, length=length, occupants=[])

# --- Helper: Find Closest Exit ---
    def find_path_to_exit(self, start_node):
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
    def add_people(self):
        self.people = []
        person_id = 0
        for node, data in self.G.nodes(data=True):
            if data["Type"] == "Classroom":
                data["width"] = max(data["students"], 1)  # Make width = student count
                for _ in range(int(data["students"])):
                    path = self.find_path_to_exit(node)
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

    def simulate(self):

        self._init_visuals()
        TICKS = 2000
        print_interval = 10
        for tick in range(TICKS):
            if tick % 2 == 0:
                self.draw_graph(tick)
                pass

            exited = self._count_exited()

            if tick % print_interval == 0:
                self._print_status(tick, exited)

            if self._all_exited(exited):
                print(f"All escaped in {tick} ticks")
                break

            for person in self.people:
                if person.status == "exited":
                    continue

                person.time_elapsed += 1

                if person.status == "waiting":
                    self._handle_waiting_person(person)
                elif person.status == "moving":
                    self._handle_moving_person(person)

    def _init_visuals(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.pos = {n: d["pos"] for n, d in self.G.nodes(data=True)}

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
            # print(f"✅ Person {person.id} exited at node {person.current_node} with no more path.")
            return

        next_node = person.path[0]
        
        if self._can_enter_node(next_node):
            self._start_moving(person, next_node)
        # else:
            # Node is full
            # print(f"⏸️ Person {person.id} at {person.current_node} blocked — {next_node} full.")


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
        person.on_edge = (person.current_node, next_node)
        person.position_on_edge = 0.0
        self.G.nodes[person.current_node]["occupants"].remove(person.id)
        self.G.edges[person.current_node, next_node]["occupants"].append(person.id)
        person.status = "moving"
        # print(f"➡️ Person {person.id} started moving from {person.current_node} to {next_node}")


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
            print(f"🏁 Person {person.id} exited at {end} in {person.time_elapsed} ticks")
        else:
            person.status = "waiting"
            # print(f"🛬 Person {person.id} arrived at {end} and is now waiting")



a = EvacuationSimulator()


# --- Simulation Loop ---
a.simulate()



