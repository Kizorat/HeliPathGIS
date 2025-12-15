import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.strtree import STRtree
from shapely.prepared import prep
import networkx as nx
import numpy as np
import os

# --------------------- PARAMETRI ---------------------
VELOCITA_KMH_ELI = 180
VELOCITA_MS_ELI = VELOCITA_KMH_ELI * 1000 / 3600

VELOCITA_KMH_AUTO = 100
VELOCITA_MS_AUTO = VELOCITA_KMH_AUTO * 1000 / 3600

MAX_DIST_FLY = 20000  # massimo collegamento tra nodi volo
GRID_STEP = 3000
crs_proj = "EPSG:32632"

# --------------------- CARICAMENTO DATI ---------------------
start = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\start_point.geojson")
eliports = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Area atterraggio elicottero.shp")
roads = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Strade_Lombardia.geojson")
hospitals = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\hospital_point.geojson")
no_fly = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\no_fly_zone_Lombardia_pulito.gpkg")

# uniformiamo CRS
for layer in [start, eliports, roads, hospitals, no_fly]:
    layer.to_crs(crs_proj, inplace=True)

start_point = start.geometry.iloc[0]
print("✔️ Punto di partenza:", start_point)

# --------------------- GRAFO STRADALE ---------------------
G_auto = nx.Graph()

for idx, row in roads.iterrows():
    geom = row.geometry
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
    elif geom.geom_type == "MultiLineString":
        coords = [c for line in geom.geoms for c in line.coords]
    else:
        continue

    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        dist = Point(p1).distance(Point(p2))
        G_auto.add_edge(p1, p2, weight=dist)

# --------------------- STRtree dei nodi stradali ---------------------
road_nodes = [Point(x, y) for (x, y) in G_auto.nodes]
tree_nodes = STRtree(road_nodes)

def nearest_node(point):
    idx = tree_nodes.nearest(point)
    geom = road_nodes[idx]
    return (geom.x, geom.y)

start_node = nearest_node(start_point)

# --------------------- TROVA L’ELIPORTO MIGLIORE ---------------------
lengths = nx.single_source_dijkstra_path_length(G_auto, start_node, weight="weight")

best_heli = None
best_dist = float("inf")
best_path_auto = None

for helipoint in eliports.geometry:
    heli_node = nearest_node(helipoint)
    if heli_node in lengths:
        dist = lengths[heli_node]
        if dist < best_dist:
            best_dist = dist
            best_heli = helipoint
            best_path_auto = nx.shortest_path(G_auto, start_node, heli_node, weight="weight")

if best_heli is None:
    raise Exception("❌ Nessun eliporto raggiungibile su strada.")

# --------------------- CALCOLO PERCORSO AUTO ---------------------
auto_segments = [LineString([best_path_auto[i], best_path_auto[i + 1]]) for i in range(len(best_path_auto) - 1)]
dist_auto = sum(s.length for s in auto_segments)
tempo_auto_min = dist_auto / VELOCITA_MS_AUTO / 60
print(f"🚗 Percorso auto: {dist_auto/1000:.2f} km ({tempo_auto_min:.1f} min)")

# --------------------- NO-FLY ZONES ---------------------
no_fly_polys = no_fly[no_fly["no_fly"] == 1].explode(ignore_index=True).geometry.tolist()
prepared_no_fly = [prep(poly) for poly in no_fly_polys]

def intersects_no_fly(geom):
    if geom.is_empty:
        return False
    for poly in prepared_no_fly:
        if poly.intersects(geom):
            return True
    return False

# --------------------- GRAFO DI VOLO ---------------------
hospital = hospitals.geometry.iloc[0]

# Inizializzo nodi con eliporto e ospedale
nodes = [best_heli, hospital]

# Aggiungo vertici dei poligoni no-fly
for poly in no_fly_polys:
    nodes.extend([Point(x, y) for x, y in poly.exterior.coords])
    for ring in poly.interiors:
        nodes.extend([Point(x, y) for x, y in ring.coords])

# Aggiungo punti della griglia solo se NON sono dentro no-fly
buffer_dist = MAX_DIST_FLY * 2
xmin = min(best_heli.x, hospital.x) - buffer_dist
xmax = max(best_heli.x, hospital.x) + buffer_dist
ymin = min(best_heli.y, hospital.y) - buffer_dist
ymax = max(best_heli.y, hospital.y) + buffer_dist

for x in np.arange(xmin, xmax + GRID_STEP, GRID_STEP):
    for y in np.arange(ymin, ymax + GRID_STEP, GRID_STEP):
        p = Point(x, y)
        if not intersects_no_fly(p):
            nodes.append(p)

# Costruzione grafo: collego solo nodi entro MAX_DIST_FLY e che NON intersecano no-fly
G_fly = nx.Graph()
for i, p1 in enumerate(nodes):
    for j in range(i + 1, len(nodes)):
        p2 = nodes[j]
        if p1.distance(p2) > MAX_DIST_FLY:
            continue
        line = LineString([p1, p2])
        if not intersects_no_fly(line):
            G_fly.add_edge((p1.x, p1.y), (p2.x, p2.y), weight=p1.distance(p2))

# --------------------- CALCOLO PERCORSO VOLO ---------------------
start_fly = (best_heli.x, best_heli.y)
end_fly = (hospital.x, hospital.y)

path_fly = nx.shortest_path(G_fly, start_fly, end_fly, weight="weight")
fly_segments = [LineString([path_fly[i], path_fly[i + 1]]) for i in range(len(path_fly) - 1)]
dist_fly = sum(s.length for s in fly_segments)
tempo_fly_min = dist_fly / VELOCITA_MS_ELI / 60
print(f"🚁 Percorso volo: {dist_fly/1000:.2f} km ({tempo_fly_min:.1f} min)")

# --------------------- RISULTATO FINALE ---------------------
distanza_totale_m = dist_auto + dist_fly
tempo_totale_min = tempo_auto_min + tempo_fly_min

all_segments = auto_segments + fly_segments
segment_type = ["auto"] * len(auto_segments) + ["eli"] * len(fly_segments)

gdf_out = gpd.GeoDataFrame(
    {
        "distanza_totale_m": [distanza_totale_m] * len(all_segments),
        "tempo_totale_min": [tempo_totale_min] * len(all_segments),
        "type": segment_type
    },
    geometry=all_segments,
    crs=crs_proj
)

output = r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Percorso_completo.geojson"
if os.path.exists(output):
    os.remove(output)

gdf_out.to_file(output, driver="GeoJSON")
print("✅ FILE SALVATO!")
print("📄", output)
