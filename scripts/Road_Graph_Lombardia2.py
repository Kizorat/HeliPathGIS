import os
import sys
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
import networkx as nx
from tqdm import tqdm

# --------------------- PARAMETRI ---------------------
VELOCITA_KMH_AUTO = 100
VELOCITA_MS_AUTO = VELOCITA_KMH_AUTO * 1000 / 3600
crs_proj = "EPSG:32632"

# --------------------- CARICAMENTO DATI ---------------------
start = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\start_point.geojson")
reticolo = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Polygon_Centroid.gpkg")
roads = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Strade_Lombardia.geojson")
hospitals = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\hospital_point.geojson")
points_hospital = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Centroidi Ospedali.gpkg")

for layer in [start, reticolo, roads, hospitals, points_hospital]:
    layer.to_crs(crs_proj, inplace=True)

start_point = start.geometry.iloc[0]
hospital_point = hospitals.geometry.iloc[0]

# --------------------- CONTROLLI ---------------------
if not any(start_point.within(poly) for poly in reticolo.geometry):
    sys.exit("❌ ERRORE: selezionare un punto di partenza presente nel reticolo!")

if not any(hospital_point.within(poly) for poly in points_hospital.geometry):
    sys.exit("❌ ERRORE: selezionare un hospital_point presente nei centroidi Ospedali!")

print("✔️ Start_point e hospital_point validi")

# --------------------- CREAZIONE GRAFO STRADALE ---------------------
G_auto = nx.Graph()
print("⛓️ Costruzione grafo stradale...")

for idx, row in tqdm(roads.iterrows(), total=len(roads), desc="⛓️ Costruzione grafo stradale"):
    geom = row.geometry
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
    elif geom.geom_type == "MultiLineString":
        coords = [c for line in geom.geoms for c in line.coords]
    else:
        continue
    for i in range(len(coords)-1):
        p1, p2 = coords[i], coords[i+1]
        G_auto.add_edge(p1, p2, weight=Point(p1).distance(Point(p2)))

print("✔️ Grafo stradale pronto.")

# --------------------- STRTREE PER NEAREST NODE ---------------------
road_nodes = [Point(x, y) for (x, y) in G_auto.nodes]
tree_nodes = STRtree(road_nodes)

def nearest_node(point):
    idx = tree_nodes.nearest(point)
    nearest_geom = road_nodes[idx]
    return (nearest_geom.x, nearest_geom.y)

start_node = nearest_node(start_point)
end_node = nearest_node(hospital_point)

# --------------------- CALCOLO PERCORSO ---------------------
print("🔍 Calcolo percorso stradale...")

try:
    path = nx.shortest_path(G_auto, start_node, end_node, weight="weight")
except nx.NetworkXNoPath:
    sys.exit("❌ Nessun percorso disponibile tra start_point e hospital_point.")

segments = [LineString([path[i], path[i+1]]) for i in range(len(path)-1)]
dist_total = sum(seg.length for seg in segments)
tempo_total_min = dist_total / VELOCITA_MS_AUTO / 60

print(f"🚗 Percorso stradale: {dist_total/1000:.2f} km ({tempo_total_min:.1f} min)")

# --------------------- SALVATAGGIO ---------------------
gdf_out = gpd.GeoDataFrame(
    {"distanza_totale_m":[dist_total]*len(segments),
     "tempo_totale_min":[tempo_total_min]*len(segments)},
    geometry=segments,
    crs=crs_proj
)

output = r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Percorso_stradale.geojson"
if os.path.exists(output):
    os.remove(output)

gdf_out.to_file(output, driver="GeoJSON")
print("✅ FILE SALVATO! (sovrascritto se già esisteva)")
print("📄", output)
