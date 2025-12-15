import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import networkx as nx
from tqdm import tqdm

# --- PARAMETRI ---
VELOCITA_KMH = 180   # velocità dell'elicottero
VELOCITA_MS = VELOCITA_KMH * 1000 / 3600  # conversione in m/s

# --- 1️⃣ Carica dati ---
start_gdf = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\ProgettoQGIS3.0\Area atterraggio elicottero.shp")
hospitals = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\ProgettoQGIS3.0\Centroide Ospedali.shp")
no_fly = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\ProgettoQGIS3.0\no_fly_zone_Lombardia.gpkg")

# --- 2️⃣ Assicurati che tutti i layer abbiano lo stesso CRS ---
crs_proj = "EPSG:32632"  # UTM 32N
start_gdf = start_gdf.to_crs(crs_proj)
hospitals = hospitals.to_crs(crs_proj)
no_fly = no_fly.to_crs(crs_proj)

# --- 3️⃣ Prepara dati ---
start_point = start_gdf.geometry.iloc[32]
nearest_hospital = min(hospitals.geometry, key=lambda p: start_point.distance(p))
no_fly = no_fly[no_fly['no_fly'] == 1]

print("Ospedale più vicino trovato ✔️")

# --- 4️⃣ Filtra poligoni vicini ---
buffer_distance = 20000  # 20 km
buffer_area = start_point.buffer(buffer_distance).union(nearest_hospital.buffer(buffer_distance))
no_fly_filtered = no_fly[no_fly.intersects(buffer_area)]
no_fly_union = unary_union(no_fly_filtered.geometry)

print(f"Poligoni no-fly considerati: {len(no_fly_filtered)}")

# --- 5️⃣ Crea nodi ---
nodes = [start_point, nearest_hospital]

for poly in no_fly_filtered.geometry:
    # Esterni
    if poly.geom_type == "Polygon":
        nodes.extend([Point(x, y) for x, y in poly.exterior.coords])
        # Interni (fori)
        for interior in poly.interiors:
            nodes.extend([Point(x, y) for x, y in interior.coords])

    elif poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            nodes.extend([Point(x, y) for x, y in p.exterior.coords])
            for interior in p.interiors:
                nodes.extend([Point(x, y) for x, y in interior.coords])

print(f"Totale nodi usati nel grafo: {len(nodes)}")

# --- 6️⃣ Costruisci grafo ---
G = nx.Graph()
skipped = 0
added = 0
max_dist = 20000  # max 20 km collegamento nodi

for i, p1 in enumerate(tqdm(nodes, desc="Costruzione grafo")):
    for j, p2 in enumerate(nodes):
        if i >= j:
            continue
        dist = p1.distance(p2)
        if dist > max_dist:
            continue
        line = LineString([p1, p2])
        if line.intersects(no_fly_union):
            skipped += 1
            continue
        G.add_edge((p1.x, p1.y), (p2.x, p2.y), weight=dist)
        added += 1

print(f"Archi aggiunti: {added}, archi saltati: {skipped}")

# --- 7️⃣ Calcolo percorso più breve ---
start_node = (start_point.x, start_point.y)
end_node = (nearest_hospital.x, nearest_hospital.y)

print("Calcolo percorso...")
try:
    shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
except nx.NetworkXNoPath:
    raise Exception("Nessun percorso possibile! Le no-fly zone bloccano tutte le rotte.")

# --- 8️⃣ Costruisci geometria e calcola distanza totale ---
path_lines = []
total_distance = 0  # in metri

for i in range(len(shortest_path) - 1):
    p1 = shortest_path[i]
    p2 = shortest_path[i+1]
    seg = LineString([p1, p2])
    total_distance += seg.length
    path_lines.append(seg)

# --- 9️⃣ Tempo di volo (in secondi, minuti, ore) ---
tempo_secondi = total_distance / VELOCITA_MS
tempo_minuti = tempo_secondi / 60
tempo_ore = tempo_minuti / 60

print(f"Distanza totale: {total_distance/1000:.2f} km")
print(f"Tempo stimato: {tempo_minuti:.1f} minuti ({tempo_ore:.2f} ore)")

# --- 🔟 Salva GeoJSON con attributi ---
path_gdf = gpd.GeoDataFrame(
    {
        "distanza_m": [total_distance] * len(path_lines),
        "tempo_min": [tempo_minuti] * len(path_lines),
        "tempo_ore": [tempo_ore] * len(path_lines),
    },
    geometry=path_lines,
    crs=start_gdf.crs
)

output_file = r"C:\Users\giuli\OneDrive\Desktop\ProgettoQGIS3.0\Percorso_Elicottero_aggirato_veloce3.geojson"
path_gdf.to_file(output_file, driver="GeoJSON")

print("\n✅ FILE SALVATO!")
print(f"Percorso salvato in: {output_file}")
