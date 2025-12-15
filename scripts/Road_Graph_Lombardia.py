import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString

# Punto di partenza
start_gdf = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\start_point.geojson")
start_point = start_gdf.geometry.iloc[0]

# Carica rete e ospedali
roads = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Strade_Lombardia.geojson")
end_point = gpd.read_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Centroide Ospedali.shp")

# Costruzione grafo
G = nx.Graph()
for idx, row in roads.iterrows():
    geom = row.geometry
    lines = geom.geoms if geom.geom_type=="MultiLineString" else [geom]
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords)-1):
            pt1, pt2 = coords[i], coords[i+1]
            G.add_edge(pt1, pt2, weight=Point(pt1).distance(Point(pt2)))

# Funzione nodo più vicino
def nearest_node(G, point):
    return min(G.nodes, key=lambda node: Point(node).distance(point))

# Nodo di partenza
start_node = nearest_node(G, start_point)

# Trova l'ospedale più vicino
nearest_hospital_geom = min(end_point.geometry, key=lambda p: start_point.distance(p))
end_node = nearest_node(G, nearest_hospital_geom)

# Percorso più breve
shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
path_lines = [LineString([shortest_path[i], shortest_path[i+1]]) for i in range(len(shortest_path)-1)]

# Salvataggio percorso
path_gdf = gpd.GeoDataFrame(geometry=path_lines, crs=roads.crs)
path_gdf.to_file(r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Percorso_Ospedale_auto.geojson", driver="GeoJSON")
print("Percorso calcolato e salvato!")
