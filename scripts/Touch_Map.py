from qgis.core import QgsProject
import geopandas as gpd
from shapely.geometry import Point
import os

# 1️⃣ Trova il layer dei centroidi
layer_name = "Polygon_Centroid"
layers = QgsProject.instance().mapLayersByName(layer_name)

if not layers:
    raise Exception(f"Layer '{layer_name}' non trovato!")

layer = layers[0]

# 2️⃣ Prendi la feature selezionata
selected_features = layer.selectedFeatures()

if not selected_features:
    raise Exception("Seleziona manualmente un punto nel layer Polygon_Centroid!")

feat = selected_features[0]
geom = feat.geometry()

# 3️⃣ Converte in Shapely Point
if geom.type() != 0:  # 0 = Point
    raise Exception("La geometria selezionata non è un Point!")

shapely_point = Point(geom.asPoint())

# 4️⃣ Salva il punto in GeoJSON
layer_path = layer.dataProvider().dataSourceUri()
save_folder = os.path.dirname(layer_path)
output_file = os.path.join(save_folder, "start_point.geojson")

gdf = gpd.GeoDataFrame({'id':[feat.id()]},
                       geometry=[shapely_point],
                       crs="EPSG:32632")  # usa CRS del progetto

gdf.to_file(output_file, driver="GeoJSON")
print(f"Punto selezionato salvato automaticamente in: {output_file}")
