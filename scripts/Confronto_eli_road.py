import geopandas as gpd
import os
import sys


import numpy as np





# Percorsi ai file
file_completo = r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Percorso_completo.geojson"
file_stradale = r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis\ProgettoQGIS3.0\Percorso_stradale.geojson"

# Controllo esistenza file
for f in [file_completo, file_stradale]:
    if not os.path.exists(f):
        sys.exit(f"❌ File non trovato: {f}")

# Carico GeoDataFrame
gdf_completo = gpd.read_file(file_completo)
gdf_stradale = gpd.read_file(file_stradale)

# Controllo start e end points
start_completo = gdf_completo.geometry.iloc[0].coords[0]
start_stradale = gdf_stradale.geometry.iloc[0].coords[0]
end_completo = gdf_completo.geometry.iloc[-1].coords[-1]
end_stradale = gdf_stradale.geometry.iloc[-1].coords[-1]


# Prendo il primo valore di tempo_totale_min (sono tutti uguali all'interno del GeoJSON)
tempo_completo_min = gdf_completo["tempo_totale_min"].iloc[0]
tempo_stradale_min = gdf_stradale["tempo_totale_min"].iloc[0]

# Conversione in ore e minuti
def format_time(minuti):
    h = int(minuti // 60)
    m = int(minuti % 60)
    return f"{h}h {m}m"

tempo_completo_str = format_time(tempo_completo_min)
tempo_stradale_str = format_time(tempo_stradale_min)

# Confronto tempi
if tempo_completo_min < tempo_stradale_min:
    percorso_veloce = "Percorso completo (auto + elicottero)"
    tempo_veloce = tempo_completo_str
else:
    percorso_veloce = "Percorso solo stradale"
    tempo_veloce = tempo_stradale_str

# Output leggibile
print("📊 TEMPI DI PERCORSO")
print(f"- Percorso completo (auto + elicottero): {tempo_completo_str}")
print(f"- Percorso solo stradale: {tempo_stradale_str}")
print(f"✅ Percorso più veloce: {percorso_veloce} ({tempo_veloce})")
