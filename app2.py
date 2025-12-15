import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid
from pyproj import Transformer
import traceback
import torch
import time
from collections import defaultdict
import heapq
import numpy as np
from shapely.strtree import STRtree
from shapely.prepared import prep
from scipy.spatial import KDTree
from collections import deque

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"C:\Users\giuli\OneDrive\Desktop\webapp_Qgis"
DATA_DIR = os.path.join(BASE_DIR, "ProgettoQGIS3.0")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Speed configurations
HELI_SPEED_KMH = 180
AUTO_SPEED_KMH = 100
MAX_HELI_EDGE_DIST = 20000

# Grid configuration - OTTIMIZZATO per volare solo nelle fly zones
GRID_STEP = 800  # Step più largo per griglia meno densa
MAX_EDGE_LENGTH = 10000  # Aumentato molto per connettere zone distanti
SAFETY_BUFFER_M = -300  # Buffer NEGATIVO per restringere le zone di volo (distanza dal bordo)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {DEVICE}")

# ============================================================
# APP INIT
# ============================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Transformers
transformer_to4326 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
transformer_to32632 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

# ============================================================
# DATA LOADING
# ============================================================
logger.info("📂 Loading geospatial data...")
t_load = time.perf_counter()

STRADA_FILE = os.path.join(DATA_DIR, "Strade_Lombardia.geojson")
HELI_FILE = os.path.join(DATA_DIR, "Area atterraggio elicottero.gpkg")
FLY_ZONE_FILE = os.path.join(DATA_DIR, "fly_zone_Lombardia.gpkg")  # File delle zone di volo permesse




strade = gpd.read_file(STRADA_FILE).to_crs("EPSG:4326")
strade_m = strade.to_crs("EPSG:32632")

heli_areas = gpd.read_file(HELI_FILE).to_crs("EPSG:4326")
heli_areas_m = heli_areas.to_crs("EPSG:32632")

# Carica le FLY ZONES (zone dove si può volare)
fly_zones = gpd.read_file(FLY_ZONE_FILE).to_crs("EPSG:4326")
fly_zones_m = fly_zones.to_crs("EPSG:32632")



logger.info(f"✅ Data loaded in {time.perf_counter() - t_load:.3f}s")
logger.info(f"   - Roads: {len(strade_m)} features")
logger.info(f"   - Helipads: {len(heli_areas_m)} features")
logger.info(f"   - Fly zones: {len(fly_zones_m)} features (zone di volo permesse)")

# ============================================================
# PREPARED FLY ZONE GEOMETRIES (per performance)
# ============================================================
logger.info("🔧 Preparing fly zone geometries...")
t_prep = time.perf_counter()

# Prepara geometrie delle fly zones con buffer negativo per sicurezza
# Buffer NEGATIVO: restringe le zone per garantire distanza di sicurezza dai bordi
prepared_fly_zones = []
fly_zones_buffered = []

# Prima ripariamo le geometrie se sono invalide
valid_geoms = []
for idx, geom in enumerate(fly_zones_m.geometry):
    if geom is not None and not geom.is_empty:
        if not geom.is_valid:
            logger.warning(f"   ⚠️  Geometry {idx} is invalid, repairing...")
            geom = make_valid(geom)
        
        # Applica buffer(0) per risolvere problemi di self-intersection
        try:
            geom = geom.buffer(0)
            valid_geoms.append(geom)
            bounds = geom.bounds
            logger.info(f"   Fly zone {idx}: bounds={bounds}, area={geom.area:.0f} m², valid={geom.is_valid}")
        except Exception as e:
            logger.error(f"   ❌ Failed to repair geometry {idx}: {e}")
            continue

if valid_geoms:
    # Ora possiamo fare l'unione delle geometrie riparate
    try:
        all_fly_zones = unary_union(valid_geoms)
        logger.info(f"   ✅ Successfully created union of {len(valid_geoms)} fly zones")
    except Exception as e:
        logger.error(f"   ❌ Failed to create union: {e}")
        # Fallback: usa la prima geometria valida
        all_fly_zones = valid_geoms[0]
else:
    logger.error("❌ No valid fly zone geometries found!")
    all_fly_zones = None

if all_fly_zones is not None and not all_fly_zones.is_empty:
    if SAFETY_BUFFER_M < 0:
        all_fly_zones_buffered = all_fly_zones.buffer(SAFETY_BUFFER_M)  # Buffer negativo
    else:
        all_fly_zones_buffered = all_fly_zones.buffer(SAFETY_BUFFER_M)
    
    prepared_fly_zones.append(prep(all_fly_zones_buffered))
    fly_zones_buffered.append(all_fly_zones_buffered)
    
    logger.info(f"   📏 Total fly zone area: {all_fly_zones.area:.0f} m²")
    logger.info(f"   📏 Buffered area: {all_fly_zones_buffered.area:.0f} m²")
else:
    logger.error("❌ Failed to create valid fly zone union!")

logger.info(f"✅ Prepared fly zones in {time.perf_counter() - t_prep:.3f}s")

# Test di verifica
if len(prepared_fly_zones) > 0:
    logger.info("🧪 Testing fly zone detection...")
    # Prendi un punto centrale della Lombardia per test
    test_point = Point(600000, 5030000)  # Coordinate approssimative centro Lombardia
    test_result = any(pg.contains(test_point) for pg in prepared_fly_zones)
    logger.info(f"   ✓ Test point at (600000, 5030000): {test_result} (dovrebbe essere True se nelle fly zones)")
else:
    logger.error("❌ NO FLY ZONES LOADED!")

# ============================================================
# GRAPH CACHE
# ============================================================
GRAPH_CACHE = {
    "nodes": None,
    "node_idx": None,
    "edges": None,
    "edge_src": None,
    "edge_dst": None,
    "edge_weight": None,
    "edge_dict": None
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def point_in_fly_zone(point: Point, prepared_geoms: list) -> bool:
    """Check if a point is inside any fly zone (zona di volo permessa)."""
    return any(pg.contains(point) for pg in prepared_geoms)

def line_in_fly_zone(line: LineString, prepared_geoms: list, raw_geoms: list) -> bool:
    """
    Check if a line is completely inside fly zones.
    Usa campionamento di punti per verificare che tutta la linea sia dentro.
    """
    # Metodo 1: controllo diretto - la linea deve essere INTERAMENTE dentro una fly zone
    for raw_geom in raw_geoms:
        if raw_geom.contains(line):
            return True
    
    # Metodo 2: se non è interamente contenuta, verifica che ogni punto campionato sia dentro
    # Campiona punti ogni 500m lungo la linea
    if line.length > 500:
        num_samples = max(3, int(line.length / 500))
        
        for i in range(num_samples + 1):
            t = i / num_samples  # da 0.0 a 1.0
            sample_point = line.interpolate(t, normalized=True)
            
            # Controlla se questo punto è dentro una fly zone
            inside = False
            for raw_geom in raw_geoms:
                if raw_geom.contains(sample_point):
                    inside = True
                    break
            
            if not inside:
                return False  # Un punto non è dentro -> linea non valida
        return True  # Tutti i punti campionati sono dentro
    
    # Per linee corte, controlla solo i punti estremi
    start_point = Point(line.coords[0])
    end_point = Point(line.coords[-1])
    
    start_inside = any(pg.contains(start_point) for pg in prepared_geoms)
    end_inside = any(pg.contains(end_point) for pg in prepared_geoms)
    
    return start_inside and end_inside

def snap_to_road(start_point: Point, road_geoms: list) -> Point:
    """Snap a point to the nearest road."""
    min_dist = float('inf')
    snapped_point = start_point
    
    for road in road_geoms:
        proj = road.interpolate(road.project(start_point))
        dist = start_point.distance(proj)
        if dist < min_dist:
            min_dist = dist
            snapped_point = proj
    
    return snapped_point

# ============================================================
# ROAD GRAPH CONSTRUCTION
# ============================================================
def prepare_graph_gpu(strade_layer, speed_kmh=AUTO_SPEED_KMH):
    """Build road graph once and cache it."""
    if GRAPH_CACHE["nodes"] is not None:
        logger.info("✅ Using cached road graph")
        return

    logger.info("🏗️  Building road graph...")
    t_start = time.perf_counter()

    nodes = []
    edges = []

    for idx, row in strade_layer.iterrows():
        geom = row.geometry
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                nodes.extend([p1, p2])
                dist = Point(p1).distance(Point(p2))
                travel_time = dist / (speed_kmh * 1000 / 3600)  # Convert to seconds
                edges.append((p1, p2, travel_time))

    t_nodes = time.perf_counter()
    logger.info(f"   ⏱️  Extracted nodes/edges in {t_nodes - t_start:.3f}s")

    # Unique nodes and indexing
    nodes = list({n: n for n in nodes}.values())
    node_idx = {n: i for i, n in enumerate(nodes)}

    t_index = time.perf_counter()
    logger.info(f"   ⏱️  Indexed {len(nodes)} nodes in {t_index - t_nodes:.3f}s")

    # Build adjacency dict for Dijkstra
    edge_dict = defaultdict(list)
    for s_coord, d_coord, w in edges:
        s = node_idx[s_coord]
        d = node_idx[d_coord]
        edge_dict[s].append((d, w))
        edge_dict[d].append((s, w))

    t_dict = time.perf_counter()
    logger.info(f"   ⏱️  Built adjacency dict in {t_dict - t_index:.3f}s")

    # Cache everything
    GRAPH_CACHE.update({
        "nodes": nodes,
        "node_idx": node_idx,
        "edge_dict": edge_dict
    })

    logger.info(f"✅ Road graph ready: {len(nodes)} nodes, {len(edges)} edges (total: {t_dict - t_start:.3f}s)")

# ============================================================
# DIJKSTRA SHORTEST PATH (ROAD NETWORK)
# ============================================================
def compute_auto_path_dijkstra(start_point: Point, end_point: Point):
    """Compute shortest path on road network using Dijkstra."""
    t_start = time.perf_counter()
    
    nodes = GRAPH_CACHE["nodes"]
    node_idx = GRAPH_CACHE["node_idx"]
    edge_dict = GRAPH_CACHE["edge_dict"]

    if nodes is None:
        raise ValueError("Graph not initialized. Call prepare_graph_gpu() first.")

    N = len(nodes)

    # Find nearest nodes
    t_snap = time.perf_counter()
    start_idx = min(range(N), key=lambda i: Point(nodes[i]).distance(start_point))
    end_idx = min(range(N), key=lambda i: Point(nodes[i]).distance(end_point))
    logger.info(f"   ⏱️  Node snapping: {time.perf_counter() - t_snap:.3f}s")

    if start_idx == end_idx:
        start_4326 = transformer_to4326.transform(start_point.x, start_point.y)
        end_4326 = transformer_to4326.transform(end_point.x, end_point.y)
        return [[list(start_4326), list(end_4326)]]

    # Dijkstra algorithm
    t_dijkstra = time.perf_counter()
    dist = [float('inf')] * N
    prev = [-1] * N
    dist[start_idx] = 0
    visited = [False] * N
    heap = [(0, start_idx)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        
        if u == end_idx:
            break
            
        for v, w in edge_dict[u]:
            alt = d_u + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    logger.info(f"   ⏱️  Dijkstra computation: {time.perf_counter() - t_dijkstra:.3f}s")

    # Reconstruct path
    if dist[end_idx] == float('inf'):
        logger.warning("⚠️  No path found between nodes")
        return []

    path_nodes = []
    u = end_idx
    while u != -1:
        path_nodes.insert(0, nodes[u])
        u = prev[u]

    # Convert to WGS84
    t_convert = time.perf_counter()
    line_4326 = [list(transformer_to4326.transform(x, y)) for x, y in path_nodes]
    logger.info(f"   ⏱️  Coordinate conversion: {time.perf_counter() - t_convert:.3f}s")
    logger.info(f"✅ Auto path computed in {time.perf_counter() - t_start:.3f}s")

    return [line_4326]

# ============================================================
# HELICOPTER GRID & PATHFINDING - VERSIONE CON FLY ZONES
# ============================================================
def build_heli_grid(start_p: Point, end_p: Point, step: int = GRID_STEP):
    """Build a grid for helicopter pathfinding, INSIDE fly zones only."""
    t_start = time.perf_counter()
    
    # Se non ci sono fly zones valide, ritorna None
    if not prepared_fly_zones:
        logger.error("❌ No valid fly zones available!")
        return None, None, None
    
    min_x, max_x = sorted([start_p.x, end_p.x])
    min_y, max_y = sorted([start_p.y, end_p.y])
    
    # Aumenta il margin per garantire connettività
    margin = step * 8  # Margine molto ampio
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    xs = np.arange(min_x, max_x + step, step)
    ys = np.arange(min_y, max_y + step, step)
    
    logger.info(f"   🗺️  Grid bounds: X[{min_x:.0f}, {max_x:.0f}], Y[{min_y:.0f}, {max_y:.0f}]")
    logger.info(f"   🗺️  Grid will have ~{len(xs) * len(ys)} points before filtering")
    
    # Generate all grid points
    all_points = [[x, y] for x in xs for y in ys]
    
    # Filter points: keep ONLY points inside fly zones
    valid_points = []
    rejected_points = 0
    
    for pt in all_points:
        point = Point(pt[0], pt[1])
        if point_in_fly_zone(point, prepared_fly_zones):
            valid_points.append(pt)
        else:
            rejected_points += 1
    
    # AGGIUNTA: Punti strategici ai bordi delle fly zones (punti di passaggio)
    border_points_added = 0
    
    for fly_geom in fly_zones_buffered:
        # 1. Punti al bordo (perimetro)
        boundary = fly_geom.boundary
        
        if boundary.length > 0:
            # Aumenta la densità dei punti sul bordo
            num_border_points = max(12, int(boundary.length / 400))  # Ogni 400m
            for i in range(num_border_points):
                t = i / num_border_points
                border_point = boundary.interpolate(t, normalized=True)
                
                # Aggiungi punti leggermente DENTRO il bordo
                # Trova la direzione verso l'interno approssimativa
                centroid = fly_geom.centroid
                dx = centroid.x - border_point.x
                dy = centroid.y - border_point.y
                length = max(0.1, (dx*dx + dy*dy)**0.5)
                
                # Sposta di 100m verso l'interno
                offset_x = (dx / length) * 100
                offset_y = (dy / length) * 100
                
                adjusted_point = [border_point.x + offset_x, border_point.y + offset_y]
                
                test_p = Point(adjusted_point[0], adjusted_point[1])
                if point_in_fly_zone(test_p, prepared_fly_zones):
                    # Verifica che il punto sia all'interno dell'area di ricerca
                    if (min_x <= adjusted_point[0] <= max_x and 
                        min_y <= adjusted_point[1] <= max_y):
                        valid_points.append(adjusted_point)
                        border_points_added += 1
        
        # 2. Punti all'interno della fly zone (griglia più sparsa)
        bounds = fly_geom.bounds
        # Genera punti ogni 2000m all'interno della bounding box
        x_steps = max(2, int((bounds[2] - bounds[0]) / 2000) + 1)
        y_steps = max(2, int((bounds[3] - bounds[1]) / 2000) + 1)
        
        for i in range(x_steps):
            for j in range(y_steps):
                x = bounds[0] + i * 2000
                y = bounds[1] + j * 2000
                point = Point(x, y)
                
                if fly_geom.contains(point):
                    # Verifica che sia dentro la fly zone bufferata
                    if point_in_fly_zone(point, prepared_fly_zones):
                        valid_points.append([x, y])
                        border_points_added += 1
    
    logger.info(f"   ⏱️  Grid generation: {time.perf_counter() - t_start:.3f}s")
    logger.info(f"   📊 Grid: {len(all_points)} total points, {len(valid_points)-border_points_added} valid (in fly zones), {rejected_points} rejected")
    logger.info(f"   📊 Added {border_points_added} strategic points inside fly zones")
    
    if len(valid_points) == 0:
        logger.error("❌ No valid grid points - start/end might be outside fly zones!")
        # Forza l'aggiunta di start e end anche se fuori dalle fly zones per debug
        valid_points.append([start_p.x, start_p.y])
        valid_points.append([end_p.x, end_p.y])
        logger.warning("⚠️  Forced start and end points into grid for debugging")
    
    # AGGIUNTA IMPORTANTE: Punti lungo una linea retta tra start e end (se dentro fly zones)
    line = LineString([(start_p.x, start_p.y), (end_p.x, end_p.y)])
    
    # Verifica se la linea diretta è interamente dentro le fly zones
    if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
        logger.info("   📍 Adding intermediate points along direct line (inside fly zones)")
        num_intermediate = max(2, int(line.length / (step * 1.5)))
        for i in range(1, num_intermediate):
            t = i / num_intermediate
            interp_point = line.interpolate(t, normalized=True)
            valid_points.append([interp_point.x, interp_point.y])
    else:
        logger.info("   ⚠️  Direct line is not entirely inside fly zones, will need detour")
    
    # CORREZIONE CRITICA: Prima di aggiungere start e end, stampiamo le coordinate per debug
    logger.info(f"   🔍 DEBUG: Start point coordinates: ({start_p.x:.1f}, {start_p.y:.1f})")
    logger.info(f"   🔍 DEBUG: End point coordinates: ({end_p.x:.1f}, {end_p.y:.1f})")
    
    # CORREZIONE CRITICA: Creiamo una lista di punti con identificatori
    all_points_with_ids = []
    
    # Prima aggiungiamo i punti della griglia
    for pt in valid_points:
        all_points_with_ids.append(('grid', pt[0], pt[1]))
    
    # Poi aggiungiamo start e end con identificatori speciali
    all_points_with_ids.append(('start', start_p.x, start_p.y))
    all_points_with_ids.append(('end', end_p.x, end_p.y))
    
    # Ora deduplichiamo mantenendo l'ordine e gli identificatori
    unique_points = []
    seen_coords = set()
    start_idx = None
    end_idx = None
    
    for i, (pt_type, x, y) in enumerate(all_points_with_ids):
        # Arrotonda le coordinate per la deduplicazione
        coord_key = (round(x, 1), round(y, 1))
        
        if coord_key not in seen_coords:
            seen_coords.add(coord_key)
            unique_points.append([x, y])
            
            # Registra gli indici di start e end
            if pt_type == 'start':
                start_idx = len(unique_points) - 1
            elif pt_type == 'end':
                end_idx = len(unique_points) - 1
    
    # CORREZIONE CRITICA: Se start o end non sono stati trovati (erano duplicati), 
    # trova i punti più vicini
    if start_idx is None:
        # Trova il punto più vicino a start_p
        distances = []
        for pt in unique_points:
            dist = ((pt[0] - start_p.x)**2 + (pt[1] - start_p.y)**2)**0.5
            distances.append(dist)
        start_idx = np.argmin(distances)
        logger.warning(f"   ⚠️  Start point was duplicate, using closest point at index {start_idx}")
    
    if end_idx is None:
        # Trova il punto più vicino a end_p
        distances = []
        for pt in unique_points:
            dist = ((pt[0] - end_p.x)**2 + (pt[1] - end_p.y)**2)**0.5
            distances.append(dist)
        end_idx = np.argmin(distances)
        logger.warning(f"   ⚠️  End point was duplicate, using closest point at index {end_idx}")
    
    # VERIFICA CRITICA: Controlliamo che gli indici siano corretti
    logger.info(f"   📍 Total unique grid points: {len(unique_points)}")
    logger.info(f"   📍 Start point index: {start_idx}, coordinates: ({unique_points[start_idx][0]:.1f}, {unique_points[start_idx][1]:.1f})")
    logger.info(f"   📍 End point index: {end_idx}, coordinates: ({unique_points[end_idx][0]:.1f}, {unique_points[end_idx][1]:.1f})")
    
    # VERIFICA FINALE: Le coordinate dovrebbero essere vicine a quelle originali
    start_coord_diff = ((unique_points[start_idx][0] - start_p.x)**2 + 
                       (unique_points[start_idx][1] - start_p.y)**2)**0.5
    end_coord_diff = ((unique_points[end_idx][0] - end_p.x)**2 + 
                     (unique_points[end_idx][1] - end_p.y)**2)**0.5
    
    if start_coord_diff > 1000:  # più di 1km di differenza
        logger.error(f"   ❌ Start point mismatch: diff={start_coord_diff:.1f}m")
        logger.error(f"   ❌ Original: ({start_p.x:.1f}, {start_p.y:.1f})")
        logger.error(f"   ❌ In grid: ({unique_points[start_idx][0]:.1f}, {unique_points[start_idx][1]:.1f})")
    
    if end_coord_diff > 1000:  # più di 1km di differenza
        logger.error(f"   ❌ End point mismatch: diff={end_coord_diff:.1f}m")
        logger.error(f"   ❌ Original: ({end_p.x:.1f}, {end_p.y:.1f})")
        logger.error(f"   ❌ In grid: ({unique_points[end_idx][0]:.1f}, {unique_points[end_idx][1]:.1f})")
    
    return torch.tensor(unique_points, dtype=torch.float32, device=DEVICE), start_idx, end_idx

def build_heli_edges(points: torch.Tensor, start_idx: int, end_idx: int, max_dist: float = MAX_EDGE_LENGTH):
    """Build edges for helicopter graph, ONLY inside fly zones."""
    t_start = time.perf_counter()
    
    edges = []
    weights = []
    pts = points.cpu().numpy()
    N = len(pts)
    
    checked = 0
    rejected_distance = 0
    rejected_not_in_fly_zone = 0
    
    logger.info(f"   🔍 Building edges for {N} points (only inside fly zones)...")
    
    # DEBUG: Verifica la connettività dei punti di start e end
    start_point = pts[start_idx]
    end_point = pts[end_idx]
    
    logger.info(f"   🔍 Start point (idx={start_idx}): {start_point}")
    logger.info(f"   🔍 End point (idx={end_idx}): {end_point}")
    
    # Verifica se start e end sono dentro le fly zones
    start_in_fly = point_in_fly_zone(Point(start_point[0], start_point[1]), prepared_fly_zones)
    end_in_fly = point_in_fly_zone(Point(end_point[0], end_point[1]), prepared_fly_zones)
    
    logger.info(f"   📍 Start point in fly zone: {start_in_fly}")
    logger.info(f"   📍 End point in fly zone: {end_in_fly}")
    
    # PRIMA STRATEGIA: Prova a connettere direttamente start e end se possibile
    line = LineString([tuple(start_point), tuple(end_point)])
    direct_dist = np.linalg.norm(start_point - end_point)
    
    if (direct_dist <= max_dist and 
        line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered)):
        logger.info(f"   ✅ Direct start-end connection is possible! Distance: {direct_dist:.0f}m")
        edges.append([start_idx, end_idx])
        edges.append([end_idx, start_idx])
        weights.append(direct_dist)
        weights.append(direct_dist)
    else:
        logger.info(f"   ⚠️  Direct start-end connection not possible. Distance: {direct_dist:.0f}m, in fly zones: {line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered)}")
    
    # SECONDA STRATEGIA: KDTree per vicini più prossimi
    logger.info("   🌳 Building KDTree for neighbor search...")
    tree = KDTree(pts)
    
    # Per ogni punto, trova i k vicini più vicini
    k = min(25, N-1)  # Numero di vicini da considerare (aumentato)
    distances, indices = tree.query(pts, k=k)
    
    logger.info(f"   🔗 Connecting each point to {k} nearest neighbors...")
    
    for i in range(N):
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):  # Salta se stesso (primo elemento)
            if j_idx <= i:  # Evita duplicati (considera solo j > i)
                continue
                
            checked += 1
            j = int(j_idx)
            p1, p2 = pts[i], pts[j]
            
            # Distance filter
            if dist > max_dist:
                rejected_distance += 1
                continue
            
            # Fly zone intersection check - la linea DEVE essere dentro le fly zones
            line = LineString([tuple(p1), tuple(p2)])
            if not line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                rejected_not_in_fly_zone += 1
                continue
            
            # Valid edge (bidirectional)
            edges.append([i, j])
            edges.append([j, i])
            weights.append(dist)
            weights.append(dist)
    
    # TERZA STRATEGIA: Assicurati che i punti critici (start, end) siano ben connessi
    critical_points = [start_idx, end_idx]
    
    for cp in critical_points:
        # Trova i vicini più vicini per i punti critici (aumenta k)
        cp_neighbors = tree.query(pts[cp], k=min(40, N))[1]
        neighbors_added = 0
        for neighbor in cp_neighbors[1:]:  # Salta se stesso
            neighbor = int(neighbor)
            if neighbor == cp:
                continue
                
            line = LineString([tuple(pts[cp]), tuple(pts[neighbor])])
            dist = np.linalg.norm(pts[cp] - pts[neighbor])
            
            if (dist <= max_dist * 2.5 and  # Limite molto ampio per punti critici
                line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered)):
                
                # Controlla se l'arco esiste già
                edge_exists = any((e[0] == cp and e[1] == neighbor) for e in edges)
                if not edge_exists:
                    edges.append([cp, neighbor])
                    edges.append([neighbor, cp])
                    weights.append(dist)
                    weights.append(dist)
                    neighbors_added += 1
        
        logger.info(f"   🔗 Added {neighbors_added} extra connections to critical point {cp}")
    
    logger.info(f"   ⏱️  Edge building: {time.perf_counter() - t_start:.3f}s")
    logger.info(f"   📊 Edge statistics:")
    logger.info(f"      ├─ Checked: {checked}")
    logger.info(f"      ├─ Rejected (distance > {max_dist}m): {rejected_distance}")
    logger.info(f"      ├─ Rejected (not in fly zone): {rejected_not_in_fly_zone}")
    logger.info(f"      └─ Valid edges: {len(edges)//2}")
    
    # DEBUG: Verifica se l'arco diretto è stato aggiunto
    direct_edge_exists = any((e[0] == start_idx and e[1] == end_idx) for e in edges)
    logger.info(f"   🔗 Direct edge between start and end in graph: {direct_edge_exists}")
    
    if not edges:
        logger.warning("⚠️  No valid edges found!")
        return None, None
    
    # DEBUG: Verifica la connettività del grafo con BFS
    adj = defaultdict(list)
    for edge in edges:
        if edge[0] < edge[1]:
            adj[edge[0]].append(edge[1])
            adj[edge[1]].append(edge[0])
    
    # BFS per verificare la connettività tra start e end
    visited = set()
    queue = deque([start_idx])
    visited.add(start_idx)
    
    while queue:
        node = queue.popleft()
        if node == end_idx:
            logger.info("   ✅ Graph connectivity: START and END are connected!")
            break
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    else:
        logger.warning(f"   ⚠️  Graph connectivity: START and END are NOT connected!")
        logger.warning(f"   ⚠️  Start component size: {len(visited)} nodes")
        
        # Trova la componente dell'end node
        visited_end = set()
        queue = deque([end_idx])
        visited_end.add(end_idx)
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in visited_end:
                    visited_end.add(neighbor)
                    queue.append(neighbor)
        
        logger.warning(f"   ⚠️  End component size: {len(visited_end)} nodes")
        
        # Prova a connettere le due componenti
        logger.info("   🔗 Attempting to bridge disconnected components...")
        bridge_added = 0
        
        for node1 in list(visited)[:min(15, len(visited))]:
            for node2 in list(visited_end)[:min(15, len(visited_end))]:
                p1, p2 = pts[node1], pts[node2]
                dist = np.linalg.norm(p1 - p2)
                
                if dist > max_dist * 4:
                    continue
                
                line = LineString([tuple(p1), tuple(p2)])
                if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                    edges.append([node1, node2])
                    edges.append([node2, node1])
                    weights.append(dist)
                    weights.append(dist)
                    bridge_added += 1
                    logger.info(f"   🌉 Added bridge between components (dist={dist:.0f}m)")
                    break
            
            if bridge_added > 0:
                break
        
        if bridge_added == 0:
            logger.error("   ❌ Could not bridge disconnected components!")
    
    return (
        torch.tensor(edges, dtype=torch.long, device=DEVICE),
        torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )

def dijkstra_gpu(points: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor, 
                 start_idx: int, end_idx: int):
    """Dijkstra shortest path on GPU-ready graph using heapq."""
    t_start = time.perf_counter()
    
    if edges is None:
        logger.warning("   ⚠️  No edges available for pathfinding")
        return [], float("inf")
    
    N = points.shape[0]
    
    # Converti tensori a CPU per Dijkstra più semplice (evita problemi GPU)
    edges_cpu = edges.cpu().numpy()
    weights_cpu = weights.cpu().numpy()
    
    # Costruisci lista di adiacenza
    adj = defaultdict(list)
    for i in range(len(edges_cpu)):
        u, v = int(edges_cpu[i][0]), int(edges_cpu[i][1])
        w = float(weights_cpu[i])
        adj[u].append((v, w))
    
    # Dijkstra standard con heapq
    dist = [float('inf')] * N
    prev = [-1] * N
    dist[start_idx] = 0
    visited = [False] * N
    
    heap = []
    heapq.heappush(heap, (0.0, start_idx))
    iterations = 0
    
    while heap:
        iterations += 1
        d_u, u = heapq.heappop(heap)
        
        if visited[u]:
            continue
        visited[u] = True
        
        if u == end_idx:
            logger.info(f"   ✅ Path found after {iterations} iterations!")
            break
        
        for v, w in adj[u]:
            alt = d_u + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    
    logger.info(f"   ⏱️  Dijkstra: {time.perf_counter() - t_start:.3f}s ({iterations} iterations)")
    
    # Reconstruct path
    if dist[end_idx] == float("inf"):
        logger.warning(f"   ⚠️  No path found: start and end are disconnected in the graph")
        logger.info(f"   📊 Connectivity check:")
        logger.info(f"      - Start node {start_idx} has {len(adj[start_idx])} neighbors")
        logger.info(f"      - End node {end_idx} has {len(adj[end_idx])} neighbors")
        return [], float("inf")
    
    path = []
    u = end_idx
    while u != -1:
        path.insert(0, points[u].cpu().tolist())
        u = prev[u]
    
    total_dist = float(dist[end_idx])
    logger.info(f"   📍 Path found with {len(path)} waypoints, total distance: {total_dist:.0f}m")
    
    # Calcola il numero di segmenti (LineString)
    num_segments = len(path) - 1
    logger.info(f"   📐 Path consists of {num_segments} segments")
    
    return path, total_dist

# ============================================================
# HELICOPTER ROUTE COMPUTATION
# ============================================================
def compute_heli_route(start_point_m: Point, end_point_m: Point):
    """Compute full helicopter route: auto to helipad + flight to hospital."""
    t_total = time.perf_counter()
    logger.info("🚁 === HELICOPTER ROUTE COMPUTATION ===")
    
    try:
        # Step 1: Find nearest reachable helipad
        t_heli = time.perf_counter()
        helipads_m = [(geom.x, geom.y) for geom in heli_areas_m.geometry]
        
        if not helipads_m:
            return {"success": False, "error": "No helipads defined"}
        
        helipad_tensor = torch.tensor(helipads_m, device=DEVICE)
        start_tensor = torch.tensor([start_point_m.x, start_point_m.y], device=DEVICE)
        dists = torch.norm(helipad_tensor - start_tensor, dim=1)
        reachable = torch.where(dists <= MAX_HELI_EDGE_DIST)[0]
        
        if len(reachable) == 0:
            return {"success": False, "error": f"No helipad within {MAX_HELI_EDGE_DIST}m"}
        
        best_idx = reachable[dists[reachable].argmin()]
        heli_x, heli_y = helipads_m[int(best_idx)]
        helipad_point_m = Point(heli_x, heli_y)
        
        logger.info(f"   ✅ Helipad found at ({heli_x:.0f}, {heli_y:.0f}) in {time.perf_counter() - t_heli:.3f}s")
        
        # Verifica se l'eliporto è dentro le fly zones
        helipad_in_fly = point_in_fly_zone(helipad_point_m, prepared_fly_zones)
        logger.info(f"   📍 Helipad in fly zone: {helipad_in_fly}")
        
        # Step 2: Auto path to helipad
        logger.info("🚗 Computing auto path to helipad...")
        t_auto = time.perf_counter()
        
        snapped_start = snap_to_road(start_point_m, list(strade_m.geometry))
        multilines_auto = compute_auto_path_dijkstra(snapped_start, helipad_point_m)
        
        if not multilines_auto:
            return {"success": False, "error": "No auto path to helipad"}
        
        # Calculate auto distance
        auto_dist_m = 0.0
        for seg in multilines_auto:
            for i in range(len(seg) - 1):
                x1, y1 = transformer_to32632.transform(seg[i][0], seg[i][1])
                x2, y2 = transformer_to32632.transform(seg[i+1][0], seg[i+1][1])
                auto_dist_m += Point(x1, y1).distance(Point(x2, y2))
        
        auto_km = auto_dist_m / 1000.0
        auto_time_h = auto_km / AUTO_SPEED_KMH
        
        logger.info(f"   ✅ Auto path: {auto_km:.2f} km, {auto_time_h*60:.1f} min ({time.perf_counter() - t_auto:.3f}s)")
        
        # Step 3: Helicopter flight path
        logger.info("✈️  Computing helicopter flight path...")
        t_flight = time.perf_counter()
        
        grid, start_idx, end_idx = build_heli_grid(helipad_point_m, end_point_m)
        if grid is None:
            return {"success": False, "error": "Cannot build grid - check fly zones"}
        
        edges, weights = build_heli_edges(grid, start_idx, end_idx)
        
        if edges is None:
            # Fallback: direct line (solo se dentro fly zones)
            logger.warning("⚠️  No valid flight path found, checking direct line...")
            line = LineString([(heli_x, heli_y), (end_point_m.x, end_point_m.y)])
            if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                heli_path_m = [[heli_x, heli_y], [end_point_m.x, end_point_m.y]]
                heli_dist_m = helipad_point_m.distance(end_point_m)
                logger.info("   ✅ Direct line is in fly zones, using it")
            else:
                logger.error("❌ Direct line is not in fly zones and no alternative path found")
                return {"success": False, "error": "No flight path within fly zones"}
        else:
            # Usa Dijkstra con start_idx e end_idx già calcolati
            heli_path_m, heli_dist_m = dijkstra_gpu(grid, edges, weights, start_idx, end_idx)
            
            if not heli_path_m:
                logger.warning("⚠️  Dijkstra failed, checking direct line...")
                line = LineString([(heli_x, heli_y), (end_point_m.x, end_point_m.y)])
                if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                    heli_path_m = [[heli_x, heli_y], [end_point_m.x, end_point_m.y]]
                    heli_dist_m = helipad_point_m.distance(end_point_m)
                    logger.info("   ✅ Direct line is in fly zones, using it")
                else:
                    logger.error("❌ No valid flight path found")
                    return {"success": False, "error": "No flight path within fly zones"}
        
        heli_km = heli_dist_m / 1000.0
        heli_time_h = heli_km / HELI_SPEED_KMH
        
        logger.info(f"   ✅ Flight path: {heli_km:.2f} km, {heli_time_h*60:.1f} min ({time.perf_counter() - t_flight:.3f}s)")
        
        # Step 4: Convert to WGS84
        path_heli_4326 = [[transformer_to4326.transform(x, y) for x, y in heli_path_m]]
        helipad_lon, helipad_lat = transformer_to4326.transform(heli_x, heli_y)
        
        total_km = auto_km + heli_km
        total_time_h = auto_time_h + heli_time_h
        
        logger.info(f"🏁 Total helicopter route: {total_km:.2f} km, {total_time_h*60:.1f} min (computed in {time.perf_counter() - t_total:.3f}s)")
        
        return {
            "success": True,
            "multilines_auto": multilines_auto,
            "auto_km": auto_km,
            "auto_time_h": auto_time_h,
            "path_heli_4326": path_heli_4326,
            "heli_km": heli_km,
            "heli_time_h": heli_time_h,
            "helipad_lonlat": (helipad_lon, helipad_lat),
            "total_km": total_km,
            "total_time_h": total_time_h
        }
        
    except Exception as e:
        logger.error(f"❌ Helicopter route error: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}

# ============================================================
# INITIALIZE ROAD GRAPH
# ============================================================
logger.info("🔧 Initializing road network graph...")
prepare_graph_gpu(strade_m)

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/geo")
def get_geo(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)

# ============================================================
# ENDPOINT AVANZATO PER CARICAMENTO DINAMICO
# ============================================================
@app.get("/geojson")
def get_geojson(
    filename: str,
    bbox: str = Query(None, description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    simplify: float = Query(None, description="Simplification tolerance in meters (for large files)")
):
    """Enhanced geojson endpoint with bounding box filtering and simplification"""
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    try:
        # Special handling for large building files
        if "Palazzi" in filename and bbox:
            logger.info(f"🏢 Loading buildings with bbox: {bbox}")
            
            # Parse bbox
            try:
                min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
            except ValueError:
                return JSONResponse({"error": "Invalid bbox format. Use: min_lon,min_lat,max_lon,max_lat"}, status_code=400)
            
            # Load with bounding box filter
            gdf = gpd.read_file(file_path, bbox=(min_lon, min_lat, max_lon, max_lat))
            
            if gdf.empty:
                logger.info(f"📍 No buildings found in bbox: {bbox}")
                return JSONResponse({"type": "FeatureCollection", "features": []})
            
            # Convert to 4326
            if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Simplify geometries if requested (reduces file size)
            if simplify and simplify > 0:
                logger.info(f"🔧 Simplifying geometries with tolerance: {simplify}m")
                # Convert to metric CRS for accurate simplification
                gdf_metric = gdf.to_crs("EPSG:32632")
                gdf_metric['geometry'] = gdf_metric['geometry'].simplify(tolerance=float(simplify))
                gdf = gdf_metric.to_crs("EPSG:4326")
            
            # Limit features for safety
            MAX_BUILDINGS = 50000
            if len(gdf) > MAX_BUILDINGS:
                logger.warning(f"⚠️ Too many buildings ({len(gdf)}), limiting to {MAX_BUILDINGS}")
                gdf = gdf.head(MAX_BUILDINGS)
            
            logger.info(f"✅ Loaded {len(gdf)} buildings for bbox {bbox}")
            
            return JSONResponse(gdf.__geo_interface__)
        
        # Standard loading for other files
        else:
            gdf = gpd.read_file(file_path)
            gdf = gdf.to_crs("EPSG:4326")
            return JSONResponse(gdf.__geo_interface__)
        
    except Exception as e:
        logger.error(f"❌ Error loading {filename}: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================================================
# AUTO PATH ENDPOINT
# ============================================================
@app.get("/auto-path")
def auto_path(
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    end_lat: float = Query(...),
    end_lon: float = Query(...)
):
    logger.info(f"🚗 AUTO PATH REQUEST: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    
    try:
        t_total = time.perf_counter()
        
        # Transform coordinates
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        end_x, end_y = transformer_to32632.transform(end_lon, end_lat)
        start_point = Point(start_x, start_y)
        end_point = Point(end_x, end_y)
        
        # Compute path
        multilines = compute_auto_path_dijkstra(start_point, end_point)
        
        if not multilines:
            return JSONResponse({"error": "No path found"}, status_code=404)
        
        # Calculate total distance
        total_dist_m = 0
        for segment in multilines:
            for i in range(len(segment) - 1):
                x1, y1 = transformer_to32632.transform(segment[i][0], segment[i][1])
                x2, y2 = transformer_to32632.transform(segment[i+1][0], segment[i+1][1])
                total_dist_m += Point(x1, y1).distance(Point(x2, y2))
        
        total_dist_km = total_dist_m / 1000
        total_time_h = total_dist_km / AUTO_SPEED_KMH
        hours = int(total_time_h)
        minutes = int((total_time_h - hours) * 60)
        
        logger.info(f"✅ Auto path completed in {time.perf_counter() - t_total:.3f}s: {total_dist_km:.2f} km, {hours}h{minutes}m")
        
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": multilines
                },
                "properties": {
                    "color": "yellow",
                    "distance_km": round(total_dist_km, 2),
                    "time_hhmm": f"{hours:02d}:{minutes:02d}"
                }
            }]
        }
        
        return JSONResponse(geojson)
        
    except Exception as e:
        logger.error(f"❌ Auto path error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================================================
# HELICOPTER PATH ENDPOINT
# ============================================================
@app.get("/heli-path")
def heli_path(
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    end_lat: float = Query(...),
    end_lon: float = Query(...)
):
    logger.info(f"🚁 HELI PATH REQUEST: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    
    try:
        t_total = time.perf_counter()
        
        # Transform to meters
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        end_x, end_y = transformer_to32632.transform(end_lon, end_lat)
        start_m = Point(start_x, start_y)
        end_m = Point(end_x, end_y)
        
        # Compute helicopter route
        result = compute_heli_route(start_m, end_m)
        
        if not result["success"]:
            return JSONResponse({"error": result.get("error", "Unknown error")}, status_code=404)
        
        # Build GeoJSON response
        h_a = int(result["auto_time_h"])
        m_a = int((result["auto_time_h"] - h_a) * 60)
        
        h_h = int(result["heli_time_h"])
        m_h = int((result["heli_time_h"] - h_h) * 60)
        
        h_t = int(result["total_time_h"])
        m_t = int((result["total_time_h"] - h_t) * 60)
        
        hel_lon, hel_lat = result["helipad_lonlat"]
        
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiLineString", "coordinates": result["multilines_auto"]},
                    "properties": {
                        "mode": "auto",
                        "distance_km": round(result["auto_km"], 2),
                        "time_hhmm": f"{h_a:02d}:{m_a:02d}"
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiLineString", "coordinates": result["path_heli_4326"]},
                    "properties": {
                        "mode": "heli",
                        "distance_km": round(result["heli_km"], 2),
                        "time_hhmm": f"{h_h:02d}:{m_h:02d}"
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [hel_lon, hel_lat]},
                    "properties": {"type": "helipad"}
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [end_lon, end_lat]},
                    "properties": {"type": "hospital"}
                },
                {
                    "type": "Feature",
                    "geometry": None,
                    "properties": {
                        "distance_km_total": round(result["total_km"], 2),
                        "time_hhmm_total": f"{h_t:02d}:{m_t:02d}"
                    }
                }
            ]
        }
        
        logger.info(f"✅ Heli path completed in {time.perf_counter() - t_total:.3f}s")
        return JSONResponse(geojson)
        
    except Exception as e:
        logger.error(f"❌ Heli path error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ============================================================
# BEST PATH ENDPOINT (Compare auto vs heli)
# ============================================================
@app.get("/best-path")
def best_path(
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    end_lat: float = Query(...),
    end_lon: float = Query(...)
):
    logger.info(f"🏆 BEST PATH REQUEST: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    
    try:
        t_total = time.perf_counter()
        
        # Transform to meters
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        end_x, end_y = transformer_to32632.transform(end_lon, end_lat)
        start_m = Point(start_x, start_y)
        end_m = Point(end_x, end_y)
        
        # Snap to road
        snapped_start = snap_to_road(start_m, list(strade_m.geometry))
        
        # --- Option 1: Direct auto route ---
        logger.info("📊 Evaluating direct auto route...")
        t_auto = time.perf_counter()
        
        auto_multilines = compute_auto_path_dijkstra(snapped_start, end_m)
        auto_info = {"available": False}
        
        if auto_multilines:
            total_dist_m = 0.0
            for segment in auto_multilines:
                for i in range(len(segment) - 1):
                    x1, y1 = transformer_to32632.transform(segment[i][0], segment[i][1])
                    x2, y2 = transformer_to32632.transform(segment[i+1][0], segment[i+1][1])
                    total_dist_m += Point(x1, y1).distance(Point(x2, y2))
            
            auto_km = total_dist_m / 1000.0
            auto_time_h = auto_km / AUTO_SPEED_KMH
            auto_info = {
                "available": True,
                "multilines": auto_multilines,
                "auto_km": auto_km,
                "auto_time_h": auto_time_h
            }
            logger.info(f"   ✅ Direct auto: {auto_km:.2f} km, {auto_time_h*60:.1f} min ({time.perf_counter() - t_auto:.3f}s)")
        else:
            logger.warning(f"   ⚠️  Direct auto route not found ({time.perf_counter() - t_auto:.3f}s)")
        
        # --- Option 2: Helicopter route ---
        logger.info("📊 Evaluating helicopter route...")
        t_heli = time.perf_counter()
        
        heli_info = compute_heli_route(snapped_start, end_m)
        heli_info["available"] = heli_info.get("success", False)
        
        if heli_info["available"]:
            logger.info(f"   ✅ Helicopter: {heli_info['total_km']:.2f} km, {heli_info['total_time_h']*60:.1f} min ({time.perf_counter() - t_heli:.3f}s)")
        else:
            logger.warning(f"   ⚠️  Helicopter route not found: {heli_info.get('error', 'Unknown')} ({time.perf_counter() - t_heli:.3f}s)")
        
        # --- Choose best option ---
        candidates = []
        if auto_info.get("available"):
            candidates.append(("auto", auto_info["auto_time_h"]))
        if heli_info.get("available"):
            candidates.append(("heli", heli_info["total_time_h"]))
        
        if not candidates:
            return JSONResponse({"error": "No route available (neither auto nor helicopter)"}, status_code=404)
        
        best_mode, best_time = min(candidates, key=lambda x: x[1])
        logger.info(f"🏆 Best option: {best_mode.upper()} ({best_time*60:.1f} min)")
        
        # --- Build GeoJSON response ---
        features = []
        
        # Add auto route if available
        if auto_info.get("available"):
            h = int(auto_info["auto_time_h"])
            m = int((auto_info["auto_time_h"] - h) * 60)
            features.append({
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": auto_info["multilines"]},
                "properties": {
                    "mode": "auto",
                    "distance_km": round(auto_info["auto_km"], 2),
                    "time_hhmm": f"{h:02d}:{m:02d}",
                    "is_best": best_mode == "auto"
                }
            })
        
        # Add helicopter route if available
        if heli_info.get("available"):
            # Auto segment to helipad
            h_a = int(heli_info["auto_time_h"])
            m_a = int((heli_info["auto_time_h"] - h_a) * 60)
            features.append({
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": heli_info["multilines_auto"]},
                "properties": {
                    "mode": "auto_to_helipad",
                    "distance_km": round(heli_info["auto_km"], 2),
                    "time_hhmm": f"{h_a:02d}:{m_a:02d}",
                    "is_best": best_mode == "heli"
                }
            })
            
            # Flight segment
            h_h = int(heli_info["heli_time_h"])
            m_h = int((heli_info["heli_time_h"] - h_h) * 60)
            features.append({
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": heli_info["path_heli_4326"]},
                "properties": {
                    "mode": "heli",
                    "distance_km": round(heli_info["heli_km"], 2),
                    "time_hhmm": f"{h_h:02d}:{m_h:02d}",
                    "is_best": best_mode == "heli"
                }
            })
            
            # Helipad and hospital markers
            hel_lon, hel_lat = heli_info["helipad_lonlat"]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [hel_lon, hel_lat]},
                "properties": {"type": "helipad"}
            })
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [end_lon, end_lat]},
                "properties": {"type": "hospital"}
            })
        
        # CALCOLA I VALORI TOTALI CORRETTI
        distance_km_total = 0
        time_hhmm_total = "00:00"
        
        if best_mode == "auto" and auto_info.get("available"):
            distance_km_total = auto_info["auto_km"]
            time_h = auto_info["auto_time_h"]
        elif best_mode == "heli" and heli_info.get("available"):
            distance_km_total = heli_info["total_km"]
            time_h = heli_info["total_time_h"]
        
        # Converti tempo in formato HH:MM
        hours = int(time_h)
        minutes = int((time_h - hours) * 60)
        time_hhmm_total = f"{hours:02d}:{minutes:02d}"
        
        # Crea il messaggio di confronto
        comparison_message = ""
        if auto_info.get("available") and heli_info.get("available"):
            auto_time_min = auto_info["auto_time_h"] * 60
            heli_time_min = heli_info["total_time_h"] * 60
            time_diff = abs(auto_time_min - heli_time_min)
            
            if best_mode == "auto":
                comparison_message = (
                    f"🏆 PERCORSO PIÙ VELOCE: AUTO\n\n"
                    f"🚗 Auto: {auto_info['auto_km']:.2f} km, {auto_time_min:.1f} min\n"
                    f"🚁 Elicottero: {heli_info['total_km']:.2f} km, {heli_time_min:.1f} min\n\n"
                    f"Risparmio di tempo: {time_diff:.1f} min"
                )
            else:
                comparison_message = (
                    f"🏆 PERCORSO PIÙ VELOCE: ELICOTTERO\n\n"
                    f"🚁 Elicottero: {heli_info['total_km']:.2f} km, {heli_time_min:.1f} min\n"
                    f"🚗 Auto: {auto_info['auto_km']:.2f} km, {auto_time_min:.1f} min\n\n"
                    f"Risparmio di tempo: {time_diff:.1f} min"
                )
        elif auto_info.get("available"):
            auto_time_min = auto_info["auto_time_h"] * 60
            comparison_message = (
                f"🏆 PERCORSO DISPONIBILE: AUTO\n\n"
                f"🚗 Auto: {auto_info['auto_km']:.2f} km, {auto_time_min:.1f} min\n"
                f"🚁 Elicottero: Non disponibile"
            )
        elif heli_info.get("available"):
            heli_time_min = heli_info["total_time_h"] * 60
            comparison_message = (
                f"🏆 PERCORSO DISPONIBILE: ELICOTTERO\n\n"
                f"🚁 Elicottero: {heli_info['total_km']:.2f} km, {heli_time_min:.1f} min\n"
                f"🚗 Auto: Non disponibile"
            )
        
        # Summary
        summary = {
            "best_mode": best_mode,
            "best_time_h": round(best_time, 4),
            "best_time_min": round(best_time * 60, 1),
            "auto_available": bool(auto_info.get("available", False)),
            "heli_available": bool(heli_info.get("available", False))
        }
        
        if auto_info.get("available"):
            summary["auto_time_h"] = round(auto_info["auto_time_h"], 4)
            summary["auto_km"] = round(auto_info["auto_km"], 3)
        
        if heli_info.get("available"):
            summary["heli_time_h"] = round(heli_info["total_time_h"], 4)
            summary["heli_km"] = round(heli_info["total_km"], 3)
        
        # Aggiungi la feature di summary con TUTTI i campi necessari
        features.append({
            "type": "Feature",
            "geometry": None,
            "properties": {
                "is_summary": True,
                "comparison_message": comparison_message,
                "best_mode": summary.get("best_mode", ""),
                "best_time_min": summary.get("best_time_min", 0),
                "auto_available": summary.get("auto_available", False),
                "heli_available": summary.get("heli_available", False),
                "distance_km_total": round(distance_km_total, 2),
                "time_hhmm_total": time_hhmm_total
            }
        })
        
        logger.info(f"✅ Best path completed in {time.perf_counter() - t_total:.3f}s")
        logger.info(f"   📊 Distance: {distance_km_total:.2f} km, Time: {time_hhmm_total}")
        
        return JSONResponse({"type": "FeatureCollection", "features": features})
        
    except Exception as e:
        logger.error(f"❌ Best path error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)