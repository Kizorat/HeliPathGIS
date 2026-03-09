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
from collections import defaultdict
import pickle
from shapely.prepared import prep
from scipy.spatial import KDTree
from collections import deque
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


                                                    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "QGIS_file")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

                      
HELI_SPEED_KMH = 180
AUTO_SPEED_KMH = 100
MAX_HELI_EDGE_DIST = 20000

                    
GRID_STEP = 800                                         
MAX_EDGE_LENGTH = 10000                                                
SAFETY_BUFFER_M = -300                                                                        

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f" Using device: {DEVICE}")


app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

              
transformer_to4326 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
transformer_to32632 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)


                                                 
logger.info("Loading geospatial data...")
t_load = time.perf_counter()

STRADA_FILE = os.path.join(DATA_DIR, "Strade_Lombardia.geojson")
HELI_FILE = os.path.join(DATA_DIR, "Area atterraggio elicottero.gpkg")
FLY_ZONE_FILE = os.path.join(DATA_DIR, "fly_zone_Lombardia.gpkg")                                    




strade = gpd.read_file(STRADA_FILE).to_crs("EPSG:4326")
strade_m = strade.to_crs("EPSG:32632")

heli_areas = gpd.read_file(HELI_FILE).to_crs("EPSG:4326")
heli_areas_m = heli_areas.to_crs("EPSG:32632")

                      
fly_zones = gpd.read_file(FLY_ZONE_FILE).to_crs("EPSG:4326")
fly_zones_m = fly_zones.to_crs("EPSG:32632")



logger.info(f" Data loaded in {time.perf_counter() - t_load:.3f}s")
logger.info(f"   - Roads: {len(strade_m)} features")
logger.info(f"   - Helipads: {len(heli_areas_m)} features")
logger.info(f"   - Fly zones: {len(fly_zones_m)} features (zone di volo permesse)")



logger.info(" Preparing fly zone geometries...")
t_prep = time.perf_counter()


prepared_fly_zones = []
fly_zones_buffered = []

                                               
valid_geoms = []
for idx, geom in enumerate(fly_zones_m.geometry):
    if geom is not None and not geom.is_empty:
        if not geom.is_valid:
            logger.warning(f"    Geometry {idx} is invalid, repairing...")
            geom = make_valid(geom)
        
                                                                       
        try:
            geom = geom.buffer(0)
            valid_geoms.append(geom)
            bounds = geom.bounds
            logger.info(f"   Fly zone {idx}: bounds={bounds}, area={geom.area:.0f} m², valid={geom.is_valid}")
        except Exception as e:
            logger.error(f"    Failed to repair geometry {idx}: {e}")
            continue

if valid_geoms:
                                                         
    try:
        all_fly_zones = unary_union(valid_geoms)
        logger.info(f"    Successfully created union of {len(valid_geoms)} fly zones")
    except Exception as e:
        logger.error(f"    Failed to create union: {e}")
                                                 
        all_fly_zones = valid_geoms[0]
else:
    logger.error(" No valid fly zone geometries found!")
    all_fly_zones = None

if all_fly_zones is not None and not all_fly_zones.is_empty:
    if SAFETY_BUFFER_M < 0:
        all_fly_zones_buffered = all_fly_zones.buffer(SAFETY_BUFFER_M)                   
    else:
        all_fly_zones_buffered = all_fly_zones.buffer(SAFETY_BUFFER_M)
    
    prepared_fly_zones.append(prep(all_fly_zones_buffered))
    fly_zones_buffered.append(all_fly_zones_buffered)
    
    logger.info(f"   Total fly zone area: {all_fly_zones.area:.0f} m²")
    logger.info(f"   Buffered area: {all_fly_zones_buffered.area:.0f} m²")
else:
    logger.error(" Failed to create valid fly zone union!")

logger.info(f" Prepared fly zones in {time.perf_counter() - t_prep:.3f}s")

                  
if len(prepared_fly_zones) > 0:
    logger.info(" Testing fly zone detection...")
                                                       
    test_point = Point(600000, 5030000)                                              
    test_result = any(pg.contains(test_point) for pg in prepared_fly_zones)
    logger.info(f"    Test point at (600000, 5030000): {test_result} (dovrebbe essere True se nelle fly zones)")
else:
    logger.error(" NO FLY ZONES LOADED!")


GRAPH_CACHE = {
    "nodes": None,
    "node_idx": None,
    "edges": None,
    "edge_src": None,
    "edge_dst": None,
    "edge_weight": None,
    "edge_dict": None
}


def point_in_fly_zone(point: Point, prepared_geoms: list) -> bool:
    """Return True if a point is inside any permitted fly zone."""
    return any(pg.contains(point) for pg in prepared_geoms)

def line_in_fly_zone(line: LineString, prepared_geoms: list, raw_geoms: list) -> bool:
    """
    Return True if a line is fully inside permitted fly zones.
    Uses point sampling to verify full line containment.
    """
                                                                                        
    for raw_geom in raw_geoms:
        if raw_geom.contains(line):
            return True
    
                                                                                             
                                             
    if line.length > 500:
        num_samples = max(3, int(line.length / 500))
        
        for i in range(num_samples + 1):
            t = i / num_samples                
            sample_point = line.interpolate(t, normalized=True)
            
                                                             
            inside = False
            for raw_geom in raw_geoms:
                if raw_geom.contains(sample_point):
                    inside = True
                    break
            
            if not inside:
                return False 
        return True                                        
    
                                                     
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



def prepare_graph_gpu(strade_layer, speed_kmh=AUTO_SPEED_KMH):
    """Build road graph once and cache it."""
    if GRAPH_CACHE["nodes"] is not None:
        logger.info(" Using cached road graph")
        return

    logger.info("  Building road graph...")
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
                travel_time = dist / (speed_kmh * 1000 / 3600)                      
                edges.append((p1, p2, travel_time))

    t_nodes = time.perf_counter()
    logger.info(f"     Extracted nodes/edges in {t_nodes - t_start:.3f}s")

                               
    nodes = list({n: n for n in nodes}.values())
    node_idx = {n: i for i, n in enumerate(nodes)}

    t_index = time.perf_counter()
    logger.info(f"    Indexed {len(nodes)} nodes in {t_index - t_nodes:.3f}s")

                                       
    edge_dict = defaultdict(list)
    for s_coord, d_coord, w in edges:
        s = node_idx[s_coord]
        d = node_idx[d_coord]
        edge_dict[s].append((d, w))
        edge_dict[d].append((s, w))

    t_dict = time.perf_counter()
    logger.info(f"     Built adjacency dict in {t_dict - t_index:.3f}s")

                      
    GRAPH_CACHE.update({
        "nodes": nodes,
        "node_idx": node_idx,
        "edge_dict": edge_dict
    })

    logger.info(f" Road graph ready: {len(nodes)} nodes, {len(edges)} edges (total: {t_dict - t_start:.3f}s)")




def compute_auto_path_dijkstra(start_point: Point, end_point: Point):
    """Compute shortest path on road network using Dijkstra."""
    t_start = time.perf_counter()
    
    nodes = GRAPH_CACHE["nodes"]
    node_idx = GRAPH_CACHE["node_idx"]
    edge_dict = GRAPH_CACHE["edge_dict"]

    if nodes is None:
        raise ValueError("Graph not initialized. Call prepare_graph_gpu() first.")

    N = len(nodes)

                        
    t_snap = time.perf_counter()
    start_idx = min(range(N), key=lambda i: Point(nodes[i]).distance(start_point))
    end_idx = min(range(N), key=lambda i: Point(nodes[i]).distance(end_point))
    logger.info(f"    Node snapping: {time.perf_counter() - t_snap:.3f}s")

    if start_idx == end_idx:
        start_4326 = transformer_to4326.transform(start_point.x, start_point.y)
        end_4326 = transformer_to4326.transform(end_point.x, end_point.y)
        return [[list(start_4326), list(end_4326)]]

                        
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

    logger.info(f"     Dijkstra computation: {time.perf_counter() - t_dijkstra:.3f}s")

                      
    if dist[end_idx] == float('inf'):
        logger.warning("  No path found between nodes")
        return []

    path_nodes = []
    u = end_idx
    while u != -1:
        path_nodes.insert(0, nodes[u])
        u = prev[u]

                      
    t_convert = time.perf_counter()
    line_4326 = [list(transformer_to4326.transform(x, y)) for x, y in path_nodes]
    logger.info(f"     Coordinate conversion: {time.perf_counter() - t_convert:.3f}s")
    logger.info(f" Auto path computed in {time.perf_counter() - t_start:.3f}s")

    return [line_4326]



def build_heli_grid(start_p: Point, end_p: Point, step: int = GRID_STEP):
    """Build a grid for helicopter pathfinding, INSIDE fly zones only."""
    t_start = time.perf_counter()
    
                                                   
    if not prepared_fly_zones:
        logger.error(" No valid fly zones available!")
        return None, None, None
    
    min_x, max_x = sorted([start_p.x, end_p.x])
    min_y, max_y = sorted([start_p.y, end_p.y])
    
                                                  
    margin = step * 8                       
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    xs = np.arange(min_x, max_x + step, step)
    ys = np.arange(min_y, max_y + step, step)
    
    logger.info(f"     Grid bounds: X[{min_x:.0f}, {max_x:.0f}], Y[{min_y:.0f}, {max_y:.0f}]")
    logger.info(f"     Grid will have ~{len(xs) * len(ys)} points before filtering")
    
                              
    all_points = [[x, y] for x in xs for y in ys]
    
                                                      
    valid_points = []
    rejected_points = 0
    
    for pt in all_points:
        point = Point(pt[0], pt[1])
        if point_in_fly_zone(point, prepared_fly_zones):
            valid_points.append(pt)
        else:
            rejected_points += 1
    
                                                                              
    border_points_added = 0
    
    for fly_geom in fly_zones_buffered:
                                       
        boundary = fly_geom.boundary
        
        if boundary.length > 0:
                                                    
            num_border_points = max(12, int(boundary.length / 400))             
            for i in range(num_border_points):
                t = i / num_border_points
                border_point = boundary.interpolate(t, normalized=True)
                
                                                            
                                                                   
                centroid = fly_geom.centroid
                dx = centroid.x - border_point.x
                dy = centroid.y - border_point.y
                length = max(0.1, (dx*dx + dy*dy)**0.5)
                
                                                
                offset_x = (dx / length) * 100
                offset_y = (dy / length) * 100
                
                adjusted_point = [border_point.x + offset_x, border_point.y + offset_y]
                
                test_p = Point(adjusted_point[0], adjusted_point[1])
                if point_in_fly_zone(test_p, prepared_fly_zones):
                                                                                
                    if (min_x <= adjusted_point[0] <= max_x and 
                        min_y <= adjusted_point[1] <= max_y):
                        valid_points.append(adjusted_point)
                        border_points_added += 1
        
                                                                  
        bounds = fly_geom.bounds
                                                                
        x_steps = max(2, int((bounds[2] - bounds[0]) / 2000) + 1)
        y_steps = max(2, int((bounds[3] - bounds[1]) / 2000) + 1)
        
        for i in range(x_steps):
            for j in range(y_steps):
                x = bounds[0] + i * 2000
                y = bounds[1] + j * 2000
                point = Point(x, y)
                
                if fly_geom.contains(point):
                                                                   
                    if point_in_fly_zone(point, prepared_fly_zones):
                        valid_points.append([x, y])
                        border_points_added += 1
    
    logger.info(f"     Grid generation: {time.perf_counter() - t_start:.3f}s")
    logger.info(f"    Grid: {len(all_points)} total points, {len(valid_points)-border_points_added} valid (in fly zones), {rejected_points} rejected")
    logger.info(f"    Added {border_points_added} strategic points inside fly zones")
    
    if len(valid_points) == 0:
        logger.error(" No valid grid points - start/end might be outside fly zones!")
                                                                                  
        valid_points.append([start_p.x, start_p.y])
        valid_points.append([end_p.x, end_p.y])
        logger.warning("  Forced start and end points into grid for debugging")
    
                                                                                            
    line = LineString([(start_p.x, start_p.y), (end_p.x, end_p.y)])
    
                                                                    
    if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
        logger.info("    Adding intermediate points along direct line (inside fly zones)")
        num_intermediate = max(2, int(line.length / (step * 1.5)))
        for i in range(1, num_intermediate):
            t = i / num_intermediate
            interp_point = line.interpolate(t, normalized=True)
            valid_points.append([interp_point.x, interp_point.y])
    else:
        logger.info("     Direct line is not entirely inside fly zones, will need detour")
    
                                                                                            
    logger.info(f"    DEBUG: Start point coordinates: ({start_p.x:.1f}, {start_p.y:.1f})")
    logger.info(f"    DEBUG: End point coordinates: ({end_p.x:.1f}, {end_p.y:.1f})")
    
                                                                       
    all_points_with_ids = []
    
                                             
    for pt in valid_points:
        all_points_with_ids.append(('grid', pt[0], pt[1]))
    
                                                             
    all_points_with_ids.append(('start', start_p.x, start_p.y))
    all_points_with_ids.append(('end', end_p.x, end_p.y))
    
                                                                
    unique_points = []
    seen_coords = set()
    start_idx = None
    end_idx = None
    
    for i, (pt_type, x, y) in enumerate(all_points_with_ids):
                                                       
        coord_key = (round(x, 1), round(y, 1))
        
        if coord_key not in seen_coords:
            seen_coords.add(coord_key)
            unique_points.append([x, y])
            
                                                
            if pt_type == 'start':
                start_idx = len(unique_points) - 1
            elif pt_type == 'end':
                end_idx = len(unique_points) - 1
    
                                                                                   
                              
    if start_idx is None:
                                             
        distances = []
        for pt in unique_points:
            dist = ((pt[0] - start_p.x)**2 + (pt[1] - start_p.y)**2)**0.5
            distances.append(dist)
        start_idx = np.argmin(distances)
        logger.warning(f"     Start point was duplicate, using closest point at index {start_idx}")
    
    if end_idx is None:
                                           
        distances = []
        for pt in unique_points:
            dist = ((pt[0] - end_p.x)**2 + (pt[1] - end_p.y)**2)**0.5
            distances.append(dist)
        end_idx = np.argmin(distances)
        logger.warning(f"     End point was duplicate, using closest point at index {end_idx}")
    
                                                                  
    logger.info(f"    Total unique grid points: {len(unique_points)}")
    logger.info(f"    Start point index: {start_idx}, coordinates: ({unique_points[start_idx][0]:.1f}, {unique_points[start_idx][1]:.1f})")
    logger.info(f"    End point index: {end_idx}, coordinates: ({unique_points[end_idx][0]:.1f}, {unique_points[end_idx][1]:.1f})")
    
                                                                                
    start_coord_diff = ((unique_points[start_idx][0] - start_p.x)**2 + 
                       (unique_points[start_idx][1] - start_p.y)**2)**0.5
    end_coord_diff = ((unique_points[end_idx][0] - end_p.x)**2 + 
                     (unique_points[end_idx][1] - end_p.y)**2)**0.5
    
    if start_coord_diff > 1000:                            
        logger.error(f"    Start point mismatch: diff={start_coord_diff:.1f}m")
        logger.error(f"    Original: ({start_p.x:.1f}, {start_p.y:.1f})")
        logger.error(f"    In grid: ({unique_points[start_idx][0]:.1f}, {unique_points[start_idx][1]:.1f})")
    
    if end_coord_diff > 1000:                            
        logger.error(f"    End point mismatch: diff={end_coord_diff:.1f}m")
        logger.error(f"    Original: ({end_p.x:.1f}, {end_p.y:.1f})")
        logger.error(f"    In grid: ({unique_points[end_idx][0]:.1f}, {unique_points[end_idx][1]:.1f})")
    
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
    
    logger.info(f"    Building edges for {N} points (only inside fly zones)...")
    
                                                              
    start_point = pts[start_idx]
    end_point = pts[end_idx]
    
    logger.info(f"    Start point (idx={start_idx}): {start_point}")
    logger.info(f"    End point (idx={end_idx}): {end_point}")
    
                                                      
    start_in_fly = point_in_fly_zone(Point(start_point[0], start_point[1]), prepared_fly_zones)
    end_in_fly = point_in_fly_zone(Point(end_point[0], end_point[1]), prepared_fly_zones)
    
    logger.info(f"    Start point in fly zone: {start_in_fly}")
    logger.info(f"    End point in fly zone: {end_in_fly}")
    
                                                                               
    line = LineString([tuple(start_point), tuple(end_point)])
    direct_dist = np.linalg.norm(start_point - end_point)
    
    if (direct_dist <= max_dist and 
        line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered)):
        logger.info(f"    Direct start-end connection is possible! Distance: {direct_dist:.0f}m")
        edges.append([start_idx, end_idx])
        edges.append([end_idx, start_idx])
        weights.append(direct_dist)
        weights.append(direct_dist)
    else:
        logger.info(f"     Direct start-end connection not possible. Distance: {direct_dist:.0f}m, in fly zones: {line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered)}")
    
                                                       
    logger.info("    Building KDTree for neighbor search...")
    tree = KDTree(pts)
    
                                                 
    k = min(25, N-1)                                               
    distances, indices = tree.query(pts, k=k)
    
    logger.info(f"    Connecting each point to {k} nearest neighbors...")
    
    for i in range(N):
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):                                    
            if j_idx <= i:                                          
                continue
                
            checked += 1
            j = int(j_idx)
            p1, p2 = pts[i], pts[j]
            
                             
            if dist > max_dist:
                rejected_distance += 1
                continue
            
                                                                                    
            line = LineString([tuple(p1), tuple(p2)])
            if not line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                rejected_not_in_fly_zone += 1
                continue
            
                                        
            edges.append([i, j])
            edges.append([j, i])
            weights.append(dist)
            weights.append(dist)
    
                                                                        
    critical_points = [start_idx, end_idx]
    
    for cp in critical_points:
                                                                   
        cp_neighbors = tree.query(pts[cp], k=min(40, N))[1]
        neighbors_added = 0
        for neighbor in cp_neighbors[1:]:                   
            neighbor = int(neighbor)
            if neighbor == cp:
                continue
                
            line = LineString([tuple(pts[cp]), tuple(pts[neighbor])])
            dist = np.linalg.norm(pts[cp] - pts[neighbor])
            
            if (dist <= max_dist * 2.5 and                                        
                line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered)):
                
                                                
                edge_exists = any((e[0] == cp and e[1] == neighbor) for e in edges)
                if not edge_exists:
                    edges.append([cp, neighbor])
                    edges.append([neighbor, cp])
                    weights.append(dist)
                    weights.append(dist)
                    neighbors_added += 1
        
        logger.info(f"    Added {neighbors_added} extra connections to critical point {cp}")
    
    logger.info(f"     Edge building: {time.perf_counter() - t_start:.3f}s")
    logger.info(f"     Edge statistics:")
    logger.info(f"      ├─ Checked: {checked}")
    logger.info(f"      ├─ Rejected (distance > {max_dist}m): {rejected_distance}")
    logger.info(f"      ├─ Rejected (not in fly zone): {rejected_not_in_fly_zone}")
    logger.info(f"      └─ Valid edges: {len(edges)//2}")
    
                                                        
    direct_edge_exists = any((e[0] == start_idx and e[1] == end_idx) for e in edges)
    logger.info(f"    Direct edge between start and end in graph: {direct_edge_exists}")
    
    if not edges:
        logger.warning("  No valid edges found!")
        return None, None
    
                                                       
    adj = defaultdict(list)
    for edge in edges:
        if edge[0] < edge[1]:
            adj[edge[0]].append(edge[1])
            adj[edge[1]].append(edge[0])
    
                                                        
    visited = set()
    queue = deque([start_idx])
    visited.add(start_idx)
    
    while queue:
        node = queue.popleft()
        if node == end_idx:
            logger.info("    Graph connectivity: START and END are connected!")
            break
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    else:
        logger.warning(f"     Graph connectivity: START and END are NOT connected!")
        logger.warning(f"     Start component size: {len(visited)} nodes")
        
                                           
        visited_end = set()
        queue = deque([end_idx])
        visited_end.add(end_idx)
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in visited_end:
                    visited_end.add(neighbor)
                    queue.append(neighbor)
        
        logger.warning(f"     End component size: {len(visited_end)} nodes")
        
                                              
        logger.info("    Attempting to bridge disconnected components...")
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
                    logger.info(f"    Added bridge between components (dist={dist:.0f}m)")
                    break
            
            if bridge_added > 0:
                break
        
        if bridge_added == 0:
            logger.error("    Could not bridge disconnected components!")
    
    return (
        torch.tensor(edges, dtype=torch.long, device=DEVICE),
        torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )

def dijkstra_gpu(points: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor, 
                 start_idx: int, end_idx: int):
    """Dijkstra shortest path on GPU-ready graph using heapq."""
    t_start = time.perf_counter()
    
    if edges is None:
        logger.warning("     No edges available for pathfinding")
        return [], float("inf")
    
    N = points.shape[0]
    
                                                                           
    edges_cpu = edges.cpu().numpy()
    weights_cpu = weights.cpu().numpy()
    
                                   
    adj = defaultdict(list)
    for i in range(len(edges_cpu)):
        u, v = int(edges_cpu[i][0]), int(edges_cpu[i][1])
        w = float(weights_cpu[i])
        adj[u].append((v, w))
    
                                 
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
            logger.info(f"    Path found after {iterations} iterations!")
            break
        
        for v, w in adj[u]:
            alt = d_u + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    
    logger.info(f"     Dijkstra: {time.perf_counter() - t_start:.3f}s ({iterations} iterations)")
    
                      
    if dist[end_idx] == float("inf"):
        logger.warning(f"     No path found: start and end are disconnected in the graph")
        logger.info(f"    Connectivity check:")
        logger.info(f"      - Start node {start_idx} has {len(adj[start_idx])} neighbors")
        logger.info(f"      - End node {end_idx} has {len(adj[end_idx])} neighbors")
        return [], float("inf")
    
    path = []
    u = end_idx
    while u != -1:
        path.insert(0, points[u].cpu().tolist())
        u = prev[u]
    
    total_dist = float(dist[end_idx])
    logger.info(f"    Path found with {len(path)} waypoints, total distance: {total_dist:.0f}m")
    
                                                
    num_segments = len(path) - 1
    logger.info(f"    Path consists of {num_segments} segments")
    
    return path, total_dist



def compute_heli_route(start_point_m: Point, end_point_m: Point):
    """Compute full helicopter route: auto to helipad + flight to hospital."""
    t_total = time.perf_counter()
    logger.info(" === HELICOPTER ROUTE COMPUTATION ===")
    
    try:
                                                
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
        
        logger.info(f"    Helipad found at ({heli_x:.0f}, {heli_y:.0f}) in {time.perf_counter() - t_heli:.3f}s")
        
                                                      
        helipad_in_fly = point_in_fly_zone(helipad_point_m, prepared_fly_zones)
        logger.info(f"    Helipad in fly zone: {helipad_in_fly}")
        
                                      
        logger.info(" Computing auto path to helipad...")
        t_auto = time.perf_counter()
        
        snapped_start = snap_to_road(start_point_m, list(strade_m.geometry))
        multilines_auto = compute_auto_path_dijkstra(snapped_start, helipad_point_m)
        
        if not multilines_auto:
            return {"success": False, "error": "No auto path to helipad"}
        
                                 
        auto_dist_m = 0.0
        for seg in multilines_auto:
            for i in range(len(seg) - 1):
                x1, y1 = transformer_to32632.transform(seg[i][0], seg[i][1])
                x2, y2 = transformer_to32632.transform(seg[i+1][0], seg[i+1][1])
                auto_dist_m += Point(x1, y1).distance(Point(x2, y2))
        
        auto_km = auto_dist_m / 1000.0
        auto_time_h = auto_km / AUTO_SPEED_KMH
        
        logger.info(f"    Auto path: {auto_km:.2f} km, {auto_time_h*60:.1f} min ({time.perf_counter() - t_auto:.3f}s)")
        
                                        
        logger.info("  Computing helicopter flight path...")
        t_flight = time.perf_counter()
        
        grid, start_idx, end_idx = build_heli_grid(helipad_point_m, end_point_m)
        if grid is None:
            return {"success": False, "error": "Cannot build grid - check fly zones"}
        
        edges, weights = build_heli_edges(grid, start_idx, end_idx)
        
        if edges is None:
                                                              
            logger.warning("  No valid flight path found, checking direct line...")
            line = LineString([(heli_x, heli_y), (end_point_m.x, end_point_m.y)])
            if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                heli_path_m = [[heli_x, heli_y], [end_point_m.x, end_point_m.y]]
                heli_dist_m = helipad_point_m.distance(end_point_m)
                logger.info("    Direct line is in fly zones, using it")
            else:
                logger.error(" Direct line is not in fly zones and no alternative path found")
                return {"success": False, "error": "No flight path within fly zones"}
        else:
                                                                
            heli_path_m, heli_dist_m = dijkstra_gpu(grid, edges, weights, start_idx, end_idx)
            
            if not heli_path_m:
                logger.warning("  Dijkstra failed, checking direct line...")
                line = LineString([(heli_x, heli_y), (end_point_m.x, end_point_m.y)])
                if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                    heli_path_m = [[heli_x, heli_y], [end_point_m.x, end_point_m.y]]
                    heli_dist_m = helipad_point_m.distance(end_point_m)
                    logger.info("    Direct line is in fly zones, using it")
                else:
                    logger.error(" No valid flight path found")
                    return {"success": False, "error": "No flight path within fly zones"}
        
        heli_km = heli_dist_m / 1000.0
        heli_time_h = heli_km / HELI_SPEED_KMH
        
        logger.info(f"    Flight path: {heli_km:.2f} km, {heli_time_h*60:.1f} min ({time.perf_counter() - t_flight:.3f}s)")
        
                                  
        path_heli_4326 = [[transformer_to4326.transform(x, y) for x, y in heli_path_m]]
        helipad_lon, helipad_lat = transformer_to4326.transform(heli_x, heli_y)
        
        total_km = auto_km + heli_km
        total_time_h = auto_time_h + heli_time_h
        
        logger.info(f" Total helicopter route: {total_km:.2f} km, {total_time_h*60:.1f} min (computed in {time.perf_counter() - t_total:.3f}s)")
        
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
        logger.error(f" Helicopter route error: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}




logger.info(" Initializing road network graph...")
prepare_graph_gpu(strade_m)



@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})




@app.get("/geo")
def get_geo(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)



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
                                                   
        if "Palazzi" in filename and bbox:
            logger.info(f" Loading buildings with bbox: {bbox}")
            
                        
            try:
                min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
            except ValueError:
                return JSONResponse({"error": "Invalid bbox format. Use: min_lon,min_lat,max_lon,max_lat"}, status_code=400)
            
                                           
            gdf = gpd.read_file(file_path, bbox=(min_lon, min_lat, max_lon, max_lat))
            
            if gdf.empty:
                logger.info(f" No buildings found in bbox: {bbox}")
                return JSONResponse({"type": "FeatureCollection", "features": []})
            
                             
            if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
                                                                  
            if simplify and simplify > 0:
                logger.info(f" Simplifying geometries with tolerance: {simplify}m")
                                                                   
                gdf_metric = gdf.to_crs("EPSG:32632")
                gdf_metric['geometry'] = gdf_metric['geometry'].simplify(tolerance=float(simplify))
                gdf = gdf_metric.to_crs("EPSG:4326")
            
                                       
            MAX_BUILDINGS = 50000
            if len(gdf) > MAX_BUILDINGS:
                logger.warning(f"Too many buildings ({len(gdf)}), limiting to {MAX_BUILDINGS}")
                gdf = gdf.head(MAX_BUILDINGS)
            
            logger.info(f" Loaded {len(gdf)} buildings for bbox {bbox}")
            
            return JSONResponse(gdf.__geo_interface__)
        
                                          
        else:
            gdf = gpd.read_file(file_path)
            gdf = gdf.to_crs("EPSG:4326")
            return JSONResponse(gdf.__geo_interface__)
        
    except Exception as e:
        logger.error(f" Error loading {filename}: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)




@app.get("/auto-path")
def auto_path(
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    end_lat: float = Query(...),
    end_lon: float = Query(...)
):
    logger.info(f" AUTO PATH REQUEST: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    
    try:
        t_total = time.perf_counter()
        
                               
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        end_x, end_y = transformer_to32632.transform(end_lon, end_lat)
        start_point = Point(start_x, start_y)
        end_point = Point(end_x, end_y)
        
                      
        multilines = compute_auto_path_dijkstra(start_point, end_point)
        
        if not multilines:
            return JSONResponse({"error": "No path found"}, status_code=404)
        
                                  
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
        
        logger.info(f" Auto path completed in {time.perf_counter() - t_total:.3f}s: {total_dist_km:.2f} km, {hours}h{minutes}m")
        
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
        logger.error(f" Auto path error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/heli-path")
def heli_path(
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    end_lat: float = Query(...),
    end_lon: float = Query(...)
):
    logger.info(f" HELI PATH REQUEST: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    
    try:
        t_total = time.perf_counter()
        
                             
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        end_x, end_y = transformer_to32632.transform(end_lon, end_lat)
        start_m = Point(start_x, start_y)
        end_m = Point(end_x, end_y)
        
                                  
        result = compute_heli_route(start_m, end_m)
        
        if not result["success"]:
            return JSONResponse({"error": result.get("error", "Unknown error")}, status_code=404)
        
                                
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
        
        logger.info(f" Heli path completed in {time.perf_counter() - t_total:.3f}s")
        return JSONResponse(geojson)
        
    except Exception as e:
        logger.error(f" Heli path error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)




@app.get("/best-path")
def best_path(
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    end_lat: float = Query(...),
    end_lon: float = Query(...)
):
    logger.info(f" BEST PATH REQUEST: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    
    try:
        t_total = time.perf_counter()
        
                             
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        end_x, end_y = transformer_to32632.transform(end_lon, end_lat)
        start_m = Point(start_x, start_y)
        end_m = Point(end_x, end_y)
        
                      
        snapped_start = snap_to_road(start_m, list(strade_m.geometry))
        
                                             
        logger.info(" Evaluating direct auto route...")
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
            logger.info(f"    Direct auto: {auto_km:.2f} km, {auto_time_h*60:.1f} min ({time.perf_counter() - t_auto:.3f}s)")
        else:
            logger.warning(f"     Direct auto route not found ({time.perf_counter() - t_auto:.3f}s)")
        
                                            
        logger.info(" Evaluating helicopter route...")
        t_heli = time.perf_counter()
        
        heli_info = compute_heli_route(snapped_start, end_m)
        heli_info["available"] = heli_info.get("success", False)
        
        if heli_info["available"]:
            logger.info(f"    Helicopter: {heli_info['total_km']:.2f} km, {heli_info['total_time_h']*60:.1f} min ({time.perf_counter() - t_heli:.3f}s)")
        else:
            logger.warning(f"     Helicopter route not found: {heli_info.get('error', 'Unknown')} ({time.perf_counter() - t_heli:.3f}s)")
        
                                    
        candidates = []
        if auto_info.get("available"):
            candidates.append(("auto", auto_info["auto_time_h"]))
        if heli_info.get("available"):
            candidates.append(("heli", heli_info["total_time_h"]))
        
        if not candidates:
            return JSONResponse({"error": "No route available (neither auto nor helicopter)"}, status_code=404)
        
        best_mode, best_time = min(candidates, key=lambda x: x[1])
        logger.info(f" Best option: {best_mode.upper()} ({best_time*60:.1f} min)")
        
        
        features = []
        
                                     
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
        
                                           
        if heli_info.get("available"):
                                     
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
        
                                          
        distance_km_total = 0
        time_hhmm_total = "00:00"
        
        if best_mode == "auto" and auto_info.get("available"):
            distance_km_total = auto_info["auto_km"]
            time_h = auto_info["auto_time_h"]
        elif best_mode == "heli" and heli_info.get("available"):
            distance_km_total = heli_info["total_km"]
            time_h = heli_info["total_time_h"]
        
                                         
        hours = int(time_h)
        minutes = int((time_h - hours) * 60)
        time_hhmm_total = f"{hours:02d}:{minutes:02d}"
        
                                        
        comparison_message = ""
        if auto_info.get("available") and heli_info.get("available"):
            auto_time_min = auto_info["auto_time_h"] * 60
            heli_time_min = heli_info["total_time_h"] * 60
            time_diff = abs(auto_time_min - heli_time_min)
            
            if best_mode == "auto":
                comparison_message = (
                    f" PERCORSO PIÙ VELOCE: AUTO\n\n"
                    f" Auto: {auto_info['auto_km']:.2f} km, {auto_time_min:.1f} min\n"
                    f" Elicottero: {heli_info['total_km']:.2f} km, {heli_time_min:.1f} min\n\n"
                    f"Risparmio di tempo: {time_diff:.1f} min"
                )
            else:
                comparison_message = (
                    f" PERCORSO PIÙ VELOCE: ELICOTTERO\n\n"
                    f" Elicottero: {heli_info['total_km']:.2f} km, {heli_time_min:.1f} min\n"
                    f" Auto: {auto_info['auto_km']:.2f} km, {auto_time_min:.1f} min\n\n"
                    f"Risparmio di tempo: {time_diff:.1f} min"
                )
        elif auto_info.get("available"):
            auto_time_min = auto_info["auto_time_h"] * 60
            comparison_message = (
                f" PERCORSO DISPONIBILE: AUTO\n\n"
                f" Auto: {auto_info['auto_km']:.2f} km, {auto_time_min:.1f} min\n"
                f" Elicottero: Non disponibile"
            )
        elif heli_info.get("available"):
            heli_time_min = heli_info["total_time_h"] * 60
            comparison_message = (
                f" PERCORSO DISPONIBILE: ELICOTTERO\n\n"
                f" Elicottero: {heli_info['total_km']:.2f} km, {heli_time_min:.1f} min\n"
                f" Auto: Non disponibile"
            )
        
                 
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
        
        logger.info(f" Best path completed in {time.perf_counter() - t_total:.3f}s")
        logger.info(f" Distance: {distance_km_total:.2f} km, Time: {time_hhmm_total}")
        
        return JSONResponse({"type": "FeatureCollection", "features": features})
        
    except Exception as e:
        logger.error(f" Best path error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)



HOSPITAL_CACHE = {
    "points": None,
    "kdtree": None,
    "node_indices": None,
    "hospital_data": None
}



def precompute_hospital_data():
    """Pre-compute hospital data for faster access."""
    logger.info(" Pre-computing hospital data...")
    
 
    HOSPITAL_FILE = os.path.join(DATA_DIR, "Centroidi Ospedali.gpkg")
    
    if not os.path.exists(HOSPITAL_FILE):
        logger.error(f" Hospital file not found: {HOSPITAL_FILE}")
                                                         
        hospital_points_m = [
            Point(548000, 5030000),
            Point(600000, 5045000),
            Point(580000, 5025000),
            Point(550000, 5050000),
            Point(560000, 5015000),
        ]
        hospital_data = [{"id": i, "name": f"Hospital_{i}"} for i in range(len(hospital_points_m))]
        logger.info(f" Using fallback: {len(hospital_points_m)} test hospitals")
    else:
                                       
        ospedali = gpd.read_file(HOSPITAL_FILE).to_crs("EPSG:4326")
        
                                                             
        total_hospitals_before = len(ospedali)
        
                                                      
        logger.info(f" Available columns in hospital data: {list(ospedali.columns)}")
        
                                    
        nome_column = None
        possible_nome_columns = ['nome', 'Nome', 'NOME', 'name', 'Name', 'NAME', 'denominazione', 'DENOMINAZIONE']
        
        for col in possible_nome_columns:
            if col in ospedali.columns:
                nome_column = col
                break
        
        if nome_column:
            logger.info(f" Found name column: '{nome_column}'")
            
                                               
            ospedali_non_null = ospedali[ospedali[nome_column].notna()]
            count_with_name = len(ospedali_non_null)
            
            logger.info(f" Hospitals with non-null name: {count_with_name}/{total_hospitals_before}")
            
                                                   
            if count_with_name > 0:
                ospedali = ospedali_non_null.copy()
                logger.info(f" Filtered to {len(ospedali)} hospitals with non-null name")
            else:
                logger.warning(" No hospitals with non-null name found, using all hospitals")
        else:
            logger.warning(" No name column found, using all hospitals")
        
                                 
        ospedali_m = ospedali.to_crs("EPSG:32632")
        hospital_points_m = [geom for geom in ospedali_m.geometry]
        
                                    
        hospital_data = []
        for i, (idx, row) in enumerate(ospedali.iterrows()):
            hospital_info = {"id": i}
            
                                             
            if nome_column and nome_column in row:
                hospital_info["name"] = row[nome_column]
            else:
                                                                      
                for col in ospedali.columns:
                    if any(keyword in col.lower() for keyword in ['nome', 'name', 'denom']):
                        if pd.notna(row[col]):
                            hospital_info["name"] = row[col]
                            break
                
                if "name" not in hospital_info:
                    hospital_info["name"] = f"Hospital_{i}"
            
                                                              
            for col in ospedali.columns:
                if col not in ['geometry', nome_column] and pd.notna(row[col]):
                    hospital_info[col] = row[col]
            
            hospital_data.append(hospital_info)
        
        logger.info(f" Loaded {len(hospital_points_m)} hospitals from file (filtered to those with non-null name)")
    
                                                     
    logger.info(" Pre-computing nearest road nodes for hospitals...")
    nodes = GRAPH_CACHE["nodes"]
    node_idx = GRAPH_CACHE["node_idx"]
    
    if nodes is not None:
        hospital_node_indices = []
        
                                                   
        node_points = np.array([(n[0], n[1]) for n in nodes])
        node_tree = KDTree(node_points)
        
        processed_count = 0
        total_hospitals = len(hospital_points_m)
        
        for i, hospital_point in enumerate(hospital_points_m):
                                       
            hospital_coords = (hospital_point.x, hospital_point.y)
            distances, indices = node_tree.query([hospital_coords], k=min(5, len(node_points)))
            
                                  
            hospital_node_indices.append(int(indices[0][0]))
            
            processed_count += 1
            
            if processed_count % 100 == 0 or processed_count == total_hospitals:
                logger.info(f"   Processed {processed_count}/{total_hospitals} hospitals")
        
                                      
        hospital_coords_array = np.array([(p.x, p.y) for p in hospital_points_m])
        hospital_tree = KDTree(hospital_coords_array)
        
                        
        HOSPITAL_CACHE.update({
            "points": hospital_points_m,
            "kdtree": hospital_tree,
            "node_indices": hospital_node_indices,
            "hospital_data": hospital_data
        })
        
        logger.info(f" Pre-computed hospital data for {len(hospital_points_m)} hospitals")
        
                         
        logger.info(" HOSPITAL DATA STATISTICS:")
        logger.info(f"   ├─ Total hospitals loaded: {len(hospital_points_m)}")
        
        if nome_column and 'hospital_data' in HOSPITAL_CACHE:
            names_with_count = {}
            for hospital in HOSPITAL_CACHE["hospital_data"]:
                name = hospital.get("name", "Unknown")
                names_with_count[name] = names_with_count.get(name, 0) + 1
            
            logger.info(f"   ├─ Unique hospital names: {len(names_with_count)}")
            
                                                
            sample_names = list(names_with_count.keys())[:5]
            logger.info(f"   ├─ Sample hospital names: {sample_names}")
            
                                             
        if hospital_points_m:
            x_coords = [p.x for p in hospital_points_m]
            y_coords = [p.y for p in hospital_points_m]
            logger.info(f"   ├─ Bounding box: X({min(x_coords):.0f}, {max(x_coords):.0f}), Y({min(y_coords):.0f}, {max(y_coords):.0f})")
            logger.info(f"   └─ Spatial coverage: {len(hospital_points_m)} points")
            
    else:
        logger.error(" Road graph not initialized!")




def precompute_helipad_data():
    """Pre-compute helipad data for faster access."""
    logger.info(" Pre-computing helipad data...")
    
    helipads_m = [(geom.x, geom.y) for geom in heli_areas_m.geometry]
    
    if not helipads_m:
        logger.error(" No helipads available")
        return
    
                                                    
    nodes = GRAPH_CACHE["nodes"]
    
    if nodes is not None:
        helipad_node_indices = []
        
                                                   
        node_points = np.array([(n[0], n[1]) for n in nodes])
        node_tree = KDTree(node_points)
        
        for i, (helipad_x, helipad_y) in enumerate(helipads_m):
                                      
            helipad_coords = (helipad_x, helipad_y)
            distances, indices = node_tree.query([helipad_coords], k=1)
            helipad_node_indices.append(int(indices[0]))
            
            if i % 20 == 0:
                logger.info(f"   Processed {i}/{len(helipads_m)} helipads")
        
                        
        HOSPITAL_CACHE["helipad_points"] = helipads_m
        HOSPITAL_CACHE["helipad_node_indices"] = helipad_node_indices
        HOSPITAL_CACHE["helipad_tree"] = KDTree(np.array(helipads_m))
        
        logger.info(f" Pre-computed helipad data for {len(helipads_m)} helipads")
    else:
        logger.error(" Road graph not initialized!")



logger.info(" Initializing pre-computed data...")
precompute_hospital_data()
precompute_helipad_data()



def dijkstra_single_source(start_node_idx, max_distance=None, max_time=None):
    """
    Run single-source Dijkstra from the start node
    and return distances to all nodes.
    """
    t_start = time.perf_counter()
    
    nodes = GRAPH_CACHE["nodes"]
    edge_dict = GRAPH_CACHE["edge_dict"]
    N = len(nodes)
    
    dist = [float('inf')] * N
    prev = [-1] * N
    dist[start_node_idx] = 0
    
    heap = [(0, start_node_idx)]
    nodes_processed = 0
    max_nodes_to_process = 50000                                                    
    
    while heap and nodes_processed < max_nodes_to_process:
        d_u, u = heapq.heappop(heap)
        
                                                          
        if max_distance is not None and d_u > max_distance:
            break
            
        nodes_processed += 1
        
        for v, w in edge_dict[u]:
            alt = d_u + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    
    logger.info(f"     Single-source Dijkstra processed {nodes_processed}/{N} nodes in {time.perf_counter() - t_start:.3f}s")
    return dist, prev


def find_nearest_hospital_optimized(start_point_m, max_road_distance_km=100):
    """
    Find the nearest hospital using single-source Dijkstra.
    This is much more efficient than the naive approach.
    """
    t_total = time.perf_counter()
    
                                                               
    nodes = GRAPH_CACHE["nodes"]
    start_idx = min(range(len(nodes)), key=lambda i: Point(nodes[i]).distance(start_point_m))
    
    logger.info(f" Start snapped to node {start_idx}")
    
                                                             
    max_distance_seconds = (max_road_distance_km * 1000) / (AUTO_SPEED_KMH * 1000 / 3600)
    dist, prev = dijkstra_single_source(start_idx, max_distance=max_distance_seconds)
    
                                             
    hospital_node_indices = HOSPITAL_CACHE["node_indices"]
    hospital_points = HOSPITAL_CACHE["points"]
    hospital_data = HOSPITAL_CACHE["hospital_data"]
    
    best_hospital_idx = None
    best_distance = float('inf')
    best_hospital_node_idx = None
    
                                        
    reachable_hospitals = []
    for i, node_idx in enumerate(hospital_node_indices):
        if node_idx < len(dist) and dist[node_idx] < float('inf'):
            reachable_hospitals.append((i, node_idx, dist[node_idx]))
    
    logger.info(f"    {len(reachable_hospitals)}/{len(hospital_node_indices)} hospitals are reachable")
    
    if not reachable_hospitals:
        return None, None, None, None
    
                         
    best_hospital_idx, best_hospital_node_idx, best_distance = min(
        reachable_hospitals, key=lambda x: x[2]
    )
    
                                 
    path_nodes = []
    u = best_hospital_node_idx
    while u != -1:
        path_nodes.insert(0, nodes[u])
        u = prev[u]
    
                      
    path_4326 = [list(transformer_to4326.transform(x, y)) for x, y in path_nodes]
    
                            
    total_dist_m = 0.0
    for i in range(len(path_nodes) - 1):
        total_dist_m += Point(path_nodes[i]).distance(Point(path_nodes[i+1]))
    
    total_dist_km = total_dist_m / 1000
    total_time_h = best_distance / 3600                           
    
    logger.info(f"Found nearest hospital #{best_hospital_idx} in {time.perf_counter() - t_total:.3f}s")
    logger.info(f"    Distance: {total_dist_km:.2f} km, Time: {total_time_h*60:.1f} min")
    
    return {
        "hospital_idx": best_hospital_idx,
        "hospital_point": hospital_points[best_hospital_idx],
        "hospital_name": hospital_data[best_hospital_idx]["name"],
        "distance_km": total_dist_km,
        "time_h": total_time_h,
        "path_4326": [path_4326]
    }


def find_nearest_helipad_optimized(start_point_m, max_road_distance_km=50):
    """
    Find the nearest helipad using single-source Dijkstra.
    """
    t_total = time.perf_counter()
    
                                                               
    nodes = GRAPH_CACHE["nodes"]
    start_idx = min(range(len(nodes)), key=lambda i: Point(nodes[i]).distance(start_point_m))
    
                                                             
    max_distance_seconds = (max_road_distance_km * 1000) / (AUTO_SPEED_KMH * 1000 / 3600)
    dist, prev = dijkstra_single_source(start_idx, max_distance=max_distance_seconds)
    
                                             
    helipad_node_indices = HOSPITAL_CACHE.get("helipad_node_indices", [])
    helipad_points = HOSPITAL_CACHE.get("helipad_points", [])
    
    if not helipad_node_indices:
        return None
    
    best_helipad_idx = None
    best_distance = float('inf')
    best_helipad_node_idx = None
    
                                        
    reachable_helipads = []
    for i, node_idx in enumerate(helipad_node_indices):
        if node_idx < len(dist) and dist[node_idx] < float('inf'):
            reachable_helipads.append((i, node_idx, dist[node_idx]))
    
    logger.info(f"    {len(reachable_helipads)}/{len(helipad_node_indices)} helipads are reachable")
    
    if not reachable_helipads:
        return None
    
                         
    best_helipad_idx, best_helipad_node_idx, best_distance = min(
        reachable_helipads, key=lambda x: x[2]
    )
    
                                 
    path_nodes = []
    u = best_helipad_node_idx
    while u != -1:
        path_nodes.insert(0, nodes[u])
        u = prev[u]
    
                      
    path_4326 = [list(transformer_to4326.transform(x, y)) for x, y in path_nodes]
    
                            
    total_dist_m = 0.0
    for i in range(len(path_nodes) - 1):
        total_dist_m += Point(path_nodes[i]).distance(Point(path_nodes[i+1]))
    
    total_dist_km = total_dist_m / 1000
    total_time_h = best_distance / 3600                           
    
    helipad_x, helipad_y = helipad_points[best_helipad_idx]
    
    logger.info(f"Found nearest helipad #{best_helipad_idx} in {time.perf_counter() - t_total:.3f}s")
    logger.info(f"    Distance: {total_dist_km:.2f} km, Time: {total_time_h*60:.1f} min")
    
    return {
        "helipad_idx": best_helipad_idx,
        "helipad_point": Point(helipad_x, helipad_y),
        "distance_km": total_dist_km,
        "time_h": total_time_h,
        "path_4326": [path_4326]
    }


def find_nearest_hospital_from_helipad(helipad_point_m, max_flight_distance_km=200):
    """
    Find the nearest hospital from a helipad by helicopter.
    Optimized using a two-phase strategy.
    """
    t_total = time.perf_counter()
    
    hospital_points = HOSPITAL_CACHE["points"]
    hospital_data = HOSPITAL_CACHE["hospital_data"]
    
                                                              
    logger.info(" Phase 1: Euclidean distance filtering...")
    
                                    
    hospital_tree = HOSPITAL_CACHE["kdtree"]
    
                                                   
    k = min(50, len(hospital_points))                                  
    distances_euclidean, indices = hospital_tree.query(
        [[helipad_point_m.x, helipad_point_m.y]], 
        k=k
    )
    
                                       
    candidate_hospitals = []
    for idx, dist_euclidean in zip(indices[0], distances_euclidean[0]):
        hospital_point = hospital_points[idx]
        
                                                   
        if not point_in_fly_zone(hospital_point, prepared_fly_zones):
            continue
        
                                                   
        if not point_in_fly_zone(helipad_point_m, prepared_fly_zones):
            continue
        
                                           
        dist_km = dist_euclidean / 1000
        
                                    
        if dist_km > max_flight_distance_km:
            continue
        
        candidate_hospitals.append((idx, hospital_point, dist_km))
    
    logger.info(f"   {len(candidate_hospitals)} candidate hospitals after filtering")
    
    if not candidate_hospitals:
        return None
    
                                                           
    logger.info(" Phase 2: Helicopter path calculation for candidates...")
    
    best_hospital = None
    best_flight_path = None
    best_flight_distance = float('inf')
    best_flight_time = float('inf')
    
                                                     
    candidate_hospitals.sort(key=lambda x: x[2])
    
                                        
    max_candidates = min(10, len(candidate_hospitals))
    
    for i, (hospital_idx, hospital_point, euclidean_dist) in enumerate(candidate_hospitals[:max_candidates]):
        try:
                                         
            grid, start_idx, end_idx = build_heli_grid(helipad_point_m, hospital_point)
            
            if grid is None:
                continue
            
            edges, weights = build_heli_edges(grid, start_idx, end_idx)
            
            if edges is None:
                                         
                line = LineString([(helipad_point_m.x, helipad_point_m.y), 
                                  (hospital_point.x, hospital_point.y)])
                if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                    flight_path_m = [[helipad_point_m.x, helipad_point_m.y], 
                                   [hospital_point.x, hospital_point.y]]
                    flight_distance = helipad_point_m.distance(hospital_point)
                    flight_time_h = (flight_distance / 1000) / HELI_SPEED_KMH
                    
                    if flight_time_h < best_flight_time:
                        best_hospital = (hospital_idx, hospital_point, hospital_data[hospital_idx]["name"])
                        best_flight_path = flight_path_m
                        best_flight_distance = flight_distance
                        best_flight_time = flight_time_h
                continue
            
                      
            flight_path_m, flight_distance = dijkstra_gpu(grid, edges, weights, start_idx, end_idx)
            
            if not flight_path_m:
                continue
            
            flight_time_h = (flight_distance / 1000) / HELI_SPEED_KMH
            
            if flight_time_h < best_flight_time:
                best_hospital = (hospital_idx, hospital_point, hospital_data[hospital_idx]["name"])
                best_flight_path = flight_path_m
                best_flight_distance = flight_distance
                best_flight_time = flight_time_h
                
                                             
                if flight_time_h < 0.25:                     
                    logger.info(f"    Early exit: found hospital within 15 minutes")
                    break
                    
        except Exception as e:
            logger.warning(f" Error calculating flight path to hospital {hospital_idx}: {e}")
            continue
    
    if best_hospital is None:
        logger.warning(" No reachable hospital found from helipad")
        return None
    
    hospital_idx, hospital_point, hospital_name = best_hospital
    
                               
    flight_path_4326 = [[transformer_to4326.transform(x, y) for x, y in best_flight_path]]
    
    logger.info(f"Found nearest hospital from helipad in {time.perf_counter() - t_total:.3f}s")
    logger.info(f"    Hospital: {hospital_name}, Flight distance: {best_flight_distance/1000:.2f} km, Time: {best_flight_time*60:.1f} min")
    
    return {
        "hospital_idx": hospital_idx,
        "hospital_point": hospital_point,
        "hospital_name": hospital_name,
        "distance_km": best_flight_distance / 1000,
        "time_h": best_flight_time,
        "path_4326": flight_path_4326
    }


@app.get("/emergency-route")
def emergency_route_optimized(
    start_lat: float = Query(...),
    start_lon: float = Query(...)
):
    """
    Compute the fastest route to a hospital from any start point.
    Optimized version using single-source Dijkstra and caching.
    """
    logger.info(f" OPTIMIZED EMERGENCY ROUTE REQUEST: ({start_lat}, {start_lon})")
    
    try:
        t_total = time.perf_counter()
        
                                         
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        start_point_m = Point(start_x, start_y)
        
                              
        snapped_start = snap_to_road(start_point_m, list(strade_m.geometry))
        logger.info(f" Start point snapped to road: ({snapped_start.x:.0f}, {snapped_start.y:.0f})")
        

        logger.info(" Finding nearest hospital by road (optimized)...")
        auto_result = find_nearest_hospital_optimized(snapped_start, max_road_distance_km=150)
        
        if auto_result is None:
            logger.error(" No reachable hospital found by road")
            return JSONResponse({"error": "No hospital reachable by road"}, status_code=404)


        logger.info(" Finding nearest helipad by road (optimized)...")
        helipad_result = find_nearest_helipad_optimized(snapped_start, max_road_distance_km=50)
        
        helicopter_available = False
        helicopter_result = None
        
        if helipad_result is not None:

            logger.info("Finding nearest hospital from helipad by helicopter...")
            helicopter_result = find_nearest_hospital_from_helipad(
                helipad_result["helipad_point"], 
                max_flight_distance_km=200
            )
            
            if helicopter_result is not None:
                helicopter_available = True
        

        logger.info("Comparing routes...")
        
                                                       
        auto_total_time_h = auto_result["time_h"]
        
                                               
        if helicopter_available:
            heli_total_time_h = helipad_result["time_h"] + helicopter_result["time_h"]
            heli_total_distance = helipad_result["distance_km"] + helicopter_result["distance_km"]
        else:
            heli_total_time_h = float('inf')
            heli_total_distance = 0
        
                                      
        if auto_total_time_h < heli_total_time_h:
            best_mode = "auto"
            best_time_h = auto_total_time_h
            best_distance = auto_result["distance_km"]
        else:
            best_mode = "heli"
            best_time_h = heli_total_time_h
            best_distance = heli_total_distance
        

        features = []
        
                                                     
        h_auto = int(auto_total_time_h)
        m_auto = int((auto_total_time_h - h_auto) * 60)
        
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "MultiLineString",
                "coordinates": auto_result["path_4326"]
            },
            "properties": {
                "mode": "direct_auto",
                "distance_km": round(auto_result["distance_km"], 2),
                "time_hhmm": f"{h_auto:02d}:{m_auto:02d}",
                "is_best": best_mode == "auto",
                "description": f"Direct road route to {auto_result.get('hospital_name', 'nearest hospital')}"
            }
        })
        
                                                     
        if helicopter_available:
                             
            h_auto_to_helipad = int(helipad_result["time_h"])
            m_auto_to_helipad = int((helipad_result["time_h"] - h_auto_to_helipad) * 60)
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": helipad_result["path_4326"]
                },
                "properties": {
                    "mode": "auto_to_helipad",
                    "distance_km": round(helipad_result["distance_km"], 2),
                    "time_hhmm": f"{h_auto_to_helipad:02d}:{m_auto_to_helipad:02d}",
                    "is_best": best_mode == "heli",
                    "description": "Road to nearest helipad"
                }
            })
            
                         
            h_flight = int(helicopter_result["time_h"])
            m_flight = int((helicopter_result["time_h"] - h_flight) * 60)
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": helicopter_result["path_4326"]
                },
                "properties": {
                    "mode": "heli_flight",
                    "distance_km": round(helicopter_result["distance_km"], 2),
                    "time_hhmm": f"{h_flight:02d}:{m_flight:02d}",
                    "is_best": best_mode == "heli",
                    "description": f"Helicopter flight to {helicopter_result.get('hospital_name', 'nearest hospital')}"
                }
            })
            
                                      
            helipad_lon, helipad_lat = transformer_to4326.transform(
                helipad_result["helipad_point"].x, 
                helipad_result["helipad_point"].y
            )
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [helipad_lon, helipad_lat]
                },
                "properties": {
                    "type": "helipad",
                    "description": f"Nearest helipad (road time: {h_auto_to_helipad:02d}:{m_auto_to_helipad:02d})"
                }
            })
            
                                                              
            hospital_heli_lon, hospital_heli_lat = transformer_to4326.transform(
                helicopter_result["hospital_point"].x,
                helicopter_result["hospital_point"].y
            )
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [hospital_heli_lon, hospital_heli_lat]
                },
                "properties": {
                    "type": "hospital",
                    "route_type": "helicopter",
                    "name": helicopter_result.get("hospital_name", "Hospital"),
                    "description": f"Nearest hospital from helipad (flight time: {int(helicopter_result['time_h']*60):.1f} min)"
                }
            })
        
                                                    
        hospital_auto_lon, hospital_auto_lat = transformer_to4326.transform(
            auto_result["hospital_point"].x,
            auto_result["hospital_point"].y
        )
        
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [hospital_auto_lon, hospital_auto_lat]
            },
            "properties": {
                "type": "hospital",
                "route_type": "direct_auto",
                "name": auto_result.get("hospital_name", "Hospital"),
                "description": f"Nearest hospital by road (time: {int(auto_result['time_h']*60):.1f} min)"
            }
        })
        
                          
        best_hours = int(best_time_h)
        best_minutes = int((best_time_h - best_hours) * 60)
        
        summary_message = ""
        if best_mode == "auto":
            summary_message = (
                f" PERCORSO CONSIGLIATO: AUTO\n\n"
                f" Direttamente all'ospedale: {auto_result['distance_km']:.2f} km, {int(auto_result['time_h']*60):.1f} min\n"
            )
            
            if helicopter_available:
                time_diff = heli_total_time_h * 60 - auto_result['time_h'] * 60
                summary_message += f" Con elicottero: {heli_total_distance:.2f} km, {heli_total_time_h*60:.1f} min\n"
                summary_message += f"  L'auto è più veloce di {abs(time_diff):.1f} min"
            else:
                summary_message += f" Percorso elicottero non disponibile"
        else:
            time_diff = auto_result['time_h'] * 60 - heli_total_time_h * 60
            summary_message = (
                f" PERCORSO CONSIGLIATO: ELICOTTERO\n\n"
                f" Auto + elicottero: {heli_total_distance:.2f} km, {heli_total_time_h*60:.1f} min\n"
                f" Solo auto: {auto_result['distance_km']:.2f} km, {auto_result['time_h']*60:.1f} min\n"
                f" L'elicottero è più veloce di {abs(time_diff):.1f} min"
            )
        
        features.append({
            "type": "Feature",
            "geometry": None,
            "properties": {
                "is_summary": True,
                "summary_message": summary_message,
                "best_mode": best_mode,
                "best_time_min": round(best_time_h * 60, 1),
                "best_distance_km": round(best_distance, 2),
                "auto_available": True,
                "heli_available": helicopter_available,
                "direct_auto_time_min": round(auto_result['time_h'] * 60, 1),
                "direct_auto_distance_km": round(auto_result['distance_km'], 2),
                "heli_total_time_min": round(heli_total_time_h * 60, 1) if helicopter_available else None,
                "heli_total_distance_km": round(heli_total_distance, 2) if helicopter_available else None,
                "computation_time": round(time.perf_counter() - t_total, 2)
            }
        })
        
        logger.info(f" Optimized emergency route computed in {time.perf_counter() - t_total:.3f}s")
        logger.info(f"   Best mode: {best_mode}, Time: {best_time_h*60:.1f} min, Distance: {best_distance:.2f} km")
        
        return JSONResponse({
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "computation_time_seconds": round(time.perf_counter() - t_total, 2),
                "hospitals_considered": len(HOSPITAL_CACHE["points"]),
                "optimization": "single-source_dijkstra"
            }
        })
        
    except Exception as e:
        logger.error(f" Optimized emergency route error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/emergency-route-ultra-fast")
def emergency_route_ultra_fast(
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    max_auto_distance_km: float = Query(120.0),
    max_flight_distance_km: float = Query(200.0)
):
    """
    Ultra-fast version using only Euclidean distances and rough estimates.
    Designed for response times under one second.
    """
    logger.info(f" ULTRA-FAST EMERGENCY ROUTE: ({start_lat}, {start_lon})")
    
    try:
        t_total = time.perf_counter()
        
                                         
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        start_point_m = Point(start_x, start_y)
        
                              
        snapped_start = snap_to_road(start_point_m, list(strade_m.geometry))
        
                                                                              
        hospital_points = HOSPITAL_CACHE["points"]
        hospital_tree = HOSPITAL_CACHE["kdtree"]
        
                                                       
        k = min(20, len(hospital_points))
        distances, indices = hospital_tree.query([[snapped_start.x, snapped_start.y]], k=k)
        
                                                                                         
        auto_candidates = []
        for idx, dist_m in zip(indices[0], distances[0]):
            dist_km = dist_m / 1000
            if dist_km > max_auto_distance_km:
                continue
            
                                                                 
            estimated_auto_time_h = (dist_km * 1.3) / AUTO_SPEED_KMH
            auto_candidates.append((idx, dist_km, estimated_auto_time_h))
        
        if not auto_candidates:
            return JSONResponse({"error": "No hospital within reasonable distance"}, status_code=404)
        
                                      
        best_auto_idx, best_auto_dist_km, best_auto_time_h = min(
            auto_candidates, key=lambda x: x[2]
        )
        
                                      
        helipad_tree = HOSPITAL_CACHE.get("helipad_tree")
        helipad_points = HOSPITAL_CACHE.get("helipad_points", [])
        
        helicopter_available = False
        heli_total_time_h = float('inf')
        heli_total_distance = 0
        
        if helipad_tree is not None and len(helipad_points) > 0:
                                         
            distances_helipad, indices_helipad = helipad_tree.query(
                [[snapped_start.x, snapped_start.y]], k=min(5, len(helipad_points))
            )
            
            if len(indices_helipad[0]) > 0:
                helipad_idx = indices_helipad[0][0]
                helipad_dist_m = distances_helipad[0][0]
                helipad_dist_km = helipad_dist_m / 1000
                
                if helipad_dist_km <= max_auto_distance_km:
                                                              
                    helipad_time_h = (helipad_dist_km * 1.3) / AUTO_SPEED_KMH
                    
                                                             
                    helipad_point = Point(helipad_points[helipad_idx][0], helipad_points[helipad_idx][1])
                    
                                                     
                    hospital_candidates_from_helipad = []
                    for idx, dist_m in zip(indices[0], distances[0]):
                        hospital_point = hospital_points[idx]
                        
                                             
                        if (point_in_fly_zone(helipad_point, prepared_fly_zones) and 
                            point_in_fly_zone(hospital_point, prepared_fly_zones)):
                            
                            flight_dist_km = helipad_point.distance(hospital_point) / 1000
                            if flight_dist_km <= max_flight_distance_km:
                                flight_time_h = flight_dist_km / HELI_SPEED_KMH
                                hospital_candidates_from_helipad.append((idx, flight_dist_km, flight_time_h))
                    
                    if hospital_candidates_from_helipad:
                                                              
                        best_flight_idx, best_flight_dist_km, best_flight_time_h = min(
                            hospital_candidates_from_helipad, key=lambda x: x[2]
                        )
                        
                        helicopter_available = True
                        heli_total_time_h = helipad_time_h + best_flight_time_h
                        heli_total_distance = helipad_dist_km + best_flight_dist_km
        
                                       
        if best_auto_time_h < heli_total_time_h:
            best_mode = "auto"
            best_time_h = best_auto_time_h
            best_distance = best_auto_dist_km
        else:
            best_mode = "heli"
            best_time_h = heli_total_time_h
            best_distance = heli_total_distance
        
                                          
        features = []
        
                                  
        hospital_point = hospital_points[best_auto_idx]
        hospital_lon, hospital_lat = transformer_to4326.transform(
            hospital_point.x, hospital_point.y
        )
        
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [hospital_lon, hospital_lat]
            },
            "properties": {
                "type": "hospital",
                "estimated_time_min": round(best_auto_time_h * 60, 1),
                "estimated_distance_km": round(best_auto_dist_km, 1),
                "is_best": best_mode == "auto"
            }
        })
        
                          
        summary_message = ""
        if best_mode == "auto":
            summary_message = f" Auto diretto: {best_auto_dist_km:.1f} km, {best_auto_time_h*60:.1f} min"
            if helicopter_available:
                summary_message += f"\n Elicottero: {heli_total_distance:.1f} km, {heli_total_time_h*60:.1f} min"
        else:
            summary_message = f" Auto+elicottero: {heli_total_distance:.1f} km, {heli_total_time_h*60:.1f} min"
        
        features.append({
            "type": "Feature",
            "geometry": None,
            "properties": {
                "is_summary": True,
                "summary_message": summary_message,
                "best_mode": best_mode,
                "best_time_min": round(best_time_h * 60, 1),
                "best_distance_km": round(best_distance, 1),
                "computation_time_ms": round((time.perf_counter() - t_total) * 1000, 1),
                "note": "Stime basate su distanze euclidee con fattori di correzione"
            }
        })
        
        logger.info(f" Ultra-fast emergency route computed in {(time.perf_counter() - t_total)*1000:.1f}ms")
        
        return JSONResponse({
            "type": "FeatureCollection",
            "features": features
        })
        
    except Exception as e:
        logger.error(f" Ultra-fast emergency route error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/emergency-route-legacy")
def emergency_route_legacy(
    start_lat: float = Query(...),
    start_lon: float = Query(...)
):
    """
    Compute the fastest route to a hospital from any start point.
    Returns:
    1. Direct road route to the nearest hospital
    2. Helicopter route (road -> helipad -> nearest hospital from helipad)
    3. Time comparison to select the best option
    """
    logger.info(f" EMERGENCY ROUTE REQUEST: ({start_lat}, {start_lon})")
    
    try:
        t_total = time.perf_counter()
        
                                         
        start_x, start_y = transformer_to32632.transform(start_lon, start_lat)
        start_point_m = Point(start_x, start_y)
        
                              
        snapped_start = snap_to_road(start_point_m, list(strade_m.geometry))
        logger.info(f" Start point snapped to road: ({snapped_start.x:.0f}, {snapped_start.y:.0f})")
        

        logger.info(" Loading hospitals...")
        t_hospitals = time.perf_counter()
        

        HOSPITAL_FILE = os.path.join(DATA_DIR, "Centroidi Ospedali.gpkg")
        
        if not os.path.exists(HOSPITAL_FILE):
            logger.error(f" Hospital file not found: {HOSPITAL_FILE}")
                                                          
            hospitals_m = [
                Point(548000, 5030000),                   
                Point(600000, 5045000),                    
                Point(580000, 5025000),                    
                Point(550000, 5050000),                 
                Point(560000, 5015000),                  
            ]
            logger.warning(" Using fallback hospital points")
        else:
                                           
            ospedali = gpd.read_file(HOSPITAL_FILE).to_crs("EPSG:4326")
            ospedali_m = ospedali.to_crs("EPSG:32632")
            hospitals_m = [geom for geom in ospedali_m.geometry]
            logger.info(f" Loaded {len(hospitals_m)} hospitals from file")
        
        logger.info(f"    Hospitals loaded in {time.perf_counter() - t_hospitals:.3f}s")
        
        if not hospitals_m:
            return JSONResponse({"error": "No hospitals available"}, status_code=404)


        logger.info(" Finding nearest hospital by road...")
        t_auto_hospital = time.perf_counter()
        
        best_auto_hospital = None
        best_auto_distance = float('inf')
        best_auto_path = None
        best_auto_time_h = float('inf')
        best_hospital_point_m = None
        
                                                     
        for i, hospital_point_m in enumerate(hospitals_m):
            try:
                                       
                multilines = compute_auto_path_dijkstra(snapped_start, hospital_point_m)
                
                if not multilines:
                    continue
                
                                         
                total_dist_m = 0.0
                for segment in multilines:
                    for j in range(len(segment) - 1):
                        x1, y1 = transformer_to32632.transform(segment[j][0], segment[j][1])
                        x2, y2 = transformer_to32632.transform(segment[j+1][0], segment[j+1][1])
                        total_dist_m += Point(x1, y1).distance(Point(x2, y2))
                
                total_dist_km = total_dist_m / 1000
                total_time_h = total_dist_km / AUTO_SPEED_KMH
                
                                      
                if total_time_h < best_auto_time_h:
                    best_auto_hospital = i
                    best_auto_distance = total_dist_km
                    best_auto_path = multilines
                    best_auto_time_h = total_time_h
                    best_hospital_point_m = hospital_point_m
                    
            except Exception as e:
                logger.warning(f" Error calculating path to hospital {i}: {e}")
                continue
        
        if best_auto_hospital is None:
            logger.error(" No reachable hospital found by road")
            return JSONResponse({"error": "No hospital reachable by road"}, status_code=404)
        
        logger.info(f" Nearest hospital by road: #{best_auto_hospital}, distance: {best_auto_distance:.2f} km, time: {best_auto_time_h*60:.1f} min ({time.perf_counter() - t_auto_hospital:.3f}s)")
        

        logger.info(" Finding nearest helipad by road...")
        t_heli_search = time.perf_counter()
        
                                    
        helipads_m = [(geom.x, geom.y) for geom in heli_areas_m.geometry]
        
        if not helipads_m:
            logger.error(" No helipads available")
            return JSONResponse({"error": "No helipads available"}, status_code=404)
        
        best_helipad = None
        best_helipad_distance = float('inf')
        best_helipad_path = None
        best_helipad_time_h = float('inf')
        best_helipad_point_m = None
        
                                                     
        for i, (helipad_x, helipad_y) in enumerate(helipads_m):
            try:
                helipad_point_m = Point(helipad_x, helipad_y)
                
                                       
                multilines = compute_auto_path_dijkstra(snapped_start, helipad_point_m)
                
                if not multilines:
                    continue
                
                                         
                total_dist_m = 0.0
                for segment in multilines:
                    for j in range(len(segment) - 1):
                        x1, y1 = transformer_to32632.transform(segment[j][0], segment[j][1])
                        x2, y2 = transformer_to32632.transform(segment[j+1][0], segment[j+1][1])
                        total_dist_m += Point(x1, y1).distance(Point(x2, y2))
                
                total_dist_km = total_dist_m / 1000
                total_time_h = total_dist_km / AUTO_SPEED_KMH
                
                                      
                if total_time_h < best_helipad_time_h:
                    best_helipad = i
                    best_helipad_distance = total_dist_km
                    best_helipad_path = multilines
                    best_helipad_time_h = total_time_h
                    best_helipad_point_m = helipad_point_m
                    
            except Exception as e:
                logger.warning(f" Error calculating path to helipad {i}: {e}")
                continue
        
        if best_helipad is None:
            logger.error(" No reachable helipad found by road")
                                                    
            heli_route_available = False
            helipad_nearest_hospital = None
            heli_flight_path = None
            heli_flight_distance = 0
            heli_flight_time_h = 0
        else:
            heli_route_available = True
            logger.info(f" Nearest helipad by road: #{best_helipad}, distance: {best_helipad_distance:.2f} km, time: {best_helipad_time_h*60:.1f} min ({time.perf_counter() - t_heli_search:.3f}s)")
            


            logger.info(" Finding nearest hospital from helipad by helicopter...")
            t_hospital_from_helipad = time.perf_counter()
            
            helipad_nearest_hospital = None
            helipad_hospital_distance = float('inf')
            heli_flight_path = None
            heli_flight_distance = 0
            heli_flight_time_h = float('inf')
            best_hospital_from_helipad_m = None
            
                                                                                
            for i, hospital_point_m in enumerate(hospitals_m):
                try:
                                                 
                    heli_path_m = []
                    heli_dist_m = 0
                    
                                                                                    
                    helipad_in_fly = point_in_fly_zone(best_helipad_point_m, prepared_fly_zones)
                    hospital_in_fly = point_in_fly_zone(hospital_point_m, prepared_fly_zones)
                    
                    if not (helipad_in_fly and hospital_in_fly):
                                                                  
                        continue
                    
                                                       
                    grid, start_idx, end_idx = build_heli_grid(best_helipad_point_m, hospital_point_m)
                    
                    if grid is None:
                        continue
                    
                    edges, weights = build_heli_edges(grid, start_idx, end_idx)
                    
                    if edges is None:
                                                 
                        line = LineString([(best_helipad_point_m.x, best_helipad_point_m.y), 
                                          (hospital_point_m.x, hospital_point_m.y)])
                        if line_in_fly_zone(line, prepared_fly_zones, fly_zones_buffered):
                            heli_path_m = [[best_helipad_point_m.x, best_helipad_point_m.y], 
                                         [hospital_point_m.x, hospital_point_m.y]]
                            heli_dist_m = best_helipad_point_m.distance(hospital_point_m)
                        else:
                            continue
                    else:
                                      
                        heli_path_m, heli_dist_m = dijkstra_gpu(grid, edges, weights, start_idx, end_idx)
                        
                        if not heli_path_m:
                            continue
                    
                                           
                    heli_dist_km = heli_dist_m / 1000
                    heli_time_h = heli_dist_km / HELI_SPEED_KMH
                    
                                          
                    if heli_time_h < heli_flight_time_h:
                        helipad_nearest_hospital = i
                        helipad_hospital_distance = heli_dist_km
                        heli_flight_path = heli_path_m
                        heli_flight_distance = heli_dist_m
                        heli_flight_time_h = heli_time_h
                        best_hospital_from_helipad_m = hospital_point_m
                        
                except Exception as e:
                    logger.warning(f" Error calculating helicopter path to hospital {i}: {e}")
                    continue
            
            logger.info(f"    Hospital from helipad search: {time.perf_counter() - t_hospital_from_helipad:.3f}s")
        


        logger.info(" Comparing routes...")
        
                                                       
        auto_total_time_h = best_auto_time_h
        
                                               
        if heli_route_available and helipad_nearest_hospital is not None:
            heli_total_time_h = best_helipad_time_h + heli_flight_time_h
            heli_total_distance = best_helipad_distance + helipad_hospital_distance
        else:
            heli_total_time_h = float('inf')
            heli_total_distance = 0
        
                                      
        if auto_total_time_h < heli_total_time_h:
            best_mode = "auto"
            best_time_h = auto_total_time_h
            best_distance = best_auto_distance
        else:
            best_mode = "heli"
            best_time_h = heli_total_time_h
            best_distance = heli_total_distance
        

        features = []
        
                                                     
        if best_auto_path:
            h_auto = int(best_auto_time_h)
            m_auto = int((best_auto_time_h - h_auto) * 60)
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": best_auto_path
                },
                "properties": {
                    "mode": "direct_auto",
                    "distance_km": round(best_auto_distance, 2),
                    "time_hhmm": f"{h_auto:02d}:{m_auto:02d}",
                    "is_best": best_mode == "auto",
                    "description": "Direct road route to nearest hospital"
                }
            })
        
                                                     
        if heli_route_available and helipad_nearest_hospital is not None:
                             
            h_auto_to_helipad = int(best_helipad_time_h)
            m_auto_to_helipad = int((best_helipad_time_h - h_auto_to_helipad) * 60)
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": best_helipad_path
                },
                "properties": {
                    "mode": "auto_to_helipad",
                    "distance_km": round(best_helipad_distance, 2),
                    "time_hhmm": f"{h_auto_to_helipad:02d}:{m_auto_to_helipad:02d}",
                    "is_best": best_mode == "heli",
                    "description": "Road to nearest helipad"
                }
            })
            
                                            
            if heli_flight_path:
                flight_path_4326 = [[transformer_to4326.transform(x, y) for x, y in heli_flight_path]]
                
                h_flight = int(heli_flight_time_h)
                m_flight = int((heli_flight_time_h - h_flight) * 60)
                
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": flight_path_4326
                    },
                    "properties": {
                        "mode": "heli_flight",
                        "distance_km": round(helipad_hospital_distance, 2),
                        "time_hhmm": f"{h_flight:02d}:{m_flight:02d}",
                        "is_best": best_mode == "heli",
                        "description": "Helicopter flight to nearest hospital"
                    }
                })
            
                                      
            helipad_lon, helipad_lat = transformer_to4326.transform(
                best_helipad_point_m.x, best_helipad_point_m.y
            )
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [helipad_lon, helipad_lat]
                },
                "properties": {
                    "type": "helipad",
                    "description": f"Nearest helipad (road time: {h_auto_to_helipad:02d}:{m_auto_to_helipad:02d})"
                }
            })
        
                                                    
        if best_hospital_point_m:
            hospital_lon, hospital_lat = transformer_to4326.transform(
                best_hospital_point_m.x, best_hospital_point_m.y
            )
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [hospital_lon, hospital_lat]
                },
                "properties": {
                    "type": "hospital",
                    "route_type": "direct_auto",
                    "description": f"Nearest hospital by road (time: {int(best_auto_time_h*60):.1f} min)"
                }
            })
        
                                                          
        if heli_route_available and best_hospital_from_helipad_m is not None:
            hospital_heli_lon, hospital_heli_lat = transformer_to4326.transform(
                best_hospital_from_helipad_m.x, best_hospital_from_helipad_m.y
            )
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [hospital_heli_lon, hospital_heli_lat]
                },
                "properties": {
                    "type": "hospital",
                    "route_type": "helicopter",
                    "description": f"Nearest hospital from helipad (flight time: {int(heli_flight_time_h*60):.1f} min)"
                }
            })
        
                          
        best_hours = int(best_time_h)
        best_minutes = int((best_time_h - best_hours) * 60)
        
        summary_message = ""
        if best_mode == "auto":
            summary_message = (
                f" PERCORSO CONSIGLIATO: AUTO\n\n"
                f" Direttamente all'ospedale: {best_auto_distance:.2f} km, {int(best_auto_time_h*60):.1f} min\n"
            )
            
            if heli_route_available and helipad_nearest_hospital is not None:
                heli_total_min = heli_total_time_h * 60
                time_diff = heli_total_min - (best_auto_time_h * 60)
                summary_message += f" Con elicottero: {heli_total_distance:.2f} km, {heli_total_min:.1f} min\n"
                summary_message += f"  L'auto è più veloce di {abs(time_diff):.1f} min"
            else:
                summary_message += f" Percorso elicottero non disponibile"
        else:
            summary_message = (
                f" PERCORSO CONSIGLIATO: ELICOTTERO\n\n"
                f" Auto + elicottero: {heli_total_distance:.2f} km, {heli_total_time_h*60:.1f} min\n"
                f" Solo auto: {best_auto_distance:.2f} km, {best_auto_time_h*60:.1f} min\n"
                f"  L'elicottero è più veloce di {abs(best_auto_time_h*60 - heli_total_time_h*60):.1f} min"
            )
        
        features.append({
            "type": "Feature",
            "geometry": None,
            "properties": {
                "is_summary": True,
                "summary_message": summary_message,
                "best_mode": best_mode,
                "best_time_min": round(best_time_h * 60, 1),
                "best_distance_km": round(best_distance, 2),
                "auto_available": True,
                "heli_available": heli_route_available and helipad_nearest_hospital is not None,
                "direct_auto_time_min": round(best_auto_time_h * 60, 1),
                "direct_auto_distance_km": round(best_auto_distance, 2),
                "heli_total_time_min": round(heli_total_time_h * 60, 1) if heli_route_available else None,
                "heli_total_distance_km": round(heli_total_distance, 2) if heli_route_available else None
            }
        })
        
        logger.info(f" Emergency route computed in {time.perf_counter() - t_total:.3f}s")
        logger.info(f"    Best mode: {best_mode}, Time: {best_time_h*60:.1f} min, Distance: {best_distance:.2f} km")
        
        return JSONResponse({
            "type": "FeatureCollection",
            "features": features
        })
        
    except Exception as e:
        logger.error(f" Emergency route error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)