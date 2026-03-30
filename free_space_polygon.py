from shapely.geometry import box, Point, MultiPolygon
from shapely.ops import unary_union
# Assuming WIDTH/HEIGHT are imported or defined
from config import HEIGHT, WIDTH 

GSD            = 100 / 8192   
BUFFER_M       = 0.5          
MIN_AREA_M2    = 0.2          
OVERLAP_THRESH = 0.5          

def _px_to_m(px):
    return round(px * GSD, 2)

# REPLACED: No longer need 0-999 normalization parsing
def _loc_to_pixel(coords):
    """YOLOv8 already provides [x1, y1, x2, y2] in pixels."""
    x1, y1, x2, y2 = coords
    return int(x1), int(y1), int(x2), int(y2)

def _format_polygon(geom) -> list[dict]:
    """Handles both single Polygons and MultiPolygons safely."""
    if geom is None or geom.is_empty:
        return []
    
    # If it's a MultiPolygon, we'll take the largest one or handle the first
    # For Kins, we usually want the largest 'chunk' of free space
    if geom.geom_type == 'MultiPolygon':
        # Option A: Take the largest polygon in the collection
        main_part = max(geom.geoms, key=lambda a: a.area)
        coords = list(main_part.exterior.coords)[:-1]
    elif geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)[:-1]
    else:
        return []

    return [{'x': _px_to_m(p[0]), 'z': _px_to_m(p[1])} for p in coords]

def loc_to_json(detections: list[tuple]) -> list[dict]:
    """Input: [('rooftop', [x1, y1, x2, y2]), ...]"""
    result = []
    for label, coords in detections:
        x1, y1, x2, y2 = _loc_to_pixel(coords)
        
        # Clamp coordinates to image boundaries
        x1, x2 = max(0, min(x1, WIDTH)), max(0, min(x2, WIDTH))
        y1, y2 = max(0, min(y1, HEIGHT)), max(0, min(y2, HEIGHT))

        result.append({
            'label': label,
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'bbox_normalized': {
                'x1': x1 / WIDTH,  'y1': y1 / HEIGHT,
                'x2': x2 / WIDTH,  'y2': y2 / HEIGHT,
            },
        })
    return result


def get_parent_roof(object_list: list[dict], payload: dict):
    """
    Find the rooftop bbox that contains the user's clicked coordinates.
    Returns largest containing rooftop as shapely box, or None if not found.
    """
    coords  = payload.get('coordinates', {})
    target  = Point(coords.get('x', 0), coords.get('y', 0))

    candidates = []
    for obj in object_list:
        if obj['label'] != 'rooftop':
            continue
        b = obj['bbox']
        rect = box(b['x1'], b['y1'], b['x2'], b['y2'])
        if rect.contains(target) or rect.intersects(target):
            candidates.append(rect)

    if not candidates:
        return None

    # Return largest candidate (most likely the main roof)
    return max(candidates, key=lambda r: r.area)


def process_kins_site(object_list: list[dict], payload: dict) -> dict | None:
    """
    Main geometry pipeline.
    Returns site plan dict or None if no parent roof found.
    """
    main_roof = get_parent_roof(object_list, payload)
    if main_roof is None:
        return None

    buffer_px      = BUFFER_M / GSD
    valid_obstacles = []
    inner_roofs     = []

    for obj in object_list:
        b        = obj['bbox']
        obj_geom = box(b['x1'], b['y1'], b['x2'], b['y2'])

        # Skip the main roof itself
        if obj_geom.equals(main_roof):
            continue

        intersection = main_roof.intersection(obj_geom)
        if intersection.is_empty:
            continue

        overlap_ratio = intersection.area / obj_geom.area
        if overlap_ratio < OVERLAP_THRESH:
            continue

        # Clip to roof boundary
        clipped = intersection

        if obj['label'] == 'rooftop':
            inner_roofs.append(clipped)
        else:
            # Skip tiny obstacles
            area_m2 = clipped.area * (GSD ** 2)
            if area_m2 < MIN_AREA_M2:
                continue
            # Apply safety buffer, clipped to roof boundary
            buffered = clipped.buffer(buffer_px).intersection(main_roof)
            valid_obstacles.append(buffered)

    # Merge overlapping obstacles into single shapes
    merged_obstacles = unary_union(valid_obstacles) if valid_obstacles else None

    # Subtract obstacles and inner roofs from main roof
    to_subtract   = unary_union(
        ([merged_obstacles] if merged_obstacles else []) + inner_roofs
    )
    free_space    = main_roof.difference(to_subtract) if to_subtract else main_roof

    # Format obstacles for output
    if merged_obstacles is None or merged_obstacles.is_empty:
        obs_list = []
    elif isinstance(merged_obstacles, MultiPolygon):
        obs_list = [{'points': _format_polygon(g)} for g in merged_obstacles.geoms]
    else:
        obs_list = [{'points': _format_polygon(merged_obstacles)}]

    return {
        'roof_polygon':      _format_polygon(main_roof),
        'free_space_polygon': _format_polygon(free_space),
        'obstacles':         obs_list,
        'upper_roofs':       [{'footprint': _format_polygon(r)} for r in inner_roofs],
        'stats': {
            'roof_area_m2':       round(main_roof.area * GSD**2, 2),
            'free_space_area_m2': round(free_space.area * GSD**2, 2)
            if not free_space.is_empty else 0,
            'obstacle_count':     len(obs_list),
            'inner_roof_count':   len(inner_roofs),
        }
    }