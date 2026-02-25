import numpy as np
from shapely import geometry
from typing import Optional

def calculate_wallPolygon(bounds, pWall=0.2):
    distTop = abs(bounds[0,0] - bounds[1,0])
    distBottom = abs(bounds[2,0] - bounds[3,0])
    distLeft = abs(bounds[0,1] - bounds[3,1])
    distRight = abs(bounds[1,1] - bounds[2,1])

    tl = int(bounds[0,0] + pWall*distTop), int(bounds[0,1] + pWall*distLeft)
    tr = int(bounds[1,0] - pWall*distTop), int(bounds[1,1] + pWall*distRight)
    br = int(bounds[2,0] - pWall*distBottom), int(bounds[2,1] - pWall*distRight)
    bl = int(bounds[3,0] + pWall*distBottom), int(bounds[3,1] - pWall*distLeft)
    points = np.array([tl,tr,br,bl])
    return points

def buffer_shapely_polygon(poly: geometry.Polygon, factor=0.2, swell: bool = False, *, distance_px: Optional[float] = None):
    ''' 
    Returns a resized Shapely polygon using buffering.

    Backward compatible mode (factor-based):
      - Computes a shrink/expand distance from the polygon's bounding-box
        center-to-corner distance times `factor`.

    Preferred mode (distance_px):
      - Uses an explicit buffer distance in the polygon's coordinate units (map pixels).
      - For inward shrink, the polygon is buffered by `-distance_px`.

    If swell = True, expands the polygon; otherwise shrinks it.
    '''

    # Preferred: explicit absolute distance in pixels
    if distance_px is not None:
        shrink_distance = abs(float(distance_px))
    else:
        xs = list(poly.exterior.coords.xy[0])
        ys = list(poly.exterior.coords.xy[1])
        x_center = 0.5 * min(xs) + 0.5 * max(xs)
        y_center = 0.5 * min(ys) + 0.5 * max(ys)
        min_corner = geometry.Point(min(xs), min(ys))
        center = geometry.Point(x_center, y_center)
        shrink_distance = center.distance(min_corner) * float(factor)

    if swell:
        return poly.buffer(shrink_distance)  # expand
    return poly.buffer(-shrink_distance)    # shrink

def non_null_len(iterable):
    return len(list(filter(None, iterable)))
def arg_first_non_null(iterable):
    for i, el in enumerate(iterable):
        if el is not None:
            return i
def len_comparator(item1, item2):
    return non_null_len(item1) - non_null_len(item2)
def arg_first_comparator(item1, item2):
    return arg_first_non_null(item1) - arg_first_non_null(item2)

def get_odd(v):
    if v%2 ==0:
        return v-1
    return v