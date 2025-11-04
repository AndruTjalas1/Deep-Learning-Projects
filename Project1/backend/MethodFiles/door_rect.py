# MethodFiles/door_rect.py
def door_rect(door_wall, door_width, door_offset, room_w, room_h):
    """Return (x,y,w,h) of door rectangle in room coords (origin bottom-left)."""
    if door_wall in ("Bottom","Top"):
        w = min(door_width, room_w)
        x = min(max(door_offset, 0.0), room_w - w)
        y = 0.0 if door_wall == "Bottom" else room_h - 0.3
        h = 0.3
    else:
        h = min(door_width, room_h)
        y = min(max(door_offset, 0.0), room_h - h)
        x = 0.0 if door_wall == "Left" else room_w - 0.3
        w = 0.3
    return x, y, w, h
