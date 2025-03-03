# -----------------------
# ROI Adjustment Globals & Callback
# -----------------------

import cv2

roi = None  # Format: [x, y, w, h]
dragging = False
drag_mode = None  # Options: "left", "right", "top", "bottom", "topleft", etc.

def roi_mouse_callback(event, x, y, flags, param):
    global roi, dragging, drag_mode
    if roi is None:
        return

    rx, ry, rw, rh = roi
    margin = 10  # pixels: sensitivity for edge selection

    if event == cv2.EVENT_LBUTTONDOWN:
        near_left   = abs(x - rx) < margin
        near_right  = abs(x - (rx + rw)) < margin
        near_top    = abs(y - ry) < margin
        near_bottom = abs(y - (ry + rh)) < margin

        if near_left and near_top:
            drag_mode = "topleft"
            dragging = True
        elif near_right and near_top:
            drag_mode = "topright"
            dragging = True
        elif near_left and near_bottom:
            drag_mode = "bottomleft"
            dragging = True
        elif near_right and near_bottom:
            drag_mode = "bottomright"
            dragging = True
        elif near_left:
            drag_mode = "left"
            dragging = True
        elif near_right:
            drag_mode = "right"
            dragging = True
        elif near_top:
            drag_mode = "top"
            dragging = True
        elif near_bottom:
            drag_mode = "bottom"
            dragging = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            if drag_mode == "left":
                new_rx = x
                new_rw = (rx + rw) - x
                if new_rw > margin:
                    roi[0] = new_rx
                    roi[2] = new_rw
            elif drag_mode == "right":
                new_rw = x - rx
                if new_rw > margin:
                    roi[2] = new_rw
            elif drag_mode == "top":
                new_ry = y
                new_rh = (ry + rh) - y
                if new_rh > margin:
                    roi[1] = new_ry
                    roi[3] = new_rh
            elif drag_mode == "bottom":
                new_rh = y - ry
                if new_rh > margin:
                    roi[3] = new_rh
            elif drag_mode == "topleft":
                new_rx = x
                new_rw = (rx + rw) - x
                new_ry = y
                new_rh = (ry + rh) - y
                if new_rw > margin and new_rh > margin:
                    roi[0] = new_rx
                    roi[1] = new_ry
                    roi[2] = new_rw
                    roi[3] = new_rh
            elif drag_mode == "topright":
                new_rw = x - rx
                new_ry = y
                new_rh = (ry + rh) - y
                if new_rw > margin and new_rh > margin:
                    roi[2] = new_rw
                    roi[1] = new_ry
                    roi[3] = new_rh
            elif drag_mode == "bottomleft":
                new_rx = x
                new_rw = (rx + rw) - x
                new_rh = y - ry
                if new_rw > margin and new_rh > margin:
                    roi[0] = new_rx
                    roi[2] = new_rw
                    roi[3] = new_rh
            elif drag_mode == "bottomright":
                new_rw = x - rx
                new_rh = y - ry
                if new_rw > margin and new_rh > margin:
                    roi[2] = new_rw
                    roi[3] = new_rh

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        drag_mode = None