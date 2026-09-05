import cv2
import numpy as np


def analyze_jaw_curvature(tip_bgr, debug_vis=False):
    if tip_bgr is None or tip_bgr.size == 0:
        return False, 0.0, None

    h, w = tip_bgr.shape[:2]
    if h < 15 or w < 15:
        return False, 0.0, None

    hsv = cv2.cvtColor(tip_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    metal_mask = cv2.bitwise_not(green_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)
    metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return False, 0.0, None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 80:
        return False, 0.0, None

    pts = cnt.reshape(-1, 2).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    center = mean[0]
    primary_axis = eigenvectors[0]

    angle = np.arctan2(primary_axis[1], primary_axis[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    pts_centered = pts - center
    pts_rotated = np.dot(pts_centered, R.T)

    x_coords = pts_rotated[:, 0]
    y_coords = pts_rotated[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    length = x_max - x_min
    if length < 20:
        return False, 0.0, None

    num_bins = 20
    bin_edges = np.linspace(x_min, x_max, num_bins + 1)
    centerline_x = []
    centerline_y = []

    for i in range(num_bins):
        mask_bin = (x_coords >= bin_edges[i]) & (x_coords < bin_edges[i + 1])
        if np.any(mask_bin):
            y_in_bin = y_coords[mask_bin]
            mid_y = (np.min(y_in_bin) + np.max(y_in_bin)) / 2.0
            mid_x = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            centerline_x.append(mid_x)
            centerline_y.append(mid_y)

    if len(centerline_x) < 5:
        return False, 0.0, None

    centerline_x = np.array(centerline_x)
    centerline_y = np.array(centerline_y)

    poly_coeff = np.polyfit(centerline_x, centerline_y, deg=1)
    baseline_y = np.polyval(poly_coeff, centerline_x)

    deflections = np.abs(centerline_y - baseline_y)
    max_deflection = float(np.max(deflections))

    deflection_ratio = max_deflection / length

    CURVATURE_RATIO_THRESH = 0.065
    is_curved = deflection_ratio > CURVATURE_RATIO_THRESH

    vis_img = None
    if debug_vis:
        vis_img = tip_bgr.copy()
        cv2.drawContours(vis_img, [cnt.astype(np.int32)], -1, (0, 255, 0), 1)
        res_text = "Curved (Artery)" if is_curved else "Straight (Needle)"
        cv2.putText(
            vis_img,
            f"{res_text} R:{deflection_ratio:.3f}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255) if is_curved else (255, 0, 0),
            1,
        )

    return is_curved, max_deflection, vis_img


if __name__ == "__main__":
    import glob, os
    for p in sorted(glob.glob("geometry_test/*.png")):
        img = cv2.imread(p)
        is_curved, md, vis = analyze_jaw_curvature(img, debug_vis=True)
        name = os.path.basename(p).replace(".png", "")
        print(f"{name:20s} curved={is_curved} max_defl={md:.2f}px "
              f"ratio={md/max(img.shape[:2]):.3f}")
        if vis is not None:
            cv2.imwrite(f"geometry_test/out_{name}.png", vis)
