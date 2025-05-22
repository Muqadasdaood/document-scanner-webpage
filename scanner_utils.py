import cv2
import numpy as np
import imutils
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       
    rect[2] = pts[np.argmax(s)]       
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    
    rect[3] = pts[np.argmax(diff)]    
    return rect
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
def scan_document(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    docCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
    if docCnt is not None:
        warped = four_point_transform(orig, docCnt.reshape(4, 2) * ratio)
        if warped is None or warped.shape[1] == 0:
            return None
        final_width = orig.shape[1]
        scale_ratio = final_width / warped.shape[1]
        warped = cv2.resize(warped, None, fx=scale_ratio, fy=scale_ratio)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        threshed = cv2.adaptiveThreshold(
            warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(threshed, cv2.COLOR_GRAY2RGB)
    else:
        fallback_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        threshed = cv2.adaptiveThreshold(
            fallback_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(threshed, cv2.COLOR_GRAY2RGB)