import cv2
from math import atan2, degrees, pi
import numpy as np

INVALID_DEGREE = 360
DETECT_AREA_SIZE = 1000
WINDOW_X = 2304
WINDOW_Y = 1536
STATION_POINT = [WINDOW_X / 2, 900]


def points2degree(points):
    x1 = (points[0][0][0] + points[1][0][0]) / 2
    y1 = (points[0][0][1] + points[1][0][1]) / 2
    x2 = (points[2][0][0] + points[3][0][0]) / 2
    y2 = (points[2][0][1] + points[3][0][1]) / 2
    
    return degrees(atan2(x1 - x2, -(y1 - y2)))


def get_img(device_num):
    cap = cv2.VideoCapture(device_num)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        return False, None
    
    return cap.read()


def get_qr_points(img, save_img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    
    if save_img:
        cv2.imwrite('img/result_binary.png', img_binary)
    
    qr = cv2.QRCodeDetector()
    # qr.setEpsX(100)
    # qr.setEpsY(100)
    return qr.detect(img)


def get_degree(img, save_img):
    # readable, img = get_img(0)
    
    # print(get_hsv(img))
    
    if not readable:
        return INVALID_DEGREE
    
    retval, points = get_qr_points(img, save_img)
    print('retval: ', retval)
    print('points: ', points)
    
    degree = INVALID_DEGREE
    if retval:
        if save_img:
            for i in range(len(points)):
                img = cv2.line(img, (points[i][0][0], points[i][0][1]),
                               (points[(i + 1) % len(points)][0][0], points[(i + 1) % len(points)][0][1]),
                               (0, 0, 255))
        
        degree = points2degree(points)
    
    if save_img:
        cv2.imwrite('img/result.png', img)
    
    return degree


def get_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = img_hsv.T[0].flatten().mean()
    s = img_hsv.T[1].flatten().mean()
    v = img_hsv.T[2].flatten().mean()
    return h, s, v


# 無駄に半周以上しないように角度を調整
def compress_degree_in_180(degree, one_round_value):
    if abs(degree) > (one_round_value / 2):
        return degree - (degree / abs(degree) * one_round_value)
    return degree


# @brief 黄色の矩形領域を検知する
# @param img GBR画像
# @param hsv_img HSV変換後の処理対象画像
# @detail
def detectYellow(img, hsv_img, area_size):
    hsv_range_min = [20, 30, 0]
    hsv_range_max = [40, 255, 255]
    mask = cv2.inRange(hsv_img, np.array(hsv_range_min), np.array(hsv_range_max))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    trim_mask = np.zeros_like(img)
    convex_hull_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        # 四角形に近い形のみ扱う
        if 3 < len(approx) < 10:
            M = cv2.moments(approx)
            if M['m00'] > area_size:
                convex_hull_list.append({'approx': approx, 'moment': M})
                cv2.fillConvexPoly(trim_mask, approx, color=(255, 255, 255))
        
    cv2.imwrite('img/trim_mask.png', trim_mask)
    bg_color = (255, 255, 255)
    bg_img = np.full_like(img, bg_color)
    trim_img = np.where(trim_mask == 255, img, bg_img)
    
    marker_mask = cv2.inRange(trim_img, np.array([0, 0, 0]), np.array([120, 120, 120]))
    cv2.imwrite('img/marker_mask.png', marker_mask)
    marker_contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m_convex_hull_list = []
    for contour in marker_contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        # 四角形に近い形のみ扱う
        if 3 < len(approx) < 10:
            M = cv2.moments(approx)
            if M['m00'] > area_size / 20:
                m_convex_hull_list.append({'approx': approx, 'moment': M})
                
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                draw_marker(trim_img, cx, cy, (0, 0, 255))
    
    if len(m_convex_hull_list) > 1:
        m_convex_hull_list.sort(key=lambda x: x['moment']['m00'], reverse=True)
        p1 = [int(m_convex_hull_list[0]['moment']['m10'] / m_convex_hull_list[0]['moment']['m00']),
              int(m_convex_hull_list[0]['moment']['m01'] / m_convex_hull_list[0]['moment']['m00'])]
        p2 = [int(m_convex_hull_list[1]['moment']['m10'] / m_convex_hull_list[1]['moment']['m00']),
              int(m_convex_hull_list[1]['moment']['m01'] / m_convex_hull_list[1]['moment']['m00'])]
        print('p1: ', p1, ' p2: ', p2)
        degree_car = atan2(p2[1] - p1[1], p1[0] - p2[0]) * 180 / pi
        degree_station = atan2(((p1[1] + p2[1]) / 2) - STATION_POINT[1], STATION_POINT[0] - ((p1[0] + p2[0]) / 2)) * 180 / pi
        degree = compress_degree_in_180(degree_car - degree_station, 360)
    else:
        degree = 360
        
    cv2.imwrite('img/result_mask.png', trim_img)
    
    pos_list = []
    size_list = []
    if len(convex_hull_list) > 0:
        for convex in convex_hull_list:
            if convex['moment']['m00'] > 0:
                size_list.append(convex['moment']['m00'])
                
                cx = int(convex['moment']['m10'] / convex['moment']['m00'])
                cy = int(convex['moment']['m01'] / convex['moment']['m00'])
                
                pos_list.append([cx, cy])
    
    # return pos_list, size_list, convex_hull_list
    return degree


# @brief 十字マーカーを描画する
# @param x 十字マーカーのx座標
# @param y 十字マーカーのy座標
# @param marker_color 十字マーカーの色
def draw_marker(img, x, y, marker_color):
    cv2.line(img, (x - 7, y), (x + 7, y), color=(255, 255, 255), thickness=2)
    cv2.line(img, (x, y - 7), (x, y + 7), color=(255, 255, 255), thickness=2)
    cv2.line(img, (x - 7, y), (x + 7, y), color=marker_color, thickness=1)
    cv2.line(img, (x, y - 7), (x, y + 7), color=marker_color, thickness=1)


if __name__ == '__main__':
    readable, img = get_img(0)
    print('degree : ', get_degree(img, True))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #pos_list, size_list, convex_list = detectYellow(img, img_hsv, DETECT_AREA_SIZE)
    print('degree = ', detectYellow(img, img_hsv, DETECT_AREA_SIZE))
    #for convex in convex_list:
        #cv2.drawContours(img, convex['approx'], -1, (0, 0, 255), 5)
    #cv2.imwrite('img/result_approx.png', img)
