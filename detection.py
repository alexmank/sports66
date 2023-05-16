import os
import glob
import json
import joblib
import cv2
from cv2 import aruco
import json

#from utils import *
import numpy as np





"""
#Opening and Saving Videos
wide = cv2.VideoCapture('fscam.avi')
fps = wide.get(cv2.CAP_PROP_FPS)
frames = wide.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps,frames)

for number in range(390,556):
    wide.set(cv2.CAP_PROP_POS_FRAMES, number)
    ret, frame = wide.read()
    cv2.imwrite(f'data/wide/{number}.png', frame)

zoom = cv2.VideoCapture('flir.avi')


with open('mynewjson.json') as json_file:
    data = json.load(json_file)





count =0
zoom_frames = []
number_of_pairs = 0
for wide_frame in range(390,556):
    for mirror_position in range(120):
        #print(mirror_position)
        pairs = data.get(str(mirror_position))
        #print(pairs, len(pairs))
        
        if pairs != None:
            for pair_number in range(len(pairs)):
                #print(pair_number)
                number_of_pairs += 1
                dictionary_pair = pairs[pair_number]
                dictionary = dictionary_pair[1]
                #print(dictionary_pair, "stop", dictionary, "stop", list(dictionary.keys()))
                
                if list(dictionary.keys()) == [str(wide_frame)]:
                    zoom_frame = list(dictionary_pair[0].keys())
                    zoom_frames.append(zoom_frame)
                    count += 1
                    
        else:
            print("try again")
print(count)

for number in zoom_frames:
    zoom.set(cv2.CAP_PROP_POS_FRAMES, int(number[0]))
    ret, frame = zoom.read()
    cv2.imwrite(f'data/zoom/{int(number[0])}.png', frame)



#for number in range(25880, 28974):
   # zoom.set(cv2.CAP_PROP_POS_FRAMES, number)
    #ret, frame = zoom.read()
   # cv2.imwrite(f'data/zoom/{number}.png', frame)
"""
# Detect checker board on gray scale image. Use pre_scale to do the initial pass on a lower resolution image.
# Use draw_scale to do reduce the resolution of an overlay image.
# Board origin: top-left. First axis: X to the right. Second axis: Y down
def detect_checker(gray, n=11, m=8, size=60, pre_scale=1, draw_on=None, draw_scale=1):
    assert(len(gray.shape) == 2)
    assert(gray.dtype == np.uint8)
    pre_scale = pre_scale or 1
   
    flags = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH

    if pre_scale > 1.5:
      
        gray2 = cv2.resize(gray, (gray.shape[1] // pre_scale, gray.shape[0] // pre_scale))
     
        ret, corners2 = cv2.findChessboardCorners(gray2, (n, m), flags=flags)
       
        if ret:
            corners = corners2 * pre_scale
      
    else:
        ret, corners = cv2.findChessboardCorners(gray, (n, m), flags=flags)
        
    if ret:
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (n, m), (-1, -1), criteria)
    else:
        print('else')
        corners = None

    if corners is not None:
        img_points = corners.reshape((m, n, 2))
        obj_points = np.zeros((n * m, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2) * size
        ids = np.arange(n*m)
    else:
        img_points, obj_points, ids = None, None, None

    if draw_on is not None:
        if draw_scale > 1.5:
            draw_on = cv2.resize(draw_on, (draw_on.shape[1] // draw_scale, draw_on.shape[0] // draw_scale))
        img = cv2.drawChessboardCorners(draw_on, (n, m), corners, ret)
    else:
        img = None

    return (img_points, obj_points, ids), img


# Detect charuco board on a gray scale image. Use pre_scale to do the initial pass on a lower resolution image.
# Use draw_scale to do reduce the resolution of an overlay image.
# Board origin: bottom-left. First axis: X to the right. Second axis: Y up
def detect_charuco(gray, n=12, m=9, size=60, pre_scale=1, draw_on=None, draw_scale=1):
    #***aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    

   

    #***
    board = aruco.CharucoBoard((n,m), size, size * 12 / 15, aruco_dict) #***
   
    if pre_scale > 1.5:
        gray2 = cv2.resize(gray, (gray.shape[1] // pre_scale, gray.shape[0] // pre_scale))
        m_pos2, m_ids, _ = detector.detectMarkers(gray2)
        m_pos = [m * pre_scale for m in m_pos2]
       
    else:
        m_pos, m_ids, _ = detector.detectMarkers(gray)
    
    if len(m_pos) > 0:
        print(4)
        
        count, c_pos, c_ids = aruco.interpolateCornersCharuco(m_pos, m_ids, gray, board)
        print(5)
    else:
     
        count, c_pos, c_ids = 0, None, None
       

    if count:
        print(count, "hello", c_pos)
        img_points, obj_points = np.array(c_pos).reshape((-1, 2)), board.getChessboardCorners().reshape((-1, 3))
        ids = c_ids.ravel()
    else:
        img_points, obj_points, ids = None, None, None

    if draw_on is not None:
        if draw_scale > 1.5:
            draw_on = cv2.resize(draw_on, (draw_on.shape[1] // draw_scale, draw_on.shape[0] // draw_scale))
        if len(draw_on.shape) == 2:
            draw_on = np.repeat(draw_on[:, :, None], 3, axis=2)
        if len(m_pos) > 0:
            img = aruco.drawDetectedMarkers(draw_on, np.array(m_pos) / draw_scale, m_ids)
        if c_pos is not None:
            img = aruco.drawDetectedCornersCharuco(img, np.array(c_pos) / draw_scale, c_ids, cornerColor=(0, 255, 0))
    else:
        img = None


    return (img_points, obj_points, ids), img


# Load image from file and detect calibration board using detect_func. Save or plot an overlay image if need be
def detect_single(filename, detect_func, resize=1, out_dir="detected", draw=False, save=False, plot=False, return_image=True, **kw):
    assert(type(filename) is str)

    img = cv2.imread(filename)
    if resize > 1.5:
        img = cv2.resize(img, (img.shape[1] // resize, img.shape[0] // resize))

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    points, img = detect_func(gray, draw_on=img if draw else None, **kw)
    success = points[2] is not None
    print(filename, " - Success" if success else " - Failed")

    new_filename = None
    if save:
        if draw:
         
            path = os.path.dirname(filename) + "/" + out_dir + "/"
            print(path)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            new_filename = path + os.path.basename(filename)[:-4] + ".png"
            cv2.imwrite(new_filename, img)
        else:
            print("Nothing new to save because draw=False")

    if plot:
        cv2.imshow(filename, img)

        while cv2.getWindowProperty(filename, 0) >= 0:
            cv2.waitKey(50)

    return points, new_filename, img if return_image else None


# Process all images that match a filename_template. Save valid results as json file if need be
def detect_all(filename_template, detect_func, out_dir="try2/detected", save_json=True, **kw):
    print(filename_template)
    filenames = glob.glob(filename_template)
    print(len(filenames), filenames[:10])

    jobs = [joblib.delayed(detect_single)
            (name, detect_func, return_image=False, out_dir=out_dir, **kw) for name in filenames]

    results = joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)
    
    ret = {}
    for filename, result in zip(filenames, results):
        name, points = os.path.basename(filename), result[0]

        if points[2] is not None:
            if save_json:
                points = points[0].tolist(), points[1].tolist(), points[2].tolist()
            ret[name] = {"img_points": points[0], "obj_points": points[1], "ids": points[2]}

    if save_json:
        path = os.path.dirname(filename_template) + "/" + out_dir
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        with open(path + "/corners.json", "w") as f:
            json.dump(ret, f, indent=4)

    return ret


def load_corners(filename):
    corners = {}

    for name, points in json.load(open(filename, "r")).items():
        img_points = np.array(points["img_points"]).reshape(-1, 2).astype(np.float32)
        obj_points = np.array(points["obj_points"]).reshape(-1, 3).astype(np.float32)
        ids = np.array(points["ids"]).ravel().astype(np.int32)
        corners[name] = {"img": img_points, "obj": obj_points, "idx": ids}

    return corners


def test(data_path):
    filename = data_path + "checker/checker (1).bmp"

    detect_single(filename, detect_checker, draw=True, save=True, pre_scale=4, draw_scale=1)
    points, new_filename, img = detect_single(filename, detect_checker, resize=5, draw=True, plot=True)
    print([p.shape for p in points], new_filename, img.shape)

    filename = data_path + "charuco/charuco (1).bmp"

    detect_single(filename, detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)
    points, new_filename, img = detect_single(filename, detect_charuco, resize=4, draw=True, plot=True)
    print([p.shape for p in points], new_filename, img.shape)


if __name__ == "__main__":
    # data_path = "D:/Scanner/Calibration/camera_intrinsics/data/"


   # dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    #parameters =  cv2.aruco.DetectorParameters()
    #aruco2 = cv2.aruco.ArucoDetector(dictionary, parameters)


    data_path = "data/wide/"
    filename = data_path + "527.png"
  
    
    # detect_single(filename, detect_checker, draw=True, save=True, plot=True, pre_scale=2, draw_scale=1)


    # test(data_path)

   # data_path = "data/wide/"
   # detect_all(data_path + "/*.png", detect_checker, draw=True, save=True, pre_scale=1, draw_scale=1)
    data_path = "data/zoom/"
    detect_all(data_path + "/*.png", detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)
    #detect_all(data_path + "/charuco/*.bmp", detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)

    # data_path = "D:/scanner_sim/calibration/accuracy_test/projector_calib/"
    # detect_all(data_path + "/charuco/*.bmp", detect_charuco, draw=True, save=True, pre_scale=2, draw_scale=1)