import json
import numpy as np
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def json_keys2int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x


def stripped_with_match(xs, ms):
    for x in xs:
        xi = int(x.strip(".png"))
        yield xi, "%d.png" % ms[xi][0]


def matches(metas_path, matches_path):
    metas_wide = json.load(open(metas_path + 'fscam.json')).get("all_metas")
    metas_zoom = json.load(open(metas_path + 'flir.json')).get("all_metas")

    zoom_times = np.array([z["global_timestamp"] for z in metas_zoom])
    zoom_mirrors = np.array([z["mirror_position"] for z in metas_zoom])

    result, latency = {}, 0
    for wi, frame in enumerate(metas_wide):
        wide_time = frame["global_timestamp"]

        diffs = np.abs(wide_time - zoom_times + latency)
        zi = np.argmin(diffs)
    
        result[wi] = (zi, zoom_mirrors[zi], wide_time, zoom_times[zi])

    with open(matches_path, "w") as f:
        json.dump(result, f, cls=NumpyEncoder)
        print("Saved matches to", matches_path) 

    

def regressions(corners_path, matches_path, calib_path):
    data_wide = json.load(open(corners_path % ("wide", "wide")))
    data_zoom = json.load(open(corners_path % ("zoom", "zoom")))

    matches = json.load(open(matches_path), object_hook=json_keys2int)

    valid_wide = [frame for frame, match in stripped_with_match(data_wide.keys(), matches)
                if match in data_zoom.keys() and len(data_zoom[match]["img_points"]) > 5]

    regression = {}
    wide_pixels, mirrors = [], []

    # Linear regressions
    for axis, title in zip(range(2), ["x", "y"]):
        plt.figure("Frames Registration - %s" % title.capitalize(), (16, 7.5))

        for mirror in {int(matches[w][1]) for w in valid_wide}:
            wp, zp = [], []
            wides = [w for w in valid_wide if matches[w][1] == mirror]

            for wide, zoom in zip(wides, ["%d.png" % matches[w][0] for w in wides]):
                zp.extend(data_zoom[zoom]['img_points'])
                wp.extend([data_wide["%d.png" % wide]['img_points'][i // 11][i % 11]
                        for i in data_zoom[zoom]['ids']])

            wp, zp = np.array(wp), np.array(zp)
            plt.plot(wp[:, axis], zp[:, axis], ".", label=f"pos = {mirror}")

            p = np.polyfit(wp[:, axis], zp[:, axis], 1)
            pred = np.polyval(p, wp[:, axis])

            plt.plot(wp[:, axis], pred, "r-")

            if title == "x":
                mid_wide = (1920 / 2 - p[1]) / p[0]
                wide_pixels.append(mid_wide)
                mirrors.append(mirror) 

                regression[mirror] = {"mid_wide": mid_wide, "p_hint": "y = p[0] * x + p[1]"}

            regression[mirror]["p_" + title] = (p[0], p[1]) #First tuple is x-regression, second tuple is y-regression

        plt.xlim([0, 1920 * 2])
        plt.ylim([0, 1920])
        plt.xlabel("Wide %s, pixels" % title.capitalize())
        plt.ylabel("Zoom %s, pixels" % title.capitalize())
        plt.title("Frames Registration - %s" % title.capitalize())
        plt.legend()
        plt.tight_layout()
        plt.savefig("frames_registration_%s.png" % title, dpi=240)

    wide_pixels, mirrors = np.array(wide_pixels), np.array(mirrors)
    cubic = np.polyfit(wide_pixels, mirrors, 3)
    pred = np.polyval(cubic, wide_pixels)

    # Cubic regression
    plt.figure("Mirror Calibration")
    plt.plot(wide_pixels, mirrors, "b.", label="Data")
    plt.plot(wide_pixels, pred, "r-", label="Fit")
    plt.xlabel("Wide X, pixels")
    plt.ylabel("Mirror Position")
    plt.title("Mirror Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mirror_calib.png", dpi=120)

    with open(calib_path, "w") as f:
        json.dump({"regressions": regression,
                   "cubic_p": cubic,
                   "cubic_hint": "y = p[0] * x^3 + p[1] * x^2 + p[2] * x + p[3]"},
                   f, indent=4, cls=NumpyEncoder)
        print("Saved calibrations to", calib_path) 


if __name__ == "__main__":
    metas_path = "./data/"
    matches_path = "matches.json"

    matches(metas_path, matches_path)

    corners_path = "data/%s/try2/detected/%s_corners.json"
    calib_path = "calib.json"

    regressions(corners_path, matches_path, calib_path)

    plt.show()
 