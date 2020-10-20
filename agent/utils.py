import numpy as np
import os
import yaml
from datetime import datetime

# get path of the script
__cpath__ = os.path.dirname(os.path.abspath(__file__))
__data__ = os.path.realpath(os.path.join(__cpath__, "..", "data"))
__logpath__ = os.path.join(__data__, "tests.yaml")
__fovpath__ = os.path.join(__data__, "fov-tests.yaml")
__binpath__ = os.path.join(__data__, "bin-tests.yaml")
__enpath__ = os.path.join(__data__, "EN")
__rtpath__ = os.path.join(__data__, "routes")
__ripath__ = os.path.join(__data__, "routes-img")

datestr = "%Y-%m-%d_%H-%M"

# load tests
with open(__logpath__, 'rb') as f:
    tests = yaml.safe_load(f)
with open(__fovpath__, 'rb') as f:
    fov_tests = yaml.safe_load(f)
with open(__binpath__, 'rb') as f:
    bin_tests = yaml.safe_load(f)


def create_agent_name(date, sky_type, step=.1, gfov=-np.pi/2, sfov=np.pi/2):
    """

    :param date: the date of the trial
    :type date: datetime
    :param sky_type: the sky-type
    :type sky_type: basestring
    :param step: the step size (default 10 cm)
    :type step: float
    :param gfov: the ground field of view in rads (default -90 degrees - full view)
    :type gfov: float
    :param sfov: the sky field of view in rads (default 90 degrees - full view)
    :type sfov: float
    :return:
    """
    agent_name = "%s_s%02d-%s-sky" % (date.strftime(datestr), step * 100, sky_type)
    if np.abs(gfov) < np.pi / 2:
        agent_name += "_gfov%02d" % np.abs(np.rad2deg(gfov))
    if np.abs(sfov) < np.pi / 2:
        agent_name += "_sfov%02d" % np.abs(np.rad2deg(sfov))

    return agent_name


def get_agent_features(sky_type, j=-1, fov=False, bin=False):
    """

    :param sky_type: the sky-type
    :type sky_type: basestring
    :param j: the index of the trial
    :type j: int
    :return:
    """

    if bin:
        if sky_type not in bin_tests.keys():
            raise AttributeError("There is not key named '%s' in the tests records." % sky_type)
        test_ = bin_tests[sky_type]
    elif fov:
        if sky_type not in fov_tests.keys():
            raise AttributeError("There is not key named '%s' in the tests records." % sky_type)
        test_ = fov_tests[sky_type]
    else:
        if sky_type not in tests.keys():
            raise AttributeError("There is not key named '%s' in the tests records." % sky_type)
        test_ = tests[sky_type]
    if j >= len(test_):
        raise AttributeError("Index %d out of range. List length = %d" % (j, len(test_)))
    test = test_[j]

    date = datetime.strptime("%s_%s" % (test["date"], test["time"]), datestr)
    step = test["step"] / 100.  # cm --> m

    if "gfov" in test.keys():
        gfov = -np.deg2rad(test["gfov"])  # degrees --> rad
    else:
        gfov = -np.pi / 2
    if "sfov" in test.keys():
        sfov = np.deg2rad(test["sfov"])  # degrees --> rad
    else:
        sfov = np.pi / 2

    return date, step, gfov, sfov


def get_agent_name(sky_type, j, fov=False, bin=False):
    """

    :param sky_type: the sky-type
    :type sky_type: basestring
    :param j: the index of the trial
    :type j: int
    :return:
    """
    date, step, gfov, sfov = get_agent_features(sky_type, j, fov, bin)
    date_str = date.strftime(datestr)

    agent_name = "%s_s%02d-%s-sky" % (date_str, step * 100, sky_type)
    if np.abs(gfov) < np.pi / 2:
        gfov = np.abs(np.rad2deg(gfov))  # degrees
        agent_name += "_gfov%02d" % gfov
    if np.abs(sfov) < np.pi / 2:
        sfov = np.abs(np.rad2deg(sfov))  # degrees
        agent_name += "_sfov%02d" % sfov

    return agent_name


def update_tests(sky_type, date, step, gfov=-np.pi/2, sfov=np.pi/2, bin=False):
    """

    :param sky_type: the sky-type
    :type sky_type:
    :param date: the date of the trial
    :type date:
    :param step: the step size (in meters)
    :type step:
    :param gfov: the ground field of view (in rads)
    :param sfov: the sky field of view (in rads)
    :return:
    """

    date_str = date.strftime("%Y-%m-%d_%H-%M")
    if bin:
        if sky_type not in bin_tests.keys():
            bin_tests[sky_type] = []

        bin_tests[sky_type].append({
            "date": date_str.split("_")[0],
            "time": date_str.split("_")[1],
            "step": int(step * 100),
            "gfov": int(np.abs(np.rad2deg(gfov))),
            "sfov": int(np.abs(np.rad2deg(sfov)))
        })

        try:
            # save/update tests
            with open(__binpath__, 'wb') as f:
                yaml.safe_dump(bin_tests, f, default_flow_style=False, allow_unicode=False)
            return True
        except Exception as e:
            print(e)
            return False
    elif np.abs(gfov) < np.pi / 2 or np.abs(sfov) < np.pi / 2:

        if sky_type not in fov_tests.keys():
            fov_tests[sky_type] = []

        fov_tests[sky_type].append({
            "date": date_str.split("_")[0],
            "time": date_str.split("_")[1],
            "step": int(step * 100),
            "gfov": int(np.abs(np.rad2deg(gfov))),
            "sfov": int(np.abs(np.rad2deg(sfov)))
        })

        try:
            # save/update tests
            with open(__fovpath__, 'wb') as f:
                yaml.safe_dump(fov_tests, f, default_flow_style=False, allow_unicode=False)
            return True
        except Exception as e:
            print(e)
            return False
    else:
        if sky_type not in tests.keys():
            tests[sky_type] = []

        tests[sky_type].append({
            "date": date_str.split("_")[0],
            "time": date_str.split("_")[1],
            "step": int(step * 100)
        })

        try:
            # save/update tests
            with open(__logpath__, 'wb') as f:
                yaml.safe_dump(tests, f, default_flow_style=False, allow_unicode=False)
            return True
        except Exception as e:
            print(e)
            return False


def delete_test(sky_type, j, fov=False, bin=False):
    name = get_agent_name(sky_type, j, fov=fov, bin=bin)

    enname = __enpath__ + name + ".npz"
    hroute = __rtpath__ + name + ".npz"
    imname = __ripath__ + name + ".png"

    files = [enname, hroute, imname]

    print("Are you sure you want to delete '%s'? ([Y]/n)" % name)
    s = input()

    if s in ["Y", "y", ""]:
        for f in files:
            try:
                os.remove(f)
                print("'%s' successfully deleted." % f)
            except OSError as e:
                print(e)

        if bin:
            bin_tests[sky_type].remove(bin_tests[sky_type][id])
            with open(__binpath__, 'wb') as f:
                yaml.safe_dump(bin_tests, f, default_flow_style=False, allow_unicode=False)
                print("Binary tests log updated successfully.")
        elif fov:
            fov_tests[sky_type].remove(fov_tests[sky_type][id])
            with open(__fovpath__, 'wb') as f:
                yaml.safe_dump(fov_tests, f, default_flow_style=False, allow_unicode=False)
                print("FOV tests log updated successfully.")
        else:
            tests[sky_type].remove(tests[sky_type][id])
            with open(__logpath__, 'wb') as f:
                yaml.safe_dump(tests, f, default_flow_style=False, allow_unicode=False)
                print("Tests log updated successfully.")

        return True

    else:
        print("Canceled.")

        return False


if __name__ == "__main__":
    sky_type = "live"
    id = 0
    fov = True
    bin = True

    delete_test(sky_type, id, fov=fov, bin=True)

