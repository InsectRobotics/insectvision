#!/usr/bin/env python

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2019, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"


from base import Environment, spectrum_influence, spectrum
from sky import Sky, visualise_luminance, visualise_degree_of_polarisation, visualise_angle_of_polarisation
from antworld import World, load_routes, load_world, load_route, save_route
from geometry import Route, Polygon, PolygonList, route_like
from conditions import NoneCondition, Stepper, Turner, Hybrid
from utils import get_seville_observer, eps

from ephem import Sun
