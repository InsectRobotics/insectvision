import numpy as np
import yaml
import os

# get path of the script
cpath = os.path.dirname(os.path.abspath(__file__)) + '/'

# load parameters
with open(cpath + 'Ardin2016.yaml', 'rb') as f:
    params = yaml.safe_load(f)

GAIN = -.1 / params['gain']

N_TL2 = params['central-complex']['TL2']
N_CL1 = params['central-complex']['CL1']
N_TB1 = params['central-complex']['TB1']
N_TN1 = params['central-complex']['TN1']
N_TN2 = params['central-complex']['TN2']
N_CPU4 = params['central-complex']['CPU4']
N_CPU1A = params['central-complex']['CPU1A']
N_CPU1B = params['central-complex']['CPU1B']
N_CPU1 = N_CPU1A + N_CPU1B
N_COLUMNS = params['central-complex']['columns']
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)


class CX(object):
    """
    Implements basic CX model but:
    - noise free.
    - perfect sinusoids for TB1.
    - memory can update using inverse amplitudes for TB1 (backwards motion).
    - perfect memory decay relative to speed.
    """

    def __init__(self, tn_prefs=np.pi/4, cpu4_mem_gain=GAIN):
        self.tn_prefs = tn_prefs
        self.cpu4_mem_gain = cpu4_mem_gain
        self.smoothed_flow = 0.

    def tl2_output(self, theta):
        """
        Dummy function.
        """
        return theta

    def cl1_output(self, tl2):
        """
        Dummy function.
        """
        return tl2

    def tb1_output(self, cl1, tb1=None):
        """
        Sinusoidal response to solar compass.
        :param cl1:
        :param tb1:
        :return:
        """
        return (1. + np.cos(np.pi + x + cl1)) / 2.

    def tn1_output(self, flow):
        """
        Linearly inverse sensitive to forwards and backwards motion.
        :param flow:
        :type flow: np.ndarray
        :return:
        """
        return np.clip((1. - flow) / 2., 0, 1)

    def tn2_output(self, flow):
        """
        Linearly sensitive to forwards motion only.
        :param flow:
        :type flow: np.ndarray
        :return:
        """
        return np.clip(flow, 0, 1)

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        """
        Updates memory based on current TB1 and TN activity.
        Can think of this as summing sinusoid of TB1 onto sinusoid of CPU4.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu4[8-15] store optic flow peaking at right 45 deg.

        :param cpu4_mem:
        :param tb1:
        :param tn1:
        :param tn2:
        :return:
        """
        cpu4_mem_reshaped = cpu4_mem.reshape(2, -1)

        # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
        mem_update = (.5 - tn1.reshape(2, 1)) * (1. - tb1)

        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        mem_update -= .5 * (.5 - tn1.reshape(2, 1))

        # Constant purely to visualise same as rate-based model
        cpu4_mem_reshaped += self.cpu4_mem_gain * mem_update
        return np.clip(cpu4_mem_reshaped.reshape(-1), 0., 1.)

    def cpu4_output(self, cpu4_mem):
        """
        Output activity based on memory.
        :param cpu4_mem:
        :return:
        """
        return cpu4_mem

    def cpu1_output(self, tb1, cpu4):
        """
        Offset CPU4 columns by 1 column (45 degrees) left and right wrt TB1.
        :param tb1:
        :param cpu4:
        :return:
        """

        cpu4_reshaped = cpu4.reshape((2, -1))
        cpu1 = (1. - tb1) * np.vstack([np.roll(cpu4_reshaped[1], 1),
                                       np.roll(cpu4_reshaped[0], -1)])
        return cpu1.reshape(-1)

    def motor_output(self, cpu1, random_std=.05):
        """
        Sum CPU1 to determine left or right turn.
        :param cpu1:
        :param random_std:
        :return:
        """

        cpu1_reshaped = cpu1.reshape(2, -1)
        motor_lr = np.sum(cpu1_reshaped, axis=1)

        # We need to add some randomness, otherwise agent infinitely overshoots
        motor = motor_lr[1] - motor_lr[0]
        if random_std > 0.:
            motor += np.random.normal(0., random_std)
        return motor

    def get_flow(self, heading, velocity, filter_steps=0):
        """
        Calculate optic flow depending on preference angles. [L, R]
        """
        A = np.array([[np.sin(heading - self.tn_prefs),
                       np.cos(heading - self.tn_prefs)],
                      [np.sin(heading + self.tn_prefs),
                       np.cos(heading + self.tn_prefs)]])
        flow = np.dot(A, velocity)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self.smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                  1.0 / filter_steps) * self.smoothed_flow)
            flow = self.smoothed_flow
        return flow

    def decode_cpu4(self, cpu4):
        """
        Shifts both CPU4 by +1 and -1 column to cancel 45 degree flow
        preference. When summed single sinusoid should point home.
        """
        cpu4_reshaped = cpu4.reshape(2, -1)
        cpu4_shifted = np.vstack([np.roll(cpu4_reshaped[0], 1),
                                  np.roll(cpu4_reshaped[1], -1)])
        return decode_position(cpu4_shifted, self.cpu4_mem_gain)

    def __str__(self):
        return "basic_holo"


class CXForwards(CX):
    """
    This class can't 'flip' the TB1 sinusoid, meaning it can integrate holonomically between -45 and +45
    of forwards heading.
    """

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        """
        Trying to be a bit more realistic, but only sensitive to motion in forward directions (-45 to +45 degrees).
        :param cpu4_mem:
        :param tb1:
        :param tn1:
        :param tn2:
        :return:
        """

        cpu4_mem_reshaped = cpu4_mem.reshape(2, -1)

        # Signal comes in from PB. (inverse TB1 as inhibited.)
        # This is inhibited by TN1, the faster the motion, the less inhibited
        mem_update = np.clip(.5 - tn1.reshape(2, 1), 0., 1.) * (1. - tb1)

        # Delay is proportionate to TN2
        mem_update -= .25 * tn2.reshape(2, 1)

        cpu4_mem_reshaped += self.cpu4_mem_gain * mem_update
        return np.clip(cpu4_mem_reshaped.reshape(-1), 0., 1.)

    def __str__(self):
        return "basic_pholo"


class CXAveraging(CXForwards):
    """
    Here CPU4 are averaged for each columns, to give OK path integration in most situations,
    however can get failure due to holonomic motion.
    """

    def tn1_output(self, flow):
        """
        Literaly inverse sensitive to forwards and backwards motion.
        :param flow:
        :return:
        """

        mean_flow = np.array([np.mean(flow)] * 2)
        return np.clip((1. - mean_flow) / 2., 0., 1.)

    def tn2_output(self, flow):
        """
        Literaly sensitive to forwards motion only.
        :param flow:
        :return:
        """

        mean_flow = np.array([np.mean(flow)] * 2)
        return np.clip(mean_flow, 0., 1.)

    def __str__(self):
        return "basic_av"


class CXFlipped(CX):
    """
    Here we are trying to invert TB1 preference angles to see if that results in a functioning path integrator.
    """

    def tb1_output(self, cl1, tb1=None):
        """
        Sinusoidal response to solar compass.
        :param cl1:
        :param tb1:
        :return:
        """
        return (1. + np.cos(np.pi + x - cl1)) / 2.

    def __str__(self):
        return "basic_holoflipped"


def decode_position(cpu4_reshaped, cpu4_mem_gain):
    """
    Decode position from sinusoid in to polar coordinates.
    Amplitude is distance, Angle is angle from nest outwards.
    Without offset angle gives the home vector.
    Input must have shape of (2, -1)
    """
    signal = np.sum(cpu4_reshaped, axis=0)
    fund_freq = np.fft.fft(signal)[1]
    angle = -np.angle(np.conj(fund_freq))
    distance = np.absolute(fund_freq) / cpu4_mem_gain
    return angle, distance


def update_cells(heading, velocity, tb1, memory, cx, filtered_steps=0.0):
    """Generate activity for all cells, based on previous activity and current
    motion."""
    # Compass
    tl2 = cx.tl2_output(heading)
    cl1 = cx.cl1_output(tl2)
    tb1 = cx.tb1_output(cl1, tb1)

    # Speed
    flow = cx.get_flow(heading, velocity, filtered_steps)
    tn1 = cx.tn1_output(flow)
    tn2 = cx.tn2_output(flow)

    # Update memory for distance just travelled
    memory = cx.cpu4_update(memory, tb1, tn1, tn2)
    cpu4 = cx.cpu4_output(memory)

    # Steer based on memory and direction
    cpu1 = cx.cpu1_output(tb1, cpu4)
    motor = cx.motor_output(cpu1)
    return tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor
