import numpy as np
from scipy.special import expit
from base import Network, params, RNG, GAIN

GAIN = -1. / GAIN
N_COLUMNS = params['central-complex']['columns']  # 8
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)

cxparams = params['central-complex']
cxrate_params = params['central-complex-rate']


class CX(Network):

    def __init__(self, tn_prefs=np.pi/4, gain=GAIN, noise=.0, pontin=False, **kwargs):

        if pontin:
            gain *= 5e-03
        super(CX, self).__init__(gain=gain, **kwargs)

        self.tn_prefs = tn_prefs
        self.smoothed_flow = 0.
        self.noise = noise
        self.pontin = pontin

        self.nb_tl2 = cxparams['TL2']  # 16
        self.nb_cl1 = cxparams['CL1']  # 16
        self.nb_tb1 = cxparams['TB1']  # 8
        self.nb_tn1 = cxparams['TN1']  # 2
        self.nb_tn2 = cxparams['TN2']  # 2
        self.nb_cpu4 = cxparams['CPU4']  # 16
        nb_cpu1a = cxparams['CPU1A']  # 14
        nb_cpu1b = cxparams['CPU1B']  # 2
        self.nb_cpu1 = nb_cpu1a + nb_cpu1b  # 16

        self.tl2 = np.zeros(self.nb_tl2)
        self.cl1 = np.zeros(self.nb_cl1)
        self.tb1 = np.zeros(self.nb_tb1)
        self.tn1 = np.zeros(self.nb_tn1)
        self.tn2 = np.zeros(self.nb_tn2)
        self.__cpu4 = .5 * np.ones(self.nb_cpu4)  # cpu4 memory
        self.cpu4 = np.zeros(self.nb_cpu4)  # cpu4 output
        self.cpu1 = np.zeros(self.nb_cpu1)

        # Weight matrices based on anatomy (These are not changeable!)
        self.w_tl22cl1 = -np.eye(self.nb_tl2, self.nb_cl1)
        self.w_cl12tb1 = np.tile(np.eye(self.nb_tb1), 2).T
        self.w_tb12tb1 = gen_tb_tb_weights(self.nb_tb1)
        self.w_tb12cpu4 = -np.tile(np.eye(self.nb_tb1), (2, 1)).T
        self.w_tn2cpu4 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        self.w_tb12cpu1a = -np.tile(np.eye(self.nb_tb1), (2, 1))[1:nb_cpu1a+1, :].T
        self.w_tb12cpu1b = -np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0, 0, 0]]).T
        self.w_cpu42cpu1a = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ]).T
        self.w_cpu42cpu1b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
        ]).T
        self.w_cpu1a2motor = np.array([
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]).T
        self.w_cpu1b2motor = np.array([[0, 1],
                                       [1, 0]]).T
        self.w_pontin2cpu1a = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
            ]).T
        self.w_pontin2cpu1b = np.array([
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 9
            ]).T
        self.w_cpu42pontin = np.eye(self.nb_cpu4)

        self.params = [
            self.w_tl22cl1,
            self.w_cl12tb1,
            self.w_tb12tb1,
            self.w_tb12cpu4,
            self.w_tb12cpu1a,
            self.w_tb12cpu1b,
            self.w_cpu42cpu1a,
            self.w_cpu42cpu1b,
            self.w_cpu1a2motor,
            self.w_cpu1b2motor,
            self.w_cpu42pontin,
            self.w_pontin2cpu1a,
            self.w_pontin2cpu1b
        ]

        # import matplotlib.pyplot as plt
        #
        # plt.figure()
        # plt.subplot(4, 3, 1)
        # plt.imshow(self.w_tl22cl1, vmin=-1, vmax=1)
        # plt.title("TL2-2-CL1")
        # plt.subplot(4, 3, 2)
        # plt.imshow(self.w_cl12tb1, vmin=-1, vmax=1)
        # plt.title("CL1-2-TB1")
        # plt.subplot(4, 3, 3)
        # plt.imshow(self.w_tb12tb1, vmin=-1, vmax=1)
        # plt.title("TB1-2-TB1")
        # plt.subplot(4, 3, 4)
        # plt.imshow(self.w_tb12cpu4, vmin=-1, vmax=1)
        # plt.title("TB1-2-CPU4")
        # plt.subplot(4, 3, 5)
        # plt.imshow(self.w_tn2cpu4, vmin=-1, vmax=1)
        # plt.title("TN-2-CPU4")
        # plt.subplot(4, 3, 6)
        # plt.imshow(self.w_tb12cpu1a, vmin=-1, vmax=1)
        # plt.title("TB1-2-CPU1a")
        # plt.subplot(4, 3, 7)
        # plt.imshow(self.w_tb12cpu1b, vmin=-1, vmax=1)
        # plt.title("TB1-2-CPU1b")
        # plt.subplot(4, 3, 8)
        # plt.imshow(self.w_cpu42cpu1a, vmin=-1, vmax=1)
        # plt.title("CPU4-2-CPU1a")
        # plt.subplot(4, 3, 9)
        # plt.imshow(self.w_cpu42cpu1b, vmin=-1, vmax=1)
        # plt.title("CPU4-2-CPU1b")
        # plt.subplot(4, 3, 10)
        # plt.imshow(self.w_cpu1a2motor, vmin=-1, vmax=1)
        # plt.title("CPU1a-2-motor")
        # plt.subplot(4, 3, 11)
        # plt.imshow(self.w_cpu1b2motor, vmin=-1, vmax=1)
        # plt.title("cpu1b-2-motor")
        # plt.show()

        # The cell properties (for sigmoid function)
        self.tl2_slope = cxrate_params['tl2-tuned']['slope']
        self.tl2_bias = cxrate_params['tl2-tuned']['bias']
        self.tl2_prefs = np.tile(np.linspace(0, 2 * np.pi, self.nb_tb1, endpoint=False), 2)
        # self.tl2_prefs = np.tile(np.linspace(-np.pi, np.pi, self.nb_tb1, endpoint=False), 2)
        self.cl1_slope = cxrate_params['cl1-tuned']['slope']
        self.cl1_bias = cxrate_params['cl1-tuned']['bias']
        self.tb1_slope = cxrate_params['tb1-tuned']['slope']
        self.tb1_bias = cxrate_params['tb1-tuned']['bias']
        self.cpu4_slope = cxrate_params['cpu4-tuned']['slope']
        self.cpu4_bias = cxrate_params['cpu4-tuned']['bias']
        self.cpu1_slope = cxrate_params['cpu1-tuned']['slope']
        self.cpu1_bias = cxrate_params['cpu1-tuned']['bias']
        self.motor_slope = cxrate_params['motor-tuned']['slope']
        self.motor_bias = cxrate_params['motor-tuned']['bias']
        self.pontin_slope = cxrate_params['pontin-tuned']['slope']
        self.pontin_bias = cxrate_params['pontin-tuned']['bias']

    @property
    def cpu4_mem(self):
        return self.__cpu4

    def reset(self):
        super(CX, self).reset()

        self.tl2 = np.zeros(self.nb_tl2)
        self.cl1 = np.zeros(self.nb_cl1)
        self.tb1 = np.zeros(self.nb_tb1)
        self.tn1 = np.zeros(self.nb_tn1)
        self.tn2 = np.zeros(self.nb_tn2)
        self.__cpu4 = .5 * np.ones(self.nb_cpu4)  # cpu4 memory
        self.cpu4 = np.zeros(self.nb_cpu4)  # cpu4 output
        self.cpu1 = np.zeros(self.nb_cpu1)
        self.update = True

    def __call__(self, *args, **kwargs):
        compass, flow = args[:2]
        tl2 = kwargs.get("tl2", None)
        cl1 = kwargs.get("cl1", None)
        if tl2 is None and len(args) > 2:
            tl2 = args[2]
        if cl1 is None and len(args) > 3:
            cl1 = args[3]
        self.tl2, self.cl1, self.tb1, self.tn1, self.tn2, self.cpu4, self.cpu1 = self._fprop(
            compass, flow, tl2=tl2, cl1=cl1
        )
        return self.f_motor(self.cpu1)

    def f_tl2(self, theta):
        """
        Just a dot product with the preferred angle and current heading.
        :param theta:
        :type theta: float
        :return:
        """
        output = np.cos(theta - self.tl2_prefs)
        return noisy_sigmoid(output, self.tl2_slope, self.tl2_bias, self.noise)

    def f_cl1(self, tl2):
        """
        Takes input from the TL2 neurons and gives output.
        :param tl2:
        :return:
        """
        output = tl2.dot(self.w_tl22cl1)
        return noisy_sigmoid(output, self.cl1_slope, self.cl1_bias, self.noise)

    def f_tb1(self, cl1, tb1=None):
        """
        Sinusoidal response to solar compass.

        :param cl1:
        :type cl1: np.ndarray
        :param tb1:
        :type tb1: np.ndarray
        :return:
        """
        if tb1 is None:
            output = cl1
        else:
            p = .667  # Proportion of input from CL1 vs TB1
            cl1_out = cl1.dot(self.w_cl12tb1)
            tb1_out = tb1.dot(self.w_tb12tb1)
            output = p * cl1_out + (1. - p) * tb1_out
            # output = p * cl1_out - (1. - p) * tb1_out

        return noisy_sigmoid(output, self.tb1_slope, self.tb1_bias, self.noise)

    def f_tn1(self, flow):
        """
        Linearly inverse sensitive to forwards and backwards motion.

        :param flow:
        :type flow: np.ndarray
        :return:
        """
        noise = self.rng.normal(scale=self.noise, size=flow.shape)
        return np.clip((1. - flow) / 2. + noise, 0, 1)

    def f_tn2(self, flow):
        """
        Linearly sensitive to forwards motion only.

        :param flow:
        :type flow: np.ndarray
        :return:
        """
        return np.clip(flow, 0, 1)

    def f_cpu4(self, tb1, tn1, tn2):
        """
        Output activity based on memory.

        :param tb1:
        :type tb1: np.ndarray
        :param tn1:
        :type tn1: np.ndarray
        :param tn2:
        :type tn2: np.ndarray
        :return:
        """

        if self.pontin:
            update = tn2.dot(self.w_tn2cpu4) - tb1.dot(self.w_tb12cpu4)
            update = .5 * self.gain * (np.clip(update, 0, 1) - .25)
        else:
            # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
            # update = np.clip((.5 - tn1).dot(self.w_tn2cpu4), 0., 1.)  # normal
            update = (.5 - tn1).dot(self.w_tn2cpu4)  # holonomic

            update *= self.gain * (tb1 - 1.).dot(self.w_tb12cpu4)
            # update *= self.gain * (1. - tb1).dot(self.w_tb12cpu4)

            # Both CPU4 waves must have same average
            # If we don't normalise get drift and weird steering
            update -= self.gain * .25 * tn2.dot(self.w_tn2cpu4)

        # Constant purely to visualise same as rate-based model
        cpu4 = np.clip(self.__cpu4 + update, 0., 1.)
        if self.update:
            self.__cpu4 = cpu4

        return noisy_sigmoid(cpu4, self.cpu4_slope, self.cpu4_bias, self.noise)

    def f_pontin(self, cpu4):
        inputs = cpu4.dot(self.w_cpu42pontin)
        return noisy_sigmoid(inputs, self.pontin_slope, self.pontin_bias, self.noise)

    def f_cpu1a(self, tb1, cpu4):
        """
        The memory and direction used together to get population code for heading.

        :param tb1:
        :type tb1: np.ndarray
        :param cpu4:
        :type cpu4: np.ndarray
        :return:
        """
        if self.pontin:
            pontin = self.f_pontin(cpu4)  # type: np.ndarray
            inputs = .5 * cpu4.dot(self.w_cpu42cpu1a) \
                     - .5 * pontin.dot(self.w_pontin2cpu1a) \
                     - tb1.dot(self.w_tb12cpu1a)
        else:
            inputs = cpu4.dot(self.w_cpu42cpu1a) * (tb1 - 1.).dot(self.w_tb12cpu1a)
        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias, self.noise)

    def f_cpu1b(self, tb1, cpu4):
        """
        The memory and direction used together to get population code for heading.

        :param tb1:
        :type tb1: np.ndarray
        :param cpu4:
        :type cpu4: np.ndarray
        :return:
        """
        if self.pontin:
            pontin = self.f_pontin(cpu4)  # type: np.ndarray
            inputs = .5 * cpu4.dot(self.w_cpu42cpu1b) \
                     - .5 * pontin.dot(self.w_pontin2cpu1b) \
                     - tb1.dot(self.w_tb12cpu1b)
        else:
            inputs = cpu4.dot(self.w_cpu42cpu1b) * (tb1 - 1.).dot(self.w_tb12cpu1b)
        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias, self.noise)

    def f_cpu1(self, tb1, cpu4):
        """
        Offset CPU4 columns by 1 column (45 degrees) left and right wrt TB1.

        :param tb1:
        :type tb1: np.ndarray
        :param cpu4:
        :type cpu4: np.ndarray
        :return:
        """
        cpu1a = self.f_cpu1a(tb1, cpu4)
        cpu1b = self.f_cpu1b(tb1, cpu4)
        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])

    def f_motor(self, cpu1):
        """
        Outputs a scalar where sign determines left or right turn.
        :param cpu1:
        :type cpu1: np.ndarray
        :return:
        """

        cpu1a = cpu1[1:-1]
        cpu1b = np.array([cpu1[-1], cpu1[0]])
        motor = cpu1a.dot(self.w_cpu1a2motor)
        motor += cpu1b.dot(self.w_cpu1b2motor)
        output = (motor[0] - motor[1])  # * .25  # to kill the noise a bit!
        return output

    def _fprop(self, phi, flow, tl2=None, cl1=None):
        if isinstance(phi, np.ndarray) and phi.size == 8:
            if tl2 is None:
                tl2 = np.tile(phi, 2)
            if cl1 is None:
                cl1 = np.tile(phi, 2)
            tl2 = noisy_sigmoid(tl2[::-1], self.tl2_slope, self.tl2_bias, self.noise)
            cl1 = noisy_sigmoid(cl1[::-1], self.cl1_slope, self.cl1_bias, self.noise)
            tb1 = noisy_sigmoid(phi[::-1], 5.*self.tb1_slope, self.tb1_bias, self.noise)
        else:
            tl2 = self.f_tl2(phi)
            cl1 = self.f_cl1(tl2)
            tb1 = self.f_tb1(cl1, self.tb1)
        tn1 = self.f_tn1(flow)
        tn2 = self.f_tn2(flow)
        cpu4 = self.f_cpu4(tb1, tn1, tn2)
        cpu1 = self.f_cpu1(tb1, cpu4)

        return tl2, cl1, tb1, tn1, tn2, cpu4, cpu1

    def get_flow(self, heading, velocity, filter_steps=0):
        """
        Calculate optic flow depending on preference angles. [L, R]
        """
        A = tn_axes(heading, self.tn_prefs)
        flow = velocity.dot(A)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self.smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                  1.0 / filter_steps) * self.smoothed_flow)
            flow = self.smoothed_flow
        return flow


def gen_tb_tb_weights(nb_tb1, weight=1.):
    """
    Weight matrix to map inhibitory connections from TB1 to other neurons
    """

    W = np.zeros([nb_tb1, nb_tb1])
    sinusoid = (np.cos(np.linspace(0, 2*np.pi, nb_tb1, endpoint=False)) - 1)/2  # type: np.ndarray
    for i in range(nb_tb1):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W


def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
    """
    Takes a vector v as input, puts through sigmoid and adds Gaussian noise. Results are clipped to return rate
    between 0 and 1.

    :param v:
    :type v: np.ndarray
    :param slope:
    :type slope: float
    :param bias:
    :type bias: float
    :param noise:
    :type noise: float
    """
    sig = expit(v * slope - bias) + RNG.normal(scale=noise, size=len(v))
    return np.clip(sig, 0, 1)


def image_motion_flow(velocity, v_heading, r_sensor):
    """
    :param velocity: translation (velocity - 3D)
    :type velocity: np.ndarray
    :param v_heading: agent heading direction (3D vector)
    :type v_heading: np.ndarray
    :param r_sensor: relative directions of sensors on the agent (3D vectors)
    :type r_sensor: np.ndarray
    Calculate optic flow based on movement.
    """
    flow = velocity - (r_sensor.T * velocity.dot(r_sensor.T)).T
    flow -= rotary_flow(v_heading, r_sensor)
    return flow


def rotary_flow(v_heading, r_sensor):
    """
    Clockwise rotation
    :param v_heading: agent heading direction (3D vector)
    :type v_heading: np.ndarray
    :param r_sensor: relative directions of sensors on the agent (3D vectors)
    :type r_sensor: np.ndarray
    :return:
    """
    return np.cross(v_heading, r_sensor)


def translatory_flow(r_sensor, r_pref):
    """

    :param r_sensor: relative directions of sensors on the agent (3D vectors)
    :type r_sensor: np.ndarray
    :param r_pref: agent's preferred direction
    :type r_pref: np.ndarray
    :return:
    """
    return np.cross(np.cross(r_sensor, r_pref), r_sensor)


def linear_range_model(t_flow, r_flow, w=1., n=0.):
    """
    Eq 5 in Franz & Krapp
    :param t_flow: translatory flow (wrt preferred direction)
    :type t_flow: np.ndarray
    :param r_flow: image motion flow
    :type r_flow: np.ndarray
    :param w: weight
    :type w: float
    :param n: noise
    :type n: float
    :return:
    """
    return w * ((t_flow * r_flow).sum(axis=1) + n).sum()


def tn_axes(heading, tn_prefs=np.pi/4):
    return np.array([[np.sin(heading - tn_prefs), np.cos(heading - tn_prefs)],
                     [np.sin(heading + tn_prefs), np.cos(heading + tn_prefs)]])


def get_flow(heading, velocity, r_sensors):
    """
    This is the longwinded version that does all the flow calculations,
    piece by piece. It can be refactored down to flow2() so use that for
    performance benefit.
    """
    translation = np.append(velocity, np.zeros(1))
    rotation = np.zeros(3)
    img_flow = image_motion_flow(translation, rotation, r_sensors)
    print "img_flow:", img_flow.shape
    tn_pref = tn_axes(heading)
    print "tn_axes:", tn_pref.shape

    flow_tn_1 = translatory_flow(r_sensors, tn_pref[0])
    print "trans_flow_1:", flow_tn_1.shape
    flow_tn_2 = translatory_flow(r_sensors, tn_pref[1])
    print "trans_flow_2:", flow_tn_2.shape

    lr_1 = linear_range_model(flow_tn_1, img_flow, w=.1)
    lr_2 = linear_range_model(flow_tn_2, img_flow, w=.1)

    return np.array([lr_1, lr_2])


if __name__ == "__main__":
    from world import load_world, load_routes, Hybrid
    from world.utils import shifted_datetime
    from agent import CXAgent, create_agent_name

    exps = [
        (False, False, False, True, None),     # fixed-no-pol-rgb
    ]

    for update_sky, uniform_sky, enable_pol, rgb, rng in exps:
        date = shifted_datetime()
        if rng is None:
            rng = np.random.RandomState(2018)
        RND = rng
        fov = (-np.pi/2, np.pi/2)
        # fov = (-np.pi/6, np.pi/2)
        sky_type = "uniform" if uniform_sky else "live" if update_sky else "fixed"
        if not enable_pol and "uniform" not in sky_type:
            sky_type += "-no-pol"
        if rgb:
            sky_type += "-rgb"
        step = .1       # 10 cm
        tau_phi = np.pi    # 60 deg
        condition = Hybrid(tau_x=step, tau_phi=tau_phi)
        agent_name = create_agent_name(date, sky_type, step, fov[0], fov[1])
        print agent_name

        world = load_world()
        world.enable_pol_filters(enable_pol)
        world.uniform_sky = uniform_sky
        routes = load_routes()
        world.add_route(routes[0])

        agent = CXAgent(condition=condition, live_sky=update_sky,
                        rgb=rgb, fov=fov, name=agent_name)
        agent.set_world(world)

        rt = agent.homing_routes[0]

        # 2D route
        xyphi = np.array([[x, y, phi] for x, y, _, phi in rt])
        xy = xyphi[:, :2]
        phi = (xyphi[:, 2] + np.pi) % (2 * np.pi) - np.pi  # type: np.ndarray
        print "xy:", xy.shape

        # velocities in each timestep
        v = np.vstack([np.array([0.0, 0.0]), np.diff(xy, axis=0)])
        print "v: ", v.shape

        xy2 = np.cumsum(v, axis=0) + xy[0]
        print "xy (reconstructed):", xy2.shape
        print "Is reconstruction accurate?", np.allclose(xy, xy2)

        print "__phi_z:", phi.shape
        x, y = xy[:, 0], xy[:, 1]
        phi2 = np.roll((np.arctan2(v[:, 0], -v[:, 1]) + np.pi) % (2 * np.pi) - np.pi, -1)
        print "__phi_z (reconstructed):", phi2.shape
        print "Is reconstruction accurate?", np.allclose(phi[:-1], phi2[:-1])

        n_sensors = 200
        directions = np.linspace(-np.pi, np.pi, n_sensors, endpoint=False)
        D = np.column_stack((np.sin(directions), np.cos(directions),
                             np.zeros(n_sensors)))

        print get_flow(0., np.array([0., 1.]), D) * .1
