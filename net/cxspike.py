import numpy as np
from brian2 import Network, NeuronGroup, Synapses

from cx import CX, noisy_sigmoid


class CXSpike(CX):

    def __init__(self, *args, **kwargs):
        super(CX, self).__init__(*args, **kwargs)

        eqs = '''dv/dt = (-v) / (10*ms) : 1'''
        eqs_mem = '''
        dv/dt = (-v) / (10*ms) : 1
        mem : 1
        '''

        self.tl2 = NeuronGroup(self.nb_tl2, eqs, threshold='v>1', reset='v=0', method='exact', name="TL2")
        self.cl1 = NeuronGroup(self.nb_cl1, eqs, threshold='v>1', reset='v=0', method='exact', name="CL1")
        self.tb1 = NeuronGroup(self.nb_tb1, eqs, threshold='v>1', reset='v=0', method='exact', name="TB1")
        self.tn1 = NeuronGroup(self.nb_tn1, eqs, threshold='v>1', reset='v=0', method='exact', name="TN1")
        self.tn2 = NeuronGroup(self.nb_tn2, eqs, threshold='v>1', reset='v=0', method='exact', name="TN2")
        self.cpu4 = NeuronGroup(self.nb_cpu4, eqs_mem, threshold='v>1', reset='v=0', method='exact', name="CPU4")
        self.cpu1a = NeuronGroup(self.nb_cpu1 - 2, eqs, threshold='v>1', reset='v=0', method='exact', name="CPU1a")
        self.cpu1b = NeuronGroup(2, eqs, threshold='v>1', reset='v=0', method='exact', name="CPU1a")
        self.motor = NeuronGroup(2, eqs, threshold='v>1', reset='v=0', method='exact', name='motor')

        self.tl22cl1 = Synapses(self.tl2, self.cl1,
                                '''
                                w : 1
                                a : 1
                                b : 1
                                ''',
                                on_pre='''v_post += a * v_pre * w - b''',
                                method='linear')
        self.tl22cl1.w = self.w_tl22cl1
        self.tl22cl1.a = self.cl1_slope
        self.tl22cl1.b = self.cl1_bias
        self.tl22cl1.connect()
        self.cl12tb1 = Synapses(self.cl1, self.tb1,
                                '''
                                w : 1
                                a : 1
                                b : 1
                                p : 1
                                ''',
                                on_pre='''v_post += a * p * v_pre * w - b''',
                                method='linear')
        self.cl12tb1.w = self.w_cl12tb1
        self.cl12tb1.a = self.tb1_slope
        self.cl12tb1.b = self.tb1_bias
        self.cl12tb1.p = .667
        self.cl12tb1.connect()
        self.tb12tb1 = Synapses(self.tb1, self.tb1,
                                '''
                                w : 1
                                a : 1
                                b : 1
                                ''',
                                on_pre='''v_post += a * (1 - p) * v_pre * w - b''',
                                method='linear')
        self.tb12tb1.w = self.w_tb12tb1
        self.tb12tb1.a = self.tb1_slope
        self.tb12tb1.b = self.tb1_bias
        self.tb12tb1.p = .667
        self.tb12tb1.connect()
        self.tb12cpu4 = Synapses(self.tb1, self.cpu4,
                                 '''
                                 w : 1
                                 a : 1
                                 b : 1
                                 g : 1
                                 ''',
                                 on_pre='''v_post *= a * g * (v_pre - 1) * w - b''',
                                 method='linear')
        self.tb12cpu4.w = self.w_tb12cpu4
        self.tb12cpu4.a = self.cpu4_slope
        self.tb12cpu4.b = self.cpu4_bias
        self.tb12cpu4.g = self.gain
        self.tb12cpu4.connect()
        self.tn12cpu4 = Synapses(self.tn1, self.cpu4,
                                 '''
                                 w : 1
                                 a : 1
                                 b : 1
                                 ''',
                                 on_pre='''v_post += a * (0.5 - v_pre) * w - b''',
                                 method='linear')
        self.tn12cpu4.w = self.w_tn2cpu4
        self.tn12cpu4.a = self.cpu4_slope
        self.tn12cpu4.b = self.cpu4_bias
        self.tn12cpu4.connect()
        self.tn22cpu4 = Synapses(self.tn2, self.cpu4,
                                 '''
                                 w : 1
                                 a : 1
                                 b : 1
                                 g : 1
                                 ''',
                                 on_pre='''v_post -= a * 0.25 * g * v_pre * w - b''',
                                 method='linear')
        self.tn22cpu4.w = self.w_tn2cpu4
        self.tn22cpu4.a = self.cpu4_slope
        self.tn22cpu4.b = self.cpu4_bias
        self.tn22cpu4.g = self.gain
        self.tn22cpu4.connect()
        self.cpu42cpu4 = Synapses(self.cpu4, self.cpu4,
                                  '''
                                  w = 1 : 1
                                  ''',
                                  on_pre='''v_post += v_pre * w''')
        self.cpu42cpu4.connect()
        self.tb12cpu1a = Synapses(self.tb1, self.cpu1a,
                                  '''
                                  w : 1
                                  a : 1
                                  b : 1
                                  ''',
                                  on_pre='''v_post += a * (v_pre - 1) * w - b''',
                                  method='linear')
        self.tb12cpu1a.w = self.w_tb12cpu1a
        self.tb12cpu1a.a = self.cpu1_slope
        self.tb12cpu1a.b = self.cpu1_bias
        self.tb12cpu1a.connect()
        self.tb12cpu1b = Synapses(self.tb1, self.cpu1b,
                                  '''
                                  w : 1
                                  a : 1
                                  b : 1
                                  ''',
                                  on_pre='''v_post += a * (v_pre - 1) * w - b''',
                                  method='linear')
        self.tb12cpu1b.w = self.w_tb12cpu1b
        self.tb12cpu1b.a = self.cpu1_slope
        self.tb12cpu1b.b = self.cpu1_bias
        self.tb12cpu1b.connect()
        self.cpu42cpu1a = Synapses(self.cpu4, self.cpu1a,
                                   '''
                                   w : 1
                                   a : 1
                                   b : 1
                                   ''',
                                   on_pre='''v_post += a * v_pre * w - b''',
                                   method='linear')
        self.cpu42cpu1a.w = self.w_cpu42cpu1a
        self.cpu42cpu1a.a = self.cpu1_slope
        self.cpu42cpu1a.b = self.cpu1_bias
        self.cpu42cpu1a.connect()
        self.cpu42cpu1b = Synapses(self.cpu4, self.cpu1b,
                                   '''
                                   w : 1
                                   a : 1
                                   b : 1
                                   ''',
                                   on_pre='''v_post += a * v_pre * w - b''',
                                   method='linear')
        self.cpu42cpu1b.w = self.w_tn2cpu4
        self.cpu42cpu1b.a = self.cpu1_slope
        self.cpu42cpu1b.b = self.cpu1_bias
        self.cpu42cpu1b.connect()
        self.cpu1a2motor = Synapses(self.cpu1a, self.motor,
                                    '''
                                    w : 1
                                    ''',
                                    on_pre='''v_post += v_pre * w''',
                                    method='linear')
        self.cpu1a2motor.w = self.w_cpu1a2motor
        self.cpu1a2motor.connect()
        self.cpu1b2motor = Synapses(self.cpu1b, self.motor,
                                    '''
                                    w : 1
                                    ''',
                                    on_pre='''v_post += v_pre * w''',
                                    method='linear')
        self.cpu1b2motor.w = self.w_cpu1b2motor
        self.tl22cl1.a = self.tl2_slope
        self.tl22cl1.b = self.tl2_bias
        self.cpu1b2motor.connect()

        self.net = Network(
            self.tl2, self.cl1, self.tb1, self.tb1, self.tn2, self.cpu4, self.cpu1a, self.cpu1b, self.motor,
            self.tl22cl1, self.cl12tb1, self.tb12tb1, self.tb12cpu4, self.tn12cpu4, self.tn22cpu4,
            self.tb12cpu4, self.cpu42cpu4, self.tb12cpu1a, self.tb12cpu1b, self.cpu42cpu1a, self.cpu42cpu1b,
            self.cpu1a2motor, self.cpu1b2motor
        )

    def __call__(self, *args, **kwargs):
        pass
