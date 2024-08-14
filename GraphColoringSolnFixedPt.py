import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.proc.dense.process import Dense
from lava.magma.core.model.sub.model import AbstractSubProcessModel

from cordic_fxp import cordic_16_32_vec as cordic

class OScillatoryNeuronFixedPoint(AbstractProcess):
    """
    This Process defines the oscillatory neurons. It maintains as a state variable, the phase angles of the neurons.
    It receives phase update values as input and sends out updated phase angles as output
    
    Only B matrix, C matrix excluded
    """
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        self.B_cos_in = InPort(shape=shape)
        self.B_sin_in = InPort(shape=shape)
        self.cos_out = OutPort(shape=shape)
        self.sin_out = OutPort(shape=shape)
        self.phi = Var(shape=shape, init=kwargs.get("phi", 0))
        self.sigma = Var(shape=(1,), init=kwargs.get("sigma", 1))
        self.decay = Var(shape=(1,), init=kwargs.get("decay", 1))
        self.frac_bits = Var(shape=(1,), init=kwargs.get("frac_bits", 13))
        self.K_fxp = Var(shape=(1,), init=kwargs.get("K_fxp", 1))
        self.fxp_pi = Var(shape=(1,), init=kwargs.get("fxp_pi", 1))
        self.atan_table = Var(shape=(13,), init=kwargs.get("atan_table", 0))
        self.lrc = Var(shape=shape, init=kwargs.get("lrc", 0))
        self.update_val = Var(shape=shape)

class GraphColorUpdateBasic(AbstractProcess):
    """
    This Process defines the connections between the neurons. It stores the adjacency matrix as state,
    receives phase angles as inputs, and then computes and outputs phase update values
    
    Only B matrix, C matrix excluded
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        shape_mat = kwargs.get("shape_mat", (1,1))
        B = kwargs.get("B", 0)
        self.cos_in = InPort(shape=shape)
        self.sin_in = InPort(shape=shape)
        self.B_cos_out = OutPort(shape=shape)
        self.B_sin_out = OutPort(shape=shape)
        self.B = Var(shape=shape_mat, init=B)

@implements(proc=OScillatoryNeuronFixedPoint, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
#@tag("bit_approximate_loihi", "fixed_pt")
class PyOscNeuronBasicFixedPointModel(PyLoihiProcessModel):
    B_cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    B_sin_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    cos_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    sin_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    phi: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    sigma: int = LavaPyType(int, int)
    decay: int = LavaPyType(int, int)
    frac_bits: int = LavaPyType(int, int)
    K_fxp: int = LavaPyType(int, int)
    fxp_pi: int = LavaPyType(int, int)
    atan_table: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    lrc: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    update_val: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    def run_spk(self):
        #self.phi = np.int32(self.phi)
        #self.lrc = np.int32(self.lrc)
        #fxp_half_pi = np.int32(self.fxp_pi>>1)
        fxp_half_pi = (self.fxp_pi>>1)
        
        self.sigma = (self.decay*self.sigma)>>self.frac_bits
        updated_phi = self.phi + np.int32(np.random.normal(0, self.sigma, size=self.phi.shape))
        updated_phi2 = self.phi + 0
        #print("self.phi.dtype: ", self.phi.dtype)
        #print("updated_phi.dtype: ", updated_phi.dtype)
        
        #Ensuring correct range for updated_phi
        updated_phi[updated_phi>self.fxp_pi] = updated_phi[updated_phi>self.fxp_pi] - self.fxp_pi - self.fxp_pi
        updated_phi[updated_phi<-self.fxp_pi] = updated_phi[updated_phi<-self.fxp_pi] + self.fxp_pi + self.fxp_pi
        
        updated_phi2[updated_phi2>self.fxp_pi] = updated_phi2[updated_phi>self.fxp_pi] - self.fxp_pi - self.fxp_pi
        updated_phi2[updated_phi2<-self.fxp_pi] = updated_phi2[updated_phi<-self.fxp_pi] + self.fxp_pi + self.fxp_pi
        
        updated_phi_cp2 = self.phi + np.int32(0)
        updated_phi_cp = updated_phi + np.int32(0)
        
        #print("updated_phi_cp.dtype: ", updated_phi_cp.dtype)
        
        #Input angle values must lie between -pi/2 - pi/2
        #Following correction will give cos and sin with opposite sign for values outside -pi/2 -> pi/2
        updated_phi[updated_phi>fxp_half_pi] = -self.fxp_pi + updated_phi[updated_phi>fxp_half_pi]
        updated_phi[updated_phi<-fxp_half_pi] = self.fxp_pi + updated_phi[updated_phi<-fxp_half_pi]
        
        updated_phi2[updated_phi2>fxp_half_pi] = -self.fxp_pi + updated_phi2[updated_phi2>fxp_half_pi]
        updated_phi2[updated_phi2<-fxp_half_pi] = self.fxp_pi + updated_phi2[updated_phi2<-fxp_half_pi]
        
        #print("updated_phi.dtype (2): ", updated_phi.dtype)
        
        cos_phi, sin_phi = cordic(updated_phi, self.atan_table, self.frac_bits)
        cos_phi = (self.K_fxp*cos_phi)>>self.frac_bits
        sin_phi = (self.K_fxp*sin_phi)>>self.frac_bits
        
        #print("cos_phi.dtype: ", cos_phi.dtype)
        #print("sin_phi.dtype: ", sin_phi.dtype)
        
        #correcting the sign for values outside -pi/2 -> pi/2
        cos_phi[updated_phi_cp>fxp_half_pi] = -cos_phi[updated_phi_cp>fxp_half_pi]
        sin_phi[updated_phi_cp>fxp_half_pi] = -sin_phi[updated_phi_cp>fxp_half_pi]
        cos_phi[updated_phi_cp<-fxp_half_pi] = -cos_phi[updated_phi_cp<-fxp_half_pi]
        sin_phi[updated_phi_cp<-fxp_half_pi] = -sin_phi[updated_phi_cp<-fxp_half_pi]
        
        #print("cos_phi.dtype (2): ", cos_phi.dtype)
        #print("sin_phi.dtype (2): ", sin_phi.dtype)
        
        #send values
        self.cos_out.send(cos_phi)
        self.sin_out.send(sin_phi)
        #receive values
        #Bcos_phi = (self.B_cos_in.recv()) #This output is coming as float64
        #Bsin_phi = (self.B_sin_in.recv()) #This output is coming as float64
        Bcos_phi = np.int32(self.B_cos_in.recv()) #16-bit overflow DOES happen, but it is managed by int32
        Bsin_phi = np.int32(self.B_sin_in.recv())
        
        #print("B_cos_in.dtype: ", Bcos_phi.dtype)
        #print("B_sin_in.dtype: ", Bsin_phi.dtype)
        
        cost_term = ((Bcos_phi*sin_phi)>>self.frac_bits) - ((Bsin_phi*cos_phi)>>self.frac_bits)
        cost_update = (self.lrc*cost_term)>>self.frac_bits
        self.update_val = cost_update
        
        #print("cost_term.dtype: ", cost_term.dtype)
        #print("cost_update.dtype: ", cost_update.dtype)
        
        #update phi
        self.phi = updated_phi_cp + cost_update
        self.phi[self.phi>self.fxp_pi] = self.phi[self.phi>self.fxp_pi] - self.fxp_pi - self.fxp_pi
        self.phi[self.phi<-self.fxp_pi] = self.phi[self.phi<-self.fxp_pi] + self.fxp_pi + self.fxp_pi
        #print("self.phi.dtype (final): ", self.phi.dtype)

@implements(proc=GraphColorUpdateBasic, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
#@tag("bit_approximate_loihi", "fixed_pt")
class PyGraphColorUpdateBasicModel(AbstractSubProcessModel):
    def __init__(self, proc):
        B = proc.proc_params.get("B")
        
        self.dense_Bcos = Dense(weights=B, num_message_bits=24)
        self.dense_Bsin = Dense(weights=B, num_message_bits=24)
        
        proc.in_ports.cos_in.connect(self.dense_Bcos.in_ports.s_in)
        proc.in_ports.sin_in.connect(self.dense_Bsin.in_ports.s_in)
        
        self.dense_Bcos.out_ports.a_out.connect(proc.out_ports.B_cos_out)
        self.dense_Bsin.out_ports.a_out.connect(proc.out_ports.B_sin_out)






