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

class OScillatoryNeuron(AbstractProcess):
    """
    This Process defines the oscillatory neurons. It maintains as a state variable, the phase angles of the neurons.
    It receives phase update values as input and sends out updated phase angles as output
    """
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        self.B_cos_in = InPort(shape=shape)
        self.B_sin_in = InPort(shape=shape)
        self.C_cos_in = InPort(shape=shape)
        self.C_sin_in = InPort(shape=shape)
        self.cos_out = OutPort(shape=shape)
        self.sin_out = OutPort(shape=shape)
        self.phi = Var(shape=shape, init=kwargs.get("phi", 0))
        self.sigma = Var(shape=(1,), init=kwargs.get("sigma", 0.1))
        self.decay = Var(shape=(1,), init=kwargs.get("decay", 1))
        self.lrc = Var(shape=shape, init=kwargs.get("lrc", 0))
        self.lrr = Var(shape=shape, init=kwargs.get("lrr", 0))
        self.update_val = Var(shape=shape)

class OScillatoryNeuronBasic(AbstractProcess):
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
        self.sigma = Var(shape=(1,), init=kwargs.get("sigma", 0.1))
        self.decay = Var(shape=(1,), init=kwargs.get("decay", 1))
        self.period = Var(shape=(1,), init=kwargs.get("period", 100))
        self.step = Var(shape=(1,), init=kwargs.get("step", 0))
        self.lrc = Var(shape=shape, init=kwargs.get("lrc", 0))
        self.update_val = Var(shape=shape)

class OScillatoryNeuronCart(AbstractProcess):
    """
    This Process defines the oscillatory neurons. It maintains as a state variable, the real and imaginary parts of the
    oscillator variable (using cartesian coordinates here). 
    
    Only B matrix, C matrix excluded
    """
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        self.x_in = InPort(shape=shape)
        self.y_in = InPort(shape=shape)
        self.x_out = OutPort(shape=shape)
        self.y_out = OutPort(shape=shape)
        self.x = Var(shape=shape, init=kwargs.get("x", 0))
        self.y = Var(shape=shape, init=kwargs.get("y", 0))
        self.sigma = Var(shape=(1,), init=kwargs.get("sigma", 0.1))
        self.decay = Var(shape=(1,), init=kwargs.get("decay", 1))
        self.period = Var(shape=(1,), init=kwargs.get("period", 100))
        self.step = Var(shape=(1,), init=kwargs.get("step", 0))
        self.lrc = Var(shape=shape, init=kwargs.get("lrc", 0))
        self.n_stable = Var(shape=(1,), init=kwargs.get("n_stable", 1))
        self.update_x = Var(shape=shape)
        self.update_y = Var(shape=shape)

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

class GraphColorUpdate(AbstractProcess):
    """
    This Process defines the connections between the neurons. It stores the adjacency matrix as state,
    receives phase angles as inputs, and then computes and outputs phase update values
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        shape_mat = kwargs.get("shape_mat", (1,1))
        B = kwargs.get("B", 0)
        C = kwargs.get("C", 0)
        self.cos_in = InPort(shape=shape)
        self.sin_in = InPort(shape=shape)
        self.B_cos_out = OutPort(shape=shape)
        self.B_sin_out = OutPort(shape=shape)
        self.C_cos_out = OutPort(shape=shape)
        self.C_sin_out = OutPort(shape=shape)
        self.B = Var(shape=shape_mat, init=B)
        self.C = Var(shape=shape_mat, init=C)

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

class GraphColorUpdateCart(AbstractProcess):
    """
    This Process defines the connections between the neurons. It stores the adjacency matrix as state,
    receives x/|z| and y/|z| as inputs
    
    Only B matrix, C matrix excluded
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        shape_mat = kwargs.get("shape_mat", (1,1))
        B = kwargs.get("B", 0)
        self.x_in = InPort(shape=shape)
        self.y_in = InPort(shape=shape)
        self.B_x_out = OutPort(shape=shape)
        self.B_y_out = OutPort(shape=shape)
        self.B = Var(shape=shape_mat, init=B)

@implements(proc=OScillatoryNeuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyOscNeuronModel(PyLoihiProcessModel):
    B_cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    B_sin_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    C_cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    C_sin_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    cos_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    sin_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    phi: np.ndarray = LavaPyType(np.ndarray, float)
    sigma: float = LavaPyType(float, float)
    decay: float = LavaPyType(float, float)
    lrc: np.ndarray = LavaPyType(np.ndarray, float)
    lrr: np.ndarray = LavaPyType(np.ndarray, float)
    update_val: np.ndarray = LavaPyType(np.ndarray, float)
    def run_spk(self):
        self.sigma = self.decay*self.sigma
        updated_phi = self.phi + np.random.normal(0, self.sigma, size=self.phi.shape)
        cos_phi = np.cos(updated_phi)
        sin_phi = np.sin(updated_phi)
        self.cos_out.send(cos_phi)
        self.sin_out.send(sin_phi)
        Bcos_phi = self.B_cos_in.recv()
        Bsin_phi = self.B_sin_in.recv()
        Ccos_phi = self.C_cos_in.recv()
        Csin_phi = self.C_sin_in.recv()
        cost_term = self.lrc*(Bcos_phi*sin_phi - Bsin_phi*cos_phi)
        reward_term = -self.lrr*(Ccos_phi*sin_phi - Csin_phi*cos_phi)
        #cost_term = 0
        #reward_term = 0
        self.update_val = (cost_term + reward_term)
        self.phi = updated_phi + (cost_term + reward_term)
        self.phi[self.phi>np.pi] = self.phi[self.phi>np.pi] - 2*np.pi
        self.phi[self.phi<-np.pi] = self.phi[self.phi<-np.pi] + 2*np.pi
        #self.cos_out.send(cos_phi)
        #self.sin_out.send(sin_phi)

@implements(proc=OScillatoryNeuronBasic, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyOscNeuronBasicModel(PyLoihiProcessModel):
    B_cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    B_sin_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    cos_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    sin_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    phi: np.ndarray = LavaPyType(np.ndarray, float)
    sigma: float = LavaPyType(float, float)
    decay: float = LavaPyType(float, float)
    period: float = LavaPyType(float, float)
    step: float = LavaPyType(float, float)
    lrc: np.ndarray = LavaPyType(np.ndarray, float)
    update_val: np.ndarray = LavaPyType(np.ndarray, float)
    def run_spk(self):
        self.sigma = self.decay*self.sigma
        updated_phi = self.phi + np.random.normal(0, self.sigma, size=self.phi.shape)
        cos_phi = np.cos(updated_phi)
        sin_phi = np.sin(updated_phi)
        self.cos_out.send(cos_phi)
        self.sin_out.send(sin_phi)
        Bcos_phi = self.B_cos_in.recv()
        Bsin_phi = self.B_sin_in.recv()
        wv = np.sin(self.step*2*np.pi/self.period)
        cost_term = self.lrc*(Bcos_phi*sin_phi - Bsin_phi*cos_phi)
        self.update_val = cost_term
        self.phi = updated_phi + cost_term
        self.phi[self.phi>np.pi] = self.phi[self.phi>np.pi] - 2*np.pi
        self.phi[self.phi<-np.pi] = self.phi[self.phi<-np.pi] + 2*np.pi
        self.step = self.step + 1
        #self.cos_out.send(cos_phi)
        #self.sin_out.send(sin_phi)

@implements(proc=OScillatoryNeuronFixedPoint, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_approximate_loihi", "fixed_pt")
class PyOscNeuronBasicFixedPointModel(PyLoihiProcessModel):
    B_cos_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    B_sin_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    cos_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    sin_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    phi: np.ndarray = LavaPyType(np.ndarray, int)
    sigma: int = LavaPyType(int, int)
    decay: int = LavaPyType(int, int)
    frac_bits: int = LavaPyType(int, int)
    K_fxp: int = LavaPyType(int, int)
    fxp_pi: int = LavaPyType(int, int)
    atan_table: np.ndarray = LavaPyType(np.ndarray, int)
    lrc: np.ndarray = LavaPyType(np.ndarray, int)
    update_val: np.ndarray = LavaPyType(np.ndarray, int)
    def run_spk(self):
        self.phi = np.int32(self.phi)
        self.lrc = np.int32(self.lrc)
        fxp_half_pi = np.int32(self.fxp_pi>>1)
        
        self.sigma = (self.decay*self.sigma)>>self.frac_bits
        updated_phi = self.phi + np.int32(np.random.normal(0, self.sigma, size=self.phi.shape))
        
        print("self.phi.dtype: ", self.phi.dtype)
        print("updated_phi.dtype: ", updated_phi.dtype)
        
        #Ensuring correct range for updated_phi
        updated_phi[updated_phi>self.fxp_pi] = updated_phi[updated_phi>self.fxp_pi] - self.fxp_pi - self.fxp_pi
        updated_phi[updated_phi<-self.fxp_pi] = updated_phi[updated_phi<-self.fxp_pi] + self.fxp_pi + self.fxp_pi
        updated_phi_cp = updated_phi + np.int32(0)
        
        print("updated_phi_cp.dtype: ", updated_phi_cp.dtype)
        
        #Input angle values must lie between -pi/2 - pi/2
        #Following correction will give cos and sin with opposite sign for values outside -pi/2 -> pi/2
        updated_phi[updated_phi>fxp_half_pi] = -self.fxp_pi + updated_phi[updated_phi>fxp_half_pi]
        updated_phi[updated_phi<-fxp_half_pi] = self.fxp_pi + updated_phi[updated_phi<-fxp_half_pi]
        
        print("updated_phi.dtype (2): ", updated_phi.dtype)
        
        cos_phi, sin_phi = cordic(updated_phi, self.atan_table, self.frac_bits)
        cos_phi = (self.K_fxp*cos_phi)>>self.frac_bits
        sin_phi = (self.K_fxp*sin_phi)>>self.frac_bits
        
        print("cos_phi.dtype: ", cos_phi.dtype)
        print("sin_phi.dtype: ", sin_phi.dtype)
        
        #correcting the sign for values outside -pi/2 -> pi/2
        cos_phi[updated_phi_cp>fxp_half_pi] = -cos_phi[updated_phi_cp>fxp_half_pi]
        sin_phi[updated_phi_cp>fxp_half_pi] = -sin_phi[updated_phi_cp>fxp_half_pi]
        cos_phi[updated_phi_cp<-fxp_half_pi] = -cos_phi[updated_phi_cp<-fxp_half_pi]
        sin_phi[updated_phi_cp<-fxp_half_pi] = -sin_phi[updated_phi_cp<-fxp_half_pi]
        
        print("cos_phi.dtype (2): ", cos_phi.dtype)
        print("sin_phi.dtype (2): ", sin_phi.dtype)
        
        #send values
        self.cos_out.send(cos_phi)
        self.sin_out.send(sin_phi)
        #receive values
        Bcos_phi = (self.B_cos_in.recv())
        Bsin_phi = (self.B_sin_in.recv())
        #Bcos_phi = np.int32(self.B_cos_in.recv())
        #Bsin_phi = np.int32(self.B_sin_in.recv())
        
        print("B_cos_in.dtype: ", Bcos_phi.dtype)
        print("B_sin_in.dtype: ", Bsin_phi.dtype)
        
        cost_term = ((Bcos_phi*sin_phi)>>self.frac_bits) - ((Bsin_phi*cos_phi)>>self.frac_bits)
        cost_update = (self.lrc*cost_term)>>self.frac_bits
        self.update_val = cost_update
        
        print("cost_term.dtype: ", cost_term.dtype)
        print("cost_update.dtype: ", cost_update.dtype)
        
        #update phi
        self.phi = updated_phi_cp + cost_update
        self.phi[self.phi>self.fxp_pi] = self.phi[self.phi>self.fxp_pi] - self.fxp_pi - self.fxp_pi
        self.phi[self.phi<-self.fxp_pi] = self.phi[self.phi<-self.fxp_pi] + self.fxp_pi + self.fxp_pi
        print("self.phi.dtype (final): ", self.phi.dtype)

sigma_period = 300

@implements(proc=OScillatoryNeuronCart, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyOscNeuronCartModel(PyLoihiProcessModel):
    x_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    y_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    x_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    y_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    x: np.ndarray = LavaPyType(np.ndarray, float)
    y: np.ndarray = LavaPyType(np.ndarray, float)
    sigma: float = LavaPyType(float, float)
    decay: float = LavaPyType(float, float)
    period: float = LavaPyType(float, float)
    step: float = LavaPyType(float, float)
    n_stable: float = LavaPyType(float, float)
    lrc: np.ndarray = LavaPyType(np.ndarray, float)
    update_x: np.ndarray = LavaPyType(np.ndarray, float)
    update_y: np.ndarray = LavaPyType(np.ndarray, float)
    def run_spk(self):
        #Compute update
        mod_z = ((self.x**2) + (self.y**2))**0.5
        phi = np.arctan2(self.y, self.x)
        self.x_out.send(self.x/mod_z)
        self.y_out.send(self.y/mod_z)
        xj_in = self.x_in.recv()
        yj_in = self.y_in.recv()
        x_update = self.lrc*((((self.x**2) - (self.y**2))/(mod_z**3)) - (1/mod_z))*xj_in + self.lrc*((2*self.x*self.y)/(mod_z**3))*yj_in
        y_update = -self.lrc*((((self.x**2) - (self.y**2))/(mod_z**3)) + (1/mod_z))*yj_in + self.lrc*((2*self.x*self.y)/(mod_z**3))*xj_in
        x_update_mod = -self.lrc*((mod_z**2) - 1)*self.x
        y_update_mod = -self.lrc*((mod_z**2) - 1)*self.y
        x_update_stable = self.lrc*self.n_stable*(self.y/mod_z)*np.sin(self.n_stable*phi)
        y_update_stable = -self.lrc*self.n_stable*(self.x/mod_z)*np.sin(self.n_stable*phi)
        self.update_x = x_update + x_update_mod + x_update_stable
        self.update_y = y_update + y_update_mod + y_update_stable
        self.x = self.x + self.update_x
        self.y = self.y + self.update_y
        #Add noise
        if self.step%sigma_period == sigma_period-1:
            self.sigma = self.sigma/2
        mod_z = ((self.x**2) + (self.y**2))**0.5
        phi = np.arctan2(self.y, self.x)
        updated_phi = phi + np.random.normal(0, self.sigma, size=phi.shape)
        self.x = mod_z*np.cos(updated_phi)
        self.y = mod_z*np.sin(updated_phi)
        self.step = self.step + 1

@implements(proc=GraphColorUpdate, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyGraphColorUpdateModel(AbstractSubProcessModel):
    def __init__(self, proc):
        B = proc.proc_params.get("B")
        C = proc.proc_params.get("C")
        self.dense_Bcos = Dense(weights=B, num_message_bits=24)
        self.dense_Bsin = Dense(weights=B, num_message_bits=24)
        self.dense_Ccos = Dense(weights=C, num_message_bits=24)
        self.dense_Csin = Dense(weights=C, num_message_bits=24)
        
        proc.in_ports.cos_in.connect(self.dense_Bcos.in_ports.s_in)
        proc.in_ports.sin_in.connect(self.dense_Bsin.in_ports.s_in)
        proc.in_ports.cos_in.connect(self.dense_Ccos.in_ports.s_in)
        proc.in_ports.sin_in.connect(self.dense_Csin.in_ports.s_in)
        
        self.dense_Bcos.out_ports.a_out.connect(proc.out_ports.B_cos_out)
        self.dense_Bsin.out_ports.a_out.connect(proc.out_ports.B_sin_out)
        self.dense_Ccos.out_ports.a_out.connect(proc.out_ports.C_cos_out)
        self.dense_Csin.out_ports.a_out.connect(proc.out_ports.C_sin_out)

@implements(proc=GraphColorUpdateBasic, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_approximate_loihi", "fixed_pt")
class PyGraphColorUpdateBasicModel(AbstractSubProcessModel):
    def __init__(self, proc):
        B = proc.proc_params.get("B")

        self.dense_Bcos = Dense(weights=B, num_message_bits=32)
        self.dense_Bsin = Dense(weights=B, num_message_bits=32)
        
        proc.in_ports.cos_in.connect(self.dense_Bcos.in_ports.s_in)
        proc.in_ports.sin_in.connect(self.dense_Bsin.in_ports.s_in)
        
        self.dense_Bcos.out_ports.a_out.connect(proc.out_ports.B_cos_out)
        self.dense_Bsin.out_ports.a_out.connect(proc.out_ports.B_sin_out)

@implements(proc=GraphColorUpdateCart, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyGraphColorUpdateCartModel(AbstractSubProcessModel):
    def __init__(self, proc):
        B = proc.proc_params.get("B")

        self.dense_B_x = Dense(weights=B, num_message_bits=32)
        self.dense_B_y = Dense(weights=B, num_message_bits=32)
        
        proc.in_ports.x_in.connect(self.dense_B_x.in_ports.s_in)
        proc.in_ports.y_in.connect(self.dense_B_y.in_ports.s_in)
        
        self.dense_B_x.out_ports.a_out.connect(proc.out_ports.B_x_out)
        self.dense_B_y.out_ports.a_out.connect(proc.out_ports.B_y_out)






