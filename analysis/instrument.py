from dataclasses import dataclass
import numpy as np
from .xrd import caglioti_fwhm

@dataclass
class InstrumentConfig:
    name: str = 'Generic Cu Kα diffractometer'
    wavelength_A: float = 1.5406
    U: float = 0.0000
    V: float = 0.0000
    W: float = 0.0100
    zero_shift_deg: float = 0.0
    radiation: str = 'Cu Kα'
    def instrumental_fwhm(self,two_theta_deg):
        return caglioti_fwhm(two_theta_deg,self.U,self.V,self.W)
    def corrected_two_theta(self,two_theta_obs_deg):
        return np.asarray(two_theta_obs_deg,float)-self.zero_shift_deg
    def as_dict(self):
        return self.__dict__.copy()