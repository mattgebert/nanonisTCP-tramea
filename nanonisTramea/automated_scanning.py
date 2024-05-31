MAGNETIC_FIELD_SIGNAL_NAME = ""

from nanonisTCP import nanonisTCP

class measurementActions:
    """
    Class to define building block actions that can be performed for a measurement.
    """
    
    def __init__(self,NanonisTCP: nanonisTCP):
        self.nanonisTCP = NanonisTCP
        self.version = NanonisTCP.version
    
    def magnetic_field_setpoint(self, value):
        """
        Set the magnetic field setpoint.
        """
        self._set_signal_value(MAGNETIC_FIELD_SIGNAL_NAME, value)