from enum import Enum
from signal_names import OxfordNanonisSignalNames
tramea_signals = OxfordNanonisSignalNames

from nanonisTCP import nanonisTCP
from nanonisTCP.Signals import Signals
from nanonisTCP.Swp1D import Swp1D
from nanonisTCP.UserOut import UserOut
import pandas as pd

class measurement_control:
    """
    Class to define building block actions that can be performed for a measurement.
    
    Setups up modules for signals, sweeps, and user outputs.
    """
    def __init__(self, NTCP: nanonisTCP):
        self.nanonisTCP = NTCP
        self.version = NTCP.version
        
        sig = Signals(self.nanonisTCP)
        swp = Swp1D(self.nanonisTCP)
        out = UserOut(self.nanonisTCP)
        self.mod_sig = sig
        self.mod_swp = swp
        self.mod_out = out
        
        # Collect the names of the measurement signals.
        meas_signals = sig.MeasNamesGet()
        # Collect the names of the sweep signals:
        _, sweep_signals = swp.SwpSignalGet()
        
        # Each measurement channel has its own `measurement` signal number.
        self.signals_meas = {name: i
                     for i, name in enumerate(meas_signals)}
        self.signals_sweep = sweep_signals
        
        # print("Signals:\t", self.sigs)
        print("Measurements:\t", self.signals_meas, "\n")
        print("Sweepers:\t", self.signals_sweep, "\n")
    
    def measurement_index(self, channel_name: str | OxfordNanonisSignalNames) -> int:
        if isinstance(channel_name, OxfordNanonisSignalNames):
            channel_name = channel_name.value
        if channel_name in self.signals_meas:
            return self.signals_meas[channel_name]
        else:
            raise ValueError(f"Channel {channel_name} is not defined in the measurement channels list.")
        
    def get_sweep_signal(self) -> str | OxfordNanonisSignalNames:
        name, names = self.mod_swp.SwpSignalGet()
        if name in OxfordNanonisSignalNames._value2member_map_:
            return OxfordNanonisSignalNames(name)
        else:
            return name
        
    def set_sweep_signal(self, signal_name: str | OxfordNanonisSignalNames):
        """
        Sets the sweep signal of the 1D Sweeper module (1DSwp.SwpSignalSet).
        """
        if isinstance(signal_name, OxfordNanonisSignalNames):
            signal_name = signal_name.value
        if signal_name not in self.signals_sweep:
            raise ValueError(f"Signal {signal_name} is not defined in the available measurement channels list.")
        else:
            self.mod_swp.SwpSignalSet(signal_name)    
        
    def get_acquisition_channels(self) -> list[str | OxfordNanonisSignalNames]:
        """
        Returns the list of acquisition channels.
        """
        channels = self.mod_swp.AcqChsGet()
        channels = [
            OxfordNanonisSignalNames(channel)
            if channel in OxfordNanonisSignalNames._value2member_map_
            else channel
            for channel in channels
        ]
        return 
        
    def set_acquisition_channels(self, channels: list[OxfordNanonisSignalNames] | list[int] | list[str]):
        dtype = type(channels[0])
        for channel in channels:
            if type(channel) != dtype:
                raise ValueError("All channels must be of the same type.")
        
        if dtype == int:
            channel_indexes = channels
        elif dtype == str:
            channel_indexes = [
                self.measurement_index(channel) for channel in channels
                ]
        elif dtype == OxfordNanonisSignalNames:
            channel_indexes = [
                self.measurement_index(channel.value) for channel in channels
                ]
        else:
            raise ValueError("Invalid channel data type.")
        self.mod_swp.AcqChsSet(channel_indexes)
        
    def get_sweep_parameters(self) -> tuple[float, float, int, int, bool, bool, float]:
        """
        Returns the sweep parameters of the 1D Sweeper module (1DSwp.SwpParamsGet).
        
        Returns
        -------
        parameters : tuple[float, float, int, int, bool, bool, float]
            The sweep parameters including:
                Initial settling time in milliseconds.
                
                Maximum slew rate in units/s.
                
                Number of steps.
                
                Period in milliseconds.
                
                Autosave enabled?
                
                Save dialog enabled?
                
                Settling time in milliseconds.
        """
        return self.mod_swp.PropsGet()

        
    def set_sweep_parameters(self, get_data:bool, sweep_direction:int, save_basename:str, reset_signal:bool):
        """
        Sets the sweep parameters of the 1D Sweeper module (1DSwp.SwpParamsSet).
        
        Parameters
        ----------
        initial_settling_time : float
            Initial settling time in milliseconds.
        maximum_slew_rate : float
            Maximum slew rate in units/s.
        number_of_steps : int
            Number of steps.
        period : int
            Period in milliseconds.
        autosave : bool
            Autosave.
        save_dialog : bool
            Save dialog.
        settling_time : float
            Settling time in milliseconds.
        """
        return self.mod_swp.PropsSet(get_data, sweep_direction, save_basename, reset_signal)
        
    # def 
        
        # # CUSTOM LIMITS
        
        # return


if __name__ == "__main__":
    # Connect to the Nanonis controller.
    TCP_IP = '127.0.0.1'
    TCP_PORT = 6501
    NTCP = nanonisTCP(TCP_IP, TCP_PORT)
    control = measurement_control(NTCP)
    set_list = [
        tramea_signals.DC_INPUT1,
        tramea_signals.DC_INPUT2,
        tramea_signals.DC_INPUT3,
        tramea_signals.DC_INPUT4,
        tramea_signals.TIME,
        tramea_signals.HE3_PROBE_TEMPERATURE,
        tramea_signals.MAGNETIC_FIELD,
    ]
    control.set_acquisition_channels(set_list)
    control.set_sweep_signal(tramea_signals.MAGNETIC_FIELD_SETPOINT)
    print(control.get_sweep_parameters())
    # control.set_sweep_parameters(
    #     initial_settling_time=300,
    #     maximum_slew_rate=28.4,
    #     number_of_steps=100,
    #     period=1000,
    #     autosave=True,
    #     save_dialog=False,
    #     settling_time=300
    # )
    
    NTCP.close_connection()