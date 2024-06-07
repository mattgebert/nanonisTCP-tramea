from enum import Enum
from signal_names import OxfordNanonisSignalNames
tramea_signals = OxfordNanonisSignalNames

from nanonisTCP import nanonisTCP
from nanonisTCP.Signals import Signals
from nanonisTCP.Swp1D import Swp1D
from nanonisTCP.UserOut import UserOut
import pandas as pd
import datetime as dt
import pytz as tz
import sys, os
import time

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
        self.current_sweeper, sweep_signals = swp.SwpSignalGet()
        
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
            self.current_sweeper = OxfordNanonisSignalNames(name)
        else:
            self.current_sweeper = name
        return self.current_sweeper
        
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

        
    def set_sweep_parameters(self, 
                             initial_settling_time: float,
                             maximum_slew_rate: float,
                             number_of_steps: int,
                             period: int,
                             autosave: bool,
                             save_dialog: bool,
                             settling_time: float):
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
        return self.mod_swp.PropsSet(
            initial_settling_time=initial_settling_time,
            maximum_slew_rate=maximum_slew_rate,
            number_of_steps=number_of_steps,
            period=period,
            autosave=autosave,
            save_dialog=save_dialog,
            settling_time=settling_time
        )
    
    def _check_limits(self, limits: tuple[float, float], hard_limits: tuple[float, float]) -> bool:
        """
        Check if the limits are within the hard limits.
        
        Parameters
        ----------
        limits : tuple[float, float]
            The limits to check.
        hard_limits : tuple[float, float]
            The hard limits.
            
        Returns
        -------
        bool
            True if the limits are valid inside the hard limits.
        """
        if (limits[0] < hard_limits[0] or limits[1] > hard_limits[1]
                or limits[0] > limits[1]
            ):
            return False
        else:
            return True
    
    def set_limits(self, limits: tuple[float, float]) -> None:
        """
        Sets the limits of the 1D Sweeper module (1DSwp.LimitsSet).
        
        Parameters
        ----------
        limits : tuple[float, float]
            The sweeper signal limits, lower then upper.
        """
        if self.current_sweeper is tramea_signals.MAGNETIC_FIELD_SETPOINT:
            hard_limits = (-14,14)
            if not self._check_limits(limits, hard_limits):
                raise ValueError(f"Field limits must be within {hard_limits}, and limits[0] < limits[1].")
        elif self.current_sweeper is tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT:
            hard_limits = (0, 300)
            if not self._check_limits(limits, hard_limits):
                raise ValueError(f"Temperature limits must be within {hard_limits}, and limits[0] < limits[1].")
        elif (self.current_sweeper is tramea_signals.DC_INPUT1
              or self.current_sweeper is tramea_signals.DC_INPUT2
              or self.current_sweeper is tramea_signals.DC_INPUT3
              or self.current_sweeper is tramea_signals.DC_INPUT4
              ):
            hard_limits = (-10, 10)
            if not self._check_limits(limits, hard_limits):
                raise ValueError(f"DC Voltage limits must be within {hard_limits}, and limits[0] < limits[1].")
        elif (self.current_sweeper is tramea_signals.AC_OUTPUT1_AMP
              or self.current_sweeper is tramea_signals.AC_OUTPUT2_AMP
              or self.current_sweeper is tramea_signals.AC_OUTPUT3_AMP
              or self.current_sweeper is tramea_signals.AC_OUTPUT4_AMP  
              ):
            hard_limits = (0, 2)
            if not self._check_limits(limits, hard_limits):
                raise ValueError(f"AC Voltage limits must be within {hard_limits}, and limits[0] < limits[1].")
        self.mod_swp.LimitsSet(*limits)
        return
    
    def get_limits(self) -> tuple[float, float]:
        """
        Returns the limits of the 1D Sweeper module (1DSwp.LimitsGet).
        
        Returns
        -------
        limits : tuple[float, float]
            The sweeper signal limits. Lower then upper.
        """
        return self.mod_swp.LimitsGet()
    
    def start(self, get_data: bool = True, sweep_direction: int = 1, save_basename: str = "test_TCP_data", reset_signal: bool = False) -> tuple[int, list[str], pd.DataFrame]:
        """
        Starts the 1D Sweeper module (1DSwp.Start).
        
        Parameters
        ----------
        get_data : bool
            Get data.
        sweep_direction : int
            Sweep direction. 1 for forward, 0 for reverse.
        save_basename : str
            Save basename.
        reset_signal : bool
            Reset signal.
        
        Returns
        -------
        sweep : tuple[int, list[str], pd.DataFrame]
            The sweep data including:
                Channel name size.
                
                Channel names.
                
                Data.
        """
        if sweep_direction not in [0, 1]:
            raise ValueError("Sweep direction must be forward (1) or reverse (0).")
        
        return self.mod_swp.Start(
            get_data=get_data,
            sweep_direction=sweep_direction,
            save_basename=save_basename,
            reset_signal=reset_signal
        )

aus_timezone = tz.timezone('Australia/Melbourne')
initial_he3_condensation = dt.datetime(
    year=2024,
    month=6,
    day=8,
    hour=8,
    minute=0,
    tzinfo=aus_timezone
)

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
    field_measurement_params = {
        "initial_settling_time" : 300,
        "maximum_slew_rate" : 28.4,
        "number_of_steps" : 100,
        "period" : 1000,
        "autosave" : True,
        "save_dialog" : False,
        "settling_time" : 300
    }
    temperature_setpoint_params = {
        "initial_settling_time" : 300,
        "maximum_slew_rate" : 28.4,
        "number_of_steps" : 100,
        "period" : 1000,
        "autosave" : False,
        "save_dialog" : False,
        "settling_time" : 300
    }
    
    print(control.get_sweep_parameters())
    control.set_sweep_parameters(**field_measurement_params)
    
    control.set_sweep_signal(tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT)
    control.set_limits((0, 0.1))
    
    #### Use a scan to perform a setpoint change.
    t_start = dt.datetime.now(aus_timezone)
    # control.start(
    #     get_data=False,
    #     sweep_direction=0,
    #     save_basename="test_TCP_data",
    #     reset_signal=False
    # )
    # while dt.datetime.now(aus_timezone) - t_start < dt.timedelta(hours=3):
    #     time.sleep(30)
    
    # print(dt.datetime.now())
    
    NTCP.close_connection()