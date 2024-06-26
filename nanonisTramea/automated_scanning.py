from signal_names import OxfordNanonisSignalNames
tramea_signals = OxfordNanonisSignalNames
from nanonisTCP import nanonisTCP
from nanonisTCP.Signals import Signals
from nanonisTCP.Swp1D import Swp1D
from nanonisTCP.UserOut import UserOut
from nanonisTCP.LockIn import LockIn
import warnings
import pandas as pd
import datetime as dt
import numpy as np
import time
import numpy.typing as npt
import asyncio
from typing import Self

# Create dictionaries to convert between custom sensor and setpoint signals.
SENSOR_TO_SETPOINTS = {
    tramea_signals.HE3_PROBE_TEMPERATURE : tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT,
    tramea_signals.MAGNETIC_FIELD : tramea_signals.MAGNETIC_FIELD_SETPOINT,
    tramea_signals.HE4_VTI_TEMPERATURE: tramea_signals.HE4_VTI_TEMPERATURE_SETPOINT,
    tramea_signals.HE4_EXTRA_TEMP_SENSOR: tramea_signals.HE4_EXTRA_TEMP_SENSOR_SETPOINT
}
if len(SENSOR_TO_SETPOINTS) != len(set(SENSOR_TO_SETPOINTS.values())):
    raise ValueError("Duplicate setpoint signals defined in SENSOR_TO_SETPOINTS.")
SETPOINTS_TO_SENSORS = {
    val: key
    for key, val in SENSOR_TO_SETPOINTS.items()
}

#Pre-defined base temperatures
_BASE_TEMPERATURES = {
    tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT:0.256,
    tramea_signals.HE4_VTI_TEMPERATURE_SETPOINT:1.7,
    tramea_signals.HE4_EXTRA_TEMP_SENSOR_SETPOINT:1.25
}

#Pre-defined limits for various signal ranges
_LIMITS = {
    tramea_signals.MAGNETIC_FIELD_SETPOINT:(-14,14),
    tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT:(0, 300),
    tramea_signals.HE4_VTI_TEMPERATURE_SETPOINT:(1.25, 300),
    tramea_signals.HE4_EXTRA_TEMP_SENSOR_SETPOINT:(1.25, 300),
    tramea_signals.DC_OUTPUT1:(-10, 10),
    tramea_signals.DC_OUTPUT2:(-10, 10),
    tramea_signals.DC_OUTPUT3:(-10, 10),
    tramea_signals.DC_OUTPUT4:(-10, 10),
    tramea_signals.DC_OUTPUT5:(-10, 10),
    tramea_signals.DC_OUTPUT6:(-10, 10),
    tramea_signals.DC_OUTPUT7:(-10, 10),
    tramea_signals.DC_OUTPUT8:(-10, 10),
    tramea_signals.AC_OUTPUT1_AMP:(0, 5),
    tramea_signals.AC_OUTPUT2_AMP:(0, 5),
    tramea_signals.AC_OUTPUT3_AMP:(0, 5),
    tramea_signals.AC_OUTPUT4_AMP:(0, 5),
}

class measurement_control:
    """
    Class to define building block actions that can be performed for a measurement.
    
    As this class operates on an async communicator, this class needs to be 
    initialised by by creating with `await measurement_control.create(NTCP)`.
    
    """
    def __init__(self, NTCP: nanonisTCP):
        self.nanonisTCP = NTCP
        self.version = NTCP.version
        self._mod_sig = None
        self._mod_swp = None
        self._mod_out = None
        self._mod_lock = None
        self._signals_meas = None
        self._signals_sweep = None
        self._signals_sig = None
        self._async_timeout_params = None
    
    @classmethod
    async def create(cls, NTCP: nanonisTCP) -> Self:
        """
        Creates a measurement control object.
        """
        self = cls(NTCP)
        assert(isinstance(self, measurement_control))
        self._async_timeout_params = None
        
        
        sig = Signals(self.nanonisTCP)
        swp = Swp1D(self.nanonisTCP)
        out = UserOut(self.nanonisTCP)
        lock = LockIn(self.nanonisTCP)
        self._mod_sig = sig
        self._mod_swp = swp
        self._mod_out = out
        self._mod_lock = lock
        
        # Collect the names of the signals used by the signal module.
        sig_signals = await sig.NamesGet()
        # Collect each data signal:
        self._signals_sig = {
            name: index
            for index, name in enumerate(sig_signals)
        }
        print("Signals:\t", self.signals_sig, "\n")
        # Collect the names of the signals used in the measurement windows.
        meas_signals = await sig.MeasNamesGet()
        # Each measurement channel has its own `measurement` signal number.
        self._signals_meas = {name: i
                     for i, name in enumerate(meas_signals)}
        print("Measurements:\t", self.signals_meas, "\n")
        # Collect the names of the sweep signals:
        self.current_sweeper, self._signals_sweep = await swp.SwpSignalGet()
        print("Sweepers:\t", self.signals_sweep, "\n")
        if self.current_sweeper == "":
            await swp.SwpSignalSet(self.signals_sweep[0])
        return self
        
    @property 
    def mod_swp(self) -> Swp1D:
        """
        Returns the 1D Sweeper module.
        """
        return self._mod_swp
    
    @property 
    def mod_sig(self) -> Signals:
        """
        Returns the Signals module.
        """
        return self._mod_sig
    
    @property
    def mod_out(self) -> UserOut:
        """
        Returns the UserOutput module (DC Tramea outputs).
        """
        return self._mod_out
    
    @property
    def mod_lock(self) -> LockIn:
        """
        Returns the LockIn module (AC Tramea outputs).
        """
        return self._mod_lock
    
    @property
    def signals_meas(self) -> dict[str, int]:
        """
        Returns the measurement signals.
        
        Measurement signals are signals that can be measured through other routines,
        such as Swp1D.
        """
        return self._signals_meas
    
    @property
    def signals_sweep(self) -> list[str]:
        """
        Returns the sweep signals.
        
        Sweep signals are signals that can be used to sweep a parameter.
        """
        return self._signals_sweep
    
    @property
    def signals_sig(self) -> dict[str, int]:
        """
        Returns the full set of data signals.
        
        Each signal has a unique index that can be used to directly measure the signal.
        These signals are used to directly measure sensor data. They are not always 
        present in the measurement channels, but most are.
        """
        return self._signals_sig
    
        
    async def measurement_index(self, channel_name: str | OxfordNanonisSignalNames) -> int:
        """
        Returns the list index of a measurement channel name, if it exists,
        used for 1DSwp module. 
        
        Parameters
        ----------
        channel_name : str | OxfordNanonisSignalNames
            The channel name to find the index of.
            
        Returns
        -------
        int
            The index of the channel name in the measurement channels list.
        
        Raises
        ------
        ValueError
            If the channel name is not defined in the measurement channels list.
        """
        if isinstance(channel_name, OxfordNanonisSignalNames):
            channel_name = channel_name.value
        if channel_name in self.signals_meas:
            return self.signals_meas[channel_name]
        else:
            raise ValueError(f"Channel {channel_name} is not defined in the measurement channels list.")
        
    async def swp_get_sweep_signal(self) -> str | OxfordNanonisSignalNames:
        name, names = await self.mod_swp.SwpSignalGet()
        if name in OxfordNanonisSignalNames._value2member_map_:
            self.current_sweeper = OxfordNanonisSignalNames(name)
        else:
            self.current_sweeper = name
        return self.current_sweeper
        
    async def swp_set_sweep_signal(self, signal_name: str | OxfordNanonisSignalNames):
        """
        Sets the sweep signal of the 1D Sweeper module (1DSwp.SwpSignalSet).
        """
        if isinstance(signal_name, OxfordNanonisSignalNames):
            signal = signal_name # Store enum.
            signal_name = signal_name.value
        elif signal_name in OxfordNanonisSignalNames._value2member_map_:
            signal = OxfordNanonisSignalNames(signal_name)
        else:
            signal = signal_name
        if signal_name not in self.signals_sweep:
            raise ValueError(f"Signal {signal_name} is not defined in the available measurement channels list.")
        else:
            await self.mod_swp.SwpSignalSet(signal_name)  
        # Wait a delay for UI to update.
        await asyncio.sleep(1)
        self.current_sweeper = signal
        return 
        
    async def swp_get_acquisition_channels(self) -> list[str | OxfordNanonisSignalNames]:
        """
        Returns the list of acquisition channels.
        """
        channels = await self.mod_swp.AcqChsGet()
        channels = [
            OxfordNanonisSignalNames(channel)
            if channel in OxfordNanonisSignalNames._value2member_map_
            else channel
            for channel in channels
        ]
        return channels
        
    async def swp_set_acquisition_channels(self, 
                                       channels: list[OxfordNanonisSignalNames | int | str]
                                       ):
        """
        Sets the acquisition channels of the 1D Sweeper module (1DSwp.AcqChsSet).
        
        Parameters
        ----------
        channels : list[OxfordNanonisSignalNames | int | str]
            List of channels to set as acquisition channels. 
            Can be channel index, channel name, OxfordNanonisSignalNames, or a combination of all.
        """
        channel_indexes = []
        for channel in channels:
            dtype = type(channel)
            if dtype == int:
                channel_indexes.append(channel)
            elif dtype == str:
                channel_indexes.append(await self.measurement_index(channel))
            elif dtype == OxfordNanonisSignalNames:
                channel_indexes.append(await self.measurement_index(channel.value))
            else:
                raise ValueError("Invalid channel data type.")
        await self.mod_swp.AcqChsSet(channel_indexes)
        # Wait a delay for UI to update.
        await asyncio.sleep(1)
        return
        
    async def swp_get_parameters(self) -> tuple[float, float, int, int, bool, bool, float]:
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
        return await self.mod_swp.PropsGet()
        
    async def swp_set_parameters(self, 
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
        await self.mod_swp.PropsSet(
            initial_settling_time=initial_settling_time,
            maximum_slew_rate=maximum_slew_rate,
            number_of_steps=number_of_steps,
            period=period,
            autosave=autosave,
            save_dialog=save_dialog,
            settling_time=settling_time
        )
        # Wait a delay for UI to update.
        await asyncio.sleep(1)
        return
    
    @staticmethod
    def _limits(signal_name: str | OxfordNanonisSignalNames) -> tuple[float, float]:
        """Pre-defined limits for various signal ranges"""
        if isinstance(signal_name, str):
            signal_name = OxfordNanonisSignalNames(signal_name)
        if signal_name in _LIMITS:
            return _LIMITS[signal_name]
        else:
            raise ValueError(f"Signal {signal_name} not defined in limits.")
    
    @staticmethod
    def _check_limits(limits: tuple[float, float], hard_limits: tuple[float, float]) -> bool:
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
    
    async def swp_set_limits(self, limits: tuple[float, float]) -> None:
        """
        Sets the limits of the 1D Sweeper module (1DSwp.LimitsSet).
        
        Parameters
        ----------
        limits : tuple[float, float]
            The sweeper signal limits, lower then upper.
        """
        hard_limits = self._limits(self.current_sweeper)
        if not self._check_limits(limits, hard_limits):
            param_name = ""
            match self.current_sweeper:
                case tramea_signals.MAGNETIC_FIELD_SETPOINT:
                    param_name = "Field"
                case (tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT |
                      tramea_signals.HE4_VTI_TEMPERATURE_SETPOINT |
                      tramea_signals.HE4_EXTRA_TEMP_SENSOR_SETPOINT):
                    param_name = "Temperature"
                case (tramea_signals.DC_INPUT1 |
                      tramea_signals.DC_INPUT2 |
                      tramea_signals.DC_INPUT3 |
                      tramea_signals.DC_INPUT4):
                    param_name = "DC Voltage"
                case (tramea_signals.AC_OUTPUT1_AMP |
                      tramea_signals.AC_OUTPUT2_AMP |
                      tramea_signals.AC_OUTPUT3_AMP |
                      tramea_signals.AC_OUTPUT4_AMP):
                    param_name = "AC Voltage"
                case _:
                    raise ValueError(f"Signal type response for {self.current_sweeper} not defined.")
            fstring = "Provided " + param_name + f" limits ({limits}) must be within {hard_limits}, and limits[0] < limits[1], for {self.current_sweeper}"
            raise ValueError(fstring)
        await self.mod_swp.LimitsSet(*limits)
        # Wait a delay for UI to update.
        await asyncio.sleep(1)
        return
    
    async def swp_get_limits(self) -> tuple[float, float]:
        """
        Returns the limits of the 1D Sweeper module (1DSwp.LimitsGet).
        
        Returns
        -------
        limits : tuple[float, float]
            The sweeper signal limits. Lower then upper.
        """
        return await self.mod_swp.LimitsGet()
    
    async def swp_ramp_start(self, get_data: bool = True, 
                             sweep_direction: int = 1, 
                             save_basename: str = "test_TCP_data", 
                             reset_signal: bool = False,
                             sweep_timeout:float = None) -> tuple[list[str], pd.DataFrame]:
        """
        Starts the 1D Sweeper module (1DSwp.Start).
        
        Additionally calls the stop function, restoring sweep signal and parameters
        from `_async_timeout_params` attribute to original values.
        Can accomodate a timeout to stop the sweep if it takes too long, or 
        if the ramp is being used to define a setpoint.
        
        Parameters
        ----------
        get_data : bool
            Get data.
        sweep_direction : int
            Sweep direction. 1 for forward, 0 for reverse.
        save_basename : str
            Save basename.
        reset_signal : bool
            Reset to initial value?
        sweep_timeout : float
            Timeout in seconds. Minimum time will be 2 seconds to ensure 
            the stop function can be called after sweep initialisation.
            By default None, implying no timeout.
        
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
        
        if sweep_timeout is not None and sweep_timeout > 0:
            print("Timeout: ", sweep_timeout)
            sweep_timeout = max(sweep_timeout, 2)
            try:
                sweep = await asyncio.wait_for(
                    fut=self.mod_swp.Start(
                        get_data=False,
                        sweep_direction=sweep_direction,
                        save_basename=save_basename,
                        reset_signal=False
                    ), 
                    timeout=sweep_timeout
                )
                # Use stop to restore the signal and parameters if they were changed.
                await self.swp_ramp_stop()
                return sweep
            except asyncio.TimeoutError:
                print("Timeout reached, stopping sweep ramp.")
                await self.swp_ramp_stop()
                return None
        else:
            sweep = await self.mod_swp.Start(
                get_data=get_data,
                sweep_direction=sweep_direction,
                save_basename=save_basename,
                reset_signal=reset_signal
            )
            # Use to restore the signal and parameters if they were changed.
            await self.restore_sweep_vars()
            return sweep
        
    async def swp_ramp_stop(self,
                            timeout=5.0) -> None:
        """ 
        Stops the async 1DSwp ramp. 
        
        Parameters
        ----------
        timeout : float
            Timeout in seconds, before trying again.
        """
        print("A")
        exec = False
        timeouts = 0
        while not exec:
            try:
                print("B")
                await asyncio.wait_for(self.mod_swp.Stop(), timeout)
                exec = True
            except asyncio.TimeoutError:
                print("C")
                timeout+=1
                if timeouts > 10:
                    raise TimeoutError(f"Attempted to stop ramp {timeouts} times. Aborting.")
                pass
        # Reset the signal to the original signal if it was changed.
        await self.restore_sweep_vars()
        
    async def restore_sweep_vars(self) -> None:
        """
        Restores the sweep signal and parameters from `_async_timeout_params`.
        """
        if self._async_timeout_params:
            sweep, params, lims = self._async_timeout_params
            await self.swp_set_sweep_signal(sweep)
            await self.swp_set_parameters(*params)
            await self.swp_set_limits(lims)
            self._async_timeout_params = None
        return
        
    async def out_set_slew_rate(self, output_index: int, slew_rate: float) -> None:
        """
        Sets the slew rate of the output module (UserOut.SlewRateSet).
        
        Parameters
        ----------
        output_index : int
            Output index.
        slew_rate : float
            Slew rate in units/s.
        """
        return await self.mod_out.SlewRateSet(output_index, slew_rate)
    
    async def out_get_slew_rate(self, output_index: int) -> float:
        """
        Returns the slew rate of the output module (UserOut.SlewRateGet).
        
        Parameters
        ----------
        output_index : int
            Output index.
            
        Returns
        -------
        slew_rate : float
            Slew rate in units/s.
        """
        return await self.mod_out.SlewRateGet(output_index)
    
    async def out_time_measure_sigs(self,
                    output_idx: int | OxfordNanonisSignalNames,
                    output_setpoints: list[float],
                    meas_sigs: list[str | OxfordNanonisSignalNames],
                    meas_to_average: int = 1,
                    meas_period: float = 1
                    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Changes UserOutput and measures Signals data over time.
        
        Uses the output module to change the control signal to the setpoints, and measure
        the data from the measurement signals. 
        
        Parameters
        ----------
        output_idx : int
            DC output index to change the value of.
        output_setpoints : list[float]
            List of control setpoints for output_idx.
        meas_sigs : list[str | OxfordNanonisSignalNames]
            List of signals to measure. Not measurement listed signals, rather 
            the signals module (128) available signals.
        meas_to_average : int
            Number of measurements to average per period.
        meas_period : float
            Time between measurements in seconds.
        
        Returns
        -------
        vals_mean : npt.NDArray
            Mean values of the setpoint + measurement signals.
        vals_std : npt.NDArray
            Standard devations of the setpoint + measurement signals.
        """
        # Get the output signal name.
        if isinstance(output_idx, int):
            output_name = "Output " + str(output_idx) + " (V)"
        elif isinstance(output_idx, OxfordNanonisSignalNames):
            if "Output" in output_idx.value and " (V)" in output_idx.value:
                output_name = output_idx.value
                output_idx = int(output_idx.value.replace("Output ", "").replace(" (V)", ""))
            else:
                raise ValueError(f"{output_idx} does not match the expected pattern ('Output <integer> (V)') of an output channel name.")
        else:
            raise ValueError("Output index must be an output channel integer or an equivalent OxfordNanonisSignalNames.")
        if output_name not in self.signals_sig:
            raise ValueError(f"Output {output_idx} is not a valid output channel index.")
        output_sig_idx = self.signals_sig[output_name]
        # Get the signal range
        lims = await self.mod_out.LimitsGet(output_idx) # upper then lower...
        if np.any(output_setpoints < lims[1]) or np.any(output_setpoints > lims[0]):
            raise ValueError(f"Setpoints '{output_setpoints}' must be within the range {lims}.")
        
        # Get initial value to reset to.
        default_val = await self.mod_sig.ValGet(self.signals_sig[output_name])
        
        # Set first setpoint.
        await self.mod_out.ValSet(output_idx, output_setpoints[0])
        
        # Collect the measurement signal indexes.
        meas_idxs = []
        for signal in meas_sigs:
            if isinstance(signal, OxfordNanonisSignalNames):
                signal = signal.value
            if signal not in self.signals_sig:
                raise ValueError(f"Signal {signal} is not defined in the available signals list.")
            meas_idxs.append(self.signals_sig[signal])
        
        # Collect the data in a timing loop.
        data_ave = []
        data_std = []
        for setpoint in output_setpoints:
            # Set setpoint.
            await self.mod_out.ValSet(output_idx, setpoint)
            # Wait for measurement to stabilise
            await asyncio.sleep(meas_period)
            # Check stabilisation
            setpoint_measure = await self.mod_sig.ValGet(output_sig_idx)
            while setpoint_measure - setpoint > 0.05*setpoint:
                print(f"Waiting for output {setpoint_measure} to stabilise to setpoint {setpoint}.")
                await asyncio.sleep(meas_period)
                setpoint_measure = await self.mod_sig.ValGet(output_sig_idx)
                
            # Measure values
            vals = []
            for i in range(meas_to_average):
                # Compile and store a list of vals
                vals.append([await self.mod_sig.ValGet(output_sig_idx)] + [
                    await self.mod_sig.ValGet(idx)
                    for idx in meas_idxs
                ])
                # If repeating measurements, wait for measurement period.
                if i != meas_to_average - 1:
                    await asyncio.sleep(meas_period)
            # Average the values to a single value
            data_ave.append(np.mean(vals, axis=0))
            data_std.append(np.std(vals, axis=0))
        # Pack into np array.
        data_ave = np.array(data_ave)
        data_std = np.array(data_std)
        
        # Reset the voltage value to the default value.
        await self.mod_out.ValSet(output_idx, default_val)
        
        return data_ave, data_std
                        
    async def swp_time_measure(self, 
                     output_index: int, 
                     measure_time: float, 
                     measure_channels: list[str | OxfordNanonisSignalNames],
                     default_output_lims: tuple[float, float] = (-0.001, 0.001),
                     measure_period: int = 300
                     ) -> dict[str | OxfordNanonisSignalNames, npt.NDArray]:
        """
        Uses Measures some data overtime without changing a critical signal, does not save data to file.
        
        Output index should be an unconnected analog output channel,
        and slew rate will be adjusted to measure over right time.
        
        Parameters
        ----------
        output_index : int
            Output index.
        time : float
            Time to measure in seconds. Used to calculate the voltage sweep slew rate.
        measure_channels : list[str | OxfordNanonisSignalNames]
            List of channels to measure data.
        default_output_lims : tuple[float, float]
            Channel output voltage limits, by default (-0.001, 0.001) volts.
        measure_period: int
            Time between measurements in milliseconds. By default 300 ms.
            
        Returns
        -------
        keyData : dict[str | OxfordNanonisSignalNames, npt.NDArray]
            Dictionary of channel names corresponding data.
        """
        assert default_output_lims[0] < default_output_lims[1], "Limits must be in increasing order."
        channel_str = tramea_signals("Output " + str(output_index) + " (V)")
        channel_signal_index = self.signals_sig[channel_str.value]
        
        
        slew_rate = (default_output_lims[1] - default_output_lims[0]) / measure_time
        # Store the current settings
        current_settings = await self.swp_get_parameters()
        current_sweep = await self.swp_get_sweep_signal()
        current_lims = await self.swp_get_limits()
        current_slew = await self.out_get_slew_rate(output_index)
        current_channels = await self.swp_get_acquisition_channels()
        # Setup time measure settings
        await self.swp_set_sweep_signal(channel_str)
        assert await self.swp_get_sweep_signal() == channel_str, "Sweep signal not set correctly."
        await self.swp_set_limits(default_output_lims)
        await self.swp_set_parameters(
            initial_settling_time=300,
            maximum_slew_rate=slew_rate,
            number_of_steps=current_settings[2],
            period=measure_period,
            autosave=False,
            save_dialog=False,
            settling_time=current_settings[6]
        )
        await self.swp_set_acquisition_channels(measure_channels)
        # Set starting value of the DC sweeper
        # Use a fast rate to initialise
        await self.out_set_slew_rate(output_index, 10.0) #default voltage / s for Tramea.
        await self.mod_out.ValSet(output_index, default_output_lims[0])
        await self.out_set_slew_rate(output_index, slew_rate)
        time.sleep(1)
        # Wait for value to reach start value: TODO: cannot check output value...
        threshold = 0.05 * abs(default_output_lims[1]-default_output_lims[0])
        channel_value_delta = abs(await self.mod_sig.ValGet(channel_signal_index) - default_output_lims[0])
        while channel_value_delta > threshold:
            # Wait, then update channel value delta.
            time.sleep(1)
            val = await self.mod_sig.ValGet(channel_signal_index)
            channel_value_delta = abs(
                 val - default_output_lims[0]
            )
        # Measure the data:
        channel_names, data = await self.swp_ramp_start(
            get_data=True,
            sweep_direction=1,
            save_basename="time_measure_stability",
            reset_signal=False
        )
        # Restore initial settings
        await self.out_set_slew_rate(output_index, current_slew)
        await self.swp_set_sweep_signal(current_sweep)
        await self.swp_set_limits(current_lims)
        await self.swp_set_parameters(*current_settings)
        if current_channels is not None:
            await self.swp_set_acquisition_channels(current_channels)
        # Return values
        channel_names = [OxfordNanonisSignalNames(channel) 
                         for channel in channel_names
                         if channel in OxfordNanonisSignalNames._value2member_map_]
        
        keyedData = {channel : data[i] 
                   for i, channel in enumerate(channel_names)}
        return keyedData
    
    async def swp_check_unstability(self, 
                            signals: list[str | OxfordNanonisSignalNames],
                            standard_deviations: list[float],
                            setpoints: list[float] = None,
                            time_to_measure: float = 60*10,
                            meas_period: int = 300,
                            output_index: int = 8,
                            default_output_lims: tuple[float, float] = (-0.001, 0.001)
                            ) -> tuple[list[str | OxfordNanonisSignalNames],
                                       dict[str | OxfordNanonisSignalNames, npt.NDArray],
                                       dict[str | OxfordNanonisSignalNames, npt.NDArray]]:
        """
        Returns signals that are unstable within a certain timeframe.
        
        Determines unstable signals by calculating the standard deviation
        is outside the provided standard deviation bound. If setpoints are 
        provided, checks the mean difference from the setpoint is also within
        the provided standard deviation value.
        
        Parameters
        ----------
        signals : list[str | OxfordNanonisSignalNames]
            List of signals to check.
        standard_deviations : list[float]
            List of standard deviations to check.
        time_to_measure : float
            Time to measure in seconds. Used to calculate the voltage sweep slew rate.
        meas_period : int
            Time between measurements in milliseconds.
        output_index : int
            Nanonis Tramea DC output index to use for time measurement.
        default_output_lims : tuple[float, float]
            Default output voltage limits for the time measurement.
            By default (-0.001, 0.001) volts.
            
        Returns
        -------
        unstable_signals : list[str | OxfordNanonisSignalNames]
            List of unstable signals. If length > 0 then some signals are unstable.
        aves : dict[str | OxfordNanonisSignalNames, float]
            Dictionary of mean values measured.
        stds : dict[str | OxfordNanonisSignalNames, float]
            Dictionary of standard deviations for each signal. 
            If setpoints are provided, an unstable signal might not 
            be outside the provided standard deviation.
        """
        assert len(signals) == len(standard_deviations), "Signals and standard deviations must be the same length."
        assert setpoints is None or len(signals) == len(setpoints), "Signals and setpoints must be the same length."
        
        data = await self.swp_time_measure(
            output_index=output_index,
            measure_time=time_to_measure,
            measure_channels=signals,
            default_output_lims=default_output_lims,
            measure_period=meas_period
        )
        stds = {}
        aves = {}
        unstable_signals = []
        for i, signal in enumerate(signals):
            std_dev = standard_deviations[i]
            if setpoints and setpoints[i]:
                setpoint = setpoints[i]
                # Calculate mean difference from setpoint.
                diff = data[signal] - setpoint
                mean = diff.mean()
                std = diff.std()
                stds[signal] = std
                # Ensure mean diff to setpoint, and std are within std_dev limit.
                if std > std_dev or mean > std_dev:
                    unstable_signals.append(signal)
            else:
                std = data[signal].std()
                stds[signal] = std
                if std > std_dev:
                    unstable_signals.append(signal)
            # Store the average value of the signal.
            aves[signal] = data[signal].mean()                
        return unstable_signals, aves, stds, 
    
    
    async def sig_check_unstability(self,
        signals: list[str | OxfordNanonisSignalNames],
        standard_deviations: list[float],
        setpoints: list[float] = None,
        meas_to_average: int = 3,
        meas_period: float = 0.3
        ) -> tuple[list[str | OxfordNanonisSignalNames],
                   dict[str | OxfordNanonisSignalNames, npt.NDArray],
                   dict[str | OxfordNanonisSignalNames, npt.NDArray]]:
        """
        Uses the signals module to check the stability of signals over time.
        
        Parameters
        ----------
        signals : list[str | OxfordNanonisSignalNames]
            List of signals to check.
        standard_deviations : list[float]
            List of standard deviations to check.
        setpoints : list[float]
            List of setpoints to check the signals against.
        meas_to_average : int
            Number of measurements to average per period.
        meas_period : float
            Time between measurements in seconds.
            
        Returns
        -------
        tuple[list[str], dict[str, float], dict[str, float]
            unstable_signals : list[str]
                List of unstable signals. If length > 0 then some signals are unstable.
            aves : dict[str, float]
                Dictionary of mean values measured.
            stds : dict[str, float]
                Dictionary of standard deviations for each signal.
        """
        signals = signals.copy()
        # Check each signal is in the signals_sig list.
        for i, signal in enumerate(signals):
            if isinstance(signal, OxfordNanonisSignalNames):
                signal = signal.value
                signals[i] = signal
            if signal not in self.signals_sig:
                raise ValueError(f"Signal {signal} is not defined in the available signals list.")
            
        # Check the lengths of the signals, standard deviations and setpoints.
        if len(standard_deviations) != len(signals):
            raise ValueError("Signals and standard deviations must be the same length.")
        if setpoints and len(setpoints) != len(signals):
            raise ValueError("Signals and setpoints must be the same length.")
        
        # Begin the measurement.
        data = []
        for i in range(meas_to_average):
            data.append([
                await self.mod_sig.ValGet(self.signals_sig[signal])
                for signal in signals
            ])
            await asyncio.sleep(meas_period)
        data = np.array(data)
            
        # Construct statistics
        stds = {}
        aves = {}
        unstable_signals = []
        for i, signal in enumerate(signals):
            std_dev = standard_deviations[i]
            if setpoints and setpoints[i]:
                setpoint = setpoints[i]
                # Calculate mean difference from setpoint.
                diff = data[:, i] - setpoint
                mean = diff.mean()
                std = diff.std()
                stds[signal] = std
                # Ensure mean diff to setpoint, and std are within std_dev limit.
                if std > std_dev or mean > std_dev:
                    unstable_signals.append(signal)
                # Store the average value of the signal.
                aves[signal] = data[signal].mean()
            else:
                std = data[:, i].std()
                stds[signal] = std
                aves[signal] = data[signal].mean()
                if std > std_dev:
                    unstable_signals.append(signal)
        return unstable_signals, aves, stds
    
    async def get_parameter_value(self,
                                signal_name: str | OxfordNanonisSignalNames,
                                output_index: int = 8,
                                ) -> float:
        """
        Returns a single value of a signal. 
        
        If the signal can be measured directly, will use the signal module.
        If the signal is only accessible via the sweep module, will use the sweep module,
        which will take some additional time. To do this, also uses the output module
        to scan across an unused output channel to perform the measurement.
        
        Parameters
        ----------
        signal_name : str | OxfordNanonisSignalNames
            The signal name.
        output_index : int
            The output index to use, if requiring a sweep measurement for the signal.
            By default channel 8.
            
        Returns
        -------
        value : float
            The value of the signal.
        """
        # Convert to Signal Name
        if isinstance(signal_name, str):
            signal_name = OxfordNanonisSignalNames(signal_name)
        # Check if signal is in the signal list.
        if signal_name.value in self.signals_sig:
            return await self.mod_sig.ValGet(self.signals_sig[signal_name.value])
        elif signal_name.value in self.signals_meas:
            # 8th Output Index is already covered if being measured by the signals module above.
            data = await self.swp_time_measure(
                output_index=output_index,
                measure_time=3,
                measure_period=300,
                measure_channels=[signal_name.value],
            )
            return data[signal_name].mean()
        else:
            raise ValueError(f"Signal {signal_name} is not defined in the available signals list.")
        
    
    async def set_parameter_setpoint(self, 
                               parameter: OxfordNanonisSignalNames | str, 
                               setpoint: float,
                               approaching_setpoint: float = None,
                               slew_rate: float | None = None,
                               sweep_timeout: float | None = None) -> None:
        """
        Sets the setpoint of a parameter, using the appropriate submodule.
        
        If the parameter is a temperature or magnetic field setpoint, the function
        will use the Swp1D module to sweep the setpoint to the new value and wait 
        for it to stabilise. If the parameter is a DC or AC output, the function 
        will use the output to the setpoint value.
        
        Parameters
        ----------
        parameter : OxfordNanonisSignalNames
            The parameter to set.
        setpoint : float
            The setpoint to set.
        approaching_setpoint : float
            The setpoint value to approach the setpoint.
            By default is calculated at 1.001 * setpoint.
        slew_rate: float | None
            Only applies to DC/AC voltages. The slew rate to set for the parameter.
            By default is None.
        sweep_timeout: float | None
            The timeout for the sweep to complete. Only used with 
            _set_parameter_setpoint_using_ramp. By default is None.
            Only parameters that can be changed via the sweep module can use this 
            (i.e., parameters not listed in Signals module).
        """
        # Check if signal is temperature or field:
        if isinstance(parameter, str):
            # Setpoint needs to be a known, to measure the corresponding signal.
            parameter = OxfordNanonisSignalNames(parameter)
        
        # Set param and param_setpoint disctinctions
        if parameter in SENSOR_TO_SETPOINTS:
            param_setpoint = SENSOR_TO_SETPOINTS[parameter]
            param = parameter
        elif parameter in SETPOINTS_TO_SENSORS:
            param = SETPOINTS_TO_SENSORS[parameter]
            param_setpoint = parameter
        else:
            param_setpoint = parameter
            param = parameter
            
        # Check if the parameter setpoint is known:
        if not(param_setpoint in SETPOINTS_TO_SENSORS
               or param_setpoint.value in self.signals_sig):
            raise ValueError(
                f"Parameter '{parameter}' must have a defined setpoint:sensor correspondence, or be defined in signals module.")
        
        # Hard limit check the setpoint
        if param_setpoint in _LIMITS:
            if approaching_setpoint is not None:
                ramp = (setpoint, approaching_setpoint) if setpoint < approaching_setpoint else (approaching_setpoint, setpoint)
                if not self._check_limits(ramp, _LIMITS[parameter]):
                    raise ValueError(f"Approach '{approaching_setpoint}' and Setpoint '{setpoint}' must be within the limits '{_LIMITS[parameter]}'.")
            else:
                if not self._check_limits((setpoint, setpoint), _LIMITS[parameter]):
                    raise ValueError(f"Setpoint '{setpoint}' must be within the limits {_LIMITS[parameter]}.")
        else:
            warnings.warn(f"Setpoint '{setpoint}' does not have defined limits to check against.")
        
        if approaching_setpoint is None:
            # Read the current value to set initial setpoint.
            approaching_setpoint = await self.get_parameter_value(param)
        
        # Run the setpoint change
        match param_setpoint:
            case (tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT | tramea_signals.HE4_VTI_TEMPERATURE_SETPOINT
                | tramea_signals.HE4_EXTRA_TEMP_SENSOR_SETPOINT):
                # As He3 Temp has no direct set method, use 1D Swp to set the field.
                await self._set_parameter_setpoint_using_swp(
                    param_setpoint=param_setpoint,
                    setpoint=setpoint,
                    approaching_setpoint=approaching_setpoint,
                    sweep_timeout=sweep_timeout
                )                
            case tramea_signals.MAGNETIC_FIELD_SETPOINT:
                # As Field has no direct set method, use 1D Swp to set the field.
                await self._set_parameter_setpoint_using_swp(
                    param_setpoint=param_setpoint,
                    setpoint=setpoint,
                    approaching_setpoint=approaching_setpoint,
                    sweep_timeout=sweep_timeout
                )
            case (tramea_signals.DC_OUTPUT1 | tramea_signals.DC_OUTPUT2
                | tramea_signals.DC_OUTPUT3 | tramea_signals.DC_OUTPUT4 
                | tramea_signals.DC_OUTPUT5 | tramea_signals.DC_OUTPUT6 
                | tramea_signals.DC_OUTPUT7 | tramea_signals.DC_OUTPUT8):
                # "Output 1 (V)"
                idx = int(param_setpoint.value.replace("Output ", "").replace(" (V)", ""))
                if slew_rate:
                    # Store the original slew rate and set the new value.
                    init_slew = await self.out_get_slew_rate(idx)
                    await self.out_set_slew_rate(idx, slew_rate)
                # Set the new voltage.
                await self.mod_out.ValSet(idx, setpoint)
                if slew_rate:
                    # Retore the old slewrate
                    await self.out_set_slew_rate(idx, init_slew)
            case (tramea_signals.AC_OUTPUT1_AMP | tramea_signals.AC_OUTPUT2_AMP
                | tramea_signals.AC_OUTPUT3_AMP | tramea_signals.AC_OUTPUT4_AMP):
                # "LI Mod 1 Amp (V)"
                idx = int(param_setpoint.value.replace("LI Mod ", "").replace(" AMP (V)", ""))
                await self.mod_lock.ModAmpSet(idx, setpoint)
            case _:
                raise ValueError(f"Parameter '{param_setpoint}' must be a known signal or setpoint.")
                    
    async def _set_parameter_setpoint_using_swp(
        self,
        param_setpoint: OxfordNanonisSignalNames | str,
        setpoint: float, 
        approaching_setpoint: float,
        sweep_timeout: float = None
    ):  
        """
        Sets the setpoint of a parameter using the 1D Sweeper module.
        """
        if isinstance(param_setpoint, str):
            param_setpoint = OxfordNanonisSignalNames(param_setpoint)
        # Check the parameter is a sweep signal.
        if param_setpoint.value not in self.signals_sweep:
            raise ValueError(f"Parameter {param_setpoint} is not a valid sweep signal.")
        # Check the parameter has a sensor correspondence.
        if param_setpoint not in SETPOINTS_TO_SENSORS:
            raise ValueError(f"Parameter {param_setpoint} must have a defined sensor correspondence.")
        else:
            param = SETPOINTS_TO_SENSORS[param_setpoint]
        
        # If the setpoint is very close to the approaching setpoint, measure the value for stability.
        sgn = True if setpoint > 0 else False
        if ((sgn and approaching_setpoint < 1.01*setpoint and approaching_setpoint > 0.99 * setpoint) 
            or (not sgn and approaching_setpoint > 1.01*setpoint and approaching_setpoint < 0.99 * setpoint)):
            # Magnitude wthin 1% of setpoint. Measure the value for stability for 30s.
            uSig, uAves, uStds = await self.swp_check_unstability(
                signals=[param],
                standard_deviations=[0.01*abs(setpoint)],
                time_to_measure=30
            )
            if param.value in uSig:
                # Assume the setpoint is not stably set.
                # Ramp away 10% then ramp towards.
                approaching_setpoint = 0.9*setpoint
                # Edge cases for base temperatures...
                if param_setpoint in _BASE_TEMPERATURES:
                    # Temperatures positive definite.
                    base = _BASE_TEMPERATURES[param_setpoint]
                    if approaching_setpoint < base:
                        # instead move slightly higher.
                        approaching_setpoint = 1.1*setpoint if setpoint < base else base
                        # If too close to the existing value, ramp away further.
                        while uAves[param.value] * 0.95 < approaching_setpoint or uAves[param.value] * 1.05 > approaching_setpoint:
                            approaching_setpoint *= 1.1
                
                print(f"{param_setpoint} unstable but close to desired value '{setpoint}'. Ramping away to {approaching_setpoint} then back to {setpoint}.")
                # Use this function to sweep away.
                self._set_parameter_setpoint_using_swp(
                    param_setpoint=param_setpoint,
                    setpoint=approaching_setpoint,
                    approaching_setpoint=uAves[param.value]
                )
                # Now perform the ramp towards the setpoint, below.
            else:
                # Assume the setpoint is stable at the desired value.
                return
            
        # Determine the approaching direction:
        if setpoint == approaching_setpoint:
            raise ValueError("Setpoint and approaching setpoint cannot be the same.")
 
        direction = 1 if setpoint > approaching_setpoint else 0
        magnitude = 1 if abs(setpoint) > abs(approaching_setpoint) else 0
        if direction and magnitude:
            # i.e. (1, 2). Must be +ve, larger setpoint and larger abs
            limits = (approaching_setpoint, setpoint)
        elif direction and not magnitude:
            # i.e. (-2, -1) Must be -ve, larger sepoint but smaller abs
            limits = (approaching_setpoint, setpoint)
        elif not direction and magnitude:
            # i.e. (-2, -1) Must be -ve, smaller setpoint but larger abs
            limits = (setpoint, approaching_setpoint)
        elif not direction and not magnitude:
            # i.e. (1, 2) Must be +ve, smaller setpoint and smaller abs
            limits = (setpoint, approaching_setpoint)
            
        limits = (approaching_setpoint, setpoint) if direction else (setpoint, approaching_setpoint)
        
        # Store old sweep signal and limits
        current_sweep = await self.swp_get_sweep_signal()
        current_settings = await self.swp_get_parameters()
        current_limits = await self.swp_get_limits()
        self._async_timeout_params = (current_sweep, current_settings, current_limits)
        
        # Set the new sweep signal and limits
        await self.swp_set_sweep_signal(param_setpoint.value)
        await self.swp_set_limits(limits)
        await self.swp_set_parameters(
            initial_settling_time=300,
            maximum_slew_rate=current_settings[1],
            number_of_steps=current_settings[2],
            period=1000,
            autosave=False,
            save_dialog=False,
            settling_time=current_settings[6]
        )
        
        print(sweep_timeout, limits)
        # Wait a delay for UI to update.
        await asyncio.sleep(1)
        
        # Start the sweep
        await self.swp_ramp_start(
            get_data=False,
            sweep_direction=direction,
            save_basename="setpoint_change",
            reset_signal=False,
            sweep_timeout=sweep_timeout
        )
        
        return
    
    async def use_ramp_to_recondense(self):
        """"
        An asyncronous function to recondense the He3 probe using the ramp signal.
        
        Requires asynchronous, because the ramp will never reach the 0 K setpoint.
        Therefore a `stop` signal must be sent to the Nanonis controller asynchronously
        after the setpoint is initialised.
        """
        init_params = await self.swp_get_parameters()
        init_lims = await self.swp_get_limits()
        init_sweep = await self.swp_get_sweep_signal()
        await self.swp_set_sweep_signal(tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT)
        # this is the setpoint required to recondense the He3 probe
        base_temp_reset = (0, 0.001) #ramp from 0K
        await self.swp_set_limits(base_temp_reset)
        
        temperature_setpoint_params = {
            "initial_settling_time" : 300,
            "maximum_slew_rate" : init_params[1],
            "number_of_steps" : init_params[2],
            "period" : 1000,
            "autosave" : False,
            "save_dialog" : False,
            "settling_time" : init_params[6]
        }
        await self.swp_set_parameters(**temperature_setpoint_params)
        # Save init_params + sweep signal in case of async timeout.
        self._async_timeout_params = (init_sweep, init_params, init_lims)
        
        await self.swp_ramp_start(
            get_data=False,
            sweep_direction=1,
            save_basename="recondense_he3_probe",
            reset_signal=False
        )
            
    async def condense_he3_probe(self,
                                 base: float = 0.26, 
                                 std: float = 0.01,
                                 time: int = 120) -> None:
        """
        Condenses the He3 probe by ramping the temperature to 0 K.
        
        Calls condensation ramp for 5 minutes, before performing a loop check 
        that the temperature is stable over a period of 120 seconds. If not,
        continues to wait for stabilisation within 'std'.
        
        Parameters
        ----------
        base : float
            The base temperature the He3 probe should recondense to.
        std : float
            The standard deviation the temperature should be within.
        time : int
            The time to measure the temperature stability in seconds.
        """
        print("He3 probe condensation beginning ...")
        try:
            await asyncio.wait_for(self.use_ramp_to_recondense(),
                               timeout=dt.timedelta(minutes=5).total_seconds())
        except asyncio.TimeoutError:
            await asyncio.sleep(2)
            # Stop_ramp has print commands before & after.
            await self.swp_ramp_stop()
            # Wait to check stability.
            await asyncio.sleep(2)
        # Wait for the He3 probe to recondense to the base temperature.
        print("Checking He3 probe temperature stability...")
        tsig = tramea_signals.HE3_PROBE_TEMPERATURE
        uSigs, uAves, uStd = await self.swp_check_unstability(
            signals=[tsig],
            standard_deviations=[std],
            setpoints=[base],
            time_to_measure=time,
            output_index=8
        )
        
        while len(uSigs) > 0:
            print(f"Temp. unstable: {uAves[tsig]} +- {uStd[tsig]} K. Re-measuring...")
            uSigs, uAves, uStd = await self.swp_check_unstability(
                signals=[tsig],
                standard_deviations=[std],
                setpoints=[base],
                time_to_measure=time,
                output_index=8
            )
        print("He3 probe re-condensed.")
        return
    
    async def set_parameter_and_stabilise(self,
                                          parameter: OxfordNanonisSignalNames | str,
                                          setpoint: float,
                                          std: float,
                                          time: int,
                                          approach_setpoint: float = None,
                                          sweep_timeout: float = None) -> float:
        """
        Sets a parameter to a setpoint and waits for it to stabilise.
        
        Parameters
        ----------
        parameter : OxfordNanonisSignalNames | str
            The parameter to set.
        setpoint : float
            The setpoint to set.
        std : float
            The standard deviation the parameter should be within.
        time : int
            The time to measure the stability in seconds.
        approach_setpoint : float
            The setpoint value to approach the setpoint.
            By default None (see set_parameter_setpoint for default behaviour).
            
        Returns
        -------
        float
            The averaged value of the parameter.
        """
        # Check if signal is temperature or field:
        if isinstance(parameter, str):
            # Setpoint needs to be a known, to measure the corresponding signal.
            parameter = OxfordNanonisSignalNames(parameter)
        
        # Set param and param_setpoint disctinctions
        if parameter in SENSOR_TO_SETPOINTS:
            param_setpoint = SENSOR_TO_SETPOINTS[parameter]
            param = parameter
        elif parameter in SETPOINTS_TO_SENSORS:
            param = SETPOINTS_TO_SENSORS[parameter]
            param_setpoint = parameter
        else:
            param_setpoint = parameter
            param = parameter
            
        # If parameter cannot be set as a measurement parameter
        if param.value not in self.signals_meas and param.value not in self.signals_sig:
            raise ValueError(
                f"Parameter {param} not measurable via signals or sweep modules.")
        
        # Set the setpoint
        await self.set_parameter_setpoint(
            parameter=param_setpoint,
            setpoint=setpoint,
            approaching_setpoint=approach_setpoint,
            sweep_timeout=sweep_timeout
        )

        # Depending on the parameter, use the signals or sweep module to measure.
        if param.value in self.signals_sig:
            check_unstability = self.sig_check_unstability
            check_args = [
                [param],
                [std],
                [setpoint],
                time,
                5,
            ]
        else:
            check_unstability = self.swp_check_unstability
            check_args = [
                [param],
                [std],
                [setpoint],
                time
            ]
        
        # Check the stability of the parameter.
        uSig, uAves, uStd = await check_unstability(*check_args)
        
        while len(uSig) > 0:
            # Some meas / sweep signals are not in the signals_sig list.
            print(f"Parameter unstable: {uAves[param]} +- {uStd[param]}. Re-measuring...")
            uSig, uAves, uStd = await check_unstability(*check_args)
        print(param)
        print(uSig, uAves)
        return uAves[param].mean()
    
def generate_log_setpoints(limits: tuple[float, float],
                           datapoints: int, 
                           forbidden_range: tuple[float, float]=None) -> npt.NDArray:
    """
    Generate logarithmically spaced setpoints for a given range.
    
    Parameters
    ----------
    limits : tuple[float, float]
        The limits of the range to generate setpoints for.
    datapoints : int
        The number of datapoints to generate.
    forbidden_range : tuple[float, float], optional
        The range to exclude from the setpoints, by default None.
    """
    if forbidden_range:
        if forbidden_range[0] > forbidden_range[1]:
            raise ValueError("Forbidden range must be in increasing order.")
        if limits[0] > forbidden_range[0] or limits[1] < forbidden_range[1]:
            raise ValueError("Forbidden range must be within limits.")
    if limits[0] > limits[1]:
        raise ValueError("Limits must be in increasing order.")
    
    excOn = (forbidden_range is not None)
    if not excOn:
        logrange = np.log(limits)
        logpoints = np.linspace(logrange[0],logrange[1], datapoints)
        points = np.exp(logpoints)
    else:
        logrange = np.log(limits)
        logexc_range = np.log(forbidden_range)
        d1 = logexc_range[0] - logrange[0] # logrange[0] < logexc_range[0]
        d2 = logrange[1] - logexc_range[1] # logrange[1] > logexc_range[1]
        numpoints1 = int(round(datapoints * (d1) / (d1 + d2))) #fraction of points
        numpoints2 = int(round(datapoints * (d2) / (d1 + d2))) #fraction of points
        logpoints1 = np.linspace(logrange[0], logexc_range[0], numpoints1)
        logpoints2 = np.linspace(logexc_range[1], logrange[1], numpoints2)
        logpoints = np.r_[logpoints1, logpoints2]
        points = np.exp(logpoints)
    return points
