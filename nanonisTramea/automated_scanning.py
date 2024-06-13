from signal_names import OxfordNanonisSignalNames
tramea_signals = OxfordNanonisSignalNames
from nanonisTCP import nanonisTCP
from nanonisTCP.Signals import Signals
from nanonisTCP.Swp1D import Swp1D
from nanonisTCP.UserOut import UserOut
import pandas as pd
import datetime as dt
import numpy as np
import time
import numpy.typing as npt
import asyncio
from typing import Self

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
        self._signals_meas = None
        self._signals_sweep = None
        self._signals_sig = None
    
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
        self._mod_sig = sig
        self._mod_swp = swp
        self._mod_out = out
        
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
        Returns the UserOutput module (DC/AC Tramea outputs).
        """
        return self._mod_out
    
    @property
    def signals_meas(self) -> dict[str, int]:
        """
        Returns the measurement signals.
        """
        return self._signals_meas
    
    @property
    def signals_sweep(self) -> list[str]:
        """
        Returns the sweep signals.
        """
        return self._signals_sweep
    
    @property
    def signals_sig(self) -> dict[str, int]:
        """
        Returns the signals.
        """
        return self._signals_sig
    
        
    async def measurement_index(self, channel_name: str | OxfordNanonisSignalNames) -> int:
        if isinstance(channel_name, OxfordNanonisSignalNames):
            channel_name = channel_name.value
        if channel_name in self.signals_meas:
            return self.signals_meas[channel_name]
        else:
            raise ValueError(f"Channel {channel_name} is not defined in the measurement channels list.")
        
    async def get_sweep_signal(self) -> str | OxfordNanonisSignalNames:
        name, names = await self.mod_swp.SwpSignalGet()
        if name in OxfordNanonisSignalNames._value2member_map_:
            self.current_sweeper = OxfordNanonisSignalNames(name)
        else:
            self.current_sweeper = name
        return self.current_sweeper
        
    async def set_sweep_signal(self, signal_name: str | OxfordNanonisSignalNames):
        """
        Sets the sweep signal of the 1D Sweeper module (1DSwp.SwpSignalSet).
        """
        if isinstance(signal_name, OxfordNanonisSignalNames):
            signal_name = signal_name.value
        if signal_name not in self.signals_sweep:
            raise ValueError(f"Signal {signal_name} is not defined in the available measurement channels list.")
        else:
            await self.mod_swp.SwpSignalSet(signal_name)  
        time.sleep(1) #add a delay before limits can be changed...  
        
    async def get_acquisition_channels(self) -> list[str | OxfordNanonisSignalNames]:
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
        
    async def set_acquisition_channels(self, channels: list[OxfordNanonisSignalNames | int | str]):
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
        return
        
    async def get_sweep_parameters(self) -> tuple[float, float, int, int, bool, bool, float]:
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

        
    async def set_sweep_parameters(self, 
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
        return await self.mod_swp.PropsSet(
            initial_settling_time=initial_settling_time,
            maximum_slew_rate=maximum_slew_rate,
            number_of_steps=number_of_steps,
            period=period,
            autosave=autosave,
            save_dialog=save_dialog,
            settling_time=settling_time
        )
    
    async def _check_limits(self, limits: tuple[float, float], hard_limits: tuple[float, float]) -> bool:
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
    
    async def set_limits(self, limits: tuple[float, float]) -> None:
        """
        Sets the limits of the 1D Sweeper module (1DSwp.LimitsSet).
        
        Parameters
        ----------
        limits : tuple[float, float]
            The sweeper signal limits, lower then upper.
        """
        if self.current_sweeper is tramea_signals.MAGNETIC_FIELD_SETPOINT:
            hard_limits = (-14,14)
            if not await self._check_limits(limits, hard_limits):
                raise ValueError(f"Field limits must be within {hard_limits}, and limits[0] < limits[1].")
        elif self.current_sweeper is tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT:
            hard_limits = (0, 300)
            if not await self._check_limits(limits, hard_limits):
                raise ValueError(f"Temperature limits must be within {hard_limits}, and limits[0] < limits[1].")
        elif (self.current_sweeper is tramea_signals.DC_INPUT1
              or self.current_sweeper is tramea_signals.DC_INPUT2
              or self.current_sweeper is tramea_signals.DC_INPUT3
              or self.current_sweeper is tramea_signals.DC_INPUT4
              ):
            hard_limits = (-10, 10)
            if not await self._check_limits(limits, hard_limits):
                raise ValueError(f"DC Voltage limits must be within {hard_limits}, and limits[0] < limits[1].")
        elif (self.current_sweeper is tramea_signals.AC_OUTPUT1_AMP
              or self.current_sweeper is tramea_signals.AC_OUTPUT2_AMP
              or self.current_sweeper is tramea_signals.AC_OUTPUT3_AMP
              or self.current_sweeper is tramea_signals.AC_OUTPUT4_AMP  
              ):
            hard_limits = (0, 2)
            if not await self._check_limits(limits, hard_limits):
                raise ValueError(f"AC Voltage limits must be within {hard_limits}, and limits[0] < limits[1].")
        await self.mod_swp.LimitsSet(*limits)
        return
    
    async def get_limits(self) -> tuple[float, float]:
        """
        Returns the limits of the 1D Sweeper module (1DSwp.LimitsGet).
        
        Returns
        -------
        limits : tuple[float, float]
            The sweeper signal limits. Lower then upper.
        """
        return await self.mod_swp.LimitsGet()
    
    async def start(self, get_data: bool = True, sweep_direction: int = 1, save_basename: str = "test_TCP_data", reset_signal: bool = False) -> tuple[list[str], pd.DataFrame]:
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
        
        return await self.mod_swp.Start(
            get_data=get_data,
            sweep_direction=sweep_direction,
            save_basename=save_basename,
            reset_signal=reset_signal
        )
        
    async def set_slew_rate(self, output_index: int, slew_rate: float) -> None:
        """
        Sets the slew rate of the 1D Sweeper module (1DSwp.SlewRateSet).
        
        Parameters
        ----------
        output_index : int
            Output index.
        slew_rate : float
            Slew rate in units/s.
        """
        return await self.mod_out.SlewRateSet(output_index, slew_rate)
    
    async def get_slew_rate(self, output_index: int) -> float:
        """
        Returns the slew rate of the 1D Sweeper module (1DSwp.SlewRateGet).
        
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
    
    async def get_signal_value(self, signal_name: str | OxfordNanonisSignalNames) -> float:
        """
        Returns the value of a signal.
        
        Parameters
        ----------
        signal_name : str | OxfordNanonisSignalNames
            The signal name.
            
        Returns
        -------
        value : float
            The value of the signal.
        """
        if isinstance(signal_name, OxfordNanonisSignalNames):
            signal_name = signal_name.value
        if signal_name not in self.signals_sig:
            raise ValueError(f"Signal {signal_name} is not defined in the available signals list.")
        return await self.mod_sig.ValGet(self.signals_sig[signal_name])
    
    async def time_measure(self, 
                     output_index: int, 
                     measure_time: int, 
                     measure_channels: list[str | OxfordNanonisSignalNames],
                     default_output_lims: tuple[float, float] = (-0.001, 0.001)
                     ) -> dict[str | OxfordNanonisSignalNames, npt.NDArray]:
        """
        Measures some data overtime without changing a critical signal, does not save data to file.
        
        Output index should be an unconnected analog output channel,
        and slew rate will be adjusted to measure over right time.
        
        Parameters
        ----------
        output_index : int
            Output index.
        time : int
            Time in seconds.
        measure_channels : list[str | OxfordNanonisSignalNames]
            List of channels to measure data.
        default_output_lims : tuple[float, float]
            Channel output voltage limits, by default (-0.001, 0.001) volts.
            
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
        current_settings = await self.get_sweep_parameters()
        current_sweep = await self.get_sweep_signal()
        current_lims = await self.get_limits()
        current_slew = await self.get_slew_rate(output_index)
        current_channels = await self.get_acquisition_channels()
        # Setup time measure settings
        await self.set_sweep_signal(channel_str)
        assert await self.get_sweep_signal() == channel_str, "Sweep signal not set correctly."
        await self.set_limits(default_output_lims)
        await self.set_sweep_parameters(
            initial_settling_time=300,
            maximum_slew_rate=slew_rate,
            number_of_steps=current_settings[2],
            period=1000,
            autosave=False,
            save_dialog=False,
            settling_time=current_settings[6]
        )
        await self.set_acquisition_channels(measure_channels)
        # Set starting value of the DC sweeper
        # Use a fast rate to initialise
        await self.set_slew_rate(output_index, 10.0) #default voltage / s for Tramea.
        await self.mod_out.ValSet(output_index, default_output_lims[0])
        time.sleep(1)
        await self.set_slew_rate(output_index, slew_rate)
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
        channel_names, data = await self.start(
            get_data=True,
            sweep_direction=1,
            save_basename="time_measure_stability",
            reset_signal=False
        )
        # Restore initial settings
        await self.set_slew_rate(output_index, current_slew)
        await self.set_sweep_signal(current_sweep)
        await self.set_limits(current_lims)
        await self.set_sweep_parameters(*current_settings)
        if current_channels is not None:
            await self.set_acquisition_channels(current_channels)
        # Return values
        channel_names = [OxfordNanonisSignalNames(channel) 
                         for channel in channel_names
                         if channel in OxfordNanonisSignalNames._value2member_map_]
        
        keyedData = {channel : data[i] 
                   for i, channel in enumerate(channel_names)}
        return keyedData
    
    async def check_unstability(self, 
                            signals: list[str | OxfordNanonisSignalNames],
                            standard_deviations: list[float],
                            setpoints: list[float] = None,
                            time_to_measure: int = 60*10,
                            output_index: int = 8,
                            default_output_lims: tuple[float, float] = (-0.001, 0.001)
                            ):
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
        time_to_measure : int
            Time to measure in seconds.
        output_index : int
            Nanonis Tramea DC output index to use for time measurement.
        default_output_lims : tuple[float, float]
            Default output voltage limits for the time measurement.
            
        Returns
        -------
        unstable_signals : list[str | OxfordNanonisSignalNames]
            List of unstable signals. If length > 0 then some signals are unstable.
        stds : dict[str | OxfordNanonisSignalNames, float]
            Dictionary of standard deviations for each signal. 
            If setpoints are provided, an unstable signal might not 
            be outside the provided standard deviation.
        """
        #TODO: Replace this function with a direct signal getVal measurement,
        # Unless the user likes a UI for this.
        assert len(signals) == len(standard_deviations), "Signals and standard deviations must be the same length."
        assert setpoints is None or len(signals) == len(setpoints), "Signals and setpoints must be the same length."
        
        data = await self.time_measure(
            output_index=output_index,
            measure_time=time_to_measure,
            measure_channels=signals,
            default_output_lims=default_output_lims
        )
        stds = {}
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
                # Ensure mean and std are within std_dev limit.
                if data[signal].std() > std_dev or mean > std_dev:
                    unstable_signals.append(signal)
            else:
                std = data[signal].std()
                stds[signal] = std
                if data[signal].std() > std_dev:
                    unstable_signals.append(signal)
        return unstable_signals, stds
    
    async def set_parameter_setpoint(self, 
                               parameter: OxfordNanonisSignalNames | str, 
                               setpoint: float,
                               approaching_setpoint: float = None) -> None:
        """
        Sets the setpoint of a parameter.
        
        Parameters
        ----------
        parameter : OxfordNanonisSignalNames
            The parameter to set.
        setpoint : float
            The setpoint to set.
        approaching_setpoint : float
            The setpoint value to approach the setpoint.
            By default is calculated at 1.001 * setpoint.
        """
        if isinstance(parameter, str):
            # Setpoint needs to be a known, to measure the corresponding signal.
            parameter = OxfordNanonisSignalNames(parameter)
        ## Check known mapping.
        # Define the setpoint signals.
        setpoint_signals = {
            tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT: tramea_signals.HE3_PROBE_TEMPERATURE,
            tramea_signals.MAGNETIC_FIELD_SETPOINT: tramea_signals.MAGNETIC_FIELD
        }
        if parameter not in setpoint_signals:
            raise ValueError("Parameter must be either temperature or magnetic field setpoint.")        
        
        # Setup the approach setpoint if not given.
        if approaching_setpoint is None:
            match parameter:
                case tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT:
                    # Temperature is positive definite.
                    approaching_setpoint = setpoint + 0.001 
                case tramea_signals.MAGNETIC_FIELD_SETPOINT:
                    # Field can be positive or negative, but can't exceed certain values.
                    approaching_setpoint = setpoint*0.999 # Don't exceed setpoint magnitude.
                
        # Determine the approaching direction:
        if setpoint == approaching_setpoint:
            raise ValueError("Setpoint and approaching setpoint cannot be the same.")
 
        direction = 1 if setpoint > approaching_setpoint else 0
        limits = (approaching_setpoint, setpoint) if direction else (setpoint, approaching_setpoint)
        
        # Store old sweep signal and limits
        current_sweep = await self.get_sweep_signal()
        current_limits = await self.get_limits()
        current_settings = await self.get_sweep_parameters()
        
        # Set the new sweep signal and limits
        await self.set_sweep_signal(parameter)
        await self.set_limits(limits)
        await self.set_sweep_parameters(
            initial_settling_time=300,
            maximum_slew_rate=current_settings[1],
            number_of_steps=current_settings[2],
            period=1000,
            autosave=False,
            save_dialog=False,
            settling_time=current_settings[6]
        )
        
        # Start the sweep
        await self.start(
            get_data=False,
            sweep_direction=direction,
            save_basename="setpoint_change",
            reset_signal=False
        )
        
        # Restore settings:
        await self.set_sweep_signal(current_sweep)
        await self.set_limits(current_limits)
        await self.set_sweep_parameters(*current_settings)
        return
    
    async def use_ramp_to_recondense(self):
        """"
        An asyncronous function to recondense the He3 probe using the ramp signal.
        
        Requires asynchronous, because the ramp will never reach the 0 K setpoint.
        Therefore a `stop` signal must be sent to the Nanonis controller asynchronously
        after the setpoint is initialised.
        """
        init_params = await self.get_sweep_parameters()
        init_sweep = await self.get_sweep_signal()
        await self.set_sweep_signal(tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT)
        # this is the setpoint required to recondense the He3 probe
        base_temp_reset = (0, 0.001) #ramp from 0K
        await self.set_limits(base_temp_reset)
        
        temperature_setpoint_params = {
            "initial_settling_time" : 300,
            "maximum_slew_rate" : init_params[1],
            "number_of_steps" : init_params[2],
            "period" : 1000,
            "autosave" : False,
            "save_dialog" : False,
            "settling_time" : init_params[6]
        }
        await self.set_sweep_parameters(**temperature_setpoint_params)
        # Save init_params + sweep signal in case of async timeout.
        self._async_timeout_params = (init_sweep, init_params)
        
        await self.start(
            get_data=False,
            sweep_direction=1,
            save_basename="recondense_he3_probe",
            reset_signal=False
        )
        
        await self.set_sweep_signal(init_sweep)
        await self.set_sweep_parameters(**init_params)
        self._async_timeout_params = None
    
    async def stop_ramp(self) -> None:
        """ Stops the async 1DSwp ramp. """
        print("Stopping ramp signal...")
        await self.mod_swp.Stop()
        print("Stopped.")
        if self._async_timeout_params:
            sweep, params = self._async_timeout_params
            await self.set_sweep_signal(sweep)
            await self.set_sweep_parameters(**params)
            self._async_timeout_params = None
            
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
            print("Recondensation initalised. Stopping ramp signal.")
            await self.stop_ramp()
        # Wait for the He3 probe to recondense to the base temperature.
        uSigs, uStd = await self.check_unstability(
            signals=[tramea_signals.HE3_PROBE_TEMPERATURE],
            standard_deviations=[std],
            setpoints=[base],
            time_to_measure=time,
            output_index=8
        )
        while len(uSigs) > 0:
            tVal = await self.get_signal_value(tramea_signals.HE3_PROBE_TEMPERATURE)
            print(f"Temp. unstable: {tVal} +- {uStd[tramea_signals.HE3_PROBE_TEMPERATURE]} K. Re-measuring...")
            uSigs, uStd = await self.check_unstability(
                signals=[tramea_signals.HE3_PROBE_TEMPERATURE],
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
                                          approach_setpoint: float = None) -> None:
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
        """
        # Check if signal is temperature or field:
        if isinstance(parameter, str):
            # Setpoint needs to be a known, to measure the corresponding signal.
            parameter = OxfordNanonisSignalNames(parameter)
        # Set param and param_setpoint disctinctions
        match parameter:
            case tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT:
                param_setpoint = tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT
                param = tramea_signals.HE3_PROBE_TEMPERATURE
            case tramea_signals.HE3_PROBE_TEMPERATURE:
                param_setpoint = tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT
                param = tramea_signals.HE3_PROBE_TEMPERATURE
            case tramea_signals.MAGNETIC_FIELD_SETPOINT:
                param_setpoint = tramea_signals.MAGNETIC_FIELD_SETPOINT
                param = tramea_signals.MAGNETIC_FIELD
            case tramea_signals.MAGNETIC_FIELD:
                param_setpoint = tramea_signals.MAGNETIC_FIELD_SETPOINT
                param = tramea_signals.MAGNETIC_FIELD
            case _:
                param_setpoint = parameter
                param = parameter
        
        await self.set_parameter_setpoint(
            parameter=param_setpoint,
            setpoint=setpoint,
            approaching_setpoint=approach_setpoint
        )
        uSig, uStd = await self.check_unstability(
            signals=[param],
            standard_deviations=[std],
            setpoints=[setpoint],
            time_to_measure=time
        )
        while len(uSig) > 0:
            # Some meas / sweep signals are not in the signals_sig list.
            if param.name in self.signals_sig:
                val = await self.get_signal_value(param)
                print(f"Parameter unstable: {val} +- {uStd[param]}. Re-measuring...")
            else:
                print(f"Parameter unstable: {param} (Variation:{uStd[param]}). Re-measuring...")
            uSig, uStd = await self.check_unstability(
                signals=[param],
                standard_deviations=[std],
                setpoints=[setpoint],
                time_to_measure=time
            )
        return
    
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
