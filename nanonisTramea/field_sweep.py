"""
Script to measure the multiple properties of a transport device in an Oxford He3 Probe / VTI cryostat.

This script assumes the following:
- The cryostat is connected to the Nanonis Tramea system (modules for the VTI and magnet are required).
- The names of the signals are defined in the tramea_signals.py file (you can edit these to match your system).
- The He3 temperature port is connected to the VTI controller as the extra temperature sensor.
- As we are using the VTI port, we are assuming we have no control over the He3 Temperature (including recondensation)
- Both He3 heater and He4 heater are set to the same base setpoint. Then He3 heater is used to ramp to the desired setpoint.
"""

from automated_scanning import measurement_control, tramea_signals, generate_log_setpoints
from nanonisTCP.nanonisTCP import nanonisTCP
import datetime as dt
import pytz as tz
import numpy as np
import asyncio
import os

aus_timezone = tz.timezone('Australia/Melbourne')

# Acquisition channels
SOURCE_ACQ_CHANNELS = [
    tramea_signals.DC_INPUT1,
    tramea_signals.DC_INPUT2,
    tramea_signals.DC_INPUT3,
    tramea_signals.DC_INPUT4,
]
TEMP_ACQ_CHANNELS = SOURCE_ACQ_CHANNELS + [
    tramea_signals.TIME,
    tramea_signals.HE4_EXTRA_TEMP_SENSOR,
    tramea_signals.HE4_VTI_TEMPERATURE,
    tramea_signals.MAGNETIC_FIELD
]
FIELD_ACQ_CHANNELS = TEMP_ACQ_CHANNELS #Same as temperature acquisition channels
# Measurement parameters
TEMP_MEAS_PARAMS = {
        "initial_settling_time":    300,
        "period":                   1000,
        "autosave":                 True,
        "save_dialog":              False,
        # "maximum_slew_rate":init_params[1],
        # "number_of_steps":init_params[2],
        # "settling_time":init_params[6],
    }
CONTROL_V_SIGNAL = tramea_signals.DC_OUTPUT2
CONTROL_V_DEFAULT = 1.0 #volts

# GENERATE SWEEPING PARAMS
SWEEP_VOLTAGES = generate_log_setpoints(
        limits=(1e-6, 5),
        datapoints=50,
        forbidden_range=None
    )
SWEEP_TEMPERATURES = np.round(
    generate_log_setpoints(
        limits=(1.8, 100),
        datapoints=20,
        forbidden_range=None
), 5)
SWEEP_FIELD_LIMITS = (-7, 7)

## Source settings
VOLT_MEAS_PARAMS = {
    "output_idx":CONTROL_V_SIGNAL,
    "output_setpoints":SWEEP_VOLTAGES,
    "meas_sigs":SOURCE_ACQ_CHANNELS,
    "meas_to_average":20,
    "meas_period":0.3
}

# File saving
docs = os.path.join("", "C:/Users/Nanonis Tramea/Documents")
usr_folder = os.path.join(docs, "Data saving/User/Matt/2024-05-22 TbIG BT #25 Chip#1/")
dir = os.path.join(usr_folder, "2024-06-21 Sixth HighT Characterisation")

async def program() -> None:
    """Async function to run the measurement program.    
    """
    
    # Connect to the Nanonis controller.
    TCP_IP = '127.0.0.1'
    TCP_PORT = 6501
    NTCP = await nanonisTCP.create(TCP_IP, TCP_PORT)
    assert isinstance(NTCP, nanonisTCP) # For typing...
    control = await measurement_control.create(NTCP)
    
    ## Setup the measurement parameters.
    # Open 1D sweeper
    await control.mod_swp.Open()
    init_params = await control.swp_get_parameters()
    init_params = {
        "initial_settling_time":init_params[0],
        "maximum_slew_rate":init_params[1],
        "number_of_steps":init_params[2],
        "period":init_params[3],
        "autosave":init_params[4],
        "save_dialog":init_params[5],
        "settling_time":init_params[6],
    }
    # Update temp meas params to use default values if not specified
    TEMP_MEAS_PARAMS.update({
        param:val 
        for param,val in init_params.items() 
        if param not in TEMP_MEAS_PARAMS
    })
    
    
    # Printout the measurement parameters
    print(dt.datetime.now(), "Scanning voltages are set: ", list(SWEEP_VOLTAGES))
    print(dt.datetime.now(), "Scanning temperatures are set: ", list(SWEEP_TEMPERATURES))
    print(dt.datetime.now(), "Field sweep limits are set: ", SWEEP_FIELD_LIMITS)
    
    # Setup the default voltage
    await control.set_parameter_setpoint(CONTROL_V_SIGNAL, CONTROL_V_DEFAULT)
    # Setup the default 1D Swp acquisition channels
    await control.swp_set_acquisition_channels(TEMP_ACQ_CHANNELS)
    
    # current_temp = await control.get_parameter_value(tramea_signals.HE4_VTI_TEMPERATURE)
    temp = current_temp = await control.get_parameter_value(tramea_signals.HE4_EXTRA_TEMP_SENSOR)
    name_format_field = f"Sweep_{temp:3.3f}K_1V_DC_10MOhmR_100mTpMin"
    # Check if the files already exist
    # Field
    exist1 = os.path.exists(os.path.join(dir, name_format_field + "00001.dat")) and os.path.exists(os.path.join(dir, name_format_field + "00002.dat"))
    if exist1:
        print(dt.datetime.now(), f"Temperature {temp}K already measured. Skipping and finishing program.")
        return
    
    # 2. Setup the magnetic field.
    print(dt.datetime.now(), "Ramping field to initial field", SWEEP_FIELD_LIMITS[0], "T...")
    # Get initial value.
    val = await control.get_parameter_value(tramea_signals.MAGNETIC_FIELD)
    await control.set_parameter_and_stabilise(
        parameter=tramea_signals.MAGNETIC_FIELD_SETPOINT,
        setpoint=SWEEP_FIELD_LIMITS[0],
        std=0.05,
        time=20,
        approach_setpoint=val               # Approach the setpoint from the current value  
    )
    print(dt.datetime.now(), "Field stabilised.")
            
    # Wait 5 seconds
    await asyncio.sleep(5)
    
    # 3. Sweep magnetic field.
    print(dt.datetime.now(), "Setting up magnetic field sweep...")
    await control.swp_set_acquisition_channels(FIELD_ACQ_CHANNELS)
    await control.swp_set_sweep_signal(tramea_signals.MAGNETIC_FIELD_SETPOINT)
    await control.swp_set_limits(SWEEP_FIELD_LIMITS)
    await control.swp_set_parameters(**TEMP_MEAS_PARAMS)
    print(dt.datetime.now(), "Running magnetic field up-sweep...")
    await control.swp_ramp_start(
        get_data=True,
        sweep_direction=1,
        save_basename=f"FieldSweep_{temp:3.3f}K",
        reset_signal=False
    )
    print(dt.datetime.now(), "Running magnetic field down-sweep...")
    await control.swp_ramp_start(
        get_data=True,
        sweep_direction=0,
        save_basename=f"FieldSweep_{temp:3.3f}K",
        reset_signal=False
    )

if __name__ == "__main__":
    asyncio.run(program())