from automated_scanning import measurement_control, tramea_signals, generate_log_setpoints
from nanonisTCP.nanonisTCP import nanonisTCP
import datetime as dt
import pytz as tz
import numpy as np
import asyncio
import os
aus_timezone = tz.timezone('Australia/Melbourne')

SWEEP_TEMPERATURES = np.round(
    generate_log_setpoints(
        limits=(1.8, 100),
        datapoints=20,
        forbidden_range=None
), 5)
print(SWEEP_TEMPERATURES)

async def main():
    
    # Connect to the Nanonis controller.
    TCP_IP = '127.0.0.1'
    TCP_PORT = 6501
    NTCP = await nanonisTCP.create(TCP_IP, TCP_PORT)
    assert isinstance(NTCP, nanonisTCP) # For typing...
    control = await measurement_control.create(NTCP)
    
    # Setup the measurement parameters.
    # Open 1D sweeper
    await control.mod_swp.Open()
    init_params = await control.swp_get_parameters()
    # Acquisition channels
    await control.swp_set_acquisition_channels([
        tramea_signals.TIME,
        tramea_signals.DC_INPUT1,
        tramea_signals.DC_INPUT2,
        tramea_signals.DC_INPUT3,
        tramea_signals.DC_INPUT4,
        tramea_signals.MAGNETIC_FIELD,
        tramea_signals.HE4_EXTRA_TEMP_SENSOR,
        tramea_signals.HE4_VTI_TEMPERATURE,
    ])
    # Field settings
    temp_measurement_params = {
        "initial_settling_time" : 300,
        "maximum_slew_rate" : init_params[1],
        "number_of_steps" : init_params[2],
        "period" : 1000,
        "autosave" : True,
        "save_dialog" : False,
        "settling_time" : init_params[6]
    }
    # Field limits
    temp_measurement_limits = (1.5, 2)
    
    await control.swp_set_sweep_signal(tramea_signals.HE4_EXTRA_TEMP_SENSOR_SETPOINT)
    await control.swp_set_limits(temp_measurement_limits)
    await control.swp_set_parameters(**temp_measurement_params)

if __name__ == "__main__":
    asyncio.run(main())