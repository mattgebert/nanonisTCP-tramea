from automated_scanning import measurement_control, tramea_signals, generate_log_setpoints
from nanonisTCP.nanonisTCP import nanonisTCP
import datetime as dt
import pytz as tz
import numpy as np
import asyncio
import os


aus_timezone = tz.timezone('Australia/Melbourne')
he3_condensation_max_time = dt.timedelta(days=2, hours=6)
    
async def main():
    # Connect to the Nanonis controller.
    TCP_IP = '127.0.0.1'
    TCP_PORT = 6501
    NTCP = await nanonisTCP.create(TCP_IP, TCP_PORT)
    assert isinstance(NTCP, nanonisTCP) # For typing...
    control = await measurement_control.create(NTCP)
    
    ## Define He3 probe condensation time.
    he3_condensation_time = dt.datetime(
        year=2024,
        month=6,
        day=18,
        hour=10,
        minute=00,
        tzinfo=aus_timezone
    )
    
    ## Setup the measurement parameters.
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
        tramea_signals.HE3_PROBE_TEMPERATURE,
        tramea_signals.MAGNETIC_FIELD,
    ])
    # Field settings
    field_measurement_params = {
        "initial_settling_time" : 300,
        "maximum_slew_rate" : init_params[1],
        "number_of_steps" : init_params[2],
        "period" : 1000,
        "autosave" : True,
        "save_dialog" : False,
        "settling_time" : init_params[6]
    }
    field_measurement_limits = (-7, 7)
    await control.swp_set_sweep_signal(tramea_signals.MAGNETIC_FIELD_SETPOINT)
    await control.swp_set_limits(field_measurement_limits)
    await control.swp_set_parameters(**field_measurement_params)
    
    # Temperature settings
    temperatures = np.round(generate_log_setpoints(
        limits=(0.27, 1.4),
        datapoints=20,
        forbidden_range=None
    ), 5)
    temperatures = temperatures[8:] #MANUAL SUBSAMPLE TODO REMOVE
    print(dt.datetime.now(), "Scanning temperatures are set: ", list(temperatures))
    
    # Ramp field to initial value:
    print(dt.datetime.now(), "Ramping field to initial value...")
    await control.set_parameter_and_stabilise(
            parameter=tramea_signals.MAGNETIC_FIELD_SETPOINT,
            setpoint=field_measurement_limits[0],
            std=0.005,
            time=20
        )
    # print("Field stabilised at ", 
        #   await control.get_signal_value(tramea_signals.MAGNETIC_FIELD_SETPOINT))
    print(dt.datetime.now(), "Field Stabilised.")
    
    for temp in temperatures:
        name_format = f"Sweep_{temp:3.3f}K_1V_DC_10MOhmR_100mTpMin"
        # Check if temperature is already completed
        docs = os.path.join("", "C:/Users/Nanonis Tramea/Documents")
        dir = os.path.join(docs, 
            "Data saving/User/Matt/2024-05-22 TbIG BT #25 Chip#1/2024-06-10 Third Measurement Sweep +- 7T"
        )
        if (os.path.exists(os.path.join(dir, name_format + "00001.dat")) and 
            os.path.exists(os.path.join(dir, name_format + "00002.dat"))):
            print(dt.datetime.now(), f"Temperature {temp:3.3f} already measured, skipping...")
            continue # skip to next for loop entry.
        
        # Measurement loop!
        print(dt.datetime.now(), "Checking if recondensation is required...")
        if (dt.datetime.now(aus_timezone) 
            - he3_condensation_time) > he3_condensation_max_time:
            print(dt.datetime.now(), "Recondensing He3 probe...")
            await control.condense_he3_probe()
            # update the time of the last recondensation
            he3_condensation_time = dt.datetime.now(aus_timezone)
            print(dt.datetime.now(), "Done")
        else:
            print(dt.datetime.now(), "Not required!")
        
        # Setup the correct temperature, and wait for stabilisation:
        print(dt.datetime.now(), "Setting temperature to ", temp, " K..."	)
        await control.set_parameter_and_stabilise(
            parameter=tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT,
            setpoint=temp,
            std=0.005,
            time=120
        )
        print(dt.datetime.now(), "Stabilised Temperature")
        # print("Stabilised at ", 
            #   await control.get_signal_value(tramea_signals.HE3_PROBE_TEMPERATURE))
        
        # Run the field sweep measurements, forward and backward.
        # These commmands shouldn't be neccessary (as set_parameter_and_stabilise 
        # should reset the signal and parameters for the field)
        await control.swp_set_sweep_signal(tramea_signals.MAGNETIC_FIELD_SETPOINT)
        await control.swp_set_limits(field_measurement_limits)
        await control.swp_set_parameters(**field_measurement_params)
        
        print(dt.datetime.now(), "Starting field sweep...")
        # Forward sweep
        await control.swp_ramp_start(
            get_data=False,
            sweep_direction=1,
            save_basename=name_format,
            reset_signal=False
        )
        print(dt.datetime.now(), "Starting reverse sweep...")
        # Reverse sweep
        await control.swp_ramp_start(
            get_data=False,
            sweep_direction=0,
            save_basename=name_format,
            reset_signal=False
        )
        print(dt.datetime.now(), "Field sweep at ", temp, " K complete.")
    
    await NTCP.close_connection()
    

if __name__ == "__main__":
    asyncio.run(main())