from automated_scanning import measurement_control, tramea_signals, generate_log_setpoints
from nanonisTCP.nanonisTCP import nanonisTCP
import datetime as dt
import pytz as tz
import asyncio


aus_timezone = tz.timezone('Australia/Melbourne')
he3_condensation_max_time = dt.timedelta(days=2, hours=12)
    
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
        day=10,
        hour=13,
        minute=0,
        tzinfo=aus_timezone
    )
    
    ## Setup the measurement parameters.
    # Open 1D sweeper
    await control.mod_swp.Open()
    init_params = await control.get_sweep_parameters()
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
    await control.set_sweep_signal(tramea_signals.MAGNETIC_FIELD_SETPOINT)
    await control.set_limits(field_measurement_limits)
    await control.set_sweep_parameters(**field_measurement_params)
    
    # Temperature settings
    temperatures = generate_log_setpoints(
        limits=(0.27, 1.4),
        datapoints=20,
        forbidden_range=None
    )
    print("Scanning temperatures are set: ", list(temperatures))
    
    # Ramp field to initial value:
    print("Ramping field to initial value...")
    await control.set_parameter_and_stabilise(
            parameter=tramea_signals.MAGNETIC_FIELD_SETPOINT,
            setpoint=field_measurement_limits[0],
            std=0.005,
            time=20
        )
    # print("Field stabilised at ", 
        #   await control.get_signal_value(tramea_signals.MAGNETIC_FIELD_SETPOINT))
    print("Field Stabilised.")
    
    for temp in temperatures:
        name_format = f"Sweep_{temp}K_1V_DC_10MOhmR_100mTpMin"
        # Measurement loop!
        print("Checking if recondensation is required...")
        if dt.datetime.now(aus_timezone) - he3_condensation_time > he3_condensation_max_time:
            await control.condense_he3_probe()
            # update the time of the last recondensation
            he3_condensation_time = dt.datetime.now(aus_timezone)
        else:
            print("Done!")
        
        # Setup the correct temperature, and wait for stabilisation:
        await control.set_parameter_and_stabilise(
            parameter=tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT,
            setpoint=temp,
            std=0.005,
            time=120
        )
        print("Stabilised Temperature")
        # print("Stabilised at ", 
            #   await control.get_signal_value(tramea_signals.HE3_PROBE_TEMPERATURE))
        
        # Run the field sweep measurements, forward and backward.
        # These commmands shouldn't be neccessary (as set_parameter_and_stabilise 
        # should reset the signal and parameters for the field)
        await control.set_sweep_signal(tramea_signals.MAGNETIC_FIELD_SETPOINT)
        await control.set_limits(field_measurement_limits)
        await control.set_sweep_parameters(**field_measurement_params)
        
        print("Starting field sweep...")
        # Forward sweep
        await control.start(
            get_data=False,
            sweep_direction=1,
            save_basename=name_format,
            reset_signal=False
        )
        # Reverse sweep
        await control.start(
            get_data=False,
            sweep_direction=0,
            save_basename=name_format,
            reset_signal=False
        )
        print("Field sweep at ", temp, " K complete.")
    
    await NTCP.close_connection()
    

if __name__ == "__main__":
    asyncio.run(main())