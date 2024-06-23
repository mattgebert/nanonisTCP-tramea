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
        day=19,
        hour=11,
        minute=00,
        tzinfo=aus_timezone
    )
    
    ## Setup the measurement parameters.
    # Open 1D sweeper
    await control.mod_swp.Open()
    init_params = await control.swp_get_parameters()
    # Acquisition channels
    source_acq_channels = [
        tramea_signals.DC_INPUT1,
        tramea_signals.DC_INPUT2,
        tramea_signals.DC_INPUT3,
        tramea_signals.DC_INPUT4,
    ]
    temp_acq_channels = source_acq_channels + [
        tramea_signals.TIME,
        tramea_signals.HE3_PROBE_TEMPERATURE,
        tramea_signals.MAGNETIC_FIELD
    ]
    await control.swp_set_acquisition_channels(temp_acq_channels)
    temp_meas_params = {
        "initial_settling_time":300,
        "maximum_slew_rate":init_params[1],
        "number_of_steps":init_params[2],
        "period":1000,
        "autosave":True,
        "save_dialog":False,
        "settling_time":init_params[6],
    }
    
    
    # Voltage settings
    control_voltage = tramea_signals.DC_OUTPUT2
    default_voltage = 1.0 #volts
    voltages = generate_log_setpoints(
        limits=(1e-6, 5),
        datapoints=50,
        forbidden_range=None
    )
    control.set_parameter_setpoint(
        parameter=control_voltage,
        setpoint=default_voltage
    )
    print(dt.datetime.now(), "Scanning voltages are set: ", list(voltages))
    
    # Temperature settings
    temperatures = np.r_[[0.256], np.round(
        generate_log_setpoints(
            limits=(0.27, 1.4),
            datapoints=20,
            forbidden_range=None
        ), 5)]
    print(dt.datetime.now(), "Scanning temperatures are set: ", list(temperatures))
    
    # Ramp voltage to initial value:
    print(dt.datetime.now(), f"Ramping voltage to initial value {voltages[0]}...")
    await control.set_parameter_setpoint(control_voltage, voltages[0])
    
    print(dt.datetime.now(), f"Measuring voltage stability...")
    ave_val = await control.set_parameter_and_stabilise(
            parameter=control_voltage,
            setpoint=voltages[0],
            std=0.1*abs(voltages[0]), # 10% of the setpoint value
            time=2
        )
    print("Voltage stabilised at: ", ave_val, ", most recent val is: ",
          await control.get_parameter_value(control_voltage))
    
    for i, temp in enumerate(temperatures):
        # Define the filename
        name_format = f"Poll_{temp:3.3f}K_10MOhmR_Voltage_Acquisition"
        # Check if temperature is already completed
        docs = os.path.join("", "C:/Users/Nanonis Tramea/Documents")
        dir = os.path.join(docs, 
            "Data saving/User/Matt/2024-05-22 TbIG BT #25 Chip#1/2024-06-19 Fifth Source Temperature Characterisation"
        )
        filename = os.path.join(dir, name_format + "00001.dat")
        exist1 = os.path.exists(filename)
        if i < len(temperatures) - 1:
            name_format2 = f"Sweep_{temp:3.3f}K_to_{temperatures[i+1]:3.3f}K_10MOhmR_Temperature_Acquisition"
            exist2 = os.path.exists(os.path.join(dir, name_format2 + "00001.dat"))
            if exist1 and exist2:
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
            time=60*5 #Stabilise for 5 minutes.
        )
        await asyncio.sleep(2*60) #wait for 2 minutes more.
        print(dt.datetime.now(), "Stabilised Temperature")
        
        print(dt.datetime.now(), "Starting voltage sweep...")
        # Run the voltage sweep measurements.
        vals, errs = await control.out_time_measure_sigs(
            output_idx=control_voltage,
            output_setpoints=voltages,
            meas_sigs=source_acq_channels,
            meas_to_average=20,
            meas_period=0.3
        )
        print(dt.datetime.now(), "Voltage sweep at ", temp, " K complete.")
        
        # Reset voltage to control parameter to default value (this should be automatically
        # done in control.out_time_measure_sigs, but repeat for saftey).
        await control.set_parameter_setpoint(
            parameter=control_voltage,
            setpoint=default_voltage
        )
        
        # Save the data.
        print(dt.datetime.now(), "Writing data to '", filename, "'.")
        with open(filename, "w") as f:
            # HEADER
            f.write("Experiment:\tTemperature and source dependent resistance\n")
            f.write(f"Saved Date:\t{dt.datetime.now(aus_timezone)}\n")
            f.write(f"Sweep signal ENUM:\t{control_voltage.name}\n")
            f.write(f"Sweep signal:\t{control_voltage.value}\n")
            #DATA
            f.write("\n[DATA]\n")
            # DATA NAMES
            f.write(f"{control_voltage.name}\t")
            f.write(f"Std {control_voltage.name}\t")
            for channel in source_acq_channels:
                f.write(f"{channel.name}\t")
                f.write(f"Std {channel.name}")
                if channel != source_acq_channels[-1]:
                    f.write("\t")
                else:
                    f.write("\n")
            # DATA VALUES
            for j in range(vals.shape[0]):
                for k in range(vals.shape[1]):
                    f.write(f"{vals[j, k]}\t")
                    f.write(f"{errs[j, k]}")
                    if k != vals.shape[1] - 1:
                        f.write("\t")
                    else:
                        f.write("\n")
        print(dt.datetime.now(), "Finished writing.")
                        
        # If another temperature setpoint exists, ramp to the temperature setpoint 
        # and record the data.
        if i < len(temperatures) - 1:
            temp_meas_limits = (temp, temperatures[i+1])
            
            await control.swp_set_sweep_signal(tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT)
            await control.swp_set_limits(temp_meas_limits)
            await control.swp_set_parameters(**temp_meas_params)
            
            print(dt.datetime.now(), f"Measuring ramp temperature between {temp} and {temperatures[i+1]}...")
            # Forward sweep
            await control.swp_ramp_start(
                get_data=False,
                sweep_direction=1,
                save_basename=name_format2,
                reset_signal=False
            )
            print(dt.datetime.now(), "Temperature sweep complete.")
            
            # Wait 10 minutes to stabilise it.
            await asyncio.sleep(10*60)
                        
    await NTCP.close_connection()
    

if __name__ == "__main__":
    asyncio.run(main())