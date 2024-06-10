from automated_scanning import *
import asyncio

aus_timezone = tz.timezone('Australia/Melbourne')
he3_condensation_max_time = dt.timedelta(days=2, hours=12)
    
async def main():
    # Connect to the Nanonis controller.
    TCP_IP = '127.0.0.1'
    TCP_PORT = 6501
    NTCP = await nanonisTCP.create(TCP_IP, TCP_PORT)
    assert isinstance(NTCP, nanonisTCP) # For typing...
    control = measurement_control(NTCP)
    
    ## Perform He3 probe re-condensation if necessary.
    he3_condensation_time = dt.datetime(
        year=2024,
        month=6,
        day=10,
        hour=9,
        minute=0,
        tzinfo=aus_timezone
    )
    if dt.datetime.now(aus_timezone) - he3_condensation_time > he3_condensation_max_time:
        await control.condense_he3_probe()
        # update the time of the last recondensation
        he3_condensation_time = dt.datetime.now(aus_timezone)
    print("Done!")
    
    # set_list = [
    #     tramea_signals.DC_INPUT1,
    #     tramea_signals.DC_INPUT2,
    #     tramea_signals.DC_INPUT3,
    #     tramea_signals.DC_INPUT4,
    #     tramea_signals.TIME,
    #     tramea_signals.HE3_PROBE_TEMPERATURE,
    #     tramea_signals.MAGNETIC_FIELD,
    # ]
    # init_params = control.get_sweep_parameters()
    # # Use initial parameters to set un-needed values.
    # field_measurement_params = {
    #     "initial_settling_time" : 300,
    #     "maximum_slew_rate" : init_params[1],
    #     "number_of_steps" : init_params[2],
    #     "period" : 1000,
    #     "autosave" : True,
    #     "save_dialog" : False,
    #     "settling_time" : init_params[6]
    # }
    # temperature_setpoint_params = {
    #     "initial_settling_time" : 300,
    #     "maximum_slew_rate" : init_params[1],
    #     "number_of_steps" : init_params[2],
    #     "period" : 1000,
    #     "autosave" : False,
    #     "save_dialog" : False,
    #     "settling_time" : init_params[6]
    # }
    # temperature_lims = [
    #     (2.700000000000000178e-01, 0.271),
    #     (2.944305948215972801e-01, 0.295),
    #     (3.210717598777762527e-01, 0.322),
    #     (3.501235157082619454e-01, 0.351),
    #     (3.818039814481941630e-01, 0.383),
    #     (4.163510124557255332e-01, 0.417),
    #     (4.540239861181982195e-01, 0.455),
    #     (4.951057492446434560e-01, 0.496),
    #     (5.399047416655294862e-01, 0.541),
    #     (5.887573119836394531e-01, 0.590),
    #     (6.420302428626211144e-01, 0.643),
    #     (7.001235048129484939e-01, 0.701),
    #     (7.634732591505845045e-01, 0.764),
    #     (8.325551326744078384e-01, 0.834),
    #     (9.078877886485174908e-01, 0.909),
    #     (9.900368209001763065e-01, 0.991),
    #     (1.079619000270081841e+00, 1.081),
    #     (1.177306905297105022e+00, 1.178),
    #     (1.283833971904446480e+00, 1.285),
    #     (1.399999999999999911e+00, 1.401),
    # ]
        
    # # Initialize temperature setting.
    # control.set_sweep_signal(tramea_signals.HE3_PROBE_TEMPERATURE_SETPOINT)
    # control.set_limits(temperature_lims[0])
    # print("Initial limits:", control.get_limits())
    # control.set_acquisition_channels(set_list)
    # control.set_sweep_parameters(**temperature_setpoint_params)
    
    # # Run a quick temperature sweep to set temperature.
    # control.start(
    #     get_data=False,
    #     sweep_direction=0,
    #     save_basename="temperature_setpoint",
    #     reset_signal=False
    # )
    # # Wait for 30s for temperature to start adjusting.
    # print("Waiting for temperature to adjust...")
    # time.sleep(30)
    
    # # Check temperature is stable
    # print(f"Checking temperature stability at {temperature_lims[0][0]} K...")
    # stability_params = {
    #     "signals" : [tramea_signals.HE3_PROBE_TEMPERATURE],
    #     "standard_deviations" : [0.005],
    #     "setpoints" : [temperature_lims[0][0]],
    #     "time_to_measure" : 10,#120,
    #     "output_index" : 8
    # }
    # sigs, devs = control.check_unstability(**stability_params)
    # # While temperature is not stable, repeat check
    # while len(sigs) > 0:
    #     print(f"Temperature unstable... {devs[tramea_signals.HE3_PROBE_TEMPERATURE]} standard deviation")
    #     sigs, devs = control.check_unstability(**stability_params)
    # val = control.get_signal_value(tramea_signals.HE3_PROBE_TEMPERATURE)
    # print("Temperature stable at ", val)
        
    # # Run a magnetic field sweep!
    # control.set_sweep_signal(tramea_signals.MAGNETIC_FIELD_SETPOINT)
    # magnetic_field_range = (-7.0, 7.0)
    # control.set_limits(magnetic_field_range)
    
    # # Use analog output 8 to allow a slow time measurement by changing slew rate.
    
    # #### Use a scan to perform a setpoint change.
    # t_start = dt.datetime.now(aus_timezone)
    # # control.start(
    # #     get_data=False,
    # #     sweep_direction=0,
    # #     save_basename="test_TCP_data",
    # #     reset_signal=False
    # # )
    # # while dt.datetime.now(aus_timezone) - t_start < dt.timedelta(hours=3):
    # #     time.sleep(30)
    
    # # print(dt.datetime.now())
    
    await NTCP.close_connection()
    

if __name__ == "__main__":
    asyncio.run(main())