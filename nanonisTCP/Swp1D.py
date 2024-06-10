# -*- coding: utf-8 -*-
"""
Created on Thurs May 30 3:11PM

@author: Daniel
"""
from nanonisTCP import nanonisTCP
import numpy as np
import asyncio

UINT32 = 4
UINT16 = 2
FLOAT32 = 4
INT_BYTES = 4

class Swp1D:
    """
    Tramea 1D Sweep
    """
    def __init__(self,NanonisTCP: nanonisTCP):
        self.nanonisTCP = NanonisTCP
        self.version = NanonisTCP.version
    
    
    async def AcqChsSet(self, channel_indexes: list[int]) -> None:
        """
        Sets the list of recorded channels of the 1D Sweeper.

        Parameters
        channel_indexes :    indexes of the recorded channels. The 
                            indexes correspond to the list of Measurement in the Nanonis software.
        """
        # Body Length Calculation
        body_len_bytes = INT_BYTES + len(channel_indexes)*INT_BYTES
        
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.AcqChsSet', body_size=body_len_bytes)
        
        ## Arguments
        # Number of channels:
        hex_rep += self.nanonisTCP.to_hex(len(channel_indexes), 4)
        # Channel indexes
        for ch in channel_indexes:
            hex_rep += self.nanonisTCP.to_hex(ch, 4)
        
        await self.nanonisTCP.send_command(hex_rep)
        await self.nanonisTCP.receive_response(0)
    
    async def AcqChsGet(self) -> list[int]:
        
        """
        Returns the list of recorded channels of the 1D Sweeper.

        Returns
        channel_indexes :    indexes of the recorded channels. The 
                            indexes correspond to the list of Measurement in the Nanonis software.
        """
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.AcqChsGet', body_size=0)
        
        ## Arguments
        # hex_rep += self.nanonisTCP.float32_to_hex(bias)                         # bias (float 32)
        
        await self.nanonisTCP.send_command(hex_rep)
        response = await self.nanonisTCP.receive_response()
    
            
        byte_counter = 0 # current response byte
        num_channels = self.nanonisTCP.hex_to_int32(response[byte_counter:byte_counter+INT_BYTES])
        byte_counter += INT_BYTES
        
        
        channel_indexes = []
        for n in range(byte_counter, (num_channels+1)*INT_BYTES, INT_BYTES):
            ch_index = self.nanonisTCP.hex_to_int32(response[n:n+INT_BYTES])
            channel_indexes.append(ch_index)
            byte_counter += INT_BYTES
        
        return channel_indexes
    
    async def SwpSignalGet(self) -> tuple[str, list[str]]:
        """
        Returns the selected Sweep signal in the 1D Sweeper.

        Returns
        -------
        tuple[str, list[str]]
            Sweep channel name, and list of signal names.
        """
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.SwpSignalGet', body_size=0)
        
        await self.nanonisTCP.send_command(hex_rep)
        
        response = await self.nanonisTCP.receive_response()
        
        sweep_channel_name_size = self.nanonisTCP.hex_to_int32(response[0:4])
        sweep_channel_name = response[4:4+sweep_channel_name_size].decode()
        idx = 4+sweep_channel_name_size
        
        
        channel_name_size = self.nanonisTCP.hex_to_int32(response[idx:idx+4])
        num_channels = self.nanonisTCP.hex_to_int32(response[idx+4:idx+8])
        idx += 8
        channel_names = []
        for n in range(num_channels):
            size = self.nanonisTCP.hex_to_int32(response[idx:idx+4])
            idx += 4
            signal_name = response[idx:idx+size].decode()
            idx += size
            channel_names.append(signal_name)
        return (sweep_channel_name, channel_names)
        
    async def SwpSignalSet(self, sweep_channel_name:str) -> None:
        """Sets the Sweep signal in the 1D Sweeper"""
        
        # Body Length Calculation
        body_len_bytes = INT_BYTES + len(sweep_channel_name)
        
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.SwpSignalSet', body_size=body_len_bytes)
        
        ## Arguments
        # Sweep channel length:
        hex_rep += self.nanonisTCP.to_hex(len(sweep_channel_name), 4)
        # Sweep channel name
        hex_rep += self.nanonisTCP.string_to_hex(sweep_channel_name)
        
        await self.nanonisTCP.send_command(hex_rep)
        await self.nanonisTCP.receive_response(0)
        
        return
    
    async def LimitsGet(self) -> tuple[float, float]:
        """Returns the limits of the 1D Sweeper"""
        
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.LimitsGet', body_size=0)
        await self.nanonisTCP.send_command(hex_rep)
        response = await self.nanonisTCP.receive_response(8)
        lower_limit = self.nanonisTCP.hex_to_float32(response[0:4])
        upper_limit = self.nanonisTCP.hex_to_float32(response[4:8])
        return (lower_limit, upper_limit)
    
    async def LimitsSet(self, lower_limit:float, upper_limit:float) -> None:
        """Sets the limits of the 1D Sweeper"""
        
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.LimitsSet', body_size=2*FLOAT32)
        
        
        # print(lower_limit)
        # print(upper_limit)
        
        
        # print(self.nanonisTCP.hex_to_float32(
        #     bytes.fromhex(self.nanonisTCP.float32_to_hex(upper_limit))
        #     ))
        ## Arguments
        # Lower limit
        hex_rep += self.nanonisTCP.float32_to_hex(lower_limit)
        # Upper limit
        hex_rep += self.nanonisTCP.float32_to_hex(upper_limit)
        
        await self.nanonisTCP.send_command(hex_rep)
        await self.nanonisTCP.receive_response(0)
        return
        
    async def PropsSet(self, 
                 initial_settling_time: float,
                 maximum_slew_rate: float,
                 number_of_steps: int,
                 period: int,
                 autosave: bool,
                 save_dialog: bool,
                 settling_time: float) -> None:
        """Sets the properties of the 1D Sweeper
        
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
        body_len_bytes = 3*FLOAT32 + 3*UINT32 + 1*UINT16
        if period > 2**16-1:
            raise ValueError("Period is too large for a 16 bit unsigned number.")
        
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.PropsSet', body_size=body_len_bytes)
        
        ## Arguments
        # Initial settling time
        hex_rep += self.nanonisTCP.float32_to_hex(initial_settling_time)
        # Maximum slew rate
        hex_rep += self.nanonisTCP.float32_to_hex(maximum_slew_rate)
        # Number of steps
        hex_rep += self.nanonisTCP.to_hex(number_of_steps, 4)
        # Period
        hex_rep += self.nanonisTCP.to_hex(period, 2)
        # Autosave
        hex_rep += self.nanonisTCP.to_hex(autosave, 4)
        # Save dialog
        hex_rep += self.nanonisTCP.to_hex(save_dialog, 4)
        # Settling time
        hex_rep += self.nanonisTCP.float32_to_hex(settling_time)
        
        await self.nanonisTCP.send_command(hex_rep)
        await self.nanonisTCP.receive_response(0)
        
    async def PropsGet(self) -> tuple[float, float, int, int, bool, bool, float]:
        """Returns the properties of the 1D Sweeper"""
        
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.PropsGet', body_size=0)
        
        await self.nanonisTCP.send_command(hex_rep)
        response = await self.nanonisTCP.receive_response(3*FLOAT32 + 3*UINT32 + UINT16)
        
        initial_settling_time = self.nanonisTCP.hex_to_float32(response[0:4])
        maximum_slew_rate = self.nanonisTCP.hex_to_float32(response[4:8])
        number_of_steps = self.nanonisTCP.hex_to_int32(response[8:12])
        period = self.nanonisTCP.hex_to_uint16(response[12:14])
        autosave = bool(self.nanonisTCP.hex_to_uint32(response[14:18]))
        save_dialog = bool(self.nanonisTCP.hex_to_uint32(response[18:22]))
        settling_time = self.nanonisTCP.hex_to_float32(response[22:26])
        
        return (initial_settling_time, maximum_slew_rate, number_of_steps, period, autosave, save_dialog, settling_time)
    
    async def Start(self, get_data:bool, sweep_direction:int, save_basename:str, reset_signal:bool):
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.Start', body_size=3*UINT32 + INT_BYTES + len(save_basename))
        
        # Arguments
        hex_rep += self.nanonisTCP.to_hex(get_data, 4)
        hex_rep += self.nanonisTCP.to_hex(sweep_direction,4)
        hex_rep += self.nanonisTCP.to_hex(len(save_basename), 4)
        hex_rep += self.nanonisTCP.string_to_hex(save_basename)
        hex_rep += self.nanonisTCP.to_hex(reset_signal, 4)
       
        # Send command and recieve
        await self.nanonisTCP.send_command(hex_rep)
        response = await self.nanonisTCP.receive_response()
        
        # Response
        channel_name_size = self.nanonisTCP.hex_to_int32(response[0:4])
        num_channels = self.nanonisTCP.hex_to_int32(response[4:8])
        channel_names = []
        idx=8
        for n in range(num_channels):
            size = self.nanonisTCP.hex_to_int32(response[idx:idx+4])
            idx += 4
            signal_name = response[idx:idx+size].decode()
            idx += size
            channel_names.append(signal_name)
        
        data_rows = self.nanonisTCP.hex_to_int32(response[idx:idx+4])
        data_cols = self.nanonisTCP.hex_to_int32(response[idx+4:idx+8])
        
        idx = idx + 8
        data = []
        for i in range(data_rows):
            col = []
            for j in range(data_cols):
                col.append(self.nanonisTCP.hex_to_float32(response[idx : idx + 4]))
                idx += 4
            data.append(col)
        return (channel_names, np.array(data))
    
    async def Stop(self):
        """Stops the sweep of the 1D Sweeper"""
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.Stop', body_size=0)
        await self.nanonisTCP.send_command(hex_rep)
        response = await self.nanonisTCP.receive_response(0)
        return
    
    async def Open(self) -> None:
        """Opens the 1D Sweeper module"""
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.Open', body_size=0)
        await self.nanonisTCP.send_command(hex_rep)
        response = await self.nanonisTCP.receive_response(0)
        return