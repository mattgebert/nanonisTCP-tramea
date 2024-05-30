# -*- coding: utf-8 -*-
"""
Created on Thurs May 30 3:11PM

@author: Daniel
"""
from nanonisTCP import nanonisTCP
import numpy as np

UINT32 = 4
INT_BYTES = 4

class Swp1D:
    """
    Tramea 1D Sweep
    """
    def __init__(self,NanonisTCP: nanonisTCP):
        self.nanonisTCP = NanonisTCP
        self.version = NanonisTCP.version
    
    def AcqChshGet(self) -> list[int]:
        
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
        
        self.nanonisTCP.send_command(hex_rep)
        
        response = self.nanonisTCP.receive_response()
    
            
        byte_counter = 0 # current response byte
        num_channels = self.nanonisTCP.hex_to_int32(response[byte_counter:byte_counter+INT_BYTES])
        byte_counter += INT_BYTES
        
        
        channel_indexes = []
        for n in range(byte_counter, (num_channels+1)*INT_BYTES, INT_BYTES):
            ch_index = self.nanonisTCP.hex_to_int32(response[n:n+INT_BYTES])
            channel_indexes.append(ch_index)
            byte_counter += INT_BYTES
        
        return channel_indexes
    
    
    def Start(self, get_data:bool, sweep_direction:int, save_basename:str, reset_signal:bool):
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('1DSwp.Start', body_size=3*UINT32 + INT_BYTES + len(save_basename))
        
        # Arguments
        hex_rep += self.nanonisTCP.to_hex(get_data, 4)
        hex_rep += self.nanonisTCP.to_hex(sweep_direction,4)
        hex_rep += self.nanonisTCP.to_hex(len(save_basename), 4)
        hex_rep += self.nanonisTCP.string_to_hex(save_basename)
        hex_rep += self.nanonisTCP.to_hex(reset_signal, 4)
       
        # Send command and recieve
        self.nanonisTCP.send_command(hex_rep)
        response = self.nanonisTCP.receive_response()
        
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
        return (channel_name_size, channel_names, np.array(data))