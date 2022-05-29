# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:40:40 2022

@author: Julian Ceddia
"""
"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!WARNING: RUNNING run_test() WILL CHANGE SETTINGS IN NANONIS. RUN IT ON A SIM!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

from nanonisTCP import nanonisTCP
from nanonisTCP.Current import Current

"""
Set up the TCP connection and interface
"""
def run_test(TCP_IP='127.0.0.1', TCP_PORT=6501):
    # Listening Port: see Nanonis File>Settings>TCP Programming Interface
    NTCP = nanonisTCP(TCP_IP, TCP_PORT)                                         # Nanonis TCP interface
    try:
        currentModule = Current(NTCP)                                           # Nanonis TCP Current Module
        
        """
        Current Get
        """
        current     = currentModule.Get()
        # current100  = currentModule.Get100()
        # currentBEEM = currentModule.BEEMGet()
        print("Current Get")
        print("Current:      " + str(current) + " A")
        # print("100 Current:  " + str(current100))
        # print("BEEM Current: " + str(currentBEEM))
        print("----------------------------------------------------------------------")
        
        """
        Current Gains Get/Set
        """
        currentModule.GainSet(13)
        gains,gain_index = currentModule.GainsGet()
        print("Current Gains")
        print("Gain: " + gains[gain_index])
        print("----------------------------------------------------------------------")
        
        """
        Calibration Set/Get
        """
        currentModule.CalibrSet(calibration=1.1e-9, offset=-1.69e-12)
        calibration,offset = currentModule.CalibrGet()
        print("Calibration")
        print("Calibration Factor: " + str(calibration) + " (A/V)")
        print("Offset:             " + str(offset) + " A")
        print("----------------------------------------------------------------------")
    except:
        print('error')
    NTCP.close_connection()
    return print('end of test')