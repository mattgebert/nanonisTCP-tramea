from nanonisTCP import nanonisTCP
from nanonisTCP.Swp1D import Swp1D

TCP_IP = '127.0.0.1'
TCP_PORT = 6501

NTCP = nanonisTCP(TCP_IP, TCP_PORT)


sweeper = Swp1D(NTCP)

channels = sweeper.AcqChsGet()

# sweep = sweeper.Start(get_data=True,
#                       sweep_direction=1,
#                       save_basename="test_TCP_data",
#                       reset_signal=False)

# channel_name_size, channel_names, data = sweep

print(channels)

# print(channel_names, data)

NTCP.close_connection()