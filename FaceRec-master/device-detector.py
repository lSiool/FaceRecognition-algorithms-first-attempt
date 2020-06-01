from device_detector import DeviceDetector

ua = 'Mozilla/5.0 (Linux; Android 4.3; C5502 Build/10.4.1.B.0.101) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.136 Mobile Safari/537.36'

# Parse UA string and load data to dict of 'os', 'client', 'device' keys
device = DeviceDetector(ua).parse()

# Use helper methods to extract data by attribute

print(device.is_bot(),      # >>> False

device.os_name(),     # >>> Android
device.os_version(),  # >>> 4.3
device.engine(),      # >>> WebKit

device.device_brand_name(),  # >>> Sony
device.device_brand(),       # >>> SO
device.device_model(),       # >>> Xperia ZR
device.device_type())        # >>> smartphone