#!/usr/bin/env python3
"""
Read calibration data from a Looking Glass Go device via USB HID.
Saves the result as a JSON file for use by the RGBD player.

Based on the Looking Glass Factory USB HID protocol:
- Send a command to EP OUT to request calibration
- Read the response from EP IN containing JSON calibration data
"""

import json
import sys
import os

def read_calibration_hidapi():
    """Read calibration using hidapi (preferred method)."""
    try:
        import hid
    except ImportError:
        print("hidapi not available, trying pyusb...")
        return None

    # Looking Glass Go: VID=0x05df, PID=0x16c0
    VID = 0x05df
    PID = 0x16c0

    device = hid.device()
    try:
        device.open(VID, PID)
        device.set_nonblocking(0)
        
        serial = device.get_serial_number_string()
        product = device.get_product_string()
        print(f"Connected to: {product} (S/N: {serial})")

        # Send calibration read command
        # Looking Glass HID protocol: send feature report to request calibration
        # Command byte 0x00 = get calibration
        cmd = [0x00] * 65  # 64 bytes + report ID
        device.write(cmd)

        # Read response - calibration comes in chunks
        calibration_data = b""
        while True:
            data = device.read(64, timeout_ms=1000)
            if not data:
                break
            calibration_data += bytes(data)
            # Check if we have a complete JSON
            try:
                text = calibration_data.decode('utf-8', errors='ignore').rstrip('\x00')
                if text and text[-1] == '}':
                    json.loads(text)
                    break
            except (json.JSONDecodeError, IndexError):
                continue

        device.close()

        if calibration_data:
            text = calibration_data.decode('utf-8', errors='ignore').rstrip('\x00')
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                print(f"Received data but not valid JSON ({len(text)} bytes)")
                return None
        return None

    except Exception as e:
        print(f"HID error: {e}")
        try:
            device.close()
        except:
            pass
        return None


def read_calibration_pyusb():
    """Read calibration using pyusb."""
    try:
        import usb.core
        import usb.util
    except ImportError:
        print("pyusb not available")
        return None

    VID = 0x05df
    PID = 0x16c0

    dev = usb.core.find(idVendor=VID, idProduct=PID)
    if dev is None:
        print("Looking Glass Go not found via pyusb")
        return None

    serial = usb.util.get_string(dev, dev.iSerial)
    product = usb.util.get_string(dev, dev.iProduct)
    print(f"Connected to: {product} (S/N: {serial})")

    try:
        if dev.is_kernel_driver_active(0):
            dev.detach_kernel_driver(0)
    except Exception as e:
        print(f"Warning: {e}")

    dev.set_configuration()

    # Try reading HID feature report for calibration
    # The calibration is typically in a GET_REPORT (feature) request
    calibration_data = b""
    
    try:
        # Try HID GET_REPORT (Feature report, report ID 0)
        # bmRequestType: 0xA1 (Device to Host, Class, Interface)
        # bRequest: 0x01 (GET_REPORT)
        # wValue: 0x0300 (Feature report, report ID 0)
        data = dev.ctrl_transfer(0xA1, 0x01, 0x0300, 0, 4096, timeout=2000)
        if data:
            calibration_data = bytes(data)
    except Exception as e:
        print(f"Feature report read failed: {e}")

    if not calibration_data:
        # Try reading from interrupt endpoint
        cfg = dev.get_active_configuration()
        intf = cfg[(0, 0)]
        ep_in = usb.util.find_descriptor(intf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        if ep_in:
            # Send a request first via EP OUT
            ep_out = usb.util.find_descriptor(intf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)
            if ep_out:
                try:
                    cmd = bytes([0] * 64)
                    ep_out.write(cmd)
                except:
                    pass

            # Read response
            try:
                while True:
                    data = ep_in.read(64, timeout=1000)
                    if not data:
                        break
                    calibration_data += bytes(data)
                    try:
                        text = calibration_data.decode('utf-8', errors='ignore').rstrip('\x00')
                        if text and text[-1] == '}':
                            json.loads(text)
                            break
                    except:
                        continue
            except Exception as e:
                print(f"Interrupt read done: {e}")

    usb.util.dispose_resources(dev)

    if calibration_data:
        text = calibration_data.decode('utf-8', errors='ignore').rstrip('\x00')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Received {len(text)} bytes but not valid JSON")
            # Try to find JSON within the data
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except:
                    pass
    return None


def get_default_lkg_go_calibration(serial="unknown"):
    """Return reasonable default calibration for Looking Glass Go."""
    # These are approximate values for LKG Go (5.5" 1440x2560 display)
    # Each device has unique calibration - these are reasonable starting defaults
    return {
        "configVersion": "1.0",
        "serial": serial,
        "pitch": {"value": 49.81804275512695},
        "slope": {"value": -5.480000019073486},
        "center": {"value": 0.15700000524520874},
        "viewCone": {"value": 40.0},
        "invView": {"value": 1},
        "verticalAngle": {"value": 0.0},
        "DPI": {"value": 491.0},
        "screenW": {"value": 1440.0},
        "screenH": {"value": 2560.0},
        "flipImageX": {"value": 0},
        "flipImageY": {"value": 0},
        "flipSubp": {"value": 0},
        "CellPatternMode": {"value": 0},
        "_note": "DEFAULT VALUES - device-specific calibration preferred"
    }


def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else "lkg_calibration.json"
    
    print("Attempting to read Looking Glass Go calibration...")
    
    # Try hidapi first
    calib = read_calibration_hidapi()
    
    # Try pyusb if hidapi failed
    if calib is None:
        calib = read_calibration_pyusb()
    
    if calib is not None:
        print("Successfully read calibration from device!")
        result = {"configValue": calib}
    else:
        print("Could not read calibration from device.")
        print("Using default Looking Glass Go calibration values.")
        result = {"configValue": get_default_lkg_go_calibration("6B84A2750F37")}
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Calibration saved to: {output_file}")
    
    # Print key values
    cv = result["configValue"]
    if isinstance(cv.get("pitch"), dict):
        print(f"  pitch:  {cv['pitch']['value']}")
        print(f"  slope:  {cv['slope']['value']}")
        print(f"  center: {cv['center']['value']}")
        print(f"  invView: {cv['invView']['value']}")
    else:
        print(f"  Raw calibration data: {json.dumps(cv, indent=2)[:200]}")


if __name__ == "__main__":
    main()
