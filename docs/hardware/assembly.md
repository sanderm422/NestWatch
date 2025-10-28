# Hardware Assembly Guide

The NestWatch feeder is built around a Raspberry Pi 4 Model B paired with a
Picamera 2 module and a 3D-printed feeder enclosure. The high-level assembly
steps are summarised below.

1. **Print the CAD models**
   - Export the STL files from the `hardware/` directory to your slicer.
   - Use PETG or ASA filament for UV and moisture resistance.
2. **Prepare the Raspberry Pi enclosure**
   - Mount the Raspberry Pi 4 onto the backplate using M2.5 spacers.
   - Route the Picamera 2 ribbon cable through the camera slot.
3. **Install the camera module**
   - Seat the Picamera 2 into the camera cradle.
   - Secure with M2 screws and ensure the lens is aligned with the feeder opening.
4. **Power management**
   - Solder the DC barrel connector or USB-C pigtail to the internal power bus.
   - Install the buck converter if running from a 12V solar setup.
5. **Mount sensors and wiring**
   - Optional sensors (weight, motion) can be mounted on the accessory rail.
   - Route cables through the internal channels to keep the seed reservoir clear.
6. **Final assembly**
   - Attach the roof and feeder tray, ensuring the gasket provides a water-tight seal.
   - Mount the unit on a pole or wall bracket using the rear keyhole slots.

Refer to `docs/hardware/bill_of_materials.md` for a detailed parts list.
