TECHNICAL DATA FOR DE440 AND DE441:
===================================

The  planetary and lunar ephemerides called DE440 and DE441 have been generated
by fitting numerically integrated orbits to ground-based and space-based
observations. 

Compared to the previous general-purpose ephemerides DE430, seven
years of new data have been added to compute DE440 and DE441, with improved
dynamical models and data calibration. 

The orbit of Jupiter has improved substantially 
by fitting to the Juno radio range and Very Long Baseline Array (VLBA) data of the Juno spacecraft.
The orbit of Saturn has been improved by radio range and VLBA data of the Cassini spacecraft, 
with improved estimation of the spacecraft orbit. 
The orbit of Pluto has been improved from use of stellar
occultation data reduced against the GAIA star catalog. 

The ephemerides DE440 and DE441 are fitted to the same data set, 
but DE441 assumes no damping between lunar liquid core and solid mantle, 
which avoids a divergence when integrated backwards in time. 

Therefore, DE441 is less accurate than DE440 for the current
century, but covers a much longer duration of years -13,200 to +17,191, 
compared to DE440 covering years 1,550 to 2,650.



WHERE TO DOWNLOAD DEs:
======================

https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/



WHERE TO DOWNLOAD KERNELS:
==========================

# Leap seconds (LSK) — needed for UTC <-> ET conversion.
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls

# JPL planetary ephemeris (use DE440 as you prefer; change name to de432s/de440/de441 as you like)
# (DE440 is current in your tests; DE432S is also fine if you prefer)
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp

# Planetary constants kernel (text PCK) — contains body radii, etc.
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc

# (Optional but recommended) gravitational constants / masses for consistency
wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc || true