You can download the main DEM from here for mapping (Rename it to GLD100.tif after download):
https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_WAC_GLD100_DTM_79S79N_100m_v1.1.tif

Geospatial Information for GLD100:
(Taken from https://astrogeology.usgs.gov/search/map/moon_lroc_wac_dtm_gld100_118m)

Target: Moon
System: Earth
Minimum Latitude: -79
Maximum Latitude: 79
Minimum Longitude: -180
Maximum Longitude: 180
Direct Spatial Reference Method: Raster
Object Type: Grid Cell
Raster Row Count (lines): 47912
Raster Column Count (samples): 109165
Bit Type (8, 16, 32): 16
Quad Name: 
Radius A: 1737400
Radius C: 1737400
Bands: 1
Pixel Resolution (meters/pixel): 118.45058759
Scale (pixels/degree): 256
Map Projection Name: Equirectangular
Latitude Type: Planetocentric
Longitude Direction: Positive East
Longitude Domain: -180 to 180



If you need more advanced DEM options, you can download the SLDEM2015 database which can be found here:
https://imbrium.mit.edu/DATA/SLDEM2015/

SLDEM2015 Information:

The LOLA team has prepared higher-level map products describing the lunar 
surface slopes, based on the SLDEM2015 shape data. Two parameters are available
globally: maximum slope (in degrees) and slope azimuth (in degrees).

The maximum slope is the angle between the surface normal vector and the radial
direction.

The slope azimuth is the angle between the horizontally-projected surface 
normal vector and the unit vector pointing to the North Pole. It is defined 
as increasing clockwise over the range [-180,180] degrees.

These slope and azimuth maps are made available at various resolutions: 64 
pixels per degree (ppd; ~500m at the equator), 128 ppd (~250m at the equator), 
256 ppd (~120m at the equator), and 512 ppd (~60m at the equator). LOLA
topographic maps are also available at these resolutions.

Except for the highest resolution of 512 ppd, these global maps are available 
as single products. The 256 ppd and 512 ppd maps are available as tiled 
products, with 12 and 48 tiles respectively. Each map is archived in two 
formats: a FLOAT_IMG version (single precision binary tables) and a 
GeoJPEG2000 version (for its smaller filesize and ease of import in GIS 
programs).


=======
SUMMARY
=======


GLD100 (LRO WAC)
----------------
118 m/pixel, global, already mosaicked, easy GeoTIFF.
File is ~10 GB, we already found it:
https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_WAC_GLD100_DTM_79S79N_100m_v1.1.tif
Coordinates: planetocentric lat, east-positive lon, domain -180..180.
Ready to go, minimal headaches.

SLDEM2015 (LOLA/LROC merged)
----------------------------
Higher quality DEM.
Global map at 64–256 ppd available as single tiles → manageable.
512 ppd is the best (~60 m) but comes in 48 separate tiles (JP2/IMG). You’d need to mosaic them yourself. That’s a pain if you’ve never done it — you’d need gdal_merge.py or QGIS. Each tile is 2–4 GB, total dataset is ~100 GB+.
If you only care about the limb visible from Kastelorizo at eclipse time, you’d only need a narrow strip near the apparent limb. But identifying and stitching just that strip is… extra work.