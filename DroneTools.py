import os, re, math, arcpy
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
RGB_RESAMPLE = Image.BICUBIC
MS_RESAMPLE  = Image.BILINEAR
BLACK_THRESH = 10
MS_NODATA    = 0
RGB_SUBDIR   = "rgb"
MS_SUBDIR    = "multispectral"

# ----------------------------------------------------------------------------
# Core Logic (The "Engine")
# ----------------------------------------------------------------------------
def read_metadata(path):
    im = Image.open(path)
    W, H = im.size
    exif = im.getexif()
    sub  = exif.get_ifd(0x8769)
    focal_mm   = float(sub.get(0x920A, 0) or 0)
    focal_35mm = float(sub.get(0xA405, 0) or 0)
    with open(path, "rb") as fh:
        blob = fh.read()
    s, e = blob.find(b"<x:xmpmeta"), blob.find(b"</x:xmpmeta>")
    xmp = blob[s:e + 12].decode("utf-8", "replace") if s != -1 and e != -1 else ""
    def x(tag, default=None):
        m = re.search(rf'drone-dji:{tag}="([^"]*)"', xmp) or re.search(rf"<drone-dji:{tag}>([^<]*)</", xmp)
        return m.group(1) if m else default
    return dict(path=path, W=W, H=H, lat=float(x("GpsLatitude")), lon=float(x("GpsLongitude")),
                rel_alt=float(x("RelativeAltitude")), gimbal_yaw=float(x("GimbalYawDegree")),
                gimbal_roll=float(x("GimbalRollDegree", "0")), calib_focal_px=float(x("CalibratedFocalLength", "0") or 0),
                focal_mm=focal_mm, focal_35mm=focal_35mm, is_tiff=path.lower().endswith((".tif", ".tiff")))

def wgs84_to_utm(lon, lat):
    a = 6378137.0; f = 1/298.257223563; e2 = f*(2-f); ep2 = e2/(1-e2); k0 = 0.9996
    zone = int((lon+180)//6)+1; lon0 = math.radians((zone-1)*6 - 180 + 3)
    phi = math.radians(lat); lam = math.radians(lon); N = a/math.sqrt(1-e2*math.sin(phi)**2)
    T = math.tan(phi)**2; C = ep2*math.cos(phi)**2; A = (lam-lon0)*math.cos(phi)
    M = a*((1-e2/4-3*e2**2/64-5*e2**3/256)*phi-(3*e2/8+3*e2**2/32+45*e2**3/1024)*math.sin(2*phi)+(15*e2**2/256+45*e2**3/1024)*math.sin(4*phi)-(35*e2**3/3072)*math.sin(6*phi))
    x = 500000 + k0*N*(A+(1-T+C)*A**3/6+(5-18*T+T**2+72*C-58*ep2)*A**5/120)
    y = k0*(M+N*math.tan(phi)*(A**2/2+(5-T+9*C+4*C**2)*A**4/24+(61-58*T+T**2+600*C-330*ep2)*A**6/720))
    if lat < 0: y += 10000000
    return x, y, (32600 if lat >= 0 else 32700)+zone

def esri_wkt_utm(epsg):
    zone = epsg-32600 if epsg < 32700 else epsg-32700
    hemi = "N" if epsg < 32700 else "S"
    cm = float((zone-1)*6 - 180 + 3); fn = 0.0 if hemi == "N" else 10000000.0
    return (f'PROJCS["WGS_1984_UTM_Zone_{zone}{hemi}",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",{fn}],PARAMETER["Central_Meridian",{cm}],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')

def process_images(paths, out_dir, takeoff_offset):
    os.makedirs(out_dir, exist_ok=True)
    meta = [read_metadata(p) for p in paths]
    _, _, epsg = wgs84_to_utm(meta[0]["lon"], meta[0]["lat"])
    for md in meta:
        eff_alt = md["rel_alt"] + takeoff_offset
        gsd = (eff_alt / md["calib_focal_px"]) if md["calib_focal_px"] > 0 else (eff_alt * (md["focal_mm"] * md["W"] / math.hypot(md["W"], md["H"])) / md["focal_mm"])
        beta = (md["gimbal_yaw"] + (180.0 if abs(md["gimbal_roll"]) > 90 else 0.0) + 180.0) % 360.0 - 180.0
        Xc, Yc, _ = wgs84_to_utm(md["lon"], md["lat"])
        im = Image.open(md["path"])
        if md["is_tiff"]:
            a = np.asarray(im).astype(np.int32)
            rot = Image.fromarray(a, mode="I").rotate(-beta, expand=True, resample=MS_RESAMPLE, fillcolor=MS_NODATA)
            rot_img = Image.fromarray(np.clip(np.asarray(rot), 0, 65535).astype(np.uint16))
            dest = os.path.join(out_dir, MS_SUBDIR); os.makedirs(dest, exist_ok=True)
            rot_img.save(os.path.join(dest, os.path.basename(md["path"])), tiffinfo={42113: str(MS_NODATA)})
        else:
            # Create a transparent (RGBA) image instead of RGB
            rgb = im.convert("RGBA")
            # Rotate with a transparent fill color (0,0,0,0)
            rot_img = rgb.rotate(-beta, expand=True, resample=RGB_RESAMPLE, fillcolor=(0, 0, 0, 0))
            dest = os.path.join(out_dir, RGB_SUBDIR); os.makedirs(dest, exist_ok=True)
            rot_img.save(os.path.join(dest, os.path.splitext(os.path.basename(md["path"]))[0] + ".png"))
        
        # Write world file + PRJ
        W2, H2 = rot_img.size
        C = Xc - (W2 - 1) / 2 * gsd; F = Yc + (H2 - 1) / 2 * gsd
        ext = ".tfw" if md["is_tiff"] else ".pgw"
        with open(os.path.join(dest, os.path.splitext(os.path.basename(md["path"]))[0] + ext), "w") as f:
            f.write(f"{gsd:.10f}\n0.0\n0.0\n{-gsd:.10f}\n{C:.4f}\n{F:.4f}\n")
        with open(os.path.join(dest, os.path.splitext(os.path.basename(md["path"]))[0] + ".prj"), "w") as f:
            f.write(esri_wkt_utm(epsg))

# ----------------------------------------------------------------------------
# ArcGIS Tool Entry Point (Corrected)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Get the raw input string
    raw_paths = arcpy.GetParameterAsText(0)
    
    # Split by semicolon and strip away the extra quotes that ArcGIS adds
    image_paths = [p.strip("' ") for p in raw_paths.split(';')]
    
    takeoff_offset = float(arcpy.GetParameterAsText(1))
    
    # Get the parent folder from the first cleaned path
    parent_folder = os.path.dirname(image_paths[0])
    out_dir = os.path.join(parent_folder, "georeferenced")
    
    arcpy.AddMessage(f"Processing {len(image_paths)} images...")
    process_images(image_paths, out_dir, takeoff_offset)
    arcpy.AddMessage(f"Done! Outputs saved to: {out_dir}")