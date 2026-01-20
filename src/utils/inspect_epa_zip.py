"""Quick diagnostic to see what's in the EPA zip file"""
import requests
from zipfile import ZipFile
from io import BytesIO

EPA_SLD_URL = "https://edg.epa.gov/EPADataCommons/public/OA/SLD/SmartLocationDatabaseV3.zip"

print("Downloading EPA Smart Location Database to inspect contents...")
print("This will download ~527MB...")

response = requests.get(EPA_SLD_URL, stream=True)
response.raise_for_status()

print("\nContents of the zip file:")
print("=" * 80)

with ZipFile(BytesIO(response.content)) as zf:
    for name in zf.namelist():
        info = zf.getinfo(name)
        print(f"{name:60s} {info.file_size:>15,} bytes")

print("=" * 80)
