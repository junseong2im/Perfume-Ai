import zipfile
z = zipfile.ZipFile("cloud.zip", "r")
total = 0
for i in z.infolist():
    ext = i.filename.rsplit(".", 1)[-1] if "." in i.filename else ""
    if ext in ("csv", "json", "py", "txt"):
        kb = i.file_size / 1024
        total += i.file_size
        print(f"  {i.filename:50s} {kb:8.1f} KB")
print(f"\n  Total: {len(z.infolist())} files, {total/1024/1024:.1f} MB (uncompressed)")
z.close()
