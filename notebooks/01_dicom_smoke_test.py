# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: RSNA (Python 3.12)
#     language: python
#     name: rsna
# ---

  # %%
  from pathlib import Path

  root = Path("/home/shaurya/rsna-data/raw/competition")

  for item in sorted(root.iterdir()):
      print(item.name, "directory" if item.is_dir() else "file")



  # %%
  from pathlib import Path
  import os
  import pydicom

  root = Path("/home/shaurya/rsna-data/raw/competition")


  def first_dicom(directory):
      for current, _, filenames in os.walk(directory):
          for filename in filenames:
              if filename.lower().endswith(".dcm"):
                  return Path(current) / filename

      raise FileNotFoundError(f"No DICOM found under {directory}")


  for split in ["train_series", "test_series"]:
      print(f"{split}: searching...", flush=True)

      path = first_dicom(root / split)
      print(f"{split}: file found", flush=True)

      ds = pydicom.dcmread(path)
      print(f"{split}: header decoded", flush=True)

      pixels = ds.pixel_array

      print(
          f"{split}: pixels decoded; "
          f"shape={pixels.shape}; "
          f"dtype={pixels.dtype}; "
          f"transfer_syntax={ds.file_meta.TransferSyntaxUID}; "
          f"range=({pixels.min()}, {pixels.max()})",
          flush=True,
      )

# %%
