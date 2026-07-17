# Converting Labelme JSON files to PNG label images
# Requires: Labelme (Python package): installable via pip install labelme
# Warning: Depending on the Labelme version and operating system, the exact command for exporting JSON to PNG may vary
# (e.g., labelme_json_to_dataset or labelme_export_json)

# Configuration: Set input- and output folders and Labelme command


import os
import shutil
import subprocess
from pathlib import Path

# === Configuration ===
input_dir = Path("./masks")                 # folder with Labelme JSON files
output_dir = Path("./labels_png")           # output folder for PNG label images
labelme_cmd = "labelme_export_json"         #"labelme_json_to_dataset"     # or "labelme_export_json" 
cleanup_intermediate = True                 # True => delete intermediate folders created by Labelme after processing
overwrite = True                            # True => overwrite existing PNG files in output_dir

# === Preprocessing ===
output_dir.mkdir(parents=True, exist_ok=True)

json_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".json")

if not json_files:
    raise SystemExit(f"No JSON files found in {input_dir}.")

for json_path in json_files:
    stem = json_path.stem
    print(f"Processing: {json_path}")

    # Labelme generates a folder <stem>_json (for labelme_json_to_dataset) per default 
    # or sometimes <stem> (depending on Tool/Version). 
    
    try:
        # example:
        # labelme_json_to_dataset <json_path> -o <output_dir_for_this_json>
        # labelme_export_json usually requires the JSON only, Output is <stem> then
        proc_args = []

        if labelme_cmd == "labelme_json_to_dataset":
            out_tmp_dir = json_path.with_name(f"{stem}_json")
            proc_args = [labelme_cmd, str(json_path), "-o", str(out_tmp_dir)]
        else:
            # Fallback: "labelme_export_json" without -o, creates folder <stem>
            out_tmp_dir = json_path.with_name(stem)
            proc_args = [labelme_cmd, str(json_path)]

        # Run Tool
        result = subprocess.run(proc_args, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[Error] Labelme Command failed for {json_path.name}")
            print("STDERR:", result.stderr.strip())
            # next JSON
            continue

        # check folder names (depending on version *_json, or without Suffix)
        candidate_dirs = [out_tmp_dir, json_path.with_name(stem), json_path.with_name(f"{stem}_json")]
        existing_dirs = [d for d in candidate_dirs if d.exists() and d.is_dir()]

        if not existing_dirs:
            print(f"[Error] No output folder for {json_path.name} found (expected {out_tmp_dir}).")
            print("STDOUT:", result.stdout.strip())
            continue

        # first existing folder
        produced_dir = existing_dirs[0]

        # search for label.png (Default Name in Labelme)
        label_png = produced_dir / "label.png"
        if not label_png.exists():
            # some setups create 'label.png' in a subfolder (e.g. 'mask' or 'img')
            # Fallback:
            candidates = list(produced_dir.rglob("label.png"))
            if candidates:
                label_png = candidates[0]
            else:
                print(f"[Error] label.png not found in {produced_dir}")
                continue

        # output path: <output_dir>/<json_basename>.png
        target_png = output_dir / f"{stem}.png"

        if target_png.exists() and not overwrite:
            print(f"[Notification] Target already exists and overwrite=False: {target_png}")
        else:
            shutil.copy2(label_png, target_png)
            print(f"OK: {target_png}")

    except FileNotFoundError:
        print(f"[Error] Command '{labelme_cmd}' not found. Is Labelme installed?")
        break
    except Exception as e:
        print(f"[Error] Unexpected error for {json_path.name}: {e}")
        continue
    finally:
        if cleanup_intermediate:
            if 'produced_dir' in locals() and produced_dir.exists():
                try:
                    shutil.rmtree(produced_dir)
                except Exception as e:
                    print(f"[WARNING] {produced_dir} could not be deleted: {e}")

print("Finished.")