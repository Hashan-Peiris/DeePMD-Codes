#!/usr/bin/env python3
"""
Big Chungus – Automated Forcefield Testing Script

This script does the following:
  1. Scans an external parent directory for version directories matching "*Ver_*".
  2. Checks local TEST_DATA directories (named like "X.TEST_DATA") for a marker file ("external_version.txt")
     that indicates which external version has been tested.
  3. For any external version not yet tested, it:
       a. Uses an existing FC directory (with "graph.pb") if available; otherwise creates a new FC directory,
          copies files (so originals remain) into it, and runs:
              dp freeze -o graph.pb
              dp compress -i graph.pb -o graph-compress.pb
          Note: If dp compress fails (e.g. model not compressible), a warning is logged and the script continues.
       b. Creates a new TEST_DATA directory for the missing external version,
          writes the external version name into "external_version.txt",
          creates symlinks to the external deepmd_data and (if available) scripts,
          copies the model files from the FC directory,
          runs "dp test" (capturing its output block and appending it to RESULTS.txt, with the full output printed),
          and then (if found) runs Plot.py and Outliers.py via conda.
          
Usage:
    python run_dp_tests.py [--dry-run]
"""

import os
import glob
import subprocess
import shutil
import sys
import argparse
import logging

# Global variable for dry-run mode.
DRY_RUN = False

# Configure logging
logger = logging.getLogger("dp_test")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Log to file
file_handler = logging.FileHandler("dp_test.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def run_cmd(cmd, cwd=None, capture_output=False, check=True):
    """
    Run a command using subprocess; obey dry-run mode.
    If capture_output is True, capture stdout and stderr (combined).
    """
    logger.info("Running command: %s", " ".join(cmd))
    if DRY_RUN:
        logger.info("Dry run enabled, not executing command: %s", " ".join(cmd))
        # Mimic a CompletedProcess with empty stdout.
        class DummyCompletedProcess:
            def __init__(self):
                self.stdout = ""
                self.returncode = 0
        return DummyCompletedProcess()
    else:
        if capture_output:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=check
            )
        else:
            result = subprocess.run(cmd, cwd=cwd, check=check)
        logger.info("Command finished with return code %d", result.returncode)
        return result


def get_external_versions(external_parent):
    """
    Return a list of tuples (basename, full_path) for directories in external_parent that match "*Ver_*".
    """
    pattern = os.path.join(external_parent, "*Ver_*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    dirs.sort()
    return [(os.path.basename(d), d) for d in dirs]


def get_test_versions():
    """
    Scan current directory for directories with "TEST_DATA" in their name and read their
    external_version.txt marker file (if present). Returns a dict mapping version name -> test directory.
    """
    test_dirs = glob.glob("*TEST_DATA*")
    versions = {}
    for td in test_dirs:
        if os.path.isdir(td):
            marker = os.path.join(td, "external_version.txt")
            if os.path.exists(marker):
                with open(marker, "r") as f:
                    ver = f.read().strip()
                versions[ver] = td
    return versions


def get_next_number(prefix):
    """
    Finds the maximum number used in directories that match the pattern "<number>.<prefix>".
    Returns the next available number.
    """
    pattern = f"*{os.extsep}{prefix}"
    dirs = glob.glob(pattern)
    max_num = 0
    for d in dirs:
        base = os.path.basename(d)
        parts = base.split(".")
        if parts[0].isdigit():
            num = int(parts[0])
            if num > max_num:
                max_num = num
    return max_num + 1


def get_existing_fc_directory():
    """
    Checks for an existing FC directory (ending with ".FC") that contains a valid model file ("graph.pb").
    If found, returns its name. Otherwise, returns None.
    """
    fc_dirs = [d for d in os.listdir(".") if d.endswith(".FC") and os.path.isdir(d)]
    valid_fc = []
    for d in fc_dirs:
        if os.path.exists(os.path.join(d, "graph.pb")):
            valid_fc.append(d)
    if valid_fc:
        # Return the one with the smallest number (assumes naming like "1.FC")
        valid_fc.sort(key=lambda x: int(x.split(".")[0]))
        logger.info("Using existing FC directory: %s", valid_fc[0])
        return valid_fc[0]
    return None


def create_fc_directory():
    """
    Create a new FC directory using the next available number.
    """
    num = get_next_number("FC")
    fc_dir = f"{num}.FC"
    if DRY_RUN:
        logger.info("Dry run: would create FC directory %s", fc_dir)
    else:
        os.makedirs(fc_dir, exist_ok=True)
    return fc_dir


def copy_files_to_fc(fc_dir, script_name):
    """
    Copy all files and directories from the current working directory (except:
      - this script file,
      - directories that include '.FC' or 'TEST_DATA' in their name)
    into the fc_dir.
    """
    for item in os.listdir("."):
        if item == script_name:
            continue
        if os.path.isdir(item) and (".FC" in item or "TEST_DATA" in item):
            continue
        src = os.path.join(os.getcwd(), item)
        dst = os.path.join(fc_dir, item)
        if os.path.isfile(src):
            logger.info("Copying file %s to %s", src, dst)
            if DRY_RUN:
                logger.info("Dry run: would copy file %s to %s", src, dst)
            else:
                shutil.copy2(src, dst)
        elif os.path.isdir(src):
            logger.info("Copying directory %s to %s", src, dst)
            if DRY_RUN:
                logger.info("Dry run: would copy directory %s to %s", src, dst)
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)


def run_dp_commands(fc_dir):
    """
    In the FC directory, run the forcefield generation commands:
      dp freeze -o graph.pb
      dp compress -i graph.pb -o graph-compress.pb
      
    Note: If dp compress fails (e.g. model not compressible), a warning is logged and the script continues.
    """
    cwd = os.getcwd()
    os.chdir(fc_dir)
    try:
        run_cmd(["dp", "freeze", "-o", "graph.pb"], capture_output=False)
        try:
            run_cmd(["dp", "compress", "-i", "graph.pb", "-o", "graph-compress.pb"], capture_output=False)
        except subprocess.CalledProcessError as e:
            logger.warning("dp compress failed, moving on. Error: %s", e)
    finally:
        os.chdir(cwd)


def copy_model_files(fc_dir, test_dir):
    """
    Copy the generated model files from the FC directory into the test directory.
    """
    for fname in ["graph.pb", "graph-compress.pb"]:
        src = os.path.join(fc_dir, fname)
        dst = os.path.join(test_dir, fname)
        logger.info("Copying model file %s to %s", src, dst)
        if DRY_RUN:
            logger.info("Dry run: would copy %s to %s", src, dst)
        else:
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                logger.warning("File %s does not exist, skipping copy.", src)


def create_test_directory(test_dir, ext_ver_basename, ext_ver_path, fc_dir):
    """
    Create the test directory:
      - Write the external version name into 'external_version.txt'
      - Create a symlink for deepmd_data from the external version directory.
      - Create symlinks for each file in the external version's scripts folder (if the folder exists).
      - Copy the model files from the FC directory.
    """
    if DRY_RUN:
        logger.info("Dry run: would create test directory %s", test_dir)
    else:
        os.makedirs(test_dir, exist_ok=True)
    # Write marker file
    marker_file = os.path.join(test_dir, "external_version.txt")
    logger.info("Writing marker file %s", marker_file)
    if not DRY_RUN:
        with open(marker_file, "w") as f:
            f.write(ext_ver_basename)

    # Link deepmd_data directory
    deepmd_source = os.path.join(ext_ver_path, "deepmd_data")
    deepmd_link = os.path.join(test_dir, "deepmd_data")
    if not os.path.exists(deepmd_link):
        logger.info("Linking %s -> %s", deepmd_source, deepmd_link)
        if DRY_RUN:
            logger.info("Dry run: would link %s -> %s", deepmd_source, deepmd_link)
        else:
            os.symlink(deepmd_source, deepmd_link)

    # Link all files from the external 'scripts' folder (if it exists)
    scripts_source = os.path.join(ext_ver_path, "scripts")
    if os.path.isdir(scripts_source):
        for filename in os.listdir(scripts_source):
            src_file = os.path.join(scripts_source, filename)
            link_name = os.path.join(test_dir, filename)
            if not os.path.exists(link_name):
                logger.info("Linking %s -> %s", src_file, link_name)
                if DRY_RUN:
                    logger.info("Dry run: would link %s -> %s", src_file, link_name)
                else:
                    os.symlink(src_file, link_name)
    else:
        logger.warning("Scripts folder not found in %s. Plot.py and Outliers.py will not be linked.", ext_ver_path)

    # Copy model files
    copy_model_files(fc_dir, test_dir)


def run_dp_test_and_analysis(test_dir):
    """
    In the test directory, run:
       dp test -m graph.pb -s ./deepmd_data/ -d RESULTS
    Capture the output block (from the starting and ending markers) and append it to RESULTS.txt.
    Also print the full dp test output to the terminal.
    Then, if present, run Plot.py and Outliers.py using conda.
    """
    cwd = os.getcwd()
    os.chdir(test_dir)
    try:
        logger.info("Running dp test in %s", test_dir)
        result = run_cmd(
            ["dp", "test", "-m", "graph.pb", "-s", "./deepmd_data/", "-d", "RESULTS"],
            capture_output=True
        )
        dp_test_output = result.stdout
        # Print dp test output to terminal
        print(dp_test_output)
        logger.info("dp test output printed to terminal.")

        # Extract the output block between the markers.
        lines = dp_test_output.splitlines()
        start_index = None
        end_index = None
        for i, line in enumerate(lines):
            if "# ---------------output of dp test---------------" in line:
                start_index = i
            if "# -----------------------------------------------" in line and start_index is not None:
                end_index = i
                break
        extracted = ""
        if start_index is not None and end_index is not None:
            extracted_lines = lines[start_index:end_index + 1]
            extracted = "\n".join(extracted_lines) + "\n\n"  # append a blank line
            logger.info("Extracted dp test output block.")
        else:
            logger.warning("Could not find dp test output markers in dp test output.")

        # Append the extracted block to RESULTS.txt
        results_file = "RESULTS.txt"
        logger.info("Writing extracted output to %s", results_file)
        if not DRY_RUN:
            with open(results_file, "a") as f:
                f.write(extracted)

        # Run analysis scripts using conda, if they exist in this directory.
        if os.path.exists("Plot.py"):
            run_cmd(["conda", "run", "-n", "SciFy2", "python", "Plot.py"], capture_output=False)
        else:
            logger.warning("Plot.py not found in %s; skipping analysis.", test_dir)

        if os.path.exists("Outliers.py"):
            run_cmd(["conda", "run", "-n", "SciFy2", "python", "Outliers.py"], capture_output=False)
        else:
            logger.warning("Outliers.py not found in %s; skipping analysis.", test_dir)
    finally:
        os.chdir(cwd)


def main():
    global DRY_RUN

    parser = argparse.ArgumentParser(description="Automated forcefield testing script.")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without executing them.")
    args = parser.parse_args()
    DRY_RUN = args.dry_run
    if DRY_RUN:
        logger.info("Dry run mode enabled. No changes will be made.")

    # Set the external parent directory where version directories reside.
    EXTERNAL_PARENT = (
        "/data/home/mpeiris1/VASP/BATTERY_STUFF/Ca-S_Projects/"
        "1.Ca-S_Reaction/8.DeepMD/1.Test/7.TEST_MODELS/0.TEST_DATA"
    )
    logger.info("Scanning external parent directory: %s", EXTERNAL_PARENT)
    
    # Get list of external version directories.
    external_versions = get_external_versions(EXTERNAL_PARENT)
    if not external_versions:
        logger.error("No external version directories found in %s", EXTERNAL_PARENT)
        sys.exit(1)
    logger.info("Found external version directories:")
    for ver_basename, ver_path in external_versions:
        logger.info("  %s -> %s", ver_basename, ver_path)
    
    # Get external versions that have already been tested locally.
    tested_versions = get_test_versions()
    logger.info("Already tested external versions: %s", list(tested_versions.keys()))
    
    # Identify missing external versions.
    missing = []
    for ver_basename, ver_path in external_versions:
        if ver_basename not in tested_versions:
            missing.append((ver_basename, ver_path))
    if not missing:
        logger.info("All external versions have been tested. Nothing to do.")
        sys.exit(0)
    
    logger.info("Missing external versions to test:")
    for ver_basename, _ in missing:
        logger.info("  %s", ver_basename)
    
    # Determine FC directory: use an existing one if available, otherwise create a new one.
    fc_dir = get_existing_fc_directory()
    if not fc_dir:
        fc_dir = create_fc_directory()
        # Copy files (except this script and any existing FC/TEST_DATA dirs) into the new FC directory.
        script_name = os.path.basename(__file__)
        copy_files_to_fc(fc_dir, script_name)
        # Run freeze and compress
        run_dp_commands(fc_dir)
    else:
        logger.info("Using existing model from %s", fc_dir)
    
    # For each missing external version, create a new TEST_DATA directory and run the tests.
    for ext_ver_basename, ext_ver_path in missing:
        test_num = get_next_number("TEST_DATA")
        test_dir = f"{test_num}.TEST_DATA"
        logger.info("Creating test directory %s for external version %s", test_dir, ext_ver_basename)
        create_test_directory(test_dir, ext_ver_basename, ext_ver_path, fc_dir)
        run_dp_test_and_analysis(test_dir)
    
    logger.info("All missing tests have been run.")


if __name__ == "__main__":
    main()
