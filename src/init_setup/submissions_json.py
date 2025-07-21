import os
import zipfile

from init_setup.ticker_cusip_data import cik_to_ticker
from utils.mappings import SUBMISSIONS_FILERS_DIR, CIK_TO_FILER_AND_AUM, SUBMISSIONS_STOCKS_DIR, CIK_TO_FILER

# Your set of CIKs (strings, zero-padded 10 digits)
ciks = CIK_TO_FILER_AND_AUM.keys()
zip_path = r"C:\Users\andre\Downloads\submissions.zip"

# Convert CIKs to expected filenames inside the ZIP
target_filenames = {f"CIK{str(int(cik)).zfill(10)}.json" for cik in ciks}

# Create output directory if it doesn't exist
os.makedirs(SUBMISSIONS_FILERS_DIR, exist_ok=True)


def extract_cik_jsons(ciks, zip_path, output_dir):
    """
    Extract JSON files for given CIKs from a ZIP archive to an output directory.

    Args:
        ciks (iterable of str): Iterable of CIK strings (can be zero-padded or int convertible).
        zip_path (str): Path to the ZIP file containing JSON files.
        output_dir (str): Directory to save the extracted JSON files.

    Returns:
        tuple: (list_of_saved_files, list_of_missing_files)
    """
    # Normalize CIKs and prepare target filenames (CIK zero-padded to 10 digits, e.g. 'CIK0000123456.json')
    target_filenames = {f"CIK{str(int(cik)).zfill(10)}.json" for cik in ciks}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    missing_files = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zip_files = set(zf.namelist())
        matching_files = target_filenames.intersection(zip_files)

        # Save found JSON files
        for filename in matching_files:
            with zf.open(filename) as f:
                data = f.read()
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'wb') as out_f:
                out_f.write(data)
            saved_files.append(filename)
            print(f"Saved {filename} to {output_path}")

        # Identify missing files
        missing_files = sorted(target_filenames - matching_files)
        if missing_files:
            print("\nThe following CIK files were NOT found in the ZIP:")
            for missing in missing_files:
                print(missing)
        else:
            print("\nAll requested CIK files were found.")

    return (saved_files, missing_files)


def main():

    zip_path = 'C:/Users/andre/Downloads/submissions.zip'
    extract_cik_jsons(cik_to_ticker.keys(), zip_path, SUBMISSIONS_STOCKS_DIR)
    extract_cik_jsons(CIK_TO_FILER.keys(), zip_path, SUBMISSIONS_FILERS_DIR)


if __name__ == "__main__":
    main()