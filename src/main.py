from utils.filings_util import generate_13f_and_add_extra_cols
from utils.mappings import CIK_TO_FILER_OVER_250B


def main():

    cik_to_filers = [
        # CIK_TO_FILER_1B_TO_3B,
        # CIK_TO_FILER_3B_TO_5B,
        # CIK_TO_FILER_5B_TO_10B,
        # CIK_TO_FILER_10B_TO_25B,
        # CIK_TO_FILER_25B_TO_50B,
        # CIK_TO_FILER_50B_TO_250B,
        CIK_TO_FILER_OVER_250B
    ]

    # STEP 0 : import raw 13f for ciks (start with CIK_TO_FILER_OVER_250B as it has fewer filers)
    # for cik_to_filer in cik_to_filers:
        # for cik in cik_to_filer.keys():
            # download_filing_to_csv(cik, latest_n_filings=5, use_requests=False)
        # generate_13f_and_add_extra_cols(cik_to_filer.keys())


    # STEP 1 : delete all csvs, clean-slate
    # delete_files_in_subfolders(BASE_DIR_FINAL)

    # STEP 2 : generate final 13f from raw-parsed 13f  (make sure prop_below_1 is commented in add_quarter_end_price)
    # for cik_to_filer in cik_to_filers:
    #     generate_13f_and_add_extra_cols(cik_to_filer.keys())

    # STEP 3 : make sure STOCKS_SHS_Q_END_PRICES_FILE has no quarter_end_price values
    # remove_column_values_from_file(file=STOCKS_SHS_Q_END_PRICES_FILE)

    # STEP 4 : save quarter_end_prices from the 13fs most frequent Q end prices
    # add_quarter_end_price_to_sh_outstanding_file(year_quarter_list=[ ['2025', 'Q2'], ['2025', 'Q1'], ['2024', 'Q4'], ['2024', 'Q3'], ['2024', 'Q2'], ['2024', 'Q1'], ['2023', 'Q4']])

    # STEP 5 : Run below AFTER uncommenting prop_below_1 part (reported in thousands - needs correction)
    # for cik_to_filer in cik_to_filers:
    #     add_quarter_end_price(cik_to_filer)


    # STEP 6 : check if any new 13f was filed
    # found_ciks = check_latest_13f(CIK_TO_FILER.keys())
    # print(found_ciks)
    # found_ciks = ['1094749']
    # for cik in found_ciks:
    #     download_filing_to_csv(cik, latest_n_filings=1, use_requests=False)
    # generate_13f_and_add_extra_cols(found_ciks)


    # STEP 7 : correct files that report in thousands
    # correct_share_values_thousands('2024', 'Q2')
    # correct_share_values_thousands('2024', 'Q3')
    # correct_share_values_thousands('2024', 'Q4')
    # correct_share_values_thousands('2025', 'Q1')

    # STEP 8 : create filer accession filing_date file
    # create_filer_accession_metadata_file()


if __name__ == "__main__":
    main()