import logging
import os

from constants import files
from constants import columns as c
from utils import load_pandas_df_from_s3


def create_deflated_input_dfs():
    initial_loans_df = load_pandas_df_from_s3(files.S3_BUCKET, files.TEST)
    initial_loans_df[c.Loans.Loan_Status] = None

    factor = 0.9
    for i in range(0, 9):
        loans_df_with_deflation = initial_loans_df.copy()
        loans_df_with_deflation[c.Loans.ApplicantIncome] = initial_loans_df[c.Loans.ApplicantIncome] * factor
        loans_df_with_deflation[c.Loans.CoapplicantIncome] = initial_loans_df[c.Loans.CoapplicantIncome] * factor
        loans_df_with_deflation[c.Loans.LoanAmount] = loans_df_with_deflation[c.Loans.LoanAmount] / factor

        for i in range(len(loans_df_with_deflation)):
            if loans_df_with_deflation[c.Loans.ApplicantIncome].iloc[i] < 1000:
                loans_df_with_deflation.at[i, c.Loans.Credit_History] = (
                        loans_df_with_deflation[c.Loans.Credit_History].iloc[i] - 2 * (1-factor))

        destination = os.path.join(files.LOCAL_ROOT_PATH, "data", "interim", f"loans_deflation_{factor:.2f}.csv")
        factor -= 0.1

        loans_df_with_deflation.to_csv(destination, index=False)


if __name__ == "__main__":
    create_deflated_input_dfs()