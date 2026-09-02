#Not working yet (BUGS)

from creditly.data.load_data import load_lending_club, load_census_income
from creditly.data.profile import profile_dataframe, describe_numeric
from creditly.data.merge import aggregate_census_to_zip3, merge_loans_with_income
from creditly.features.split import train_test_split_manual
from creditly.features.missing import drop_mcar_rows, impute_mar_by_group, flag_and_fill_mnar
from creditly.features.encoding import one_hot_encode, TargetEncoder


def derive_binary_target(df):

    resolved = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    resolved["loan_status_binary"] = (
        resolved["loan_status"].isin(["Charged Off", "Default"]).astype(int)
    )
    return resolved


def run_day1():
    loans = load_lending_club("data/raw/lending_club/accepted_2007_to_2018Q4.csv")
    census = load_census_income("data/raw/census_acs/census_income.csv")

    profile_dataframe(loans, name="lending_club (raw)")
    describe_numeric(loans)
    profile_dataframe(census, name="census_income (raw)")

    zip3_income = aggregate_census_to_zip3(
        census,
        zcta_col="zip code tabulation area",
        income_col="S1901_C01_012E",
    )

    merged = merge_loans_with_income(loans, zip3_income)
    profile_dataframe(merged, name="merged (post-join)")

   
    assert merged.shape[0] == loans.shape[0], "row count changed — merge likely duplicated rows"

    merged.to_csv("data/interim/loans_merged_zip3.csv", index=False)
    return merged


def run_day2(merged_df):
   
    df = derive_binary_target(merged_df)

   
    df = drop_mcar_rows(df, columns=["dti"])
    df = impute_mar_by_group(df, target_col="emp_length", group_col="issue_d")
    df = flag_and_fill_mnar(df, column="mths_since_last_delinq", fill_value=0)

   
    # train/test independence.
    train_df, test_df = train_test_split_manual(df, test_size=0.2, seed=42)

    # Step 4a: one-hot — safe on both sets independently, no target used.
    train_df = one_hot_encode(train_df, columns=["term", "grade"])
    test_df = one_hot_encode(test_df, columns=["term", "grade"])

    
    encoder = TargetEncoder(column="zip3", smoothing=10.0)
    encoder.fit(train_df, target=train_df["loan_status_binary"])

    train_df["zip3_encoded"] = encoder.transform(train_df)
    test_df["zip3_encoded"] = encoder.transform(test_df)

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    return train_df, test_df


def main():
    merged = run_day1()
    run_day2(merged)


if __name__ == "__main__":
    main()
