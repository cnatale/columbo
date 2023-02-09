############## Columbo - An Automated Anti-Manipulation Detective ###############
# This is a training and test validation pipeline. If running locally and there's no
# training_dataset.csv file in the same directory, it attempts to connect to an
# AWS Redshift db to pull in fresh samples of gaming and non-manipulation tests in equal
# numbers.

# IMPORTANT: For this application to run in Google Colab, it requires a copy of the training_dataset.csv
# file included in Columbo's Google Drive folder. Copy the file into the main directory of the Google Colab folder,
# which is accessible on the left-hand side of the Colab page UI.

# NOTE: This project was developed using Python 3.10.
# When running in Google Colab running, older versions of many required
# packages are installed due to it running Python 3.8. Everything seems to work, with the exception
# of the feature importance chart not displaying the actual names of features.
# I've included a copy of the feature importance bar chart in the project's Google Drive.
# It's in the 'visualizations' folder.

if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    import redshift_connector
    import pandas as pd
    from matplotlib import pyplot
    from xgboost import plot_importance, plot_tree


    # disable when running in Google Colab or else the app will throw an exception
    # due to not being able to connect to VPN-protected Redshift server
    conn = redshift_connector.connect(
        host=os.getent('REDSHIFT_HOST'),
        database=os.getenv('REDSHIFT_DB'),
        port=os.getenv('REDSHIFT_PORT'),
        user=os.getenv('REDSHIFT_USER'),
        password=os.getenv('REDSHIFT_PASSWORD')
    )
    cursor = conn.cursor()

    # check if csv data file exists. if so, use it for training data
    training_dataset_filename = 'training_dataset.csv'

    # columns to remove from Redshift query response
    cols_to_drop = ["country_id",
                    "result_date"
                    # "sim_mobile_carrier_id",
                    # "network_mobile_carrier_id",
                    # "raw_sim_carrier_id",
                    # "raw_network_carrier_id"
                    ]

    country_id = 11

    if os.path.exists(training_dataset_filename):
    # if False:
        training_dataset = pd.read_csv(training_dataset_filename)
    else:
        # get blacklisted data
        cursor.execute("""SELECT country_id, download_kbps, is_blacklisted, result_date,
        sim_mobile_carrier_id, network_mobile_carrier_id, raw_sim_carrier_id, raw_network_carrier_id
        FROM prod_analytics.mobile_base
        JOIN prod_analytics.mobile_blacklist_reasons ON mobile_base.device_id=mobile_blacklist_reasons.device_id
        WHERE is_blacklisted=TRUE AND reason_code=99
        AND country_id=""" + str(country_id) + """
        AND result_date >= '2019-01-01'
        LIMIT 500""")

        blacklisted_result: pd.DataFrame = cursor.fetch_dataframe()
        blacklisted_result.drop(cols_to_drop, axis=1, inplace=True)
        blacklisted_result.is_blacklisted = blacklisted_result.is_blacklisted.replace({True: 1, False: 0})

        print("blacklisted pipeline result: ")
        print(blacklisted_result.head())
        print("blacklisted query result info: ")
        print(blacklisted_result.info())
        print("\n")

        missing_blacklisted_props = blacklisted_result.isna().mean(axis=0)
        print(missing_blacklisted_props)

        # get non-blacklisted data
        cursor.execute("""SELECT country_id, download_kbps, is_blacklisted, result_date,
        sim_mobile_carrier_id, network_mobile_carrier_id, raw_sim_carrier_id, raw_network_carrier_id
        FROM prod_analytics.mobile_base
        WHERE is_blacklisted = FALSE
        AND country_id=""" + str(country_id) + """
        LIMIT 500""")

        non_blacklisted_result: pd.DataFrame = cursor.fetch_dataframe()
        non_blacklisted_result.head()
        non_blacklisted_result.drop(cols_to_drop, axis=1, inplace=True)
        non_blacklisted_result.is_blacklisted = non_blacklisted_result.is_blacklisted.replace({True: 1, False: 0})

        print("non-blacklisted pipeline result: ")
        print(non_blacklisted_result.head())
        print("non-blacklisted result info:")
        print(non_blacklisted_result.info())
        print("\n")

        missing_non_blacklisted_props = non_blacklisted_result.isna().mean(axis=0)

        # combine non-blacklisted and blacklisted results to form the training dataset
        training_dataset = pd.concat([non_blacklisted_result, blacklisted_result], axis=0)

        # save the training dataset to a csv file
        training_dataset.to_csv('training_dataset.csv', index=False)

    # add a column representing variance from mean of download_kbps,
    # which should be more generalizable that raw download_kbps
    mean_download_kbps = training_dataset['download_kbps'].mean()
    training_dataset['download_kbps_mean_deviation'] = training_dataset['download_kbps'] - mean_download_kbps

    # drop download_kbps since analysis of their F score showed only one is really necessary
    training_dataset = training_dataset.drop("download_kbps", axis=1)

    # divide the data into feature (everything not is_blacklisted) and target (the is_blacklisted column) DataFrames
    X = training_dataset.drop("is_blacklisted", axis=1)
    y = training_dataset.is_blacklisted

    print("feature array:")
    print(X)
    print("target array:")
    print(y)

    # pipelines for processing categorical (one-hot encoded) and numeric features
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")),
               ("scale", StandardScaler())]
    )

    cat_cols = X.select_dtypes(exclude="number").columns
    num_cols = X.select_dtypes(include="number").columns

    from sklearn.compose import ColumnTransformer

    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ]
    )

    # XGBoost Training/Classification Logic
    import xgboost as xgb

    # create a classifier and train it
    xgb_cl = xgb.XGBClassifier()

    # preprocess data, then divide into randomly populated training and test subsets
    X_processed = full_processor.fit_transform(X)
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, stratify=y_processed, random_state=1121218
    )

    # fit classifier with default params, evaluate performance. this is where the model training occurs.
    from sklearn.metrics import accuracy_score

    # fit
    xgb_cl.fit(X_train, y_train)
    print(X.columns.tolist())
    print("xgb_cl.bet_booster().feature_names before adapting")
    print(xgb_cl.get_booster().feature_names)

    # disable next line when running in Google Colab b/c the version of
    # XGBoost used w/ Python 3.8 will throw an exception if feature names are
    # set manually.
    xgb_cl.get_booster().feature_names = X.columns.tolist()
    print("X.columns.tolist():")
    print(X.columns.tolist())

    # predict gaming/non-gaming speedtests using trained model on the separate test data
    preds = xgb_cl.predict(X_test)

    # print the overall accuracy score of the classifier's predictions
    print("\n")
    print("\033[1maccuracy score:")
    print(accuracy_score(y_test, preds))
    print('\033[0m')

    print("########### Additional Model Information ###########")
    print("list of features:")
    # iterating the columns
    for col in training_dataset.columns:
        print("\t" + col)
    print("list of xgboost features:")
    booster = xgb_cl.get_booster()
    features = booster.feature_names
    print(features)
    print("feature importances: ")
    print(xgb_cl.feature_importances_)

    # generate charts based on the generated model
    plot_importance(xgb_cl.get_booster())
    pyplot.show()
    plot_tree(xgb_cl.get_booster())
    pyplot.show()
    print('columns')
    print(cat_cols)
    print(num_cols)
