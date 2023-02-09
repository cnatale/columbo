#### File to test pipelining redshift results to pandas, cleaning data, and packaging for XGBoost
# -find market where you know gaming has happened, limit to just that market and time period.
# -go feature hunting.
#     -lat/lng,
#     -time,
#     -download speed,
#     -connection type.
# -think about using dimension reduction techniques like PCA. Implementation can be found in ScikitLearn
# -randomly sample half the data.
# -hold one half of data separate as validation data.
# -feed the validation data in, see what classification is,
# -then figure out error rate.
#   -^ two-fold cross validation. could walk through different model versions
#   -detail how classification rate went up or down


import torch
import redshift_connector
from dotenv import load_dotenv
from lookups import set_sizes
import pandas as pd

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

cursor.execute("""SELECT country_id, sim_mobile_carrier_id, is_blacklisted, result_date
FROM prod_analytics.mobile_base
WHERE is_blacklisted = FALSE
LIMIT 20""")

result: pandas.DataFrame = cursor.fetch_dataframe()
print("pipeline result: ")
print(result.head())
print(result.info())

# ## gets a blacklist reason code
# cursor.execute("""select * from prod_analytics.mobile_blacklist_reasons JOIN prod_analytics.mobile_blacklist_reason_codes
# ON mobile_blacklist_reason_codes.reason_code = mobile_blacklist_reasons.reason_code ORDER BY prod_analytics.mobile_blacklist_reasons.date_blacklisted
# ASC LIMIT 1;""")

# get a set of blacklisted results
# note that reason_code==99 suspected gaming
# cursor.execute("""SELECT result_id, server_id, mobile_base.device_id, result_date, received_date, timezone_id,
# mobile_base.mobile_platform_id, sim_mobile_carrier_id, network_mobile_carrier_id, isp_id, place_id, country_id,
# server_distance_km, download_kbps, upload_kbps, latency_ms, jitter, ploss_sent, ploss_recv, rssi,
# latitude, longitude, app_version, os_version, packet_loss_pct, tech_family
# FROM prod_analytics.mobile_base
# JOIN prod_analytics.mobile_blacklist_reasons ON mobile_base.device_id=mobile_blacklist_reasons.device_id
# WHERE is_blacklisted=TRUE AND reason_code=99
# LIMIT 100""")
###################################################################
