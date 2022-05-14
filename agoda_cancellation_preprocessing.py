import numpy as np
import pandas as pd

# pio.templates.default = "simple_white"
#
# redundant_features = ["h_booking_id" , "hotel_id" , "h_customer_id" , "hotel_brand_code" , "hotel_chain_code"]
# non_negative_features = ["no_of_children" , "no_of_extra_bed" ]
# positive_features =["no_of_adults" , "no_of_room"]
# ranged_features = {"hotel_star_rating" : (0,5,0.5) }
# binary_features = ["request_airport" , "request_earlycheckin" , "request_highfloor" , "request_largebed" , "request_latecheckin" , "request_nonesmoke"  , "request_twinbeds"]
# dummies = ["hotel_area_code" , "hotel_city_code" , "hotel_country_name" , "accommadation_type_name" , "charge_option" , "customer_nationality" ,
#            "guest_nationality_country_name" , "origin_country_code" , "language" , "original_payment_method" , "original_payment_type" ,
#            "original_payment_currency" , "cancellation_policy_code" ]
#
# # def canceled_between():
# #     return
#
# def fill_null(data):
#     data['cancellation_datetime'] = data['cancellation_datetime'].fillna(0)
#     data["cancelled"] = np.where(data["cancellation_datetime"] != 0, 1, 0)
#     data.drop("cancellation_datetime", 1)
#     return data.dropna().drop_duplicates()
#
# def drop_redundant(data, features):
#     for feature in features:
#         data = data.drop(feature, 1)
#     return data
#
# def binarize(data):
#     data["guest_is_not_the_customer"] = np.where(data["guest_is_not_the_customer"] == True, 1, 0)
#     data["is_user_logged_in"] = np.where(data["is_user_logged_in"] == True, 1, 0)
#     data["is_first_booking"] = np.where(data["is_user_logged_in"] == True, 1, 0)
#     return data
#
#
# def process_categorial(data):
#     for f in  dummies:
#         data = pd.get_dummies(data, prefix=f +'_', columns=[f])
#     data = binarize(data)
#     return data
#
# def process_dates(data):
#     return
#
# def process_input(data):
#     data = drop_redundant(data , redundant_features)
#     data = fill_null(data)
#     for feature in positive_features:
#         data = data[data[feature] > 0]
#     for feature in non_negative_features:
#         data = data[data[feature] >= 0]
#     for feature in binary_features:
#         data = data[data[feature].isin([0, 1])]
#     for feature in ranged_features:
#         a,b,c= ranged_features[feature]
#         data = data[data[feature].isin(range(a,b,c))]
#     return data
#
#
# def remove_outliers(data):
#     return data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
#
# def load_data(filename: str):
#     """
#     Load house prices dataset and preprocess data.
#     Parameters
#     ----------
#     filename: str
#         Path to house prices dataset
#
#     Returns
#     -------
#     Design matrix and response vector (prices) - either as a single
#     DataFrame or a Tuple[DataFrame, Series]
#     """
#     # load data:
#
#     df = pd.read_csv(filename)
#     df = process_input(df)
#     df = process_categorial(df)
#     df = remove_outliers(df)
#     return df.drop("cancelled", 1), df.cancelled
#
new_cols = ['1st_pnlty_days', '1st_pnlty_amount', '1st_pnlty_type',
              '2st_pnlty_days', '2st_pnlty_amount', '2st_pnlty_type',
              'no_show_amount', 'no_show_type']

df = pd.read_csv("C:\\Users\\User\\Desktop\\IML.HUJI-main\\dataPreprocessing\\agoda_cancellation_train.csv")

for col in new_cols:
    df[col] = np.nan


for index, s in enumerate(df.cancellation_policy_code):
  s = (s.split('_'))
  for policy in s:
      no_show = False
      counter = 0
      if 'D' in ''.join(policy):
          p = policy.split('D')
          day1 = p[0]
          fee1 = p[1][:-1]
          typ1 = p[1][-1]
          counter += 1
          if counter == 1:
              # print(day1, fee1, typ1)
              df.at[index, '1st_pnlty_days'] = day1
              df.at[index, '1st_pnlty_amount'] = fee1
              df.at[index, '1st_pnlty_type'] = typ1
          else:
              # print(day1, fee1, typ1)
              df.at[index, '2st_pnlty_days'] = day1
              df.at[index, '2st_pnlty_amount'] = fee1
              df.at[index, '2st_pnlty_type'] = typ1
      elif policy != 'UNKNOWN':  # no show case
          no_show = True
          if 'N' in ''.join(policy):
              p = policy.split('N')
              noshow_amount = p[0]
              noshow_type = 'N'
          else:
              p = policy.split('P')
              noshow_amount = p[0]
              noshow_type = 'P'
          df.at[index, 'no_show_amount'] = noshow_amount
          df.at[index, 'no_show_type'] = noshow_type
      else:
        for k in new_cols:
          df.at[index, k] = '0'

print(df.head())