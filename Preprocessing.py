#######################################################################################################################
# IMPORTS
#######################################################################################################################
from statistics import stdev

import pandas as pd
import numpy as np
from numpy import dtype
import os
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

ISO2 = {'Afghanistan': 'AF',
        'Albania': 'AL',
        'Algeria': 'DZ',
        'American Samoa': 'AS',
        'Andorra': 'AD',
        'Angola': 'AO',
        'Anguilla': 'AI',
        'Antarctica': 'AQ',
        'Antigua and Barbuda': 'AG',
        'Argentina': 'AR',
        'Armenia': 'AM',
        'Aruba': 'AW',
        'Australia': 'AU',
        'Austria': 'AT',
        'Azerbaijan': 'AZ',
        'Bahamas': 'BS',
        'Bahrain': 'BH',
        'Bangladesh': 'BD',
        'Barbados': 'BB',
        'Belarus': 'BY',
        'Belgium': 'BE',
        'Belize': 'BZ',
        'Benin': 'BJ',
        'Bermuda': 'BM',
        'Bhutan': 'BT',
        'Bolivia, Plurinational State of': 'BO',
        'Bonaire, Sint Eustatius and Saba': 'BQ',
        'Bosnia and Herzegovina': 'BA',
        'Botswana': 'BW',
        'Bouvet Island': 'BV',
        'Brazil': 'BR',
        'British Indian Ocean Territory': 'IO',
        'Brunei Darussalam': 'BN',
        'Bulgaria': 'BG',
        'Burkina Faso': 'BF',
        'Burundi': 'BI',
        'Cambodia': 'KH',
        'Cameroon': 'CM',
        'Canada': 'CA',
        'Cape Verde': 'CV',
        'Cayman Islands': 'KY',
        'Central African Republic': 'CF',
        'Chad': 'TD',
        'Chile': 'CL',
        'China': 'CN',
        'Christmas Island': 'CX',
        'Cocos (Keeling) Islands': 'CC',
        'Colombia': 'CO',
        'Comoros': 'KM',
        'Congo': 'CG',
        'Congo, the Democratic Republic of the': 'CD',
        'Cook Islands': 'CK',
        'Costa Rica': 'CR',
        'Country name': 'Code',
        'Croatia': 'HR',
        'Cuba': 'CU',
        'Curaçao': 'CW',
        'Cyprus': 'CY',
        'Czech Republic': 'CZ',
        "Côte d'Ivoire": 'CI',
        'Denmark': 'DK',
        'Djibouti': 'DJ',
        'Dominica': 'DM',
        'Dominican Republic': 'DO',
        'Ecuador': 'EC',
        'Egypt': 'EG',
        'El Salvador': 'SV',
        'Equatorial Guinea': 'GQ',
        'Eritrea': 'ER',
        'Estonia': 'EE',
        'Ethiopia': 'ET',
        'Falkland Islands (Malvinas)': 'FK',
        'Faroe Islands': 'FO',
        'Fiji': 'FJ',
        'Finland': 'FI',
        'France': 'FR',
        'French Guiana': 'GF',
        'French Polynesia': 'PF',
        'French Southern Territories': 'TF',
        'Gabon': 'GA',
        'Gambia': 'GM',
        'Georgia': 'GE',
        'Germany': 'DE',
        'Ghana': 'GH',
        'Gibraltar': 'GI',
        'Greece': 'GR',
        'Greenland': 'GL',
        'Grenada': 'GD',
        'Guadeloupe': 'GP',
        'Guam': 'GU',
        'Guatemala': 'GT',
        'Guernsey': 'GG',
        'Guinea': 'GN',
        'Guinea-Bissau': 'GW',
        'Guyana': 'GY',
        'Haiti': 'HT',
        'Heard Island and McDonald Islands': 'HM',
        'Holy See (Vatican City State)': 'VA',
        'Honduras': 'HN',
        'Hong Kong': 'HK',
        'Hungary': 'HU',
        'ISO 3166-2:GB': '(.uk)',
        'Iceland': 'IS',
        'India': 'IN',
        'Indonesia': 'ID',
        'Iran, Islamic Republic of': 'IR',
        'Iraq': 'IQ',
        'Ireland': 'IE',
        'Isle of Man': 'IM',
        'Israel': 'IL',
        'Italy': 'IT',
        'Jamaica': 'JM',
        'Japan': 'JP',
        'Jersey': 'JE',
        'Jordan': 'JO',
        'Kazakhstan': 'KZ',
        'Kenya': 'KE',
        'Kiribati': 'KI',
        "Korea, Democratic People's Republic of": 'KP',
        'Korea, Republic of': 'KR',
        'Kuwait': 'KW',
        'Kyrgyzstan': 'KG',
        "Lao People's Democratic Republic": 'LA',
        'Latvia': 'LV',
        'Lebanon': 'LB',
        'Lesotho': 'LS',
        'Liberia': 'LR',
        'Libya': 'LY',
        'Liechtenstein': 'LI',
        'Lithuania': 'LT',
        'Luxembourg': 'LU',
        'Macao': 'MO',
        'Macedonia, the former Yugoslav Republic of': 'MK',
        'Madagascar': 'MG',
        'Malawi': 'MW',
        'Malaysia': 'MY',
        'Maldives': 'MV',
        'Mali': 'ML',
        'Malta': 'MT',
        'Marshall Islands': 'MH',
        'Martinique': 'MQ',
        'Mauritania': 'MR',
        'Mauritius': 'MU',
        'Mayotte': 'YT',
        'Mexico': 'MX',
        'Micronesia, Federated States of': 'FM',
        'Moldova, Republic of': 'MD',
        'Monaco': 'MC',
        'Mongolia': 'MN',
        'Montenegro': 'ME',
        'Montserrat': 'MS',
        'Morocco': 'MA',
        'Mozambique': 'MZ',
        'Myanmar': 'MM',
        'Namibia': 'NA',
        'Nauru': 'NR',
        'Nepal': 'NP',
        'Netherlands': 'NL',
        'New Caledonia': 'NC',
        'New Zealand': 'NZ',
        'Nicaragua': 'NI',
        'Niger': 'NE',
        'Nigeria': 'NG',
        'Niue': 'NU',
        'Norfolk Island': 'NF',
        'Northern Mariana Islands': 'MP',
        'Norway': 'NO',
        'Oman': 'OM',
        'Pakistan': 'PK',
        'Palau': 'PW',
        'Palestine, State of': 'PS',
        'Panama': 'PA',
        'Papua New Guinea': 'PG',
        'Paraguay': 'PY',
        'Peru': 'PE',
        'Philippines': 'PH',
        'Pitcairn': 'PN',
        'Poland': 'PL',
        'Portugal': 'PT',
        'Puerto Rico': 'PR',
        'Qatar': 'QA',
        'Romania': 'RO',
        'Russian Federation': 'RU',
        'Rwanda': 'RW',
        'Réunion': 'RE',
        'Saint Barthélemy': 'BL',
        'Saint Helena, Ascension and Tristan da Cunha': 'SH',
        'Saint Kitts and Nevis': 'KN',
        'Saint Lucia': 'LC',
        'Saint Martin (French part)': 'MF',
        'Saint Pierre and Miquelon': 'PM',
        'Saint Vincent and the Grenadines': 'VC',
        'Samoa': 'WS',
        'San Marino': 'SM',
        'Sao Tome and Principe': 'ST',
        'Saudi Arabia': 'SA',
        'Senegal': 'SN',
        'Serbia': 'RS',
        'Seychelles': 'SC',
        'Sierra Leone': 'SL',
        'Singapore': 'SG',
        'Sint Maarten (Dutch part)': 'SX',
        'Slovakia': 'SK',
        'Slovenia': 'SI',
        'Solomon Islands': 'SB',
        'Somalia': 'SO',
        'South Africa': 'ZA',
        'South Georgia and the South Sandwich Islands': 'GS',
        'South Sudan': 'SS',
        'Spain': 'ES',
        'Sri Lanka': 'LK',
        'Sudan': 'SD',
        'Suriname': 'SR',
        'Svalbard and Jan Mayen': 'SJ',
        'Swaziland': 'SZ',
        'Sweden': 'SE',
        'Switzerland': 'CH',
        'Syrian Arab Republic': 'SY',
        'Taiwan, Province of China': 'TW',
        'Tajikistan': 'TJ',
        'Tanzania, United Republic of': 'TZ',
        'Thailand': 'TH',
        'Timor-Leste': 'TL',
        'Togo': 'TG',
        'Tokelau': 'TK',
        'Tonga': 'TO',
        'Trinidad and Tobago': 'TT',
        'Tunisia': 'TN',
        'Turkey': 'TR',
        'Turkmenistan': 'TM',
        'Turks and Caicos Islands': 'TC',
        'Tuvalu': 'TV',
        'Uganda': 'UG',
        'Ukraine': 'UA',
        'United Arab Emirates': 'AE',
        'United Kingdom': 'GB',
        'United States': 'US',
        'United States Minor Outlying Islands': 'UM',
        'Uruguay': 'UY',
        'Uzbekistan': 'UZ',
        'Vanuatu': 'VU',
        'Venezuela, Bolivarian Republic of': 'VE',
        'Viet Nam': 'VN',
        'Virgin Islands, British': 'VG',
        'Virgin Islands, U.S.': 'VI',
        'Wallis and Futuna': 'WF',
        'Western Sahara': 'EH',
        'Yemen': 'YE',
        'Zambia': 'ZM',
        'Zimbabwe': 'ZW',
        'Åland Islands': 'AX'}

#######################################################################################################################
# LOAD DATA
#######################################################################################################################
def load_original_training_data(original_set):
    main = pd.read_csv(original_set)
    main['cancellation_datetime'] = pd.to_datetime(main['cancellation_datetime'])
    main['cancellation_datetime'] = main['cancellation_datetime'].fillna(0)
    main["cancelled"] = np.where(main["cancellation_datetime"] != 0, 1, 0)
    main = main.drop("cancellation_datetime", 1)
    return main

def merge_data_frames(original_set, test_sets, labels):
    frames = []
    main = pd.read_csv(original_set)
    main['cancellation_datetime'] = pd.to_datetime(main['cancellation_datetime'])
    main['cancellation_datetime'] = main['cancellation_datetime'].fillna(0)
    main["cancelled"] = np.where(main["cancellation_datetime"] != 0, 1, 0)
    main = main.drop("cancellation_datetime", 1)

    frames.append(main)
    for i in range(len(labels)):
        test_set = pd.read_csv(test_sets[i])
        label_df = pd.read_csv(labels[i])
        # test_set["cancelled"] = label_df['h_booking_id|label'].str[-1:].astype(int)
        test_set["cancelled"] = label_df['cancel'].astype(int)
        frames.append(test_set)
    return pd.concat(frames)

local = "/Users/mayagoldman/PycharmProjects/new_data_challenge/"
original = "/Users/mayagoldman/PycharmProjects/new_data_challenge/data/agoda_cancellation_train.csv"
test_sets = [local + "data/Weekly test set/week_1_test_data.csv" , local + "data/Weekly test set/week_2_test_data.csv" , local + "data/Weekly test set/week_3_test_data.csv" , local + "data/Weekly test set/week_4_test_data.csv"] # , local + "data/Weekly test set/week_5_test_data.csv"]
labels = [local+"data/Labels/week_1_labels.csv" , local+"data/Labels/week_2_labels.csv" , local+"data/Labels/week_3_labels.csv" , local+"data/Labels/week_4_labels.csv" ]#, local+"data/Labels/week_5_labels.csv"]


#######################################################################################################################
#PREPROCESSING METHODS
#######################################################################################################################

def cancellationPolicyPreprocess(df):
    new_cols = ['1st_pnlty_days', '1st_pnlty_amount', '1st_pnlty_type',
                '2st_pnlty_days', '2st_pnlty_amount', '2st_pnlty_type',
                'no_show_amount', 'no_show_type']
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

    return df


binary_features =[ "request_highfloor", "request_largebed", "request_nonesmoke", "request_twinbeds" , "request_airport", "request_earlycheckin"]

def fill_null(df):
    for f in binary_features:
        df[f] = df[f].fillna(0)
    return df.dropna().drop_duplicates()


def replace(df):
    df["guest_is_not_the_customer"] = np.where(df["guest_is_not_the_customer"] == True, 1, 0)
    df["is_user_logged_in"] = np.where(df["is_user_logged_in"] == True, 1, 0)
    df["is_first_booking"] = np.where(df["is_first_booking"] == True, 1, 0)
    return df.replace({"guest_nationality_country_name": ISO2})


DateTimeFeatures = ['booking_datetime', 'checkin_date', 'checkout_date' , 'hotel_live_date']

def toDateTime(df):
    for col in DateTimeFeatures:
        df[col] = pd.to_datetime(df[col])
    return df


non_negative_features = []
positive_features = []


def validateRanges(df):
    for feature in positive_features:
        df = df[df[feature] > 0]
    for feature in non_negative_features:
        df = df[df[feature] >= 0]
    for feature in binary_features:
        df = df[df[feature].isin([0, 1])]
    df = df[df["hotel_star_rating"].isin([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])]
    return df

#######################################################################################################################
# PREPROCESSING METHODS - FEATURE EXTRACTION & CREATION
#######################################################################################################################
redundant_features = []
to_process = ["hotel_city_code" , "hotel_chain_code" , "hotel_brand_code" , "hotel_area_code" , "h_customer_id" , "h_booking_id" , "hotel_id" , "cancellation_policy_code"]

def remove_outliers(df):
    df = df[df["booked_ahead"] < 200]


sparcity = ["original_payment_currency" , "original_payment_method", "language" , "origin_country_code" , "guest_nationality_country_name" , "customer_nationality" ,
            "hotel_country_code" , "accommadation_type_name" ]

def group_sparce(df):
    for i in sparcity:
        df[i] = df[i].mask(df[i].map(df[i].value_counts(normalize=True)) < 0.05, i + '_Other')
    return df

def create_new_features(df):
    special_requests = (list(df.filter(regex="request")))
    df["special_requests"] = df[special_requests].sum(axis=1)

    df["booking_days"] = (df['checkout_date'] - df['checkin_date']).dt.days

    df["cost_per_night"] = df["original_selling_amount"] / (df["booking_days"] * df['no_of_adults']) #include children or not?

    # df["weekday_num_nights"] = np.busday_count(df['checkin_date'].values.astype('datetime64[D]'),
    #                                              df['checkout_date'].values.astype('datetime64[D]'))
    # df["weekend_num_nights"] = (df["booking_days"] - df["weekday_num_nights"])

    df["booked_ahead"] = (df['checkin_date'] - df["booking_datetime"]).dt.days

    df["booked_ahead"].replace({-1: 0}, inplace=True)  # same-day bookings are valued 0.


    # df["local stay"] = (df["guest_nationality_country_name"] == df["hotel_country_code"])
    # df["local stay"] = np.where(df["local stay"] == True, 1, 0)


    # df["no_of_children"] = np.where(df["no_of_children"] > 0, 1, 0)
    # df["pay_now"] = np.where(df["charge_option"] == "Pay Now", 1, 0)
    # df = df[df["hotel_live_date"].dt.year > 2008]
    # df["new_hotel"] = np.where(df["hotel_live_date"].dt.year > 2016, 1, 0)
    # df["unknown_payment"] = np.where(df["original_payment_method"] == "UNKNOWN", 1, 0)
    # df["Card"] = np.where(df["original_payment_type"] == "Credit Card", 1, 0)

    return df


categorial_features = []

def drop_redundant(df):
    for feature in to_process:
        df = df.drop(feature, 1)
    for feature in sparcity:
        df = df.drop(feature, 1)
    for feature in DateTimeFeatures :
        df = df.drop(feature, 1)

    df = df.drop("charge_option", 1)
    df = df.drop("original_payment_type", 1)

    return df

#######################################################################################################################
# DATA EXPLORATION - GRAPH METHODS
#######################################################################################################################
def produce_graph():
    if not os.path.exists("data_challenge_images"):
        os.mkdir("data_challenge_images")

    for c in df.columns:
        if dtype(df[c]):
            title = str(c) + " vs cancelled"
            fig = px.histogram(df , x= c , color = "cancelled", histnorm = "density" ,  barmode='group', height=400 , title = title )
            # fig.write_image("data_challenge_images/" + str(c) +".png")
            fig.show("browser")



#######################################################################################################################
# PREPROCESS TRAIN DATA
#######################################################################################################################

df = merge_data_frames( original, test_sets , labels)
# df = load_original_training_data(original)
df = fill_null(df)
# df = cancellationPolicyPreprocess(df)
df = toDateTime(df)
df = replace(df)
df = group_sparce(df)
df = create_new_features(df)
df = drop_redundant(df)

X = df.drop("cancelled", 1)
y = df.cancelled

#######################################################################################################################
# PREPROCESS TEST DATA
#######################################################################################################################
def preprocess_past_test_data(test, label):
    df = pd.read_csv(test)
    for f in binary_features:
        df[f] = df[f].fillna(0)
    # df = cancellationPolicyPreprocess(df)
    df = toDateTime(df)
    df = replace(df)
    df = group_sparce(df)
    df = create_new_features(df)
    df = drop_redundant(df)
    label_df = pd.read_csv(label)
    return df , label_df['cancel'].astype(int)

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


#######################################################################################################################
# PREDICTIONS USING SKLEARN
#######################################################################################################################

# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import AdaBoostRegressor
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# clf = RandomForestClassifier()
# # clf = KNeighborsClassifier()
# # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
# clf.fit(X_train , y_train)

####################### XGB CLASSIFIER #################################################################################


X_processed = full_processor.fit_transform(X)
y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
    y.values.reshape(-1, 1)
)

from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=10, shuffle=True)
xgb_cl = xgb.XGBClassifier()


lst_accu_stratified = []

for train_index, test_index in skf.split(X, y):
    x_train_fold, x_test_fold = X_processed[train_index], X_processed[test_index]
    y_train_fold, y_test_fold = y_processed[train_index], y_processed[test_index]
    xgb_cl.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(xgb_cl.score(x_test_fold, y_test_fold))

# Print the output.
print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(lst_accu_stratified) * 100, '%')
print('\nMinimum Accuracy:',
      min(lst_accu_stratified) * 100, '%')
print('\nOverall Accuracy:',
      np.mean(lst_accu_stratified) * 100, '%')
print('\nStandard Deviation is:', stdev(lst_accu_stratified))


# for i in range(len(test_sets)):
#     X_test , y_test = preprocess_past_test_data(test_sets[i], labels[i])
#     print(xgb_cl.score(X_test , y_test))

from sklearn import metrics

xgb_cl.fit(X_processed, y_processed)
X_test, y_test = preprocess_past_test_data(local + "data/Weekly test set/week_5_test_data.csv",  local+"data/Labels/week_5_labels.csv")
# print(xgb_cl.score(X_test, y_test) * 100)
y_pred = xgb_cl.predict(X_test)
print("Accuracy:",metrics.f1_score(y_test, y_pred, average= 'macro'))
cm = confusion_matrix(y_test, y_pred, labels=xgb_cl.classes_ )
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_cl.classes_)
plt = disp.plot()
print(cm)
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X_processed, y_processed, stratify=y_processed)
#
# from sklearn.metrics import accuracy_score
#
# # Init classifier
# xgb_cl = xgb.XGBClassifier()
#
# # Fit
# xgb_cl.fit(X_train, y_train)
#
# # Predict
# preds = xgb_cl.predict(X_test)
#
# # Score
# print(accuracy_score(y_test, preds))


#######################################################################################################################
# MODEL EVALUATION
#######################################################################################################################

# print(clf.score(X_test , y_test))



#######################################################################################################################
# PARAMETER TUNING
#######################################################################################################################
# param_grid = {
#     "max_depth": [3, 4, 5, 7],
#     "learning_rate": [0.1, 0.01, 0.05],
#     "gamma": [0, 0.25, 1],
#     "reg_lambda": [0, 1, 10],
#     "scale_pos_weight": [1, 3, 5],
#     "subsample": [0.8],
#     "colsample_bytree": [0.5],
# }
#
# from sklearn.model_selection import GridSearchCV
#
# # Init classifier
# xgb_cl = xgb.XGBClassifier(objective="binary:logistic")
#
# # Init Grid Search
# grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")
#
# # Fit
# _ = grid_cv.fit(X_processed, y_processed)