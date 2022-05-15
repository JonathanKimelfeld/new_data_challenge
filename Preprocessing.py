import pandas as pd
import numpy as np

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


def merge_data_frames(original_set, test_sets, labels):
    frames = []
    main = pd.read_csv(original_set)
    main['cancellation_datetime'] = pd.to_datetime(main['cancellation_datetime'])
    main['cancellation_datetime'] = main['cancellation_datetime'].fillna(0)
    main["cancelled"] = np.where(main["cancellation_datetime"] != 0, 1, 0)
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
test_sets = [local + "data/Weekly test set/week_1_test_data.csv" , local + "data/Weekly test set/week_2_test_data.csv" , local + "data/Weekly test set/week_3_test_data.csv" , local + "data/Weekly test set/week_4_test_data.csv" , local + "data/Weekly test set/week_5_test_data.csv"]
labels = [local+"data/Labels/week_1_labels.csv" , local+"data/Labels/week_2_labels.csv" , local+"data/Labels/week_3_labels.csv" , local+"data/Labels/week_4_labels.csv" , local+"data/Labels/week_5_labels.csv"]

df = merge_data_frames( original, test_sets , labels)
df.head()
df.describe()
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


def fill_null(df):
    for f in binary_features:
        df[f] = df[f].fillna(0)
    return df.dropna().drop_duplicates()


def binarize(df):
    df["guest_is_not_the_customer"] = np.where(df["guest_is_not_the_customer"] == True, 1, 0)
    df["is_user_logged_in"] = np.where(df["is_user_logged_in"] == True, 1, 0)
    df["is_first_booking"] = np.where(df["is_first_booking"] == True, 1, 0)
    return df


DateTimeFeatures = ['booking_datetime', 'checkin_date', 'checkout_date', 'cancellation_datetime']


def toDateTime(df):
    pass


non_negative_features = []
positive_features = []
ranged_features = []
binary_features = []


def validateRanges(df):
    for feature in positive_features:
        df = df[df[feature] > 0]
    for feature in non_negative_features:
        df = df[df[feature] >= 0]
    for feature in binary_features:
        df = df[df[feature].isin([0, 1])]
    for feature in ranged_features:
        df = df[df[feature].isin(ranged_features[feature])]
    return df


#######################################################################################################################
redundant_features = []


def drop_redundant(df):
    for feature in redundant_features:
        df = df.drop(feature, 1)
    return df


def remove_outliers(df):
    pass


def group_sparce(df):
    pass


categorial_features = []
