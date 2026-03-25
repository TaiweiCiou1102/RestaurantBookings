import pandas as pd
import numpy as np
from src.etl.utils import *

def clean_order():
    order = pd.read_csv(
        './data/raw/train.csv', 
        sep='\t', encoding='utf-16', on_bad_lines = 'warn'
    )

    purpose_mapping = {
        '請選擇': 'Unknown',
        'Please,Select': 'Unknown',
        'Friends,Reunion': '朋友聚餐',
        'Family,Gathering': '家人用餐',       
        'Birthday': '生日慶祝',
        'Friends': '朋友聚餐',
        '家人用餐,盡量靠W': '朋友聚餐',
        'Business,meeting':'商務聚餐',
        '家人&#65533;':'家人用餐',
        '家人用餐-生日':'生日慶祝',
        '慶生':'生日慶祝',
        'Birthday,Celebration':'生日慶祝',
        '甜蜜紀念日': '紀念日慶祝'
    }

    order_final = (order
        .pipe(fill_missing_values, col_name='purpose', fill_value='Unknown')
        .pipe(parse_datetime_column, col_name='cdate')
        .pipe(parse_datetime_column, col_name='datetime')
        .pipe(trim_whitespace, col_name = 'purpose')
        .pipe(map_categorical_values, col_name = 'purpose', mapping = purpose_mapping)
        .pipe(remove_specific_value_str, col_name = 'purpose', value = '&#65533;&#40115;&#65533;&#65533;&#26813;&#65533;')
        .pipe(convert_to_category, col_name = 'purpose')
        .pipe(convert_to_category, col_name = 'gender')
        .pipe(convert_to_category, col_name = 'status')
    )

    order_final.to_csv('./data/interim/train_silver.csv', index=False)

    return None

def clean_restaurant():
    restaurant = pd.read_csv(
        './data/raw/restaurant_revised.csv', 
        sep='\t', encoding='utf-16', on_bad_lines = 'warn'
    )
    city_mapping = {
        '桃園縣':'桃園市',
        '%':'Unknown'
    }

    cityarea_mapping = {
        '請選擇':'Unknown'
    }

    restaurant_final = (
        restaurant
        .pipe(fill_missing_values, col_name='cityarea')
        .pipe(parse_datetime_column, col_name='cdate')
        .pipe(map_categorical_values, col_name = 'city', mapping = city_mapping)
        .pipe(map_categorical_values, col_name = 'cityarea', mapping = cityarea_mapping)
        .pipe(convert_to_category, col_name='country')
        .pipe(convert_to_category, col_name='currency')
        .pipe(convert_to_category, col_name='city')
        .pipe(convert_to_category, col_name='cityarea')
        .pipe(convert_to_category, col_name='timezone')
        .pipe(convert_to_category, col_name='locale')
    )

    restaurant_final.to_csv('./data/interim/restaurant_silver.csv', index=False)
    return None

def clean_member():

    conti_cols = []
    datetime_cols = ['birthdate','cdate']
    category_cols = ['gender','city']
    binary_cols = [
        'is_vip','has_google_id','has_yahoo_id','has_weibo_id'
        ]
    ordinal_cols = []
    unstructured_cols = []
    y_cols = []

    member = pd.read_csv(
        './data/raw/member.csv', 
        sep='\t', encoding='utf-16', on_bad_lines = 'warn'
    )

    member_city_mapping = {
        '桃園縣':'桃園市',
        '0':'Unknown',
        0:'Unknown'
    }

    member_final = (
        member
        .pipe(fill_missing_values, col_name = 'city')
        .pipe(fill_missing_values, col_name = 'gender')
        .pipe(parse_datetime_column, col_name = datetime_cols[0])
        .pipe(parse_datetime_column, col_name = datetime_cols[1])
        .pipe(remove_specific_value_str, col_name = 'city', value = '??蝮?>??蝮?/option><option value')
        .pipe(map_categorical_values, col_name = 'city', mapping = member_city_mapping)
        .pipe(convert_to_category, col_name = category_cols[0])
        .pipe(convert_to_category, col_name = category_cols[1])
        )

    member_final.to_csv('./data/interim/member_silver.csv', index=False)

    return None

if __name__ == '__main__':
    clean_order()
    clean_restaurant()
    clean_member()


