# config.py

ORDER_COLUMNS = [
    'booking_id', 'member_id', 'cdate', 'restaurant_id', 'datetime', 
    'people', 'gender', 'status', 'is_required_prepay_satisfied', 'return90'
]

RESTAURANT_COLUMNS = [
    'id', 'is_hotel', 'country', 'currency', 'cityarea', 'name', 'abbr', 
    'tel', 'opening_hours', 'good_for_family', 'accept_credit_card', 
    'parking', 'outdoor_seating', 'wifi', 'wheelchair_accessible', 
    'price1', 'price2', 'lat', 'lng', 'timezone', 'locale', 'cdate'
]

MEMBER_COLUMNS = [
    'id', 'is_vip', 'gender', 'birthdate', 'city', 'has_google_id', 
    'has_yahoo_id', 'has_weibo_id', 'cdate'
]