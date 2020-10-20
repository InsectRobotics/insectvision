from datetime import datetime, timedelta


def shifted_datetime(roll_back_days=153, lower_limit=7.5, upper_limit=19.5):
    date_time = datetime.now() - timedelta(days=roll_back_days)
    if lower_limit is not None and upper_limit is not None:
        uhours = int(upper_limit // 1)
        uminutes = timedelta(minutes=(upper_limit % 1) * 60)
        lhours = int(lower_limit // 1)
        lminutes = timedelta(minutes=(lower_limit % 1) * 60)
        if (date_time - uminutes).hour > uhours or (date_time - lminutes).hour < lhours:
            date_time = date_time + timedelta(hours=12)
    return date_time
