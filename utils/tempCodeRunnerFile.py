    import datetime
    import pytz
    currtime = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    ct = str(currtime)
    date = ct[0:10]
    time = ct[10:16]