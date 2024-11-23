#cleanup script to get rid of spaces and convert to proper csv
import pandas as pd
import matplotlib.dates
from datetime import datetime

df = pd.read_csv('517_uncleaned.csv',sep='\s+',header=None);

def datetime_adjust(x):
    mtime = matplotlib.dates.num2date(x);
    hour = mtime.hour
    minute = mtime.minute;
    if (30 <= mtime.second):
        minute += 1;
    if (minute > 59):
        hour += 1;
        minute = 0;
    newDateTime = datetime(mtime.year, mtime.month, mtime.day, hour, minute)
    times = matplotlib.dates.date2num(newDateTime)
    return times;

#convert to unix epoch
df[1] = df[1] - 719163;
df[1] = df[1].apply(lambda x: datetime_adjust(x))
for times in df[1]:
    print(matplotlib.dates.num2date(times))
df[4] = df[4] + 273.15

#DHT22 calibration correction applied
df[5] = 0.9144 * df[5] + 5.6403
#df[1].to_csv('517_halfhourly_cleaned.csv',header=None);
df.to_csv('517_cleaned.csv',header=['Hour', 'Days_since_UNIX', 'PM2.5', 'PM10', 'Temp/K', 'Humidity']);
