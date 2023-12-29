import datetime as dt

def error(msg):
    print(f"💥 ERROR: {msg}")

def info(msg):
    print(f"📄 INFO [{getTimePrefix()}]:{msg}")

def warn(msg):
    print(f"⭐ WARN: {msg}")    

def debug(msg):
    print(f"💻 DEBUG [{getTimePrefix()}]: {msg}")

def success(msg):
    print(f"✅ SUCCESS: {msg}")

def progress(msg):
    print(f"🔄 PROG: {msg}")
   
def getTimePrefix():
    now = dt.datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3] 