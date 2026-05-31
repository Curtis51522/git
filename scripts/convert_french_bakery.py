# -*- coding: utf-8 -*-
import sys, os, numpy as np, pandas as pd, holidays, warnings
from datetime import date, timedelta
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RANDOM_SEED = 77
np.random.seed(RANDOM_SEED)
MY = holidays.MY()
KL = {1:(27.3,226.7,80),2:(27.8,192.8,80),3:(28.1,270.4,80),4:(28.1,301.5,82),5:(28.5,229.9,81),6:(28.4,145.8,80),7:(28.0,165.2,79),8:(28.0,174.3,79),9:(27.7,220.3,81),10:(27.5,283.8,82),11:(27.1,355.8,84),12:(27.1,280.6,83)}
MAP = {"croissant":["CROISSANT","CROISSANT AMANDES"],"donut":["ECLAIR","COOKIE","TARTELETTE"],"chiffon":["BRIOCHE","PAIN AUX RAISINS"],"bread_coconut":["SPECIAL BREAD","CEREAL BAGUETTE","CAMPAGNE"],"bread_roll":["BAGUETTE","TRADITIONAL BAGUETTE","COUPE","BANETTE","BANETTINE","PAIN BANETTE","FICELLE","PAIN","COMPLET","BOULE 400G","BOULE 200G"],"croissant_chocolate":["PAIN AU CHOCOLAT"]}
PRODS = list(MAP.keys())
SCALE = {"bread_roll":0.125,"donut":1.8,"chiffon":2.5}
def ramadan(d):
    try:
        from hijri_converter import convert
        h = convert.Gregorian(d.year,d.month,d.day).to_hijri()
        return 1 if h.month==9 else 0
    except: return 0
def weather(dt):
    m=dt.month;t,r,h=KL[m]
    temp=round(np.clip(np.random.normal(t,1.8),t-4,t+4),1)
    rain=round(np.random.exponential(r/30.0),1)
    hum=round(np.clip(np.random.normal(h,5),60,98),1)
    wt="rainy" if rain>15 else ("cloudy" if rain>3 else "sunny")
    return {"temperature":temp,"rainfall":rain,"humidity":hum,"is_rainy":1 if rain>5 else 0,"weather_sunny":1 if wt=="sunny" else 0,"weather_cloudy":1 if wt=="cloudy" else 0,"weather_rainy":1 if wt=="rainy" else 0,"weather_storm":0,"weather_type":wt}
def lag(sl,prod,fd,db):
    target=fd-timedelta(days=db)
    for _ in range(4):
        k=target.strftime("%Y-%m-%d")
        if (prod,k) in sl: return sl[(prod,k)]
        target-=timedelta(days=1)
    return 0.0
def roll7(sl,prod,fd):
    vals=[sl[(prod,(fd-timedelta(days=o)).strftime("%Y-%m-%d"))] for o in range(1,8) if (prod,(fd-timedelta(days=o)).strftime("%Y-%m-%d")) in sl]
    return float(sum(vals)/len(vals)) if vals else 0.0
print("Loading French bakery...")
df=pd.read_csv("data/kaggle/Bakery sales.csv")
df["date"]=pd.to_datetime(df["date"])
rev={v:k for k,vs in MAP.items() for v in vs}
df["product"]=df["article"].map(rev)
df=df.dropna(subset=["product"])
daily=df.groupby(["date","product"])["Quantity"].sum().reset_index()
for p,f in SCALE.items():
    m=daily["product"]==p
    daily.loc[m,"Quantity"]=(daily.loc[m,"Quantity"]*f).round().astype(int)
dates=sorted(daily["date"].unique())
rec=[];sl={};wc={"sunny":0,"cloudy":0,"rainy":0}
for d in dates:
    dow=d.weekday()
    if dow==0: continue
    w=weather(d);wc[w["weather_type"]]+=1
    dd=daily[daily["date"]==d]
    for p in PRODS:
        r=dd[dd["product"]==p]
        s=int(r["Quantity"].values[0]) if len(r) else 0
        k=d.strftime("%Y-%m-%d");sl[(p,k)]=float(s)
        rec.append({"date":d,"product":p,"sales":s,"day_of_week":dow,"is_weekend":1 if dow>=5 else 0,"day_of_month":d.day,"month":d.month,"is_public_holiday":1 if d in MY else 0,"is_ramadan":ramadan(d),"temperature":w["temperature"],"rainfall":w["rainfall"],"humidity":w["humidity"],"is_rainy":w["is_rainy"],"weather_sunny":w["weather_sunny"],"weather_cloudy":w["weather_cloudy"],"weather_rainy":w["weather_rainy"],"weather_storm":w["weather_storm"],"lag_1":0.0,"lag_7":0.0,"rolling_7d_mean":0.0})
out=pd.DataFrame(rec)
for i,row in out.iterrows():
    p=row["product"];d=row["date"]
    out.at[i,"lag_1"]=lag(sl,p,d,1)
    out.at[i,"lag_7"]=lag(sl,p,d,7)
    out.at[i,"rolling_7d_mean"]=roll7(sl,p,d)
for p in PRODS:
    pdf=out[out["product"]==p]
    print("  %s: mean=%.0f std=%.0f"%(p,pdf["sales"].mean(),pdf["sales"].std()))
out.to_csv("data/synthetic_sales_1year.csv",index=False)
print("Done: %d rows" % len(out))
