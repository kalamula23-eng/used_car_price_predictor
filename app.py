import io
import csv
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="AutoWorth · Car Price Studio", page_icon="🚘", layout="wide", initial_sidebar_state="expanded")
DATA_URL = "https://raw.githubusercontent.com/bagassenop/cardekho/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
FEATURES = ["brand", "model_name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
NUMERIC = ["year", "km_driven", "owner"]
CAT = ["brand", "model_name", "fuel", "seller_type", "transmission"]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif} .stApp{background:radial-gradient(ellipse at 8% 12%,#d8d8ff 0,transparent 30%),radial-gradient(ellipse at 88% 45%,#c8f1e8 0,transparent 29%),radial-gradient(ellipse at 55% 100%,#ffe7d6 0,transparent 32%),linear-gradient(135deg,#edf0ff 0%,#f5f3ff 48%,#e9f8f5 100%);background-attachment:fixed} .block-container{padding-top:1.6rem;max-width:1420px}
.hero{background:radial-gradient(circle at 86% 18%,#398d88 0,transparent 27%),linear-gradient(120deg,#101b35 0%,#1b3156 55%,#24656b 100%);border-radius:24px;padding:34px 40px;color:white;margin-bottom:24px;box-shadow:0 14px 38px #14284829;border:1px solid #ffffff22}
.hero h1{font-family:Manrope,sans-serif;font-size:2.35rem;letter-spacing:-1.3px;margin:4px 0 8px}.hero p{color:#d2e0ef;margin:0;font-size:1.02rem}.eyebrow{color:#77e1c0;text-transform:uppercase;letter-spacing:2px;font-weight:700;font-size:.72rem}
.metric{background:white;border:1px solid #e8edf5;border-radius:16px;padding:18px 20px;min-height:106px}.metric .label{font-size:.8rem;color:#65748b}.metric .value{font-family:Manrope,sans-serif;font-size:1.55rem;font-weight:800;color:#152642;margin-top:5px}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#ffffff,#f0f7fa);border:1px solid #dce8ef;padding:15px;border-radius:15px;box-shadow:0 5px 16px #1634510a} .stButton>button{border-radius:12px;font-weight:700;min-height:3rem;transition:all .2s ease}.stButton>button[kind="primary"]{background:linear-gradient(100deg,#157b72,#138f78);border:0;box-shadow:0 5px 14px #16856d38}.stButton>button[kind="primary"]:hover{background:linear-gradient(100deg,#11675f,#087b66);transform:translateY(-1px)}
.stSelectbox label,.stNumberInput label{font-weight:650;color:#253b58}.stAlert{border-radius:14px}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading the CarDekho listings…")
def load_data(uploaded_bytes=None):
    if uploaded_bytes:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        local_csv = Path(__file__).parent / "data" / "car_details.csv"
        if local_csv.exists():
            df = pd.read_csv(local_csv)
        else:
            with urllib.request.urlopen(DATA_URL, timeout=60) as response:
                df = pd.read_csv(response)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {"name":"name", "year":"year", "km_driven":"km_driven", "fuel":"fuel", "seller_type":"seller_type", "transmission":"transmission", "owner":"owner", "selling_price":"selling_price"}
    df = df.rename(columns=aliases)
    needed = list(aliases)
    missing = [c for c in needed if c not in df.columns]
    if missing: raise ValueError("Dataset columns are missing: " + ", ".join(missing))
    df = df[needed].dropna(subset=["selling_price", "year", "km_driven"])
    df["model_name"] = df["name"].astype(str).str.strip().str.title()
    df["brand"] = df["model_name"].str.split().str[0]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df["owner"] = df["owner"].astype(str).str.lower().map({"first owner":0,"second owner":1,"third owner":2,"fourth & above owner":3,"fourth & above":3}).fillna(pd.to_numeric(df["owner"], errors="coerce"))
    df = df.dropna(subset=["selling_price", "year", "km_driven"])
    df = df[(df.selling_price > 0) & (df.km_driven >= 0)]
    for c in CAT: df[c] = df[c].fillna("Unknown").astype(str).str.title()
    return df

@st.cache_resource(show_spinner="Training two models and checking their accuracy…")
def train(df):
    X, y = df[FEATURES], df.selling_price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERIC), ("cat", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    rf = Pipeline([("prep",prep),("model",RandomForestRegressor(n_estimators=260,min_samples_leaf=2,max_features=.9,n_jobs=-1,random_state=42))])
    et = Pipeline([("prep",prep),("model",ExtraTreesRegressor(n_estimators=260,min_samples_leaf=2,max_features=.9,n_jobs=-1,random_state=42))])
    rf.fit(X_train,y_train); et.fit(X_train,y_train)
    p1,p2=rf.predict(X_test),et.predict(X_test)
    scores=[mean_absolute_error(y_test,p1),mean_absolute_error(y_test,p2)]
    best=rf if scores[0] <= scores[1] else et
    final=best
    final.fit(X,y)
    return final, min(scores), r2_score(y_test,p1 if scores[0] <= scores[1] else p2), len(df), y_test.to_numpy(), (p1 if scores[0] <= scores[1] else p2)

def money(v):
    if v >= 10000000: return f"₹{v/10000000:.2f} Cr"
    if v >= 100000: return f"₹{v/100000:.2f} L"
    return f"₹{v:,.0f}"

with st.sidebar:
    st.markdown("## 🚘 AutoWorth")
    st.caption("USED CAR PRICE INTELLIGENCE")
    page=st.radio("Workspace",["Price estimator","Market insights","About this project"],label_visibility="collapsed")
    st.divider()
    uploaded=st.file_uploader("Use your own CSV dataset",type=["csv"],help="Expected columns: name, year, selling_price, km_driven, fuel, seller_type, transmission, owner")
    st.caption("No file? We’ll load the public CarDekho listings automatically.")

st.markdown('<div class="hero"><div class="eyebrow">Smart valuation · India</div><h1>Know what your car is worth.</h1><p>A data-powered estimate from real used-car listings. Clear, quick, and built for your next move.</p></div>',unsafe_allow_html=True)
try:
    df=load_data(uploaded.getvalue() if uploaded else None)
    model,mae,r2,n_rows,y_test,pred_test=train(df)
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.info("Check your internet connection, or upload a CarDekho-format CSV from the sidebar.")
    st.stop()

if page == "Price estimator":
    left,right=st.columns([1.12,.88],gap="large")
    with left:
        st.subheader("Tell us about the car")
        st.caption("A few details are enough to build your estimate.")
        c1,c2=st.columns(2)
        brands=sorted(df.brand.dropna().unique().tolist())
        brand=c1.selectbox("Brand",brands,index=brands.index("Maruti") if "Maruti" in brands else 0)
        year=c2.selectbox("Year",list(range(datetime.now().year,1990,-1)),index=min(5,datetime.now().year-1991))
        models=sorted(df.loc[df.brand==brand,"model_name"].dropna().unique().tolist())
        model_name=st.selectbox("Car model",models,help="Start typing a model name to find it, for example Maruti 800.")
        st.caption("Choose the closest listed model/variant. You can type in the dropdown to search.")
        km=c1.number_input("Kilometres driven",min_value=0,max_value=1000000,value=45000,step=5000)
        fuel_opts=sorted(df.fuel.unique()); fuel=c2.selectbox("Fuel type",fuel_opts,index=fuel_opts.index("Petrol") if "Petrol" in fuel_opts else 0)
        seller_opts=sorted(df.seller_type.unique()); seller=c1.selectbox("Seller type",seller_opts,index=0)
        trans_opts=sorted(df.transmission.unique()); trans=c2.selectbox("Transmission",trans_opts,index=0)
        owner=c1.selectbox("Previous owners",[0,1,2,3],format_func=lambda x:["First owner","Second owner","Third owner","4+ owners"][x])
        st.markdown(" ")
        predict=st.button("✨  Estimate my car’s value",type="primary",use_container_width=True)
        if predict:
            row=pd.DataFrame([{"brand":brand,"model_name":model_name,"year":year,"km_driven":km,"fuel":fuel,"seller_type":seller,"transmission":trans,"owner":owner}],columns=FEATURES)
            st.session_state["estimate"]=max(0,float(model.predict(row)[0]))
            st.session_state["details"]=row.iloc[0].to_dict()
            same=df[(df.model_name==model_name)&(df.fuel==fuel)&(df.transmission==trans)&(df.year.between(year-2,year+2))]
            scope="same model, fuel, transmission and nearby year"
            if same.empty:
                same=df[(df.model_name==model_name)&(df.fuel==fuel)]
                scope="same model and fuel"
            if same.empty:
                same=df[df.brand==brand]
                scope="same brand"
            st.session_state["comparables"]={"median":float(same.selling_price.median()) if not same.empty else None,"count":int(len(same)),"scope":scope}
    with right:
        st.subheader("Your valuation")
        st.caption("Estimated resale price based on comparable listings")
        if "estimate" in st.session_state:
            value=st.session_state["estimate"]
            st.markdown(f'<div style="background:linear-gradient(135deg,#e5fbf3,#eef5ff);border:1px solid #d5efe5;border-radius:22px;padding:30px;margin:16px 0 20px"><div style="color:#347c68;font-weight:700;font-size:.84rem;letter-spacing:1px">PREDICTED SELLING PRICE</div><div style="font:800 2.8rem Manrope;color:#142943;margin:9px 0">{money(value)}</div><div style="color:#63758a">Typical model error: around {money(mae)} on the held-out listings</div></div>',unsafe_allow_html=True)
            details=st.session_state["details"]
            st.markdown(f"**{details['year']} {details['model_name']}** · {details['fuel']} · {details['transmission']}  \n{details['km_driven']:,} km · {['First','Second','Third','4+'][int(details['owner'])]} owner")
            st.caption("This is an indicative estimate, not a dealer quote. Condition, city, service history and variant can change the actual price.")
            low=max(0,value-mae); high=value+mae
            st.markdown("#### Price guide")
            p1,p2=st.columns(2)
            p1.metric("Indicative range",f"{money(low)} – {money(high)}")
            comp=st.session_state.get("comparables",{})
            if comp.get("median") is not None:
                p2.metric("Similar listing median",money(comp["median"]))
                st.caption(f"Based on {comp['count']:,} dataset listings: {comp['scope']}. The range uses the model's test MAE as a rough guide; it is not a confidence interval.")
                chart_df=pd.DataFrame({"Price type":["Model estimate","Similar listing median"],"Price (₹)":[value,comp["median"]]})
                st.plotly_chart(px.bar(chart_df,x="Price (₹)",y="Price type",orientation="h",color="Price type",color_discrete_sequence=["#16856d","#f29d57"],title="Estimate compared with similar listings",template="plotly_white",text_auto=".2s"),width="stretch")
            else:
                p2.metric("Similar listings","Not available")
            report=pd.DataFrame([{**details,"estimated_price_inr":round(value),"guide_low_inr":round(low),"guide_high_inr":round(high),"similar_listing_median_inr":round(comp["median"]) if comp.get("median") is not None else "","similar_listing_count":comp.get("count",0)}])
            st.download_button("Download valuation summary (CSV)",report.to_csv(index=False),file_name="autoworth_valuation.csv",mime="text/csv",width="stretch")
        else:
            st.info("Your estimate will appear here. Adjust the details and select **Estimate my car’s value**.")
        st.markdown("#### Model snapshot")
        a,b,c=st.columns(3)
        a.metric("Listings",f"{n_rows:,}"); b.metric("Test R²",f"{r2:.2f}"); c.metric("Mean error",money(mae))
        st.caption("Metrics come from a held-out 20% test set. The best of Random Forest and Extra Trees is selected by MAE.")
elif page == "Market insights":
    st.subheader("The market, at a glance")
    st.caption(f"Explore {n_rows:,} public listings used to train the estimator.")
    a,b,c=st.columns(3); a.metric("Listings",f"{n_rows:,}");b.metric("Typical listed price",money(df.selling_price.median()));c.metric("Median distance",f"{df.km_driven.median():,.0f} km")
    l,r=st.columns(2)
    chart=df.copy(); chart["car_age"]=datetime.now().year-chart.year
    with l: st.plotly_chart(px.scatter(chart.sample(min(1200,len(chart)),random_state=42),x="km_driven",y="selling_price",color="fuel",hover_data=["brand","year"],title="Price and distance driven",labels={"km_driven":"Kilometres driven","selling_price":"Listed price (₹)"},template="plotly_white",opacity=.65),use_container_width=True)
    with r: st.plotly_chart(px.box(chart,x="fuel",y="selling_price",color="fuel",title="Price range by fuel type",labels={"fuel":"Fuel","selling_price":"Listed price (₹)"},template="plotly_white",points=False),use_container_width=True)
    st.plotly_chart(px.scatter(chart.groupby("car_age",as_index=False).selling_price.median(),x="car_age",y="selling_price",title="Median price by car age",labels={"car_age":"Car age (years)","selling_price":"Median listed price (₹)"},template="plotly_white"),use_container_width=True)
elif page == "About this project":
    st.subheader("A practical machine learning mini-project")
    st.markdown("""**Problem statement**  
Used-car buyers and sellers need a consistent starting point for pricing. This app learns from public CarDekho listings and estimates a listing price from the car's brand and model/variant, age, distance, fuel, seller, transmission, and ownership history.

**How it works**  
The app cleans the dataset, splits listings into training and test sets, compares Random Forest and Extra Trees using mean absolute error, selects the better model, then trains it on all available rows. Unseen categories are handled safely.

**Dataset**  
CarDekho public used-car listings: [CSV source]("""+DATA_URL+"""). The app loads it on startup. Upload a compatible CSV in the sidebar to use another copy.

**Limits**  
The source listings are historical and do not capture every location, trim, accident record, or vehicle condition. Treat the result as a guide, not a formal appraisal. The displayed test metrics are calculated from the loaded dataset each session.""")
    st.markdown(f"**Current run:** {n_rows:,} rows · MAE {money(mae)} · R² {r2:.3f}")
st.markdown("<hr><div style='text-align:center;color:#8390a2;font-size:.8rem'>AUTOWORTH · Machine learning for a smarter second-hand car decision</div>",unsafe_allow_html=True)
