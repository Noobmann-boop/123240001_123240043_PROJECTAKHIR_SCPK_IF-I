import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Global_Cybersecurity_Threats_2015-2024.csv
data = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")

# RENTANG UNIVERSE
# INPUT
financialLoss            = ctrl.Antecedent(np.arange(0, 101, 1), 'financialLoss')
affectedUsers            = ctrl.Antecedent(np.arange(0, 1000001, 1000), 'affectedUsers')
incidentResolutionTime   = ctrl.Antecedent(np.arange(0, 101, 1), 'incidentResolutionTime')
# OUTPUT
riskLevel                = ctrl.Consequent(np.arange(0, 101, 1), 'riskLevel')
handlingPriority         = ctrl.Consequent(np.arange(0, 101, 1), 'handlingPriority')
threatStatus             = ctrl.Consequent(np.arange(0, 101, 1), 'threatStatus')


# FUNGSI KEANGGOTAAN INPUT
# financial loss
financialLoss['rendah'] = fuzz.trapmf(financialLoss.universe, [0, 0, 25, 45])
financialLoss['sedang']  = fuzz.trimf(financialLoss.universe, [30, 50, 70])
financialLoss['tinggi'] = fuzz.trapmf(financialLoss.universe, [60, 80, 100, 100])
 
# affected users
affectedUsers['sedikit'] = fuzz.trapmf(affectedUsers.universe, [0, 0, 250000, 450000])
affectedUsers['sedang'] = fuzz.trimf(affectedUsers.universe, [300000, 500000, 700000])
affectedUsers['banyak'] = fuzz.trapmf(affectedUsers.universe, [600000, 800000, 1000000, 1000000])
 
# incident resolution time
incidentResolutionTime['cepat'] = fuzz.trapmf(incidentResolutionTime.universe, [0, 0, 20, 40])
incidentResolutionTime['sedang'] = fuzz.trimf(incidentResolutionTime.universe, [30, 50, 70])
incidentResolutionTime['lama'] = fuzz.trapmf(incidentResolutionTime.universe, [60, 80, 100, 100])


# FUNGSI KEANGGOTAAN OUTPUT
# risk level
riskLevel['rendah'] = fuzz.trapmf(riskLevel.universe, [0, 0, 20, 40])
riskLevel['sedang']  = fuzz.trimf(riskLevel.universe, [30, 50, 70])
riskLevel['tinggi'] = fuzz.trapmf(riskLevel.universe, [60, 80, 100, 100])

# handling priority
handlingPriority['normal'] = fuzz.trapmf(handlingPriority.universe, [0, 0, 40, 60])
handlingPriority['penting']  = fuzz.trimf(handlingPriority.universe, [50, 70, 85])
handlingPriority['darurat'] = fuzz.trapmf(handlingPriority.universe, [80, 90, 100, 100])

# threat status
threatStatus['aman'] = fuzz.trapmf(threatStatus.universe, [0, 0, 25, 45])
threatStatus['waspada']  = fuzz.trimf(threatStatus.universe, [35, 55, 75])
threatStatus['bahaya'] = fuzz.trapmf(threatStatus.universe, [65, 85, 100, 100])


# RULE
rule1 = ctrl.Rule(financialLoss['tinggi'] & affectedUsers['banyak']  & incidentResolutionTime['lama'],
                  (riskLevel['tinggi'], handlingPriority['darurat'], threatStatus['bahaya']))
rule2 = ctrl.Rule(financialLoss['rendah'] & incidentResolutionTime['cepat'],
                  (riskLevel['rendah'], handlingPriority['normal'], threatStatus['aman']))
rule3 = ctrl.Rule(affectedUsers['banyak']  & incidentResolutionTime['lama'],
                  (riskLevel['tinggi'], handlingPriority['darurat'], threatStatus['bahaya']))
rule4 = ctrl.Rule(financialLoss['sedang'] & affectedUsers['sedang'],
                  (riskLevel['sedang'], handlingPriority['penting'], threatStatus['waspada']))
rule5 = ctrl.Rule(financialLoss['tinggi'] | incidentResolutionTime['lama'],
                  (riskLevel['tinggi'], handlingPriority['darurat'], threatStatus['bahaya']))
rule6 = ctrl.Rule(financialLoss['rendah'] & affectedUsers['sedikit'],
                  (riskLevel['rendah'], handlingPriority['normal'], threatStatus['aman']))
rule7 = ctrl.Rule(financialLoss['sedang'] & affectedUsers['banyak'] & incidentResolutionTime['sedang'],
                  (riskLevel['tinggi'], handlingPriority['darurat'], threatStatus['bahaya']))
rule8 = ctrl.Rule(financialLoss['rendah'] & affectedUsers['banyak'],
                  (riskLevel['sedang'], handlingPriority['penting'], threatStatus['waspada']))
rule9 = ctrl.Rule(incidentResolutionTime['sedang'],
                  (riskLevel['sedang'], handlingPriority['penting'], threatStatus['waspada']))

system = ctrl.ControlSystem([
    rule1, rule2, rule3,
    rule4, rule5, rule6,
    rule7, rule8, rule9
])
sim    = ctrl.ControlSystemSimulation(system)


# FUNGSI KONVERSI OUTPUT
def kategori_riskLevel(nilai):
    if nilai <= 50:
        return "rendah"
    elif nilai <= 70:
        return "sedang"
    else:
        return "tinggi"

def kategori_handling(nilai):
    if nilai <= 50:
        return "normal"
    elif nilai <= 70:
        return "penting"
    else:
        return "darurat"
    
def kategori_threat(nilai):
    if nilai <= 50:
        return "aman"
    elif nilai <= 70:
        return "waspada"
    else:
        return "bahaya"


# STREAMLIT
st.set_page_config (
    page_title="SISTEM DETEKSI SIBER",
    layout= "centered"
)

st.title("🚨 Sistem Penentuan Risiko Serangan Siber")
st.write("Metode Fuzzy Mamdani Menggunakan Dataset Cybersecurity")


# TAB STREAMLIT, READ CSV, INPUT, GRAFIK FUNGSI KEANGGOTAAN, OUTPUT, CRISP VALUE, DEFUZZIFIKASI
tab1, tab2, tab3, tab4 = st.tabs([
    "Input Dataset",
    "Output",
    "Visualisasi Fuzzy",
    "Input Manual"
])

with tab1:
    # Menampilkan data dari csv
    st.subheader("🗂️ Dataset Cybersecurity")
    st.dataframe(data)


    # PILIH DATA CSV
    st.subheader("Pilih Data dari Dataset")

    row = st.number_input(
        f"Pilih Index Data (0 - {len(data)-1})",
        min_value=0,
        max_value=len(data)-1,
        value=0,
        step=1
    )

    sample = data.iloc[row]

    # DASHBOARD RINGKASAN DATASET
    st.divider()
    st.subheader("Ringkasan Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(data)}"
        )

    with col2:
        st.metric(
            "Rata-rata Financial Loss",
            f"${data['Financial Loss (in Million $)'].mean():.2f}M"
        )

    with col3:
        st.metric(
            "Rata-rata Affected Users",
            f"{data['Number of Affected Users'].mean():,.0f}"
        )

    with col4:
        st.metric(
            "Rata-rata Resolution Time",
            f"{data['Incident Resolution Time (in Hours)'].mean():.1f} jam"
        )


with tab2:
    # TAMPILKAN DATA TERPILIH
    st.subheader("🗂️ Data Terpilih")

    st.dataframe(sample.astype(str).to_frame())

    sim.input['financialLoss'] = sample['Financial Loss (in Million $)']
    sim.input['affectedUsers'] = sample['Number of Affected Users']
    sim.input['incidentResolutionTime'] = sample['Incident Resolution Time (in Hours)']

    st.subheader("Hasil Analisis Fuzzy (Crisp)")

    try:
        sim.compute()

        risk = sim.output['riskLevel']
        priority = sim.output['handlingPriority']
        status = sim.output['threatStatus']

        st.success(
            f"⚠️ Risk Level : {risk:.2f} "
            f"({kategori_riskLevel(risk)})"
        )

        st.warning(
            f"🚨 Handling Priority : {priority:.2f} "
            f"({kategori_handling(priority)})"
        )

        st.error(
            f"🛡 Threat Status : {status:.2f} "
            f"({kategori_threat(status)})"
        )


        # GRAFIK DEFUZZIFIKASI
        st.header("Grafik Defuzzifikasi")

        st.subheader("Risk Level")
        riskLevel.view(sim=sim)
        st.pyplot(plt.gcf())

        st.subheader("Handling Priority")
        handlingPriority.view(sim=sim)
        st.pyplot(plt.gcf())

        st.subheader("Threat Status")
        threatStatus.view(sim=sim)
        st.pyplot(plt.gcf())  

    except Exception as e:
        st.error(f"Terjadi error: {e}")

    
with tab3:

    st.header("📊 Visualisasi Fuzzy")

    pilihan = st.radio(
        "Pilih Visualisasi",
        [
            "Input Variables",
            "Output Variables",
            "Rule Base"
        ],
        horizontal=True
    )

    if pilihan == "Input Variables":

        st.subheader("📥 Input Variables")

        st.markdown("### 💰 Financial Loss")
        financialLoss.view()
        st.pyplot(plt.gcf())
        st.divider()

        st.markdown("### 👥 Affected Users")
        affectedUsers.view()
        st.pyplot(plt.gcf())
        st.divider()

        st.markdown("### ⏱ Incident Resolution Time")
        incidentResolutionTime.view()
        st.pyplot(plt.gcf())

    elif pilihan == "Output Variables":

        st.subheader("📤 Output Variables")

        st.markdown("### ⚠️ Risk Level")
        riskLevel.view()
        st.pyplot(plt.gcf())
        st.divider()

        st.markdown("### 🚨 Handling Priority")
        handlingPriority.view()
        st.pyplot(plt.gcf())
        st.divider()

        st.markdown("### 🛡 Threat Status")
        threatStatus.view()
        st.pyplot(plt.gcf())

    
    else:

        st.subheader("📜 Rule Base Fuzzy")

        st.text("""
        Rule 1
        IF Financial Loss = Tinggi AND Affected Users = Banyak AND Incident Resolution Time = Lama
        THEN Risk Level = Tinggi, Handling Priority = Darurat, Threat Status = Bahaya

        Rule 2
        IF Financial Loss = Rendah AND Incident Resolution Time = Cepat
        THEN Risk Level = Rendah, Handling Priority = Normal, Threat Status = Aman

        Rule 3
        IF Affected Users = Banyak AND Incident Resolution Time = Lama
        THEN Risk Level = Tinggi, Handling Priority = Darurat, Threat Status = Bahaya

        Rule 4
        IF Financial Loss = Sedang AND Affected Users = Sedang
        THEN Risk Level = Sedang, Handling Priority = Penting, Threat Status = Waspada

        Rule 5
        IF Financial Loss = Tinggi OR Incident Resolution Time = Lama
        THEN Risk Level = Tinggi, Handling Priority = Darurat, Threat Status = Bahaya

        Rule 6
        IF Financial Loss = Rendah AND Affected Users = Sedikit
        THEN Risk Level = Rendah, Handling Priority = Normal, Threat Status = Aman

        Rule 7
        IF Financial Loss = Sedang AND Affected Users = Banyak AND Incident Resolution Time = Sedang
        THEN Risk Level = Tinggi, Handling Priority = Darurat, Threat Status = Bahaya

        Rule 8
        IF Financial Loss = Rendah AND Affected Users = Banyak
        THEN Risk Level = Sedang, Handling Priority = Penting, Threat Status = Waspada

        Rule 9
        IF Incident Resolution Time = Sedang
        THEN Risk Level = Sedang, Handling Priority = Penting, Threat Status = Waspada
        """)

with tab4:

    st.header("✏️ Simulasi Input Manual")

    financial_manual = st.number_input(
        "Financial Loss (Million $)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    users_manual = st.number_input(
        "Affected Users",
        min_value=0,
        max_value=1000000,
        value=500000,
        step=1000
    )

    resolution_manual = st.number_input(
        "Incident Resolution Time (Hours)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    if st.button("🚀 Analisis Risiko"):

        sim_manual = ctrl.ControlSystemSimulation(system)

        sim_manual.input['financialLoss'] = financial_manual
        sim_manual.input['affectedUsers'] = users_manual
        sim_manual.input['incidentResolutionTime'] = resolution_manual

        try:

            sim_manual.compute()

            risk = sim_manual.output['riskLevel']
            priority = sim_manual.output['handlingPriority']
            status = sim_manual.output['threatStatus']

            st.divider()
            st.subheader("Hasil Analisis Fuzzy (Crisp)")

            st.success(
                f"⚠️ Risk Level : {risk:.2f} "
                f"({kategori_riskLevel(risk)})"
            )

            st.warning(
                f"📦 Handling Priority : {priority:.2f} "
                f"({kategori_handling(priority)})"
            )

            st.error(
                f"💥 Threat Status : {status:.2f} "
                f"({kategori_threat(status)})"
            )

            st.divider()
            st.subheader("📊 Visualisasi Input Manual")


            col1, col2, col3 = st.columns(3)
            with col1:
                plt.figure(figsize=(5,3))
                financialLoss.view(sim=sim_manual)
                plt.legend()
                st.pyplot(plt.gcf())

            with col2:
                plt.figure(figsize=(5,3))
                affectedUsers.view(sim=sim_manual)
                plt.legend()
                st.pyplot(plt.gcf())

            with col3:
                plt.figure(figsize=(5,3))
                incidentResolutionTime.view(sim=sim_manual)
                plt.legend()
                st.pyplot(plt.gcf())

        except Exception as e:
            st.error(f"Terjadi error: {e}")