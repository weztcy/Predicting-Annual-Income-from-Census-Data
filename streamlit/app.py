"""
Interactive Streamlit Application
Project: Predicting Annual Income from Census Data
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model, predict_income

st.set_page_config(
    page_title="Income Prediction - Census Data",
    page_icon="💼",
    layout="wide",
)

# Education mapping for user convenience
EDUCATION_MAPPING = {
    "Preschool (1)": 1.0,
    "1st-4th (2)": 2.0,
    "5th-6th (3)": 3.0,
    "7th-8th (4)": 4.0,
    "9th (5)": 5.0,
    "10th (6)": 6.0,
    "11th (7)": 7.0,
    "12th (8)": 8.0,
    "HS-grad (9)": 9.0,
    "Some-college (10)": 10.0,
    "Assoc-voc (11)": 11.0,
    "Assoc-acdm (12)": 12.0,
    "Bachelors (13)": 13.0,
    "Masters (14)": 14.0,
    "Prof-school (15)": 15.0,
    "Doctorate (16)": 16.0,
}

WORKCLASS_OPTIONS = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
    "Unknown/Missing",
]

MARITAL_STATUS_OPTIONS = [
    "Married-civ-spouse",
    "Never-married",
    "Divorced",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]

OCCUPATION_OPTIONS = [
    "Exec-managerial",
    "Prof-specialty",
    "Tech-support",
    "Sales",
    "Craft-repair",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Protective-serv",
    "Other-service",
    "Priv-house-serv",
    "Armed-Forces",
    "Unknown/Missing",
]

RELATIONSHIP_OPTIONS = [
    "Husband",
    "Wife",
    "Own-child",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]

RACE_OPTIONS = [
    "White",
    "Black",
    "Asian-Pac-Islander",
    "Amer-Indian-Eskimo",
    "Other",
]

NATIVE_COUNTRY_OPTIONS = [
    "United-States",
    "Mexico",
    "Philippines",
    "Germany",
    "Canada",
    "Puerto-Rico",
    "India",
    "Cuba",
    "England",
    "China",
    "South",
    "Jamaica",
    "Italy",
    "Dominican-Republic",
    "Japan",
    "Guatemala",
    "Vietnam",
    "Columbia",
    "Poland",
    "Haiti",
    "Portugal",
    "Taiwan",
    "Iran",
    "Greece",
    "Nicaragua",
    "Peru",
    "Ecuador",
    "Ireland",
    "France",
    "Thailand",
    "Hong",
    "Cambodia",
    "Trinadad&Tobago",
    "Yugoslavia",
    "Outlying-US(Guam-USVI-etc)",
    "Scotland",
    "Honduras",
    "Hungary",
    "Holand-Netherlands",
]


def render_header():
    st.title("💼 Predicting Annual Income from Census Data")
    st.markdown(
        "Demonstrasi machine learning end-to-end untuk klasifikasi pendapatan tahunan "
        "(**<=50K** vs **>50K**) menggunakan data historis sensus Amerika Serikat 1994 (UCI ML Repository)."
    )
    st.warning(
        "⚠️ **Peringatan Etika & Batasan Penggunaan:** Model ini dilatih pada data sensus historis tahun 1994 "
        "dan hanya ditujukan untuk keperluan edukasi dan demonstrasi akademis. Model ini **TIDAK BOLEH** digunakan "
        "untuk keputusan dunia nyata seperti seleksi kerja, persetujuan kredit, penetapan asuransi, atau bantuan sosial."
    )


def render_prediction_tab():
    st.subheader("Interactive Prediction Form")
    st.write("Masukkan karakteristik demografis dan pekerjaan untuk memprediksi probabilitas pendapatan >$50.000/tahun.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Usia (Age)", min_value=17, max_value=90, value=38)
        workclass = st.selectbox("Sektor Pekerjaan (Workclass)", WORKCLASS_OPTIONS, index=0)
        edu_label = st.selectbox("Tingkat Pendidikan (Education)", list(EDUCATION_MAPPING.keys()), index=12)
        edu_num = EDUCATION_MAPPING[edu_label]
        marital_status = st.selectbox("Status Pernikahan (Marital Status)", MARITAL_STATUS_OPTIONS, index=0)

    with col2:
        occupation = st.selectbox("Jenis Pekerjaan (Occupation)", OCCUPATION_OPTIONS, index=0)
        relationship = st.selectbox("Peran Keluarga (Relationship)", RELATIONSHIP_OPTIONS, index=0)
        race = st.selectbox("Ras (Race)", RACE_OPTIONS, index=0)
        sex = st.selectbox("Jenis Kelamin (Sex)", ["Male", "Female"], index=0)

    with col3:
        capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=99999, value=0, step=500)
        capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=5000, value=0, step=100)
        hours_per_week = st.slider("Jam Kerja per Minggu (Hours per Week)", min_value=1, max_value=99, value=40)
        native_country = st.selectbox("Negara Asal (Native Country)", NATIVE_COUNTRY_OPTIONS, index=0)

    if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
        input_data = {
            "age": float(age),
            "workclass": np.nan if "Unknown" in workclass else workclass,
            "education-num": float(edu_num),
            "marital-status": marital_status,
            "occupation": np.nan if "Unknown" in occupation else occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital-gain": float(capital_gain),
            "capital-loss": float(capital_loss),
            "hours-per-week": float(hours_per_week),
            "native-country": native_country,
        }

        try:
            preds, probas = predict_income(input_data)
            pred_class = preds[0]
            prob_high = probas[0]
            prob_low = 1.0 - prob_high

            st.divider()
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if pred_class == ">50K":
                    st.success(f"### Prediksi: **>50K**\n(Pendapatan > USD 50.000/tahun)")
                else:
                    st.info(f"### Prediksi: **<=50K**\n(Pendapatan <= USD 50.000/tahun)")

            with res_col2:
                st.write("**Probabilitas Prediksi:**")
                st.progress(prob_high)
                st.write(f"- Probabilitas `>50K` (Kelas Positif): **{prob_high*100:.2f}%**")
                st.write(f"- Probabilitas `<=50K`: **{prob_low*100:.2f}%**")

            st.caption(
                "Catatan: Prediksi dihitung menggunakan pipeline `HistGradientBoostingClassifier` terbaik "
                "dengan preprocessing terintegrasi (median imputer, mode imputer, standard scaler, one-hot encoder)."
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")


def render_performance_tab():
    st.subheader("Model Performance & Comparison")
    st.write("Ringkasan evaluasi model pada dataset pengujian (test set) sebesar 9.758 sampel:")

    comp_file = PROJECT_ROOT / "reports" / "model_comparison.csv"
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        st.dataframe(df_comp.style.format(precision=4), use_container_width=True)
    else:
        st.info("File model_comparison.csv belum tersedia.")

    st.write("### Visualisasi Evaluasi Model")
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        roc_img = PROJECT_ROOT / "reports" / "figures" / "11_roc_curves.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="Fig 11: Receiver Operating Characteristic (ROC) Curves")

        cm_img = PROJECT_ROOT / "reports" / "figures" / "10_best_model_confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Fig 10: Confusion Matrix Model Terbaik")

    with fig_col2:
        pr_img = PROJECT_ROOT / "reports" / "figures" / "12_precision_recall_curves.png"
        if pr_img.exists():
            st.image(str(pr_img), caption="Fig 12: Precision-Recall Curves")

        imp_img = PROJECT_ROOT / "reports" / "figures" / "13_feature_importance.png"
        if imp_img.exists():
            st.image(str(imp_img), caption="Fig 13: Permutation Feature Importance")


def render_fairness_tab():
    st.subheader("Fairness & Demographic Disparity Audit")
    st.write(
        "Pemeriksaan bias antarkelompok demografis (`sex` dan `race`) berdasarkan Selection Rate, Recall (TPR), "
        "False Positive Rate (FPR), Precision, dan F1 Score:"
    )

    fair_file = PROJECT_ROOT / "reports" / "fairness_report.csv"
    if fair_file.exists():
        df_fair = pd.read_csv(fair_file)
        st.dataframe(df_fair.style.format(precision=4), use_container_width=True)

    fair_img = PROJECT_ROOT / "reports" / "figures" / "14_fairness_by_sex_and_race.png"
    if fair_img.exists():
        st.image(str(fair_img), caption="Fig 14: Perbandingan Metrik Subgrup Sex dan Race", use_container_width=True)

    st.markdown(
        """
        ### Temuan Utama Analisis Fairness:
        1. **Disparitas Sex:** Model memiliki selection rate yang lebih rendah pada kelompok perempuan dibandingkan laki-laki, mencerminkan representasi historis tahun 1994 di mana hanya ~11% perempuan berpenghasilan >$50K dibandingkan ~31% laki-laki dalam data sensus.
        2. **Efek Proxy:** Menghapus fitur sensitif (`sex` dan `race`) tidak otomatis menghilangkan disparitas karena variabel lain seperti `marital-status`, `relationship` (misal 'Wife' vs 'Husband'), dan `occupation` bertindak sebagai variabel proxy.
        3. **Rekomendasi:** Model tidak boleh digunakan untuk otomasi seleksi manusia dalam konteks dunia nyata apa pun.
        """
    )


def main():
    render_header()
    tab1, tab2, tab3 = st.tabs(["🎯 Prediksi Interaktif", "📊 Performa Model", "⚖️ Audit Fairness"])

    with tab1:
        render_prediction_tab()
    with tab2:
        render_performance_tab()
    with tab3:
        render_fairness_tab()


if __name__ == "__main__":
    main()
