import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import re
import plotly.figure_factory as ff
import scipy # Ajouté pour résoudre ImportError avec create_distplot

# --- Chemins vers vos fichiers de données ---
# ATTENTION : Ces chemins ont été mis à jour pour être RELATIFS.
# Cela signifie que les fichiers Excel/CSV doivent se trouver dans le MÊME dossier
# que ce script Python lorsque vous le déployez (par exemple, sur GitHub pour Streamlit Community Cloud).
DATA_PATHS = {
    "memory": "memory_final_cleaned_clean.xlsx",
    "hitlist_db": "HITLIST_DATABASE_final_cleaned_clean.xlsx",
    "times": "Times_final_cleaned_clean.xlsx",
    "tasktimes": "TASKTIMES_final_cleaned_clean.xlsx",
    "usertcode": "USERTCODE_cleaned.xlsx",
    "performance": "AL_GET_PERFORMANCE_final_cleaned_clean.xlsx",
    "sql_trace_summary": "performance_trace_summary_final_cleaned_clean.xlsx",
    "usr02": "usr02_data.xlsx",
}

# --- Configuration de la page Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard SAP Complet Multi-Sources")

# --- Fonctions de Nettoyage et Chargement des Données (avec cache) ---

def clean_string_column(series, default_value="Non défini"):
    """
    Nettoyage d'une série de type string : supprime espaces, remplace NaN/vides/caractères non imprimables.
    """
    cleaned_series = series.astype(str).str.strip()
    cleaned_series = cleaned_series.apply(lambda x: re.sub(r'[^\x20-\x7E\s]+', ' ', x).strip())
    cleaned_series = cleaned_series.replace({'nan': default_value, '': default_value, ' ': default_value})
    return cleaned_series

def clean_column_names(df):
    """
    Nettoyage des noms de colonnes : supprime les espaces, les caractères invisibles,
    et s'assure qu'ils sont valides pour l'accès.
    """
    new_columns = []
    for col in df.columns:
        cleaned_col = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', str(col)).strip()
        cleaned_col = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned_col)
        cleaned_col = re.sub(r'_+', '_', cleaned_col)
        cleaned_col = cleaned_col.strip('_')
        new_columns.append(cleaned_col)
    df.columns = new_columns
    return df

def convert_mm_ss_to_seconds(time_str):
    """
    Convertit une chaîne de caractères au format MM:SS en secondes.
    Gère les cas où les minutes ou secondes sont manquantes ou invalides.
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return 0
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return int(minutes * 60 + seconds)
        elif len(parts) == 1:
            return int(float(parts[0]))
        else:
            return 0
    except ValueError:
        return 0

def clean_numeric_with_comma(series):
    """
    Nettoyage d'une série de chaînes numériques qui peuvent contenir des virgules
    comme séparateurs de milliers ou décimaux, et conversion en float.
    """
    cleaned_series = series.astype(str).str.replace(' ', '').str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(cleaned_series, errors='coerce').fillna(0)


@st.cache_data
def load_and_process_data(file_key, path):
    """Charge et nettoie un fichier Excel/CSV."""
    df = pd.DataFrame()
    try:
        if path.lower().endswith('.xlsx'):
            df = pd.read_excel(path)
        elif path.lower().endswith('.csv'):
            df = pd.read_csv(path)
        else:
            st.error(f"Format de fichier non supporté pour {file_key}: {path}")
            return pd.DataFrame()

        df = clean_column_names(df.copy())

        # --- Gestion spécifique des types de données et valeurs manquantes ---
        if file_key == "memory":
            numeric_cols = ['MEMSUM', 'PRIVSUM', 'USEDBYTES', 'MAXBYTES', 'MAXBYTESDI', 'PRIVCOUNT', 'RESTCOUNT', 'COUNTER']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            if 'ACCOUNT' in df.columns:
                df['ACCOUNT'] = clean_string_column(df['ACCOUNT'], 'Compte Inconnu')
            if 'MANDT' in df.columns:
                df['MANDT'] = clean_string_column(df['MANDT'], 'MANDT Inconnu')
            if 'TASKTYPE' in df.columns:
                df['TASKTYPE'] = clean_string_column(df['TASKTYPE'], 'Type de Tâche Inconnu')

            if 'ENDDATE' in df.columns and 'ENDTIME' in df.columns:
                df['ENDTIME_STR'] = df['ENDTIME'].astype(str).str.zfill(6)
                df['FULL_DATETIME'] = pd.to_datetime(df['ENDDATE'].astype(str) + df['ENDTIME_STR'], format='%Y%m%d%H%M%S', errors='coerce')
                df.drop(columns=['ENDTIME_STR'], inplace=True, errors='ignore')
            elif 'FULL_DATETIME' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['FULL_DATETIME']):
                df['FULL_DATETIME'] = pd.to_datetime(df['FULL_DATETIME'], errors='coerce')
            
            subset_cols_memory = []
            if 'USEDBYTES' in df.columns:
                subset_cols_memory.append('USEDBYTES')
            if 'ACCOUNT' in df.columns:
                subset_cols_memory.append('ACCOUNT')
            if subset_cols_memory:
                df.dropna(subset=subset_cols_memory, inplace=True)


        elif file_key == "hitlist_db":
            numeric_cols = [
                'GENERATETI', 'REPLOADTI', 'CUALOADTI', 'DYNPLOADTI', 'QUETI', 'DDICTI', 'CPICTI',
                'LOCKCNT', 'LOCKTI', 'BTCSTEPNR', 'RESPTI', 'PROCTI', 'CPUTI', 'QUEUETI', 'ROLLWAITTI',
                'GUITIME', 'GUICNT', 'GUINETTIME', 'DBP_COUNT', 'DBP_TIME', 'DSQLCNT', 'QUECNT',
                'CPICCNT', 'SLI_CNT', 'TAB1DIRCNT', 'TAB1SEQCNT', 'TAB1UPDCNT', 'TAB2DIRCNT',
                'TAB2SEQCNT', 'TAB2UPDCNT', 'TAB3DIRCNT', 'TAB3SEQCNT', 'TAB3UPDCNT', 'TAB4DIRCNT',
                'TAB4SEQCNT', 'TAB4UPDCNT', 'TAB5DIRCNT', 'TAB5SEQCNT', 'TAB5UPDCNT',
                'READDIRCNT', 'READDIRTI', 'READDIRBUF', 'READDIRREC', 'READSEQCNT', 'READSEQTI',
                'READSEQBUF', 'READSEQREC', 'PHYREADCNT', 'INSCNT', 'INSTI', 'INSREC', 'PHYINSCNT',
                'UPDCNT', 'UPDTI', 'UPDREC', 'PHYUPDCNT', 'DELCNT', 'DELTI', 'DELREC', 'PHYDELCNT',
                'DBCALLS', 'COMMITTI', 'INPUTLEN', 'OUTPUTLEN', 'MAXROLL', 'MAXPAGE',
                'ROLLINCNT', 'ROLLINTI', 'ROLLOUTCNT', 'ROLLOUTTI', 'ROLLED_OUT', 'PRIVSUM',
                'USEDBYTES', 'MAXBYTES', 'MAXBYTESDI', 'RFCRECEIVE', 'RFCSEND',
                'RFCEXETIME', 'RFCCALLTIM', 'RFCCALLS', 'VMC_CALL_COUNT', 'VMC_CPU_TIME', 'VMC_ELAP_TIME'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            if 'ENDDATE' in df.columns and 'ENDTIME' in df.columns:
                df['ENDTIME_STR'] = df['ENDTIME'].astype(str).str.zfill(6)
                df['FULL_DATETIME'] = pd.to_datetime(df['ENDDATE'].astype(str) + df['ENDTIME_STR'], format='%Y%m%d%H%M%S', errors='coerce')
                df.drop(columns=['ENDTIME_STR'], inplace=True, errors='ignore')
            elif 'FULL_DATETIME' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['FULL_DATETIME']):
                df['FULL_DATETIME'] = pd.to_datetime(df['FULL_DATETIME'], errors='coerce')

            subset_cols_hitlist = []
            if 'RESPTI' in df.columns: subset_cols_hitlist.append('RESPTI')
            if 'PROCTI' in df.columns: subset_cols_hitlist.append('PROCTI')
            if 'CPUTI' in df.columns: subset_cols_hitlist.append('CPUTI')
            if 'DBCALLS' in df.columns: subset_cols_hitlist.append('DBCALLS')
            if subset_cols_hitlist:
                df.dropna(subset=subset_cols_hitlist, inplace=True)
            if 'FULL_DATETIME' in df.columns:
                df.dropna(subset=['FULL_DATETIME'], inplace=True)

            for col in ['WPID', 'ACCOUNT', 'REPORT', 'ROLLKEY', 'PRIVMODE', 'WPRESTART', 'TASKTYPE']:
                if col in df.columns:
                    df[col] = clean_string_column(df[col])


        elif file_key == "times":
            numeric_cols = [
                'COUNT', 'LUW_COUNT', 'RESPTI', 'PROCTI', 'CPUTI', 'QUEUETI', 'ROLLWAITTI',
                'GUITIME', 'GUICNT', 'GUINETTIME', 'DBP_COUNT', 'DBP_TIME', 'READDIRCNT',
                'READDIRTI', 'READDIRBUF', 'READDIRREC', 'READSEQCNT', 'READSEQTI',
                'READSEQBUF', 'READSEQREC', 'CHNGCNT', 'CHNGTI', 'CHNGREC', 'PHYREADCNT',
                'PHYCHNGREC', 'PHYCALLS', 'VMC_CALL_COUNT', 'VMC_CPU_TIME', 'VMC_ELAP_TIME'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            subset_cols_times = []
            if 'RESPTI' in df.columns: subset_cols_times.append('RESPTI')
            if 'PHYCALLS' in df.columns: subset_cols_times.append('PHYCALLS')
            if 'COUNT' in df.columns: subset_cols_times.append('COUNT')
            if subset_cols_times:
                df.dropna(subset=subset_cols_times, inplace=True)
            
            if 'TIME' in df.columns:
                df['TIME'] = clean_string_column(df['TIME'])
            if 'TASKTYPE' in df.columns:
                df['TASKTYPE'] = clean_string_column(df['TASKTYPE'])
            if 'ENTRY_ID' in df.columns:
                df['ENTRY_ID'] = clean_string_column(df[col])

        elif file_key == "tasktimes":
            numeric_cols = [
                'COUNT', 'RESPTI', 'PROCTI', 'CPUTI', 'QUEUETI', 'ROLLWAITTI', 'GUITIME',
                'GUICNT', 'GUINETTIME', 'DBP_COUNT', 'DBP_TIME', 'READDIRCNT', 'READDIRTI',
                'READDIRBUF', 'READDIRREC', 'READSEQCNT', 'READSEQTI',
                'READSEQBUF', 'READSEQREC', 'CHNGCNT', 'CHNGTI', 'CHNGREC', 'PHYREADCNT',
                'PHYCHNGREC', 'PHYCALLS', 'CNT001', 'CNT002', 'CNT003', 'CNT004', 'CNT005', 'CNT006', 'CNT007', 'CNT008', 'CNT009'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            subset_cols_tasktimes = []
            if 'COUNT' in df.columns: subset_cols_tasktimes.append('COUNT')
            if 'RESPTI' in df.columns: subset_cols_tasktimes.append('RESPTI')
            if 'CPUTI' in df.columns: subset_cols_tasktimes.append('CPUTI')
            if subset_cols_tasktimes:
                df.dropna(subset=subset_cols_tasktimes, inplace=True)
            
            if 'TASKTYPE' in df.columns:
                df['TASKTYPE'] = clean_string_column(df['TASKTYPE'], 'Type de tâche non spécifié')
            if 'TIME' in df.columns:
                df['TIME'] = clean_string_column(df['TIME'])


        elif file_key == "usertcode":
            numeric_cols = [
                'COUNT', 'DCOUNT', 'UCOUNT', 'BCOUNT', 'ECOUNT', 'SCOUNT', 'LUW_COUNT',
                'TMBYTESIN', 'TMBYTESOUT', 'RESPTI', 'PROCTI', 'CPUTI', 'QUEUETI',
                'ROLLWAITTI', 'GUITIME', 'GUICNT', 'GUINETTIME', 'DBP_COUNT', 'DBP_TIME',
                'READDIRCNT', 'READDIRTI', 'READDIRBUF', 'READDIRREC', 'READSEQCNT',
                'READSEQTI', 'READSEQBUF', 'READSEQREC', 'CHNGCNT', 'CHNGTI', 'CHNGREC',
                'PHYREADCNT', 'PHYCHNGREC', 'PHYCALLS', 'DSQLCNT', 'QUECNT', 'CPICCNT',
                'SLI_CNT', 'VMC_CALL_COUNT', 'VMC_CPU_TIME', 'VMC_ELAP_TIME'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            # Add FULL_DATETIME creation for usertcode
            if 'ENDDATE' in df.columns and 'ENDTIME' in df.columns:
                df['ENDTIME_STR'] = df['ENDTIME'].astype(str).str.zfill(6)
                df['FULL_DATETIME'] = pd.to_datetime(df['ENDDATE'].astype(str) + df['ENDTIME_STR'], format='%Y%m%d%H%M%S', errors='coerce')
                df.drop(columns=['ENDTIME_STR'], inplace=True, errors='ignore')
            elif 'FULL_DATETIME' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['FULL_DATETIME']):
                df['FULL_DATETIME'] = pd.to_datetime(df['FULL_DATETIME'], errors='coerce')

            critical_usertcode_cols = []
            if 'RESPTI' in df.columns: critical_usertcode_cols.append('RESPTI')
            if 'ACCOUNT' in df.columns: critical_usertcode_cols.append('ACCOUNT')
            if 'COUNT' in df.columns: critical_usertcode_cols.append('COUNT')
            
            if critical_usertcode_cols:
                df.dropna(subset=critical_usertcode_cols, inplace=True)
            
            for col in ['TASKTYPE', 'ENTRY_ID', 'ACCOUNT']:
                if col in df.columns:
                    df[col] = clean_string_column(df[col])

        elif file_key == "performance": # Nouveau bloc pour AL_GET_PERFORMANCE
            # Convertir WP_CPU de MM:SS en secondes
            if 'WP_CPU' in df.columns:
                df['WP_CPU_SECONDS'] = df['WP_CPU'].apply(convert_mm_ss_to_seconds).astype(float)
            
            # Convertir WP_IWAIT en secondes (s'il est en ms, diviser par 1000)
            if 'WP_IWAIT' in df.columns:
                df['WP_IWAIT'] = pd.to_numeric(df['WP_IWAIT'], errors='coerce').fillna(0)
                # Keep WP_IWAIT as is, we will use it for plotting.
                # df['WP_IWAIT_SECONDS'] = df['WP_IWAIT'] / 1000.0 # This conversion might not be universally needed
            else:
                df['WP_IWAIT'] = 0 # Ensure column exists even if empty

            # Nettoyage des colonnes string
            for col in ['WP_SEMSTAT', 'WP_IACTION', 'WP_ITYPE', 'WP_RESTART', 'WP_ISTATUS', 'WP_TYP', 'WP_STATUS']:
                if col in df.columns:
                    df[col] = clean_string_column(df[col])
            
            # Nettoyage des colonnes numériques
            numeric_cols_perf = ['WP_NO', 'WP_IRESTRT', 'WP_PID', 'WP_INDEX']
            for col in numeric_cols_perf:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            # Supprimer les lignes avec des valeurs critiques manquantes si nécessaire
            subset_cols_perf = []
            if 'WP_CPU_SECONDS' in df.columns: subset_cols_perf.append('WP_CPU_SECONDS')
            if 'WP_STATUS' in df.columns: subset_cols_perf.append('WP_STATUS')
            if subset_cols_perf:
                df.dropna(subset=subset_cols_perf, inplace=True)
        
        elif file_key == "sql_trace_summary": # Nouveau bloc pour performance_trace_summary
            # Nettoyage des colonnes numériques avec virgule/espace
            numeric_cols_sql = ['TOTALEXEC', 'IDENTSEL', 'EXECTIME', 'RECPROCNUM', 'TIMEPEREXE', 'RECPEREXE', 'AVGTPERREC', 'MINTPERREC']
            for col in numeric_cols_sql:
                if col in df.columns:
                    df[col] = clean_numeric_with_comma(df[col]).astype(float)
            
            # Nettoyage des colonnes string
            for col in ['SQLSTATEM', 'SERVERNAME', 'TRANS_ID']:
                if col in df.columns:
                    df[col] = clean_string_column(df[col])
            
            # Supprimer les lignes avec des valeurs critiques manquantes si nécessaire
            subset_cols_sql = []
            if 'EXECTIME' in df.columns: subset_cols_sql.append('EXECTIME')
            if 'TOTALEXEC' in df.columns: subset_cols_sql.append('TOTALEXEC')
            if 'SQLSTATEM' in df.columns: subset_cols_sql.append('SQLSTATEM')
            if subset_cols_sql:
                df.dropna(subset=subset_cols_sql, inplace=True)

        elif file_key == "usr02": # Nouveau bloc pour usr02_data.xlsx
            # Nettoyage des colonnes string
            for col in ['BNAME', 'USTYP']:
                if col in df.columns:
                    df[col] = clean_string_column(df[col])
            
            # Conversion de GLTGB en datetime
            if 'GLTGB' in df.columns:
                df['GLTGB'] = df['GLTGB'].astype(str).replace('00000000', np.nan)
                df['GLTGB_DATE'] = pd.to_datetime(df['GLTGB'], format='%Y%m%d', errors='coerce')
            else:
                df['GLTGB_DATE'] = pd.NaT

        return df

    except FileNotFoundError:
        st.error(f"Erreur: Le fichier '{path}' pour '{file_key}' est introuvable. Veuillez vérifier le chemin.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement du fichier '{file_key}' : {e}. Détails : {e}")
        return pd.DataFrame()

# --- Chargement de TOUTES les données ---
dfs = {}
for key, path in DATA_PATHS.items():
    dfs[key] = load_and_process_data(key, path)

# --- Contenu principal du Dashboard ---
st.title("📊 Tableau de Bord SAP Complet Multi-Sources")
st.markdown("Explorez les performances, l'utilisation mémoire, les transactions utilisateurs et la santé du système à travers différentes sources de données.")

# --- Affichage des KPIs ---
st.markdown("---")
kpi_cols = st.columns(5)

# KPI 1: Temps de Réponse Moyen Global (Hitlist DB)
avg_resp_time = 0
if not dfs['hitlist_db'].empty and 'RESPTI' in dfs['hitlist_db'].columns:
    # Ensure RESPTI is numeric before mean calculation
    if pd.api.types.is_numeric_dtype(dfs['hitlist_db']['RESPTI']):
        avg_resp_time = dfs['hitlist_db']['RESPTI'].mean() / 1000
kpi_cols[0].metric("Temps de Réponse Moyen (s)", f"{avg_resp_time:.2f}")

# KPI 2: Utilisation Mémoire Moyenne (USEDBYTES)
avg_memory_usage = 0
if not dfs['memory'].empty and 'USEDBYTES' in dfs['memory'].columns:
    # Ensure USEDBYTES is numeric before mean calculation
    if pd.api.types.is_numeric_dtype(dfs['memory']['USEDBYTES']):
        avg_memory_usage = dfs['memory']['USEDBYTES'].mean() / (1024 * 1024)
kpi_cols[1].metric("Mémoire Moyenne (Mo)", f"{avg_memory_usage:.2f}")

# KPI 3: Total des Appels Base de Données (Hitlist DB)
total_db_calls = 0
if not dfs['hitlist_db'].empty and 'DBCALLS' in dfs['hitlist_db'].columns:
    # Ensure DBCALLS is numeric before sum calculation
    if pd.api.types.is_numeric_dtype(dfs['hitlist_db']['DBCALLS']):
        total_db_calls = dfs['hitlist_db']['DBCALLS'].sum()
kpi_cols[2].metric("Total Appels DB", f"{int(total_db_calls):,}".replace(",", " "))

# KPI 4: Total des Exécutions SQL (performance_trace_summary) - NOUVEAU KPI
total_sql_executions = 0
if not dfs['sql_trace_summary'].empty and 'TOTALEXEC' in dfs['sql_trace_summary'].columns:
    # Ensure TOTALEXEC is numeric before sum calculation
    if pd.api.types.is_numeric_dtype(dfs['sql_trace_summary']['TOTALEXEC']):
        total_sql_executions = dfs['sql_trace_summary']['TOTALEXEC'].sum()
kpi_cols[3].metric("Total Exécutions SQL", f"{int(total_sql_executions):,}".replace(",", " "))

# KPI 5: Temps CPU Moyen Global (Hitlist DB)
avg_cpu_time = 0
if not dfs['hitlist_db'].empty and 'CPUTI' in dfs['hitlist_db'].columns:
    # Ensure CPUTI is numeric before mean calculation
    if pd.api.types.is_numeric_dtype(dfs['hitlist_db']['CPUTI']):
        avg_cpu_time = dfs['hitlist_db']['CPUTI'].mean() / 1000
kpi_cols[4].metric("Temps CPU Moyen (s)", f"{avg_cpu_time:.2f}")

st.markdown("---")

# --- Barre de navigation flexible ---
tab_titles = [
    "Analyse Mémoire",
    "Transactions Utilisateurs",
    "Statistiques Horaires",
    "Décomposition des Tâches",
    "Insights Hitlist DB",
    "Performance des Processus de Travail",
    "Résumé des Traces de Performance SQL",
    "Analyse des Utilisateurs",
    "Détection d'Anomalies"
]

if 'current_section' not in st.session_state:
    st.session_state.current_section = tab_titles[0]

st.sidebar.header("Navigation Rapide")
selected_section = st.sidebar.radio(
    "Accéder à la section :",
    tab_titles,
    index=tab_titles.index(st.session_state.current_section)
)

st.session_state.current_section = selected_section

if all(df.empty for df in dfs.values()):
    st.error("Aucune source de données n'a pu être chargée. Le dashboard ne peut pas s'afficher. Veuillez vérifier les chemins et les fichiers.")
else:
    # --- Sidebar pour les filtres globaux ---
    st.sidebar.header("Filtres")

    all_accounts = pd.Index([])
    if not dfs['memory'].empty and 'ACCOUNT' in dfs['memory'].columns:
        all_accounts = all_accounts.union(dfs['memory']['ACCOUNT'].dropna().unique())
    if not dfs['usertcode'].empty and 'ACCOUNT' in dfs['usertcode'].columns:
        all_accounts = all_accounts.union(dfs['usertcode']['ACCOUNT'].dropna().unique())
    if not dfs['hitlist_db'].empty and 'ACCOUNT' in dfs['hitlist_db'].columns:
        all_accounts = all_accounts.union(dfs['hitlist_db']['ACCOUNT'].dropna().unique())
    
    selected_accounts = []
    if not all_accounts.empty:
        selected_accounts = st.sidebar.multiselect(
            "Sélectionner des Comptes",
            options=sorted(all_accounts.tolist()),
            default=[]
        )
        if selected_accounts:
            for key in ['memory', 'usertcode', 'hitlist_db']:
                if not dfs[key].empty and 'ACCOUNT' in dfs[key].columns:
                    dfs[key] = dfs[key][dfs[key]['ACCOUNT'].isin(selected_accounts)]

    selected_reports = []
    if not dfs['hitlist_db'].empty and 'REPORT' in dfs['hitlist_db'].columns:
        all_reports = dfs['hitlist_db']['REPORT'].dropna().unique().tolist()
        selected_reports = st.sidebar.multiselect(
            "Sélectionner des Rapports (Hitlist DB)",
            options=sorted(all_reports),
            default=[]
        )
        if selected_reports:
            dfs['hitlist_db'] = dfs['hitlist_db'][dfs['hitlist_db']['REPORT'].isin(selected_reports)]
    
    all_tasktypes = pd.Index([])
    if not dfs['usertcode'].empty and 'TASKTYPE' in dfs['usertcode'].columns:
        all_tasktypes = all_tasktypes.union(dfs['usertcode']['TASKTYPE'].dropna().unique())
    if not dfs['times'].empty and 'TASKTYPE' in dfs['times'].columns:
        all_tasktypes = all_tasktypes.union(dfs['times']['TASKTYPE'].dropna().unique())
    if not dfs['tasktimes'].empty and 'TASKTYPE' in dfs['tasktimes'].columns:
        all_tasktypes = all_tasktypes.union(dfs['tasktimes']['TASKTYPE'].dropna().unique())
    if not dfs['hitlist_db'].empty and 'TASKTYPE' in dfs['hitlist_db'].columns:
        all_tasktypes = all_tasktypes.union(dfs['hitlist_db']['TASKTYPE'].dropna().unique())
    if not dfs['memory'].empty and 'TASKTYPE' in dfs['memory'].columns: # Added memory to tasktype filter
        all_tasktypes = all_tasktypes.union(dfs['memory']['TASKTYPE'].dropna().unique())


    selected_tasktypes = []
    if not all_tasktypes.empty:
        selected_tasktypes = st.sidebar.multiselect(
            "Sélectionner des Types de Tâches",
            options=sorted(all_tasktypes.tolist()),
            default=[]
        )
        if selected_tasktypes:
            for key in ['usertcode', 'times', 'tasktimes', 'hitlist_db', 'memory']: # Added 'memory' to task type filter
                if not dfs[key].empty and 'TASKTYPE' in dfs[key].columns:
                    dfs[key] = dfs[key][dfs[key]['TASKTYPE'].isin(selected_tasktypes)]
    
    selected_wp_types = []
    if not dfs['performance'].empty and 'WP_TYP' in dfs['performance'].columns:
        all_wp_types = dfs['performance']['WP_TYP'].dropna().unique().tolist()
        selected_wp_types = st.sidebar.multiselect(
            "Sélectionner des Types de Processus de Travail (Performance)",
            options=sorted(all_wp_types),
            default=[]
        )
        if selected_wp_types:
            dfs['performance'] = dfs['performance'][dfs['performance']['WP_TYP'].isin(selected_wp_types)]

    df_hitlist_filtered = dfs['hitlist_db'].copy()


    # --- Contenu des sections basé sur la sélection de la barre latérale ---
    if st.session_state.current_section == "Analyse Mémoire":
        # --- Onglet 1: Analyse Mémoire (memory_final_cleaned_clean.xlsx) ---
        st.header("🧠 Analyse de l'Utilisation Mémoire")
        df_mem = dfs['memory'].copy()
        if selected_accounts:
            df_mem = df_mem[df_mem['ACCOUNT'].isin(selected_accounts)]

        if not df_mem.empty:
            st.subheader("Top 10 Utilisateurs par Utilisation Mémoire (USEDBYTES)")
            if all(col in df_mem.columns for col in ['ACCOUNT', 'USEDBYTES', 'MAXBYTES', 'PRIVSUM']) and df_mem['USEDBYTES'].sum() > 0:
                # Ensure numeric types before aggregation
                df_mem['USEDBYTES'] = pd.to_numeric(df_mem['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                df_mem['MAXBYTES'] = pd.to_numeric(df_mem['MAXBYTES'], errors='coerce').fillna(0).astype(float)
                df_mem['PRIVSUM'] = pd.to_numeric(df_mem['PRIVSUM'], errors='coerce').fillna(0).astype(float)

                top_users_mem = df_mem.groupby('ACCOUNT', as_index=False)[['USEDBYTES', 'MAXBYTES', 'PRIVSUM']].sum().nlargest(10, 'USEDBYTES')
                if not top_users_mem.empty and top_users_mem['USEDBYTES'].sum() > 0:
                    fig_top_users_mem = px.bar(top_users_mem,
                                                x='ACCOUNT', y='USEDBYTES',
                                                title="Top 10 Comptes par USEDBYTES Total",
                                                labels={'USEDBYTES': 'Utilisation Mémoire (Octets)', 'ACCOUNT': 'Compte Utilisateur'},
                                                hover_data=['MAXBYTES', 'PRIVSUM'],
                                                color='USEDBYTES', color_continuous_scale=px.colors.sequential.Plasma)
                    st.plotly_chart(fig_top_users_mem, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Utilisateurs par Utilisation Mémoire après filtrage.")
            else:
                st.info("Colonnes nécessaires (ACCOUNT, USEDBYTES, MAXBYTES, PRIVSUM) manquantes ou USEDBYTES total est zéro/vide après filtrage.")

            st.subheader("Moyenne de USEDBYTES par Client (ACCOUNT)")
            if 'ACCOUNT' in df_mem.columns and 'USEDBYTES' in df_mem.columns and df_mem['USEDBYTES'].sum() > 0:
                df_mem_account_clean = df_mem[df_mem['ACCOUNT'] != 'Compte Inconnu'].copy()
                
                if not df_mem_account_clean.empty:
                    # Ensure USEDBYTES is numeric here
                    df_mem_account_clean['USEDBYTES'] = pd.to_numeric(df_mem_account_clean['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                    df_mem_account_clean['ACCOUNT_DISPLAY'] = df_mem_account_clean['ACCOUNT'].astype(str)

                    account_counts = df_mem_account_clean['ACCOUNT_DISPLAY'].nunique()
                    if account_counts > 6:
                        top_accounts = df_mem_account_clean['ACCOUNT_DISPLAY'].value_counts().nlargest(6).index
                        df_mem_account_filtered_for_plot = df_mem_account_clean.loc[df_mem_account_clean['ACCOUNT_DISPLAY'].isin(top_accounts)].copy()
                    else:
                        df_mem_account_filtered_for_plot = df_mem_account_clean.copy()

                    avg_mem_account = df_mem_account_filtered_for_plot.groupby('ACCOUNT_DISPLAY', as_index=False)['USEDBYTES'].mean().sort_values(by='USEDBYTES', ascending=False)
                    if not avg_mem_account.empty and not avg_mem_account['USEDBYTES'].sum() == 0:
                        fig_avg_mem_account = px.bar(avg_mem_account,
                                                     x='ACCOUNT_DISPLAY', y='USEDBYTES',
                                                     title="Moyenne de USEDBYTES par Client SAP (Top 6 ou tous)",
                                                     labels={'USEDBYTES': 'Moyenne USEDBYTES (Octets)', 'ACCOUNT_DISPLAY': 'Client SAP'},
                                                     color='USEDBYTES', color_continuous_scale=px.colors.sequential.Viridis)
                        fig_avg_mem_account.update_xaxes(type='category')
                        st.plotly_chart(fig_avg_mem_account, use_container_width=True)
                    else:
                        st.info("Pas de données valides pour la moyenne de USEDBYTES par Client SAP après filtrage (peut-être tous 'Compte Inconnu' ou USEDBYTES est zéro).")
                else:
                    st.info("Aucune donnée valide pour les clients (ACCOUNT) après filtrage.")
            else:
                st.info("Colonnes 'ACCOUNT' ou 'USEDBYTES' manquantes ou USEDBYTES total est zéro/vide après filtrage.")

            st.subheader("Distribution de l'Utilisation Mémoire (USEDBYTES) - Courbe de Densité")
            if 'USEDBYTES' in df_mem.columns and df_mem['USEDBYTES'].sum() > 0:
                # Ensure USEDBYTES is numeric here
                df_mem['USEDBYTES'] = pd.to_numeric(df_mem['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                if df_mem['USEDBYTES'].nunique() > 1:
                    fig_dist_mem = ff.create_distplot([df_mem['USEDBYTES'].dropna()], ['USEDBYTES'], bin_size=df_mem['USEDBYTES'].std()/5 if df_mem['USEDBYTES'].std() > 0 else 1, show_rug=False, show_hist=False)
                    fig_dist_mem.update_layout(title_text="Distribution de l'Utilisation Mémoire (USEDBYTES) - Courbe de Densité", xaxis_title='Utilisation Mémoire (Octets)', yaxis_title='Densité')
                    fig_dist_mem.data[0].line.color = 'lightcoral'
                    st.plotly_chart(fig_dist_mem, use_container_width=True)
                else:
                    st.info("La colonne 'USEDBYTES' contient des valeurs uniques ou est vide après filtrage, impossible de créer une courbe de densité.")
            else:
                st.info("Colonne 'USEDBYTES' manquante ou total est zéro/vide après filtrage.")

            if 'FULL_DATETIME' in df_mem.columns and pd.api.types.is_datetime64_any_dtype(df_mem['FULL_DATETIME']) and not df_mem['FULL_DATETIME'].isnull().all() and 'USEDBYTES' in df_mem.columns and df_mem['USEDBYTES'].sum() > 0:
                # Ensure USEDBYTES is numeric here
                df_mem['USEDBYTES'] = pd.to_numeric(df_mem['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                hourly_mem_usage = df_mem.set_index('FULL_DATETIME')['USEDBYTES'].resample('H').mean().dropna()
                if not hourly_mem_usage.empty:
                    fig_hourly_mem = px.line(hourly_mem_usage.reset_index(), x='FULL_DATETIME', y='USEDBYTES', title="Tendance Moyenne USEDBYTES par Heure", labels={'FULL_DATETIME': 'Heure', 'USEDBYTES': 'Moyenne USEDBYTES'}, color_discrete_sequence=['purple'])
                    fig_hourly_mem.update_xaxes(dtick="H1", tickformat="%H:%M")
                    st.plotly_chart(fig_hourly_mem, use_container_width=True)
                else:
                    pass
            else:
                pass
            
            st.subheader("Comparaison des Métriques Mémoire (USEDBYTES, MAXBYTES, PRIVSUM) par Compte Utilisateur")
            mem_metrics_cols = ['USEDBYTES', 'MAXBYTES', 'PRIVSUM']
            if all(col in df_mem.columns for col in mem_metrics_cols) and 'ACCOUNT' in df_mem.columns and df_mem[mem_metrics_cols].sum().sum() > 0:
                # Ensure numeric types before aggregation
                for col in mem_metrics_cols:
                    df_mem[col] = pd.to_numeric(df_mem[col], errors='coerce').fillna(0).astype(float)
                account_mem_summary = df_mem.groupby('ACCOUNT', as_index=False)[mem_metrics_cols].sum().nlargest(10, 'USEDBYTES')
                if not account_mem_summary.empty and account_mem_summary[mem_metrics_cols].sum().sum() > 0:
                    fig_mem_comparison = px.bar(account_mem_summary, x='ACCOUNT', y=mem_metrics_cols, title="Comparaison des Métriques Mémoire par Compte Utilisateur (Top 10 USEDBYTES)", labels={'value': 'Quantité (Octets)', 'variable': 'Métrique Mémoire', 'ACCOUNT': 'Compte Utilisateur'}, barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_mem_comparison, use_container_width=True)
                else:
                    st.info("Pas de données valides pour la comparaison des métriques mémoire par compte utilisateur après filtrage.")
            else:
                st.info("Colonnes nécessaires (ACCOUNT, USEDBYTES, MAXBYTES, PRIVSUM) manquantes ou leurs totaux sont zéro/vides après filtrage pour la comparaison des métriques mémoire.")
            
            # NOUVELLE SECTION : Top Types de Tâches par Utilisation Mémoire
            st.subheader("Top Types de Tâches (TASKTYPE) par Utilisation Mémoire (USEDBYTES)")
            if 'TASKTYPE' in df_mem.columns and 'USEDBYTES' in df_mem.columns and df_mem['USEDBYTES'].sum() > 0:
                # Ensure USEDBYTES is numeric here
                df_mem['USEDBYTES'] = pd.to_numeric(df_mem['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                top_tasktype_mem = df_mem.groupby('TASKTYPE', as_index=False)['USEDBYTES'].sum().nlargest(3, 'USEDBYTES') # Ajout de 'USEDBYTES' comme critère
                if not top_tasktype_mem.empty and top_tasktype_mem['USEDBYTES'].sum() > 0:
                    fig_top_tasktype_mem = px.bar(top_tasktype_mem,
                                                x='TASKTYPE', y='USEDBYTES',
                                                title="Top 3 Types de Tâches par Utilisation Mémoire (USEDBYTES)",
                                                labels={'USEDBYTES': 'Utilisation Mémoire Totale (Octets)', 'TASKTYPE': 'Type de Tâche'},
                                                color='USEDBYTES', color_continuous_scale=px.colors.sequential.Greys)
                    st.plotly_chart(fig_top_tasktype_mem, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top Types de Tâches par Utilisation Mémoire après filtrage.")
            else:
                st.info("Colonnes 'TASKTYPE' ou 'USEDBYTES' manquantes ou USEDBYTES total est zéro/vide après filtrage pour les types de tâches mémoire.")

            st.subheader("Aperçu des Données Mémoire Filtrées")
            st.dataframe(df_mem.head())
        else:
            st.warning("Données mémoire non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Transactions Utilisateurs":
        # --- Onglet 2: Transactions Utilisateurs (USERTCODE_cleaned.xlsx) ---
        st.header("👤 Analyse des Transactions Utilisateurs")
        df_user = dfs['usertcode'].copy()
        if selected_accounts:
            if 'ACCOUNT' in df_user.columns:
                df_user = df_user[df_user['ACCOUNT'].isin(selected_accounts)]
            else:
                st.warning("La colonne 'ACCOUNT' est manquante dans les données utilisateurs pour le filtrage.")
        if selected_tasktypes:
            if 'TASKTYPE' in df_user.columns:
                df_user = df_user[df_user['TASKTYPE'].isin(selected_tasktypes)]
            else:
                st.warning("La colonne 'TASKTYPE' est manquante dans les données utilisateurs pour le filtrage.")

        if not df_user.empty:
            st.subheader("Top Types de Tâches (TASKTYPE) par Temps de Réponse Moyen")
            if 'TASKTYPE' in df_user.columns and 'RESPTI' in df_user.columns and df_user['RESPTI'].sum() > 0:
                # Ensure RESPTI is numeric before aggregation
                df_user['RESPTI'] = pd.to_numeric(df_user['RESPTI'], errors='coerce').fillna(0).astype(float)
                temp_top_tasktype_resp = df_user.groupby('TASKTYPE', as_index=False)['RESPTI'].mean().nlargest(10, 'RESPTI')
                if not temp_top_tasktype_resp.empty and temp_top_tasktype_resp['RESPTI'].sum() > 0:
                    fig_top_tasktype_resp = px.bar(temp_top_tasktype_resp,
                                                x='TASKTYPE',
                                                y='RESPTI',
                                                title="Top 10 Types de Tâches par Temps de Réponse Moyen",
                                                labels={'RESPTI': 'Temps de Réponse Moyen (ms)', 'TASKTYPE': 'Type de Tâche'},
                                                color='RESPTI', color_continuous_scale=px.colors.sequential.Cividis)
                    st.plotly_chart(fig_top_tasktype_resp, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top Types de Tâches par Temps de Réponse Moyen après filtrage.")
            else:
                st.info("Colonnes 'TASKTYPE' ou 'RESPTI' manquantes ou RESPTI total est zéro/vide après filtrage.")

            st.subheader("Nombre de Transactions par Utilisateur (Top 10)")
            if "usertcode" in dfs and not dfs["usertcode"].empty:
                df_usertcode = dfs["usertcode"].copy()
                if selected_accounts:
                    df_usertcode = df_usertcode[df_usertcode['ACCOUNT'].isin(selected_accounts)]
                if selected_tasktypes:
                    df_usertcode = df_usertcode[df_usertcode['TASKTYPE'].isin(selected_tasktypes)]

                if 'ACCOUNT' in df_usertcode.columns:
                    df_transactions_per_user = df_usertcode.groupby('ACCOUNT').size().reset_index(name='TRANSACTION_COUNT')
                    df_transactions_per_user = df_transactions_per_user.sort_values(by='TRANSACTION_COUNT', ascending=False).head(10)

                    if not df_transactions_per_user.empty:
                        fig_transactions_user = px.bar(
                            df_transactions_per_user,
                            x='ACCOUNT',
                            y='TRANSACTION_COUNT',
                            title='Top 10 des Utilisateurs par Nombre de Transactions',
                            labels={'ACCOUNT': 'Utilisateur', 'TRANSACTION_COUNT': 'Nombre de Transactions'}
                        )
                        st.plotly_chart(fig_transactions_user, use_container_width=True)
                    else:
                        st.info("Aucune donnée de transaction par utilisateur disponible après filtrage.")
                else:
                    st.info("La colonne 'ACCOUNT' est manquante dans le DataFrame 'usertcode' pour les transactions par utilisateur.")
            else:
                st.warning("Données de transactions par utilisateur (USERTCODE) non disponibles ou filtrées à vide.")

            # NOUVELLE VISUALISATION : Tendance du Temps de Réponse Moyen par Heure
            st.subheader("Tendance du Temps de Réponse Moyen par Heure")
            if 'FULL_DATETIME' in df_user.columns and pd.api.types.is_datetime64_any_dtype(df_user['FULL_DATETIME']) and not df_user['FULL_DATETIME'].isnull().all() and 'RESPTI' in df_user.columns and df_user['RESPTI'].sum() > 0:
                # Ensure RESPTI is numeric here
                df_user['RESPTI'] = pd.to_numeric(df_user['RESPTI'], errors='coerce').fillna(0).astype(float)
                hourly_resp_time = df_user.set_index('FULL_DATETIME')['RESPTI'].resample('H').mean().dropna() / 1000.0
                if not hourly_resp_time.empty:
                    fig_hourly_resp = px.line(hourly_resp_time.reset_index(), x='FULL_DATETIME', y='RESPTI',
                                                title="Tendance du Temps de Réponse Moyen par Heure (s)",
                                                labels={'FULL_DATETIME': 'Heure', 'RESPTI': 'Temps de Réponse Moyen (s)'},
                                                color_discrete_sequence=['red'])
                    fig_hourly_resp.update_xaxes(dtick="H1", tickformat="%H:%M")
                    st.plotly_chart(fig_hourly_resp, use_container_width=True)
                else:
                    st.info("Pas de données valides pour la tendance horaire du temps de réponse après filtrage.")
            else:
                st.info("Colonnes 'FULL_DATETIME' ou 'RESPTI' manquantes/invalides ou RESPTI total est zéro/vide après filtrage pour la tendance.")
            
            st.subheader("Corrélation entre Temps de Réponse et Temps CPU")
            st.markdown("""
                Ce graphique explore la relation entre le temps de réponse total d'une transaction et le temps CPU qu'elle consomme.
                * Chaque point représente une transaction.
                * Une tendance à la hausse (points allant de bas à gauche vers haut à droite) suggère que plus une transaction utilise de CPU, plus son temps de réponse est long.
                * Les points éloignés de la tendance peuvent indiquer d'autres facteurs influençant le temps de réponse (par exemple, des attentes E/S, des verrous, etc.).
                * La couleur des points indique le type de tâche, aidant à identifier les catégories de transactions qui se comportent différemment.
                """)
            
            hover_data_cols = []
            if 'ACCOUNT' in df_user.columns:
                hover_data_cols.append('ACCOUNT')
            if 'TASKTYPE' in df_user.columns:
                hover_data_cols.append('TASKTYPE')
            if 'ENTRY_ID' in df_user.columns:
                hover_data_cols.append('ENTRY_ID')

            if 'RESPTI' in df_user.columns and 'CPUTI' in df_user.columns and df_user['CPUTI'].sum() > 0 and df_user['RESPTI'].sum() > 0:
                # Ensure numeric types here
                df_user['RESPTI'] = pd.to_numeric(df_user['RESPTI'], errors='coerce').fillna(0).astype(float)
                df_user['CPUTI'] = pd.to_numeric(df_user['CPUTI'], errors='coerce').fillna(0).astype(float)
                fig_resp_cpu_corr = px.scatter(df_user, x='CPUTI', y='RESPTI',
                                                title="Temps de Réponse vs. Temps CPU",
                                                labels={'CPUTI': 'Temps CPU (ms)', 'RESPTI': 'Temps de Réponse (ms)'},
                                                hover_data=hover_data_cols,
                                                color='TASKTYPE' if 'TASKTYPE' in df_user.columns else None,
                                                log_x=True,
                                                log_y=True,
                                                # Removed: trendline="ols" - requires 'statsmodels' which causes installation issues
                                                color_discrete_sequence=px.colors.qualitative.Alphabet)
                st.plotly_chart(fig_resp_cpu_corr, use_container_width=True)
            else:
                st.info("Colonnes 'RESPTI' ou 'CPUTI' manquantes ou leurs totaux sont zéro/vide après filtrage pour la corrélation.")
            
            io_detailed_metrics_counts = ['READDIRCNT', 'READSEQCNT', 'CHNGCNT', 'PHYREADCNT']
            if 'TASKTYPE' in df_user.columns and all(col in df_user.columns for col in io_detailed_metrics_counts) and df_user[io_detailed_metrics_counts].sum().sum() > 0:
                st.subheader("Total des Opérations de Lecture/Écriture (Comptes) par Type de Tâche")
                st.markdown("""
                    Ce graphique présente le total des opérations de lecture et d'écriture par type de tâche.
                    * **READDIRCNT** : Nombre de lectures directes (accès spécifiques à des blocs de données).
                    * **READSEQCNT** : Nombre de lectures séquentielles (accès consécutives aux données).
                    * **CHNGCNT** : Nombre de changements (écritures) d'enregistrements.
                    * **PHYREADCNT** : Nombre total de lectures physiques (lectures réelles depuis le disque).
                    Ces métriques sont cruciales pour comprendre l'intensité des interactions de chaque tâche avec la base de données ou le système de fichiers.
                    """)
                # Ensure numeric types here
                for col in io_detailed_metrics_counts:
                    df_user[col] = pd.to_numeric(df_user[col], errors='coerce').fillna(0).astype(float)
                df_io_counts = df_user.groupby('TASKTYPE', as_index=False)[io_detailed_metrics_counts].sum().nlargest(10, 'PHYREADCNT')
                if not df_io_counts.empty and df_io_counts['PHYREADCNT'].sum() > 0: # Check sum of the column used for nlargest
                    fig_io_counts = px.bar(df_io_counts, x='TASKTYPE', y=io_detailed_metrics_counts,
                                           title="Total des Opérations de Lecture/Écriture (Comptes) par Type de Tâche (Top 10)",
                                           labels={'value': 'Nombre d\'Opérations', 'variable': 'Type d\'Opération', 'TASKTYPE': 'Type de Tâche'},
                                           barmode='group', color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig_io_counts, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour les opérations de lecture/écriture (comptes) après filtrage.")
            else:
                pass

            io_detailed_metrics_buffers_records = ['READDIRBUF', 'READDIRREC', 'READSEQBUF', 'READSEQREC', 'CHNGREC', 'PHYCHNGREC']
            if 'TASKTYPE' in df_user.columns and all(col in df_user.columns for col in io_detailed_metrics_buffers_records) and df_user[io_detailed_metrics_buffers_records].sum().sum() > 0:
                st.subheader("Utilisation des Buffers et Enregistrements par Type de Tâche")
                st.markdown("""
                    Ce graphique détaille l'efficacité des opérations d'E/S en montrant l'utilisation des tampons et le nombre d'enregistrements traités.
                    * **READDIRBUF** : Nombre de lectures directes via buffer.
                    * **READDIRREC** : Nombre d'enregistrements lus directement.
                    * **READSEQBUF** : Nombre de lectures séquentielles via buffer.
                    * **READSEQREC** : Nombre d'enregistrements lus séquentiellement.
                    * **CHNGREC** : Nombre d'enregistrements modifiés.
                    * **PHYCHNGREC** : Nombre total d'enregistrements physiquement modifiés.
                    Ces métriques aident à évaluer si les tâches tirent parti de la mise en cache (buffers) et l'ampleur des données traitées.
                    """)
                # Ensure numeric types here
                for col in io_detailed_metrics_buffers_records:
                    df_user[col] = pd.to_numeric(df_user[col], errors='coerce').fillna(0).astype(float)
                df_io_buffers_records = df_user.groupby('TASKTYPE', as_index=False)[io_detailed_metrics_buffers_records].sum().nlargest(10, 'READDIRREC')
                if not df_io_buffers_records.empty and df_io_buffers_records['READDIRREC'].sum() > 0: # Check sum of the column used for nlargest
                    fig_io_buffers_records = px.bar(df_io_buffers_records, x='TASKTYPE', y=io_detailed_metrics_buffers_records,
                                                     title="Utilisation des Buffers et Enregistrements par Type de Tâche (Top 10)",
                                                     labels={'value': 'Nombre', 'variable': 'Métrique', 'TASKTYPE': 'Type de Tâche'},
                                                     barmode='group', color_discrete_sequence=px.colors.sequential.Plasma)
                    st.plotly_chart(fig_io_buffers_records, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour l'utilisation des buffers et enregistrements après filtrage.")
            else:
                pass
            
            comm_metrics_filtered = ['DSQLCNT', 'SLI_CNT']
            if 'TASKTYPE' in df_user.columns and all(col in df_user.columns for col in comm_metrics_filtered) and df_user[comm_metrics_filtered].sum().sum() > 0:
                st.subheader("Analyse des Communications et Appels Système par Type de Tâche (DSQLCNT et SLI_CNT)")
                st.markdown("""
                    Ce graphique se concentre sur deux métriques clés pour les interactions des tâches avec d'autres systèmes :
                    * **DSQLCNT** : Nombre d'appels SQL dynamiques (requêtes de base de données générées dynamiquement). Un nombre élevé peut indiquer une forte interaction avec la base de données.
                    * **SLI_CNT** : Nombre d'appels SLI (System Level Interface). Ces appels représentent les interactions de bas niveau avec le système d'exploitation ou d'autres composants système.
                    Ces métriques sont essentielles pour diagnostiquer les problèmes de communication ou les dépendances externes.
                    """)
                # Ensure numeric types here
                for col in comm_metrics_filtered:
                    df_user[col] = pd.to_numeric(df_user[col], errors='coerce').fillna(0).astype(float)
                df_comm_metrics = df_user.groupby('TASKTYPE', as_index=False)[comm_metrics_filtered].sum().nlargest(4, 'DSQLCNT')
                if not df_comm_metrics.empty and df_comm_metrics['DSQLCNT'].sum() > 0: # Check sum of the column used for nlargest
                    fig_comm_metrics = px.bar(df_comm_metrics, x='TASKTYPE', y=comm_metrics_filtered,
                                                 title="Communications et Appels Système par Type de Tâche (Top 4)",
                                                 labels={'value': 'Nombre / Temps (ms)', 'variable': 'Métrique', 'TASKTYPE': 'Type de Tâche'},
                                                 barmode='group', color_discrete_sequence=px.colors.qualitative.Bold)
                    st.plotly_chart(fig_comm_metrics, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour les métriques de communication et d'appels système après filtrage.")
            else:
                st.info("Colonnes de communication (DSQLCNT, SLI_CNT) manquantes ou leurs sommes sont zéro/vides après filtrage.")


            st.subheader("Aperçu des Données Utilisateurs Filtrées")
            st.dataframe(df_user.head())
        else:
            st.warning("Données utilisateurs (USERTCODE) non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Statistiques Horaires":
        # --- Onglet 3: Statistiques Horaires (Times_final_cleaned_clean.xlsx) ---
        st.header("⏰ Statistiques Horaires du Système")
        df_times_data = dfs['times'].copy()
        if selected_tasktypes:
            if 'TASKTYPE' in df_times_data.columns:
                df_times_data = df_times_data[df_times_data['TASKTYPE'].isin(selected_tasktypes)]
            else:
                st.warning("La colonne 'TASKTYPE' est manquante dans les données horaires pour le filtrage.")
            
        if not df_times_data.empty:
            st.subheader("Évolution du Nombre Total d'Appels Physiques (PHYCALLS) par Tranche Horaire")
            if 'TIME' in df_times_data.columns and 'PHYCALLS' in df_times_data.columns and df_times_data['PHYCALLS'].sum() > 0:
                # Ensure PHYCALLS is numeric here
                df_times_data['PHYCALLS'] = pd.to_numeric(df_times_data['PHYCALLS'], errors='coerce').fillna(0).astype(float)
                df_times_data['HOUR_OF_DAY'] = df_times_data['TIME'].apply(lambda x: str(x).split(':')[0].zfill(2) if ':' in str(x) else str(x).zfill(2)[:2])
                
                # Appliquer fillna(0) sur la colonne numérique avant de grouper et de convertir en catégorie
                hourly_counts = df_times_data.groupby('HOUR_OF_DAY', as_index=False)['PHYCALLS'].sum().fillna(0)
                
                hourly_categories = [str(i).zfill(2) for i in range(24)] # Générer toutes les heures de 00 à 23
                hourly_counts['HOUR_OF_DAY'] = pd.Categorical(hourly_counts['HOUR_OF_DAY'], categories=hourly_categories, ordered=True)
                hourly_counts = hourly_counts.sort_values('HOUR_OF_DAY')

                if not hourly_counts.empty and hourly_counts['PHYCALLS'].sum() > 0:
                    fig_phycalls = px.line(hourly_counts,
                                            x='HOUR_OF_DAY', y='PHYCALLS',
                                            title="Total Appels Physiques par Tranche Horaire",
                                            labels={'HOUR_OF_DAY': 'Tranche Horaire', 'PHYCALLS': 'Total Appels Physiques'},
                                            color_discrete_sequence=px.colors.sequential.Cividis,
                                            markers=True)
                    st.plotly_chart(fig_phycalls, use_container_width=True)
                else:
                    pass
            else:
                st.info("Colonnes 'TIME' ou 'PHYCALLS' manquantes ou PHYCALLS total est zéro/vide après filtrage.")

            st.subheader("Top 5 Tranches Horaires les plus Chargées (Opérations d'E/S)")
            io_cols = ['READDIRCNT', 'READSEQCNT', 'CHNGCNT']
            if all(col in df_times_data.columns for col in io_cols) and df_times_data[io_cols].sum().sum() > 0:
                # Ensure numeric types here
                for col in io_cols:
                    df_times_data[col] = pd.to_numeric(df_times_data[col], errors='coerce').fillna(0).astype(float)
                df_times_data['TOTAL_IO'] = df_times_data['READDIRCNT'] + df_times_data['READSEQCNT'] + df_times_data['CHNGCNT']
                top_io_times = df_times_data.groupby('TIME', as_index=False)['TOTAL_IO'].sum().nlargest(5, 'TOTAL_IO').sort_values(by='TOTAL_IO', ascending=False)
                if not top_io_times.empty and top_io_times['TOTAL_IO'].sum() > 0:
                    fig_top_io = px.bar(top_io_times,
                                        x='TIME', y='TOTAL_IO',
                                        title="Top 5 Tranches Horaires par Total Opérations I/O",
                                        labels={'TIME': 'Tranche Horaire', 'TOTAL_IO': 'Total Opérations I/O'},
                                        color='TOTAL_IO', color_continuous_scale=px.colors.sequential.Inferno)
                    st.plotly_chart(fig_top_io, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les opérations I/O après filtrage.")
            else:
                st.info("Colonnes I/O manquantes (READDIRCNT, READSEQCNT, CHNGCNT) ou leur somme est zéro/vide après filtrage.")

            st.subheader("Temps Moyen de Réponse / CPU / Traitement par Tranche Horaire")
            perf_cols = ["RESPTI", "CPUTI", "PROCTI"]
            if all(col in df_times_data.columns for col in perf_cols) and df_times_data[perf_cols].sum().sum() > 0:
                # Ensure columns are numeric here too
                for col in perf_cols:
                    df_times_data[col] = pd.to_numeric(df_times_data[col], errors='coerce').fillna(0).astype(float)

                avg_times_by_hour_temp = df_times_data.groupby("TIME", as_index=False)[perf_cols].mean()
                
                if not avg_times_by_hour_temp.empty and avg_times_by_hour_temp[perf_cols].sum().sum() > 0: # Check before division
                    # Apply division and fillna(0) only to the numeric columns
                    avg_times_by_hour = avg_times_by_hour_temp.copy() # Create a copy
                    for col in perf_cols:
                        avg_times_by_hour[col] = (avg_times_by_hour[col] / 1000.0).fillna(0) # Apply fillna here
                    
                    hourly_categories_times = [
                        '00--06', '06--07', '07--08', '08--09', '09--10', '10--11', '11--12', '12--13',
                        '13--14', '14--15', '15--16', '16--17', '17--18', '18--19', '19--20', '20--21',
                        '21--22', '22--23', '23--00'
                    ]
                    # Convert 'TIME' to categorical AFTER numeric columns are handled
                    avg_times_by_hour['TIME'] = pd.Categorical(avg_times_by_hour['TIME'], categories=hourly_categories_times, ordered=True)
                    avg_times_by_hour = avg_times_by_hour.sort_values('TIME') # Removed .fillna(0) from here

                    if not avg_times_by_hour.empty and avg_times_by_hour[perf_cols].sum().sum() > 0:
                        fig_avg_times = px.line(avg_times_by_hour,
                                                x='TIME', y=perf_cols,
                                                title="Temps Moyen (s) par Tranche Horaire",
                                                labels={'value': 'Temps Moyen (s)', 'variable': 'Métrique', 'TIME': 'Tranche Horaire'},
                                                color_discrete_sequence=px.colors.qualitative.Set1,
                                                markers=True)
                        st.plotly_chart(fig_avg_times, use_container_width=True)
                    else:
                        st.info("Pas de données valides pour les temps moyens après filtrage.")
                else:
                    st.info("Pas de données valides pour les temps moyens après filtrage (la moyenne est vide ou zéro).")
            else:
                st.info("Colonnes nécessaires (RESPTI, CPUTI, PROCTI, TIME) manquantes ou leur somme est zéro/vide après filtrage.")
            
            st.subheader("Aperçu des Données Horaires Filtrées")
            st.dataframe(df_times_data.head())
        else:
            st.warning("Données horaires (Times) non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Décomposition des Tâches":
        # --- Onglet 4: Décomposition des Tâches (TASKTIMES_final_cleaned_clean.xlsx) ---
        st.header("⚙️ Décomposition des Types de Tâches")
        df_task = dfs['tasktimes'].copy()
        if selected_tasktypes:
            if 'TASKTYPE' in df_task.columns:
                df_task = df_task[df_task['TASKTYPE'].isin(selected_tasktypes)]
            else:
                st.warning("La colonne 'TASKTYPE' est manquante dans les données de temps de tâches pour le filtrage.")


        if not df_task.empty:
            st.subheader("Répartition des Types de Tâches (TASKTYPE)")
            if 'TASKTYPE' in df_task.columns and 'COUNT' in df_task.columns and df_task['COUNT'].sum() > 0:
                # Ensure COUNT is numeric here
                df_task['COUNT'] = pd.to_numeric(df_task['COUNT'], errors='coerce').fillna(0).astype(float)
                task_counts = df_task.groupby('TASKTYPE', as_index=False)['COUNT'].sum()
                task_counts.columns = ['TASKTYPE', 'Count']
                
                min_count_for_pie = task_counts['Count'].sum() * 0.01
                significant_tasks = pd.DataFrame() # Initialize as empty DataFrame
                other_tasks_count = 0

                if not task_counts.empty:
                    significant_tasks = task_counts[task_counts['Count'] >= min_count_for_pie]
                    other_tasks_count = task_counts[task_counts['Count'] < min_count_for_pie]['Count'].sum()

                if other_tasks_count > 0:
                    significant_tasks = pd.concat([significant_tasks, pd.DataFrame([{'TASKTYPE': 'Autres Petites Tâches', 'Count': other_tasks_count}])])

                if not significant_tasks.empty and significant_tasks['Count'].sum() > 0:
                    fig_task_dist = px.pie(significant_tasks, values='Count', names='TASKTYPE',
                                            title="Répartition des Types de Tâches",
                                            hole=0.3,
                                            color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig_task_dist, use_container_width=True)
                else:
                    st.info("Pas de données valides pour la répartition des types de tâches après filtrage.")
            else:
                st.info("Colonnes 'TASKTYPE' ou 'COUNT' manquantes ou COUNT total est zéro/vide après filtrage.")

            st.subheader("Top 10 TASKTYPE par Temps de Réponse (RESPTI) et CPU (CPUTI)")
            perf_cols_task = ['RESPTI', 'CPUTI']
            if 'TASKTYPE' in df_task.columns and all(col in df_task.columns for col in perf_cols_task) and df_task[perf_cols_task].sum().sum() > 0:
                # Ensure columns are numeric here too
                for col in perf_cols_task:
                    df_task[col] = pd.to_numeric(df_task[col], errors='coerce').fillna(0).astype(float)

                temp_task_perf = df_task.groupby('TASKTYPE', as_index=False)[perf_cols_task].mean()
                
                if not temp_task_perf.empty and 'RESPTI' in temp_task_perf.columns and pd.api.types.is_numeric_dtype(temp_task_perf['RESPTI']): # Check before nlargest and division
                    if temp_task_perf['RESPTI'].dropna().count() >= 10: # Check if at least 10 non-NaN values
                        top_task_perf_intermediate = temp_task_perf.nlargest(10, 'RESPTI').sort_values(by='RESPTI', ascending=False)
                        if not top_task_perf_intermediate.empty and top_task_perf_intermediate['RESPTI'].sum() > 0:
                            # Ensure columns are numeric before division
                            for col in perf_cols_task:
                                top_task_perf_intermediate[col] = pd.to_numeric(top_task_perf_intermediate[col], errors='coerce').fillna(0).astype(float)
                            
                            # Apply division only to the numeric column
                            task_perf = top_task_perf_intermediate.copy() # Create a copy
                            for col in perf_cols_task:
                                task_perf[col] = task_perf[col] / 1000.0
                            
                            if not task_perf.empty and task_perf['RESPTI'].sum() > 0:
                                fig_task_perf = px.bar(task_perf,
                                                        x='TASKTYPE', y=perf_cols_task,
                                                        title="Top 10 TASKTYPE par Temps de Réponse et CPU (s)",
                                                        labels={'value': 'Temps Moyen (s)', 'variable': 'Métrique', 'TASKTYPE': 'Type de Tâche'},
                                                        barmode='group', color_discrete_sequence=px.colors.qualitative.Bold)
                                st.plotly_chart(fig_task_perf, use_container_width=True)
                            else:
                                st.info("Pas de données valides pour les temps de performance des tâches après filtrage et sélection des 10 plus grandes valeurs (résultat vide ou zéro après division).")
                        else:
                            st.info("Pas de données valides pour les temps de performance des tâches après filtrage et sélection des 10 plus grandes valeurs (résultat intermédiaire vide).")
                    else:
                        st.info("Pas assez de données valides dans 'RESPTI' pour déterminer les Top 10 Types de Tâches après filtrage.")
                else:
                    st.info("Pas de données valides pour les temps de performance des tâches après filtrage (la moyenne est vide ou non-numérique).")
            else:
                st.info("Colonnes 'TASKTYPE', 'RESPTI' ou 'CPUTI' manquantes ou leur somme est zéro/vide après filtrage.")

            st.subheader("Décomposition des Temps d'Attente et GUI par Type de Tâche")
            st.markdown("""
                Ce graphique détaille où le temps est passé au-delà du traitement CPU pour les tâches.
                * **QUEUETI (Temps d'Attente en File)** : Temps passé par la tâche en attente dans une file d'attente. Un temps élevé peut indiquer une surcharge du système ou des goulots d'étranglement.
                * **ROLLWAITTI (Temps d'Attente de Roll-in/out)** : Temps passé par la tâche en attente de chargement ou de déchargement de la mémoire (roll-in/out).
                * **GUITIME (Temps GUI)** : Temps passé par la tâche dans l'interface graphique utilisateur.
                * **GUINETTIME (Temps Réseau GUI)** : Temps passé sur le réseau pour les interactions de l'interface graphique utilisateur.
                Ces métriques aident à identifier les causes de lenteur qui ne sont pas directement liées au CPU, comme les attentes de ressources ou les problèmes réseau.
                """)
            wait_gui_metrics = ['QUEUETI', 'ROLLWAITTI', 'GUITIME', 'GUINETTIME']
            if 'TASKTYPE' in df_task.columns and all(col in df_task.columns for col in wait_gui_metrics) and df_task[wait_gui_metrics].sum().sum() > 0:
                # Ensure numeric types here
                for col in wait_gui_metrics:
                    df_task[col] = pd.to_numeric(df_task[col], errors='coerce').fillna(0).astype(float)
                df_wait_gui = df_task.groupby('TASKTYPE', as_index=False)[wait_gui_metrics].sum().nlargest(10, 'QUEUETI')
                if not df_wait_gui.empty and df_wait_gui['QUEUETI'].sum() > 0:
                    fig_wait_gui = px.bar(df_wait_gui, x='TASKTYPE',
                                          y=wait_gui_metrics,
                                          title="Temps d'Attente et GUI par Type de Tâche (Top 10)",
                                          labels={'value': 'Temps (ms)', 'variable': 'Métrique de Temps', 'TASKTYPE': 'Type de Tâche'},
                                          barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_wait_gui, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour la décomposition des temps d'attente et GUI après filtrage.")
            else:
                st.info("Colonnes d'attente/GUI manquantes ou leurs sommes sont zéro/vides après filtrage.")

            st.subheader("Analyse des Opérations d'E/S (Lectures/Écritures) par Type de Tâche")
            st.markdown("""
                Ce graphique fournit des détails sur les opérations d'entrée/sortie (E/S) spécifiques aux tâches.
                * **READDIRCNT (Lectures Directes)** : Nombre de lectures directes d'enregistrements.
                * **READSEQCNT (Lectures Séquentielles)** : Nombre de lectures séquentielles d'enregistrements.
                * **CHNGCNT (Changements)** : Nombre de changements (écritures) d'enregistrements.
                * **PHYREADCNT (Lectures Physiques)** : Nombre total de lectures physiques (sur le disque).
                * **PHYCHNGREC (Changements Physiques)** : Nombre total d'enregistrements physiquement modifiés.
                * **READDIRREC (Enregistrements Lus Directement)** : Nombre d'enregistrements lus directement.
                Ces métriques sont essentielles pour identifier les tâches gourmandes en E/S et évaluer l'efficacité de l'accès aux données.
                """)
            # FIX: Added 'READDIRREC' to the list so it's available for nlargest
            io_metrics_tasktimes = ['READDIRCNT', 'READSEQCNT', 'CHNGCNT', 'PHYREADCNT', 'PHYCHNGREC', 'READDIRREC']
            if 'TASKTYPE' in df_task.columns and all(col in df_task.columns for col in io_metrics_tasktimes) and df_task[io_metrics_tasktimes].sum().sum() > 0:
                # Ensure numeric types here
                for col in io_metrics_tasktimes:
                    df_task[col] = pd.to_numeric(df_task[col], errors='coerce').fillna(0).astype(float)
                df_io_tasktimes = df_task.groupby('TASKTYPE', as_index=False)[io_metrics_tasktimes].sum().nlargest(10, 'READDIRREC')
                if not df_io_tasktimes.empty and df_io_tasktimes['READDIRREC'].sum() > 0:
                    fig_io_tasktimes = px.bar(df_io_tasktimes, x='TASKTYPE', y=io_metrics_tasktimes,
                                              title="Opérations d'E/S par Type de Tâche (Top 10)",
                                              labels={'value': 'Nombre d\'Opérations', 'variable': 'Métrique E/S', 'TASKTYPE': 'Type de Tâche'},
                                              barmode='group', color_discrete_sequence=px.colors.sequential.Greens)
                    st.plotly_chart(fig_io_tasktimes, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour l'analyse des opérations d'E/S après filtrage.")
            else:
                pass


            st.subheader("Aperçu des Données des Temps de Tâches Filtrées")
            st.dataframe(df_task.head())
        else:
            st.warning("Données des temps de tâches non disponibles ou filtrées à vide.")


    elif st.session_state.current_section == "Insights Hitlist DB":
        # --- NOUVEL ONGLET: Insights Détaillés de la Base de Données (Hitlist DB) ---
        st.header("🔍 Insights Détaillés de la Base de Données (Hitlist DB)")
        df_hitlist = dfs['hitlist_db'].copy()
        
        # Appliquer les filtres globaux si disponibles
        if selected_accounts:
            if 'ACCOUNT' in df_hitlist.columns:
                df_hitlist = df_hitlist[df_hitlist['ACCOUNT'].isin(selected_accounts)]
            else:
                st.warning("La colonne 'ACCOUNT' est manquante dans les données Hitlist DB pour le filtrage.")
        if selected_reports:
            if 'REPORT' in df_hitlist.columns:
                df_hitlist = df_hitlist[df_hitlist['REPORT'].isin(selected_reports)]
            else:
                st.warning("La colonne 'REPORT' est manquante dans les données Hitlist DB pour le filtrage.")
        if selected_tasktypes:
            if 'TASKTYPE' in df_hitlist.columns:
                df_hitlist = df_hitlist[df_hitlist['TASKTYPE'].isin(selected_tasktypes)]
            else:
                st.warning("La colonne 'TASKTYPE' est manquante dans les données Hitlist DB pour le filtrage.")

        if not df_hitlist.empty:
            st.subheader("Top 10 Rapports par Temps de Réponse Moyen (RESPTI)")
            if 'REPORT' in df_hitlist.columns and 'RESPTI' in df_hitlist.columns and df_hitlist['RESPTI'].sum() > 0:
                df_hitlist['RESPTI'] = pd.to_numeric(df_hitlist['RESPTI'], errors='coerce').fillna(0).astype(float)
                top_reports_resp = df_hitlist.groupby('REPORT', as_index=False)['RESPTI'].mean().nlargest(10, 'RESPTI')
                if not top_reports_resp.empty and top_reports_resp['RESPTI'].sum() > 0:
                    fig_top_reports_resp = px.bar(top_reports_resp,
                                                  x='REPORT', y='RESPTI',
                                                  title="Top 10 Rapports par Temps de Réponse Moyen (ms)",
                                                  labels={'RESPTI': 'Temps de Réponse Moyen (ms)', 'REPORT': 'Rapport'},
                                                  color='RESPTI', color_continuous_scale=px.colors.sequential.Sunset)
                    st.plotly_chart(fig_top_reports_resp, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Rapports par Temps de Réponse Moyen après filtrage.")
            else:
                st.info("Colonnes 'REPORT' ou 'RESPTI' manquantes ou RESPTI total est zéro/vide après filtrage.")

            st.subheader("Top 10 Comptes par Nombre d'Appels Base de Données (DBCALLS)")
            if 'ACCOUNT' in df_hitlist.columns and 'DBCALLS' in df_hitlist.columns and df_hitlist['DBCALLS'].sum() > 0:
                df_hitlist['DBCALLS'] = pd.to_numeric(df_hitlist['DBCALLS'], errors='coerce').fillna(0).astype(float)
                top_accounts_db_calls = df_hitlist.groupby('ACCOUNT', as_index=False)['DBCALLS'].sum().nlargest(10, 'DBCALLS')
                if not top_accounts_db_calls.empty and top_accounts_db_calls['DBCALLS'].sum() > 0:
                    fig_top_accounts_db_calls = px.bar(top_accounts_db_calls,
                                                       x='ACCOUNT', y='DBCALLS',
                                                       title="Top 10 Comptes par Nombre d'Appels Base de Données",
                                                       labels={'DBCALLS': 'Nombre Total d\'Appels DB', 'ACCOUNT': 'Compte Utilisateur'},
                                                       color='DBCALLS', color_continuous_scale=px.colors.sequential.Mint)
                    st.plotly_chart(fig_top_accounts_db_calls, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Comptes par Nombre d'Appels Base de Données après filtrage.")
            else:
                st.info("Colonnes 'ACCOUNT' ou 'DBCALLS' manquantes ou DBCALLS total est zéro/vide après filtrage.")

            st.subheader("Distribution du Temps de Réponse (RESPTI) - Courbe de Densité")
            if 'RESPTI' in df_hitlist.columns and df_hitlist['RESPTI'].sum() > 0:
                df_hitlist['RESPTI'] = pd.to_numeric(df_hitlist['RESPTI'], errors='coerce').fillna(0).astype(float)
                if df_hitlist['RESPTI'].nunique() > 1:
                    fig_dist_resp_time = ff.create_distplot([df_hitlist['RESPTI'].dropna()], ['RESPTI'],
                                                            bin_size=df_hitlist['RESPTI'].std()/5 if df_hitlist['RESPTI'].std() > 0 else 1,
                                                            show_rug=False, show_hist=False)
                    fig_dist_resp_time.update_layout(title_text="Distribution du Temps de Réponse (RESPTI)",
                                                     xaxis_title='Temps de Réponse (ms)',
                                                     yaxis_title='Densité')
                    fig_dist_resp_time.data[0].line.color = 'darkred'
                    st.plotly_chart(fig_dist_resp_time, use_container_width=True)
                else:
                    st.info("La colonne 'RESPTI' contient des valeurs uniques ou est vide après filtrage, impossible de créer une courbe de densité.")
            else:
                st.info("Colonne 'RESPTI' manquante ou total est zéro/vide après filtrage.")

            # --- DÉBUT DU BLOC SUPPRIMÉ ---
            # st.subheader("Corrélation entre Temps de Réponse (RESPTI) et Temps CPU (CPUTI)")
            # if 'RESPTI' in df_hitlist.columns and 'CPUTI' in df_hitlist.columns and df_hitlist['RESPTI'].sum() > 0 and df_hitlist['CPUTI'].sum() > 0:
            #     df_hitlist['RESPTI'] = pd.to_numeric(df_hitlist['RESPTI'], errors='coerce').fillna(0).astype(float)
            #     df_hitlist['CPUTI'] = pd.to_numeric(df_hitlist['CPUTI'], errors='coerce').fillna(0).astype(float)
            #     fig_resp_cpu_hitlist = px.scatter(df_hitlist, x='CPUTI', y='RESPTI',
            #                                       title="Temps de Réponse vs. Temps CPU (Hitlist DB)",
            #                                       labels={'CPUTI': 'Temps CPU (ms)', 'RESPTI': 'Temps de Réponse (ms)'},
            #                                       hover_data=['ACCOUNT', 'REPORT', 'TASKTYPE'],
            #                                       color='TASKTYPE' if 'TASKTYPE' in df_hitlist.columns else None,
            #                                       log_x=True,
            #                                       log_y=True,
            #                                       color_discrete_sequence=px.colors.qualitative.Plotly)
            #     st.plotly_chart(fig_resp_cpu_hitlist, use_container_width=True)
            # else:
            #     st.info("Colonnes 'RESPTI' ou 'CPUTI' manquantes ou leurs totaux sont zéro/vides après filtrage pour la corrélation.")
            # --- FIN DU BLOC SUPPRIMÉ ---

            st.subheader("Tendance du Temps de Réponse Moyen par Heure (Hitlist DB)")
            if 'FULL_DATETIME' in df_hitlist.columns and pd.api.types.is_datetime64_any_dtype(df_hitlist['FULL_DATETIME']) and not df_hitlist['FULL_DATETIME'].isnull().all() and 'RESPTI' in df_hitlist.columns and df_hitlist['RESPTI'].sum() > 0:
                df_hitlist['RESPTI'] = pd.to_numeric(df_hitlist['RESPTI'], errors='coerce').fillna(0).astype(float)
                hourly_resp_time_hitlist = df_hitlist.set_index('FULL_DATETIME')['RESPTI'].resample('H').mean().dropna() / 1000.0
                if not hourly_resp_time_hitlist.empty:
                    fig_hourly_resp_hitlist = px.line(hourly_resp_time_hitlist.reset_index(), x='FULL_DATETIME', y='RESPTI',
                                                      title="Tendance du Temps de Réponse Moyen par Heure (s) - Hitlist DB",
                                                      labels={'FULL_DATETIME': 'Heure', 'RESPTI': 'Temps de Réponse Moyen (s)'},
                                                      color_discrete_sequence=['blue'])
                    fig_hourly_resp_hitlist.update_xaxes(dtick="H1", tickformat="%H:%M")
                    st.plotly_chart(fig_hourly_resp_hitlist, use_container_width=True)
                else:
                    st.info("Pas de données valides pour la tendance horaire du temps de réponse après filtrage.")
            else:
                st.info("Colonnes 'FULL_DATETIME' ou 'RESPTI' manquantes/invalides ou RESPTI total est zéro/vide après filtrage pour la tendance.")

            st.subheader("Aperçu des Données Hitlist DB Filtrées")
            st.dataframe(df_hitlist.head())
        else:
            st.warning("Données Hitlist DB non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Performance des Processus de Travail":
        # --- Onglet 6: Performance des Processus de Travail (AL_GET_PERFORMANCE) ---
        st.header("⚡ Performance des Processus de Travail")
        df_perf = dfs['performance'].copy()

        if selected_wp_types:
            if 'WP_TYP' in df_perf.columns:
                df_perf = df_perf[df_perf['WP_TYP'].isin(selected_wp_types)]
            else:
                st.warning("La colonne 'WP_TYP' est manquante dans les données de performance pour le filtrage.")

        if not df_perf.empty:
            st.subheader("Distribution du Temps CPU des Processus de Travail (en secondes)")
            if 'WP_CPU_SECONDS' in df_perf.columns and df_perf['WP_CPU_SECONDS'].sum() > 0:
                # Ensure WP_CPU_SECONDS is numeric here
                df_perf['WP_CPU_SECONDS'] = pd.to_numeric(df_perf['WP_CPU_SECONDS'], errors='coerce').fillna(0).astype(float)
                if df_perf['WP_CPU_SECONDS'].nunique() > 1:
                    fig_cpu_dist = ff.create_distplot([df_perf['WP_CPU_SECONDS'].dropna()], ['Temps CPU (s)'],
                                                      bin_size=df_perf['WP_CPU_SECONDS'].std()/5 if df_perf['WP_CPU_SECONDS'].std() > 0 else 1,
                                                      show_rug=False, show_hist=False)
                    fig_cpu_dist.update_layout(title_text="Distribution du Temps CPU des Processus de Travail",
                                               xaxis_title='Temps CPU (secondes)',
                                               yaxis_title='Densité')
                    fig_cpu_dist.data[0].line.color = 'darkblue'
                    st.plotly_chart(fig_cpu_dist, use_container_width=True)
                else:
                    st.info("La colonne 'WP_CPU_SECONDS' contient des valeurs uniques ou est vide après filtrage, impossible de créer une courbe de densité.")
            else:
                st.info("Colonne 'WP_CPU_SECONDS' manquante ou total est zéro/vide après filtrage.")

            st.subheader("Répartition des Processus de Travail par Statut (WP_STATUS)")
            if 'WP_STATUS' in df_perf.columns and not df_perf['WP_STATUS'].empty:
                status_counts = df_perf['WP_STATUS'].value_counts().reset_index()
                status_counts.columns = ['Statut', 'Count']
                if not status_counts.empty and status_counts['Count'].sum() > 0:
                    fig_status_pie = px.pie(status_counts, values='Count', names='Statut',
                                            title="Répartition des Processus de Travail par Statut",
                                            hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_status_pie, use_container_width=True)
                else:
                    st.info("Pas de données valides pour la répartition par statut des processus de travail après filtrage.")
            else:
                st.info("Colonne 'WP_STATUS' manquante ou vide après filtrage.")

            st.subheader("Nombre de Processus de Travail par Type (WP_TYP)")
            if 'WP_TYP' in df_perf.columns and not df_perf['WP_TYP'].empty:
                type_counts = df_perf['WP_TYP'].value_counts().reset_index()
                type_counts.columns = ['Type', 'Count']
                if not type_counts.empty and type_counts['Count'].sum() > 0:
                    fig_type_bar = px.bar(type_counts, x='Type', y='Count',
                                            title="Nombre de Processus de Travail par Type",
                                            labels={'Type': 'Type de Processus', 'Count': 'Nombre'},
                                            color='Count', color_continuous_scale=px.colors.sequential.Viridis)
                    st.plotly_chart(fig_type_bar, use_container_width=True)
                else:
                    st.info("Pas de données valides pour le nombre de processus de travail par type après filtrage.")
            else:
                st.info("Colonne 'WP_TYP' manquante ou vide après filtrage.")

            st.subheader("Temps CPU Moyen par Type de Processus de Travail (en secondes)")
            if 'WP_TYP' in df_perf.columns and 'WP_CPU_SECONDS' in df_perf.columns and df_perf['WP_CPU_SECONDS'].sum() > 0:
                # Ensure WP_CPU_SECONDS is numeric here
                df_perf['WP_CPU_SECONDS'] = pd.to_numeric(df_perf['WP_CPU_SECONDS'], errors='coerce').fillna(0).astype(float)
                avg_cpu_by_type = df_perf.groupby('WP_TYP', as_index=False)['WP_CPU_SECONDS'].mean()
                if not avg_cpu_by_type.empty and avg_cpu_by_type['WP_CPU_SECONDS'].sum() > 0:
                    fig_avg_cpu_type = px.bar(avg_cpu_by_type, x='WP_TYP', y='WP_CPU_SECONDS',
                                                title="Temps CPU Moyen par Type de Processus de Travail",
                                                labels={'WP_TYP': 'Type de Processus', 'WP_CPU_SECONDS': 'Temps CPU Moyen (s)'},
                                                color='WP_CPU_SECONDS', color_continuous_scale=px.colors.sequential.Plasma)
                    st.plotly_chart(fig_avg_cpu_type, use_container_width=True)
                else:
                    st.info("Pas de données valides pour le temps CPU moyen par type de processus de travail après filtrage.")
            else:
                st.info("Colonnes 'WP_TYP' ou 'WP_CPU_SECONDS' manquantes ou total est zéro/vide après filtrage.")

            st.subheader("Nombre Total de Redémarrages par Type de Processus de Travail (WP_IRESTRT)")
            if 'WP_TYP' in df_perf.columns and 'WP_IRESTRT' in df_perf.columns and df_perf['WP_IRESTRT'].sum() > 0:
                # Ensure WP_IRESTRT is numeric here
                df_perf['WP_IRESTRT'] = pd.to_numeric(df_perf['WP_IRESTRT'], errors='coerce').fillna(0).astype(float)
                restarts_by_type = df_perf.groupby('WP_TYP', as_index=False)['WP_IRESTRT'].sum().nlargest(10, 'WP_IRESTRT')
                if not restarts_by_type.empty and restarts_by_type['WP_IRESTRT'].sum() > 0:
                    fig_restarts_type = px.bar(restarts_by_type, x='WP_TYP', y='WP_IRESTRT',
                                                title="Nombre Total de Redémarrages par Type de Processus de Travail",
                                                labels={'WP_TYP': 'Type de Processus', 'WP_IRESTRT': 'Nombre Total de Redémarrages'},
                                                color='WP_IRESTRT', color_continuous_scale=px.colors.sequential.OrRd)
                    st.plotly_chart(fig_restarts_type, use_container_width=True)
                else:
                    st.info("Pas de données valides pour le nombre de redémarrages par type de processus de travail après filtrage.")
            else:
                st.info("Colonnes 'WP_TYP' ou 'WP_IRESTRT' manquantes ou total est zéro/vide après filtrage.")

            st.subheader("Aperçu des Données de Performance Filtrées")
            st.dataframe(df_perf.head())
        else:
            st.warning("Données de performance non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Résumé des Traces de Performance SQL":
        # --- Onglet 7: Résumé des Traces de Performance SQL (performance_trace_summary_final_cleaned_clean.xlsx) ---
        st.header("📊 Résumé des Traces de Performance SQL")
        df_sql_trace = dfs['sql_trace_summary'].copy()

        if not df_sql_trace.empty:
            st.subheader("Top 10 Requêtes SQL par Temps d'Exécution Total (EXECTIME)")
            st.markdown("""
                Ce graphique identifie les 10 requêtes SQL qui ont consommé le plus de temps d'exécution cumulé.
                Il est crucial pour repérer les goulots d'étranglement globaux en termes de performance.
                """)
            if 'SQLSTATEM' in df_sql_trace.columns and 'EXECTIME' in df_sql_trace.columns and df_sql_trace['EXECTIME'].sum() > 0:
                # Ensure EXECTIME is numeric here
                df_sql_trace['EXECTIME'] = pd.to_numeric(df_sql_trace['EXECTIME'], errors='coerce').fillna(0).astype(float)
                top_sql_by_exectime = df_sql_trace.groupby('SQLSTATEM', as_index=False)['EXECTIME'].sum().nlargest(10, 'EXECTIME')
                top_sql_by_exectime['SQLSTATEM_SHORT'] = top_sql_by_exectime['SQLSTATEM'].apply(lambda x: x[:70] + '...' if len(x) > 70 else x)
                if not top_sql_by_exectime.empty and top_sql_by_exectime['EXECTIME'].sum() > 0:
                    fig_top_sql_exectime = px.bar(top_sql_by_exectime, y='SQLSTATEM_SHORT', x='EXECTIME', orientation='h',
                                                    title="Top 10 Requêtes SQL par Temps d'Exécution Total",
                                                    labels={'SQLSTATEM_SHORT': 'Instruction SQL', 'EXECTIME': 'Temps d\'Exécution Total'},
                                                    color='EXECTIME', color_continuous_scale=px.colors.sequential.Blues)
                    fig_top_sql_exectime.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_top_sql_exectime, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Requêtes SQL par Temps d'Exécution Total après filtrage.")
            else:
                st.info("Colonnes 'SQLSTATEM' ou 'EXECTIME' manquantes ou leur total est zéro/vide après filtrage.")

            st.subheader("Top 10 Requêtes SQL par Nombre Total d'Exécutions (TOTALEXEC)")
            st.markdown("""
                Ce graphique met en évidence les 10 requêtes SQL les plus fréquemment exécutées.
                Il est utile pour identifier les requêtes qui, même si elles ne sont pas individuellement lentes,
                peuvent avoir un impact significatif sur la performance globale en raison de leur volume d'exécution élevé.
                """)
            if 'SQLSTATEM' in df_sql_trace.columns and 'TOTALEXEC' in df_sql_trace.columns and df_sql_trace['TOTALEXEC'].sum() > 0:
                # Ensure TOTALEXEC is numeric here
                df_sql_trace['TOTALEXEC'] = pd.to_numeric(df_sql_trace['TOTALEXEC'], errors='coerce').fillna(0).astype(float)
                top_sql_by_totalexec = df_sql_trace.groupby('SQLSTATEM', as_index=False)['TOTALEXEC'].sum().nlargest(10, 'TOTALEXEC')
                top_sql_by_totalexec['SQLSTATEM_SHORT'] = top_sql_by_totalexec['SQLSTATEM'].apply(lambda x: x[:70] + '...' if len(x) > 70 else x)
                if not top_sql_by_totalexec.empty and top_sql_by_totalexec['TOTALEXEC'].sum() > 0:
                    fig_top_sql_totalexec = px.bar(top_sql_by_totalexec, y='SQLSTATEM_SHORT', x='TOTALEXEC', orientation='h',
                                                    title="Top 10 Requêtes SQL par Nombre Total d'Exécutions",
                                                    labels={'SQLSTATEM_SHORT': 'Instruction SQL', 'TOTALEXEC': 'Nombre Total d\'Exécutions'},
                                                    color='TOTALEXEC', color_continuous_scale=px.colors.sequential.Greens)
                    fig_top_sql_totalexec.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_top_sql_totalexec, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Requêtes SQL par Nombre Total d'Exécutions après filtrage.")
            else:
                st.info("Colonnes 'SQLSTATEM' ou 'TOTALEXEC' manquantes ou leur total est zéro/vide après filtrage.")

            st.subheader("Distribution du Temps par Exécution (TIMEPEREXE)")
            st.markdown("""
                Cette courbe de densité montre la répartition des temps d'exécution individuels par requête.
                Elle permet de comprendre si la plupart des exécutions sont rapides ou si certaines sont significativement plus lentes,
                indiquant des performances inégales.
                """)
            if 'TIMEPEREXE' in df_sql_trace.columns and df_sql_trace['TIMEPEREXE'].sum() > 0:
                # Ensure TIMEPEREXE is numeric here
                df_sql_trace['TIMEPEREXE'] = pd.to_numeric(df_sql_trace['TIMEPEREXE'], errors='coerce').fillna(0).astype(float)
                if df_sql_trace['TIMEPEREXE'].nunique() > 1:
                    fig_time_per_exe_dist = ff.create_distplot([df_sql_trace['TIMEPEREXE'].dropna()], ['TIMEPEREXE'],
                                                                bin_size=df_sql_trace['TIMEPEREXE'].std()/5 if df_sql_trace['TIMEPEREXE'].std() > 0 else 1,
                                                                show_rug=False, show_hist=False)
                    fig_time_per_exe_dist.update_layout(title_text="Distribution du Temps par Exécution",
                                                        xaxis_title='Temps par Exécution',
                                                        yaxis_title='Densité')
                    fig_time_per_exe_dist.data[0].line.color = 'darkgreen'
                    st.plotly_chart(fig_time_per_exe_dist, use_container_width=True)
                else:
                    st.info("La colonne 'TIMEPEREXE' contient des valeurs uniques ou est vide après filtrage, impossible de créer une courbe de densité.")
            else:
                st.info("Colonne 'TIMEPEREXE' manquante ou total est zéro/vide après filtrage.")

            st.subheader("Distribution du Temps Moyen par Enregistrement (AVGTPERREC) pour le serveur 'ECC-VE7-00'")
            st.markdown("""
                Cette courbe de densité montre la répartition du temps moyen par enregistrement spécifiquement pour le serveur "ECC-VE7-00".
                Elle permet d'analyser la cohérence des performances de ce serveur en termes de traitement des enregistrements.
                """)
            if 'SERVERNAME' in df_sql_trace.columns and 'AVGTPERREC' in df_sql_trace.columns:
                # Ensure AVGTPERREC is numeric here
                df_sql_trace['AVGTPERREC'] = pd.to_numeric(df_sql_trace['AVGTPERREC'], errors='coerce').fillna(0).astype(float)
                df_ecc_ve7_00 = df_sql_trace[df_sql_trace['SERVERNAME'].astype(str).str.contains('ECC-VE7-00', na=False, case=False)].copy()
                
                if not df_ecc_ve7_00.empty and df_ecc_ve7_00['AVGTPERREC'].sum() > 0:
                    avg_t_per_rec_data = df_ecc_ve7_00['AVGTPERREC'].dropna()
                    
                    if avg_t_per_rec_data.nunique() > 1:
                        fig_ecc_ve7_00_avg_time_dist = ff.create_distplot([avg_t_per_rec_data], ['AVGTPERREC'],
                                                                          bin_size=avg_t_per_rec_data.std()/5 if avg_t_per_rec_data.std() > 0 else 1,
                                                                          show_rug=False, show_hist=False)
                        fig_ecc_ve7_00_avg_time_dist.update_layout(title_text="Distribution du Temps Moyen par Enregistrement (AVGTPERREC) pour 'ECC-VE7-00'",
                                                                   xaxis_title='Temps Moyen par Enregistrement',
                                                                   yaxis_title='Densité')
                        fig_ecc_ve7_00_avg_time_dist.data[0].line.color = 'darkblue'
                        st.plotly_chart(fig_ecc_ve7_00_avg_time_dist, use_container_width=True)
                    else:
                        st.info("Données insuffisantes ou valeurs uniques pour créer une courbe de densité pour 'ECC-VE7-00' (AVGTPERREC).")
                else:
                    st.info("Aucune donnée valide pour le serveur 'ECC-VE7-00' ou la colonne 'AVGTPERREC' est vide/zéro après filtrage.")
            else:
                st.info("Colonnes 'SERVERNAME' ou 'AVGTPERREC' manquantes dans les données de traces SQL.")

            st.subheader("Top 10 Requêtes SQL par Temps Moyen par Exécution (TIMEPEREXE)")
            st.markdown("""
                Ce graphique identifie les 10 requêtes SQL qui prennent le plus de temps en moyenne à chaque exécution.
                Ceci est utile pour cibler les requêtes intrinsèquement lentes, même si elles ne sont pas exécutées très fréquemment.
                """)
            if 'SQLSTATEM' in df_sql_trace.columns and 'TIMEPEREXE' in df_sql_trace.columns and df_sql_trace['TIMEPEREXE'].sum() > 0:
                # Ensure TIMEPEREXE is numeric here
                df_sql_trace['TIMEPEREXE'] = pd.to_numeric(df_sql_trace['TIMEPEREXE'], errors='coerce').fillna(0).astype(float)
                top_sql_by_time_per_exe = df_sql_trace.groupby('SQLSTATEM', as_index=False)['TIMEPEREXE'].mean().nlargest(10, 'TIMEPEREXE')
                top_sql_by_time_per_exe['SQLSTATEM_SHORT'] = top_sql_by_time_per_exe['SQLSTATEM'].apply(lambda x: x[:70] + '...' if len(x) > 70 else x)
                if not top_sql_by_time_per_exe.empty and top_sql_by_time_per_exe['TIMEPEREXE'].sum() > 0:
                    fig_top_sql_time_per_exe = px.bar(top_sql_by_time_per_exe, y='SQLSTATEM_SHORT', x='TIMEPEREXE', orientation='h',
                                                    title="Top 10 Requêtes SQL par Temps Moyen par Exécution",
                                                    labels={'SQLSTATEM_SHORT': 'Instruction SQL', 'TIMEPEREXE': 'Temps Moyen par Exécution'},
                                                    color='TIMEPEREXE', color_continuous_scale=px.colors.sequential.Oranges)
                    fig_top_sql_time_per_exe.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_top_sql_time_per_exe, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Requêtes SQL par Temps Moyen par Exécution après filtrage.")
            else:
                st.info("Colonnes 'SQLSTATEM' ou 'TIMEPEREXE' manquantes ou leur total est zéro/vide après filtrage.")

            st.subheader("Top 10 Requêtes SQL par Nombre d'Enregistrements Traités (RECPROCNUM)")
            st.markdown("""
                Ce graphique montre les 10 requêtes SQL qui traitent le plus grand nombre d'enregistrements.
                Cela peut indiquer des requêtes qui accèdent à de grandes quantités de données, potentiellement optimisables
                par l'ajout d'index ou la refonte de la logique de récupération des données.
                """)
            if 'SQLSTATEM' in df_sql_trace.columns and 'RECPROCNUM' in df_sql_trace.columns and df_sql_trace['RECPROCNUM'].sum() > 0:
                # Ensure RECPROCNUM is numeric here
                df_sql_trace['RECPROCNUM'] = pd.to_numeric(df_sql_trace['RECPROCNUM'], errors='coerce').fillna(0).astype(float)
                top_sql_by_recprocnum = df_sql_trace.groupby('SQLSTATEM', as_index=False)['RECPROCNUM'].sum().nlargest(10, 'RECPROCNUM')
                top_sql_by_recprocnum['SQLSTATEM_SHORT'] = top_sql_by_recprocnum['SQLSTATEM'].apply(lambda x: x[:70] + '...' if len(x) > 70 else x)
                if not top_sql_by_recprocnum.empty and top_sql_by_recprocnum['RECPROCNUM'].sum() > 0:
                    fig_top_sql_recprocnum = px.bar(top_sql_by_recprocnum, y='SQLSTATEM_SHORT', x='RECPROCNUM', orientation='h',
                                                    title="Top 10 Requêtes SQL par Nombre d'Enregistrements Traités",
                                                    labels={'SQLSTATEM_SHORT': 'Instruction SQL', 'RECPROCNUM': 'Nombre d\'Enregistrements Traités'},
                                                    color='RECPROCNUM', color_continuous_scale=px.colors.sequential.Purples)
                    fig_top_sql_recprocnum.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_top_sql_recprocnum, use_container_width=True)
                else:
                    st.info("Colonnes 'SQLSTATEM' ou 'RECPROCNUM' manquantes ou leur total est zéro/vide après filtrage.")

            st.subheader("Aperçu des Données de Traces SQL Filtrées")
            st.dataframe(df_sql_trace.head())
        else:
            st.warning("Données de traces SQL non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Analyse des Utilisateurs":
        # --- Onglet 8: Analyse des Utilisateurs (usr02_data.xlsx) ---
        st.header("👤 Analyse des Utilisateurs SAP")
        df_usr02 = dfs['usr02'].copy()
        if selected_accounts: # Filter BNAME (user name) by selected_accounts (ACCOUNT) if it aligns
            if 'BNAME' in df_usr02.columns:
                df_usr02 = df_usr02[df_usr02['BNAME'].isin(selected_accounts)]
            else:
                st.warning("La colonne 'BNAME' est manquante dans les données USR02 pour le filtrage par compte.")

        if not df_usr02.empty:
            st.subheader("Répartition des Types d'Utilisateurs (USTYP)")
            if 'USTYP' in df_usr02.columns:
                user_type_counts = df_usr02['USTYP'].value_counts().reset_index()
                user_type_counts.columns = ['Type d\'Utilisateur', 'Count']
                if not user_type_counts.empty:
                    fig_user_types = px.pie(user_type_counts,
                                            values='Count',
                                            names='Type d\'Utilisateur',
                                            title='Répartition des Types d\'Utilisateurs SAP',
                                            hole=0.3)
                    st.plotly_chart(fig_user_types, use_container_width=True)
                else:
                    st.info("Aucune donnée de type d'utilisateur disponible après filtrage.")
            else:
                st.info("Colonne 'USTYP' manquante dans le DataFrame USR02.")

            st.subheader("Nombre d'Utilisateurs par Date de Dernier Logon")
            if 'GLTGB_DATE' in df_usr02.columns and not df_usr02['GLTGB_DATE'].isnull().all():
                # Compter les utilisateurs par date de logon, en ignorant les NaT
                logon_dates_counts = df_usr02['GLTGB_DATE'].value_counts().sort_index().reset_index()
                logon_dates_counts.columns = ['Date', 'Nombre d\'Utilisateurs']
                
                if not logon_dates_counts.empty and logon_dates_counts['Nombre d\'Utilisateurs'].sum() > 0:
                    fig_logon_dates = px.line(logon_dates_counts,
                                            x='Date',
                                            y='Nombre d\'Utilisateurs',
                                            title='Nombre d\'Utilisateurs par Date de Dernier Logon',
                                            labels={'Date': 'Date de Dernier Logon', 'Nombre d\'Utilisateurs': 'Nombre d\'Utilisateurs'})
                    fig_logon_dates.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        type="date"
                    )
                    st.plotly_chart(fig_logon_dates, use_container_width=True)
                else:
                    st.info("Aucune donnée de date de dernier logon valide après filtrage ou la somme des utilisateurs est zéro.")
            else:
                st.info("Aucune donnée de date de dernier logon valide après filtrage.")

            # NOUVEAU BLOC : Utilisateurs sans Dernier Logon Récent
            st.subheader("Utilisateurs sans Dernier Logon Récent (GLTGB_DATE)")
            if 'GLTGB_DATE' in df_usr02.columns and pd.api.types.is_datetime64_any_dtype(df_usr02['GLTGB_DATE']):
                # Correction pour s'assurer que current_date est un objet datetime.date pour comparaison
                # C'est implicitement géré par pd.Timestamp.now().date() puis pd.to_datetime
                current_date = pd.to_datetime(pd.Timestamp.now().date())
                inactive_threshold_months = st.slider("Considérer inactif si pas de logon depuis (mois):", 1, 24, 6)
                
                inactive_date = current_date - pd.DateOffset(months=inactive_threshold_months)
                
                inactive_users = df_usr02[df_usr02['GLTGB_DATE'] < inactive_date]

                if not inactive_users.empty:
                    st.warning(f"🚨 **{len(inactive_users)}** utilisateurs n'ont pas effectué de logon depuis le {inactive_date.strftime('%Y-%m-%d')}.")
                    st.dataframe(inactive_users[['BNAME', 'GLTGB_DATE', 'USTYP']].sort_values(by='GLTGB_DATE'))
                    
                    # Graphique des logons par date pour les utilisateurs inactifs
                    logon_dates = inactive_users['GLTGB_DATE'].value_counts().sort_index().reset_index()
                    logon_dates.columns = ['GLTGB_DATE', 'Count']
                    
                    # Filtrer les dates avant '1900-01-01' pour éviter les problèmes de visualisation de très vieilles dates
                    logon_dates = logon_dates[logon_dates['GLTGB_DATE'] > '1900-01-01']

                    if not logon_dates.empty and logon_dates['Count'].sum() > 0:
                        fig_logon_dates = px.line(logon_dates, x='GLTGB_DATE', y='Count', 
                                                title=f"Distribution des Derniers Logons pour Utilisateurs Inactifs (depuis {inactive_threshold_months} mois)",
                                                labels={'GLTGB_DATE': 'Date de Dernier Logon', 'Count': 'Nombre d\'Utilisateurs'},
                                                color_discrete_sequence=['red'])
                        fig_logon_dates.update_xaxes(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                    dict(step="all")
                                ])
                            ),
                            rangeslider=dict(visible=True),
                            type="date"
                        )
                        
                        st.plotly_chart(fig_logon_dates, use_container_width=True)
                    else:
                        st.info("Aucune donnée de date de dernier logon valide après filtrage ou la somme des utilisateurs est zéro pour les utilisateurs inactifs.")
                else:
                    st.info("Aucun utilisateur inactif détecté selon le seuil défini.")
            else:
                st.info("Colonne 'GLTGB_DATE' manquante ou ne contient pas de dates valides après filtrage pour la détection d'utilisateurs inactifs.")

            st.subheader("Aperçu des Données Utilisateurs Filtrées")
            st.dataframe(df_usr02.head())
        else:
            st.warning("Données utilisateurs (USR02) non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Détection d'Anomalies":
        st.header("🚨 Détection d'Anomalies")
        st.write("Cette section est dédiée à l'analyse et la détection des anomalies dans vos données SAP.")

        if not dfs['hitlist_db'].empty and 'RESPTI' in dfs['hitlist_db'].columns and 'FULL_DATETIME' in dfs['hitlist_db'].columns:
            df_respti = dfs['hitlist_db'].copy()
            # Appliquer les filtres globaux à ce DataFrame aussi
            if selected_accounts:
                df_respti = df_respti[df_respti['ACCOUNT'].isin(selected_accounts)]
            if selected_reports:
                df_respti = df_respti[df_respti['REPORT'].isin(selected_reports)]
            if selected_tasktypes:
                df_respti = df_respti[df_respti['TASKTYPE'].isin(selected_tasktypes)]

            df_respti['RESPTI'] = pd.to_numeric(df_respti['RESPTI'], errors='coerce').fillna(0).astype(float)
            df_respti = df_respti.dropna(subset=['FULL_DATETIME', 'RESPTI'])

            if not df_respti.empty and df_respti['RESPTI'].sum() > 0:
                st.subheader("Anomalies dans le Temps de Réponse (RESPTI)")

                mean_respti = df_respti['RESPTI'].mean()
                std_respti = df_respti['RESPTI'].std()
                
                # Seuil configurable par l'utilisateur
                std_dev_multiplier = st.slider("Multiplicateur d'écart-type pour le seuil d'anomalie :", 1.0, 5.0, 3.0, 0.1)
                anomaly_threshold = mean_respti + std_dev_multiplier * std_respti
                
                st.info(f"Seuil d'anomalie pour RESPTI (Moyenne + {std_dev_multiplier}*StdDev) : **{anomaly_threshold:.2f} ms**")

                anomalies_respti = df_respti[df_respti['RESPTI'] > anomaly_threshold]

                if not anomalies_respti.empty:
                    st.warning(f"⚠️ **Anomalies détectées** : **{len(anomalies_respti)}** transactions avec des temps de réponse anormalement élevés.")
                    st.dataframe(anomalies_respti[['FULL_DATETIME', 'RESPTI', 'ACCOUNT', 'REPORT']].sort_values(by='RESPTI', ascending=False))

                    # Visualisation améliorée des anomalies
                    fig_anomalies_respti = px.scatter(df_respti, x='FULL_DATETIME', y='RESPTI',
                                                    title='Temps de Réponse (RESPTI) avec Anomalies Mises en Évidence',
                                                    labels={'FULL_DATETIME': 'Date et Heure', 'RESPTI': 'Temps de Réponse (ms)'},
                                                    color_discrete_sequence=['blue']) # Couleur par défaut pour les points normaux
                    
                    # Ajouter le seuil comme ligne
                    fig_anomalies_respti.add_hline(y=anomaly_threshold, line_dash="dash",
                                                    annotation_text=f"Seuil d'Anomalie ({anomaly_threshold:.0f} ms)",
                                                    annotation_position="bottom right",
                                                    line_color="red",
                                                    line_width=2)
                    
                    # Ajouter les points d'anomalie en rouge vif et plus grands
                    if not anomalies_respti.empty:
                        fig_anomalies_respti.add_scatter(x=anomalies_respti['FULL_DATETIME'], y=anomalies_respti['RESPTI'],
                                                        mode='markers', name='Anomalie',
                                                        marker=dict(color='red', size=10, symbol='star', line=dict(width=1, color='DarkRed')))
                    
                    fig_anomalies_respti.update_layout(hovermode="x unified") # Améliore l'interaction au survol
                    fig_anomalies_respti.update_xaxes(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1h", step="hour", stepmode="backward"),
                                dict(count=6, label="6h", step="hour", stepmode="backward"),
                                dict(count=1, label="1j", step="day", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                    
                    st.plotly_chart(fig_anomalies_respti, use_container_width=True)

                else:
                    st.success("✅ Aucune anomalie majeure détectée pour le temps de réponse (RESPTI) selon le seuil défini.")
            else:
                st.info("Données 'RESPTI' ou 'FULL_DATETIME' vides ou n'ont pas de somme positive après filtrage dans 'hitlist_db'. Impossible d'effectuer la détection d'anomalies.")
        else:
            st.info("Le DataFrame 'hitlist_db' ou les colonnes 'RESPTI'/'FULL_DATETIME' ne sont pas disponibles ou contiennent des données invalides. Impossible d'effectuer la détection d'anomalies sur le temps de réponse.")
        
        st.markdown("---")


# Option pour afficher tous les DataFrames (utile pour le débogage)
with st.expander("🔍 Afficher tous les DataFrames chargés (pour débogage)"):
    for key, df in dfs.items():
        st.subheader(f"DataFrame: {key} (Taille: {len(df)} lignes)")
        st.dataframe(df.head())
        # Mise à jour de la checkbox avec une clé unique et un label plus clair
        if st.checkbox(f"Afficher les informations de '{key}' (df.info())", key=f"info_{key}"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            st.write(f"...")
            st.write(f"...")