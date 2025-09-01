import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect, text
import torch
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer, AutoModelForCausalLM
import io


st.set_page_config(
    page_title="Text-to-SQL dengan Phi-3",
    page_icon="🤖",
    layout="wide"
)

# --- 1. Memuat Model Phi-3 ---
@st.cache_resource
def load_model_pipeline():
    """Memuat model Phi-3-mini yang terkuantisasi dan pipeline-nya."""
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    with st.spinner("⏳ Memuat model AI (Phi-3-mini)... Ini mungkin butuh beberapa menit."):
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                quantization_config=quantization_config
            )

            model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            return model_pipeline
        except Exception as e:
            st.error(f"Gagal memuat model: {e}. Pastikan Anda memiliki koneksi internet dan RAM yang cukup (rekomendasi > 8GB).")
            return None

pipe = load_model_pipeline()


def get_schema_string(db_engine):
    """Menghasilkan representasi skema database dalam bentuk string CREATE TABLE."""
    inspector = inspect(db_engine)
    schema_str = ""
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        schema_str += f"CREATE TABLE {table_name} (\n"
        for col in columns:
            schema_str += f"  {col['name']} {col['type']},\n"
        schema_str = schema_str.rstrip(',\n') + "\n);\n\n"
    return schema_str.strip()

def execute_query(sql_query, db_engine):
    """Mengeksekusi query SQL dan mengembalikan hasilnya."""
    try:
        with db_engine.connect() as connection:
            result_df = pd.read_sql_query(text(sql_query), connection)
        return result_df, None
    except Exception as e:
        return None, f"Terjadi error saat eksekusi: {e}"


st.title("🤖 Text-to-SQL Generator dengan Phi-3")
st.markdown("Unggah dataset Anda (CSV atau Excel), ajukan pertanyaan dalam bahasa alami, dan biarkan AI mengubahnya menjadi query SQL!")

if pipe is None:
    st.stop()

if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None

col1, col2 = st.columns(2)

with col1:
    st.header("1. Unggah Dataset Anda")
    uploaded_files = st.file_uploader(
        "Pilih file CSV atau Excel",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        help="Anda bisa mengunggah beberapa file. Setiap file akan menjadi satu tabel di database."
    )

    if uploaded_files:
        try:
            engine = create_engine("sqlite:///:memory:")
            table_names = []
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                table_name = file_name.split('.')[0].lower().replace(" ", "_")
                table_names.append(table_name)

                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df.to_sql(table_name, engine, index=False, if_exists='replace')
            
            st.session_state.db_engine = engine
            st.success(f"Berhasil! Tabel **{', '.join(table_names)}** dimuat ke database.")
            
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")
            st.session_state.db_engine = None

with col2:
    if st.session_state.db_engine:
        st.header("2. Lihat Skema & Ajukan Pertanyaan")
        db_schema = get_schema_string(st.session_state.db_engine)
        with st.expander("Skema Database (Klik untuk melihat)"):
            st.code(db_schema, language="sql")
        
        user_question = st.text_area("Masukkan pertanyaan Anda di sini:", height=100, placeholder="Contoh: Berapa total penjualan untuk setiap pelanggan?")
        
        if st.button("🚀 Hasilkan SQL", use_container_width=True):
            if not user_question:
                st.warning("Pertanyaan tidak boleh kosong.")
            else:
                prompt = f"""<|user|>
Anda adalah model text-to-SQL. Tugas Anda adalah menghasilkan satu query SQLite yang valid berdasarkan skema database dan pertanyaan pengguna.
- **Hanya keluarkan query SQL.**
- Jangan sertakan penjelasan, komentar, markdown, atau titik koma di akhir.

**Skema Database:**
```sql
{db_schema}
```

**Pertanyaan Pengguna:**
{user_question}<|end|>
<|assistant|>
"""
                with st.spinner("AI sedang berpikir... 🧠"):
                    model_output = pipe(prompt, max_new_tokens=250, do_sample=False, pad_token_id=pipe.tokenizer.eos_token_id)
                
                generated_text = model_output[0]['generated_text']
                sql_query = ""
                if "<|assistant|>" in generated_text:
                    sql_query_part = generated_text.split("<|assistant|>")
                    if len(sql_query_part) > 1:
                        sql_query = sql_query_part[1].strip()
                        sql_query = sql_query.replace("<|end|>", "").strip()
                        if sql_query.startswith("```sql"):
                            sql_query = sql_query[6:].strip()
                        if sql_query.endswith("```"):
                            sql_query = sql_query[:-3].strip()

                st.subheader("✨ Generated SQL Query")
                st.code(sql_query, language="sql")
                
                if sql_query:
                    st.subheader("📊 Hasil Query")
                    with st.spinner("Mengeksekusi query..."):
                        result_df, error = execute_query(sql_query, st.session_state.db_engine)
                    
                    if error:
                        st.error(error)
                    elif result_df is not None:
                        if result_df.empty:
                            st.info("Query berhasil dieksekusi, tetapi tidak ada hasil yang dikembalikan.")
                        else:
                            st.dataframe(result_df)
    else:
        st.info("Silakan unggah file dataset terlebih dahulu untuk memulai.")
