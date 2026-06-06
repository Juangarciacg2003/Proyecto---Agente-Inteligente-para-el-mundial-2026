import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Librerías de LangChain y OpenAI/OpenRouter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

# Importación de las dos fuentes de internet distintas para RAG en Tiempo Real
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

# =====================================================================
# CONFIGURACIÓN DE CREDENCIALES
# =====================================================================
ruta_env = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ruta_env)

api_key = os.getenv("OPENROUTE_API_KEY") or os.getenv("OPENROUTE_APIKEY")

# =====================================================================
# CONFIGURACIÓN DE PÁGINA (DEBE IR PRIMERO)
# =====================================================================
st.set_page_config(
    page_title="AlonBot · Mundial 2026",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS CANCHA PROFESIONAL ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --verde-cancha:   #0d5c2f;
    --verde-oscuro:   #083d1e;
    --verde-claro:    #1a7a40;
    --amarillo:       #FFD100;
    --azul-col:       #0033A0;
    --rojo-col:       #E40428;
    --blanco:         #f0f4f0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--verde-oscuro) !important;
}

/* Fondo cancha con líneas y gradiente */
.stApp {
    background-color: var(--verde-oscuro) !important;
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px),
        radial-gradient(ellipse at 50% 0%, rgba(26,122,64,0.4) 0%, transparent 60%);
    background-size: 60px 60px, 60px 60px, 100% 100%;
}

/* Ocultar chrome de Streamlit */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 1rem 2rem !important;
    max-width: 760px !important;
}

/* ── HEADER ── */
.alonbot-header {
    background: linear-gradient(135deg, var(--verde-cancha) 0%, var(--verde-oscuro) 100%);
    border: 1px solid rgba(255,209,0,0.2);
    border-radius: 20px;
    margin-bottom: 1.4rem;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,209,0,0.1);
    position: relative;
}
.alonbot-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--azul-col), var(--amarillo), var(--rojo-col));
}
.header-inner {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    padding: 1.2rem 1.6rem;
}
.header-text h1 {
    font-family: 'Archivo Black', sans-serif;
    font-size: 2rem;
    color: var(--amarillo);
    margin: 0;
    letter-spacing: 1px;
    text-shadow: 0 2px 12px rgba(255,209,0,0.3);
    line-height: 1;
}
.header-text p {
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    margin: 4px 0 8px;
    font-weight: 500;
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
}
.badge-azul  { background: var(--azul-col); color: var(--amarillo); }
.badge-rojo  { background: var(--rojo-col); color: white; }
.badge-verde { background: rgba(255,255,255,0.1); color: rgba(255,255,255,0.8);
               border: 1px solid rgba(255,255,255,0.2); }

/* ── CHAT MESSAGES ── */
/* Contenedor general del chat */
[data-testid="stChatMessageContainer"] {
    background: rgba(0,0,0,0.25) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px) !important;
}

/* Burbuja usuario */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(0,51,160,0.5), rgba(0,68,204,0.4)) !important;
    border: 1px solid rgba(0,51,160,0.4) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 10px 16px !important;
    margin-left: auto !important;
    box-shadow: 0 4px 12px rgba(0,51,160,0.3) !important;
}

/* Burbuja bot */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,209,0,0.12) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 10px 16px !important;
    box-shadow: none !important;
}

/* Texto del chat */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: var(--blanco) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

[data-testid="stChatMessage"] strong {
    color: var(--amarillo) !important;
}

/* ── INPUT CHAT ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(255,209,0,0.3) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--amarillo) !important;
    box-shadow: 0 0 0 3px rgba(255,209,0,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.35) !important;
}
[data-testid="stChatInput"] button {
    background: var(--amarillo) !important;
    border-radius: 10px !important;
    color: var(--verde-oscuro) !important;
}

/* ── STATUS BOX ── */
[data-testid="stStatus"] {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,209,0,0.2) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.8) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,209,0,0.3); border-radius: 4px; }

/* ── ERROR BOX ── */
[data-testid="stAlert"] {
    background: rgba(228,4,40,0.12) !important;
    border: 1px solid rgba(228,4,40,0.35) !important;
    border-radius: 12px !important;
    color: #ff8a8a !important;
}
</style>
""", unsafe_allow_html=True)

# ── LOGO DESDE ASSETS ────────────────────────────────────────────────────────
ruta_logo = Path(__file__).parent / "Assets" / "logo.png"

# ── HEADER ───────────────────────────────────────────────────────────────────
# ── HEADER ───────────────────────────────────────────────────────────────────
col_logo, col_texto = st.columns([1, 2.5])

with col_logo:
    if ruta_logo.exists():
        st.image(str(ruta_logo), use_container_width=True)

with col_texto:
    st.markdown("""
    <div style="padding: 1rem 0;">
      <h1 style="font-family:'Archivo Black',sans-serif; font-size:2.2rem;
                 color:#FFD100; margin:0; text-shadow:0 2px 12px rgba(255,209,0,0.3);">
        AlonBot
      </h1>
      <p style="color:rgba(255,255,255,0.6); font-size:12px; margin:4px 0 10px;">
        El agente experto en la Tricolor y el Mundial 2026 · Conectado a Internet 🌐
      </p>
      <span class="badge badge-azul">🇨🇴 Colombia</span>
      <span class="badge badge-rojo">🏆 Mundial 2026</span>
      <span class="badge badge-verde">⚡ IA en vivo</span>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# VALIDACIÓN DE API KEY (después del header para que se vea bonito)
# =====================================================================
if not api_key:
    st.error("❌ ¡Falta la API Key! Revisa que el archivo .env tenga la variable OPENROUTE_API_KEY.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# =====================================================================
# INICIALIZACIÓN DE COMPONENTES CORE (sin cambios)
# =====================================================================
@st.cache_resource
def inicializar_agente():
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.7,
        default_headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AlonBot Web Agent"
        }
    )
    wikipedia    = WikipediaAPIWrapper(lang="es", top_k_results=1, doc_content_chars_max=4000)
    buscador_web = DuckDuckGoSearchRun()
    return llm, wikipedia, buscador_web

llm, wikipedia, buscador_web = inicializar_agente()

@tool
def consultar_fuentes_internet(jugador_o_tema: str) -> str:
    """Utiliza esta herramienta cuando necesites verificar datos biográficos, de debut,
    fichajes o estadísticas. Consulta múltiples fuentes en tiempo real."""
    try:    res_wiki = wikipedia.run(jugador_o_tema)
    except: res_wiki = "No disponible en Wikipedia."
    try:    res_web  = buscador_web.run(jugador_o_tema)
    except: res_web  = "No disponible en la búsqueda web."
    return f"-- FUENTE 1: WIKIPEDIA --\n{res_wiki}\n\n-- FUENTE 2: WEB LIVE --\n{res_web}"

herramientas         = [consultar_fuentes_internet]
llm_con_herramientas = llm.bind_tools(herramientas)

# =====================================================================
# PROMPTS DE CONFIGURACIÓN (sin cambios)
# =====================================================================
system_template = """# ROLE AND IDENTITY
Eres "AlonBot", el asistente virtual definitivo y experto de nivel élite en fútbol y deportes globales. Tu especialidad absoluta es la Copa Mundial de la FIFA 2026 (EE. UU., México y Canadá) y todo lo relacionado con la Selección Colombia de Fútbol (la "Tricolor").

# TONAL GUIDELINES & PERSONALITY
- Friendly and Approachable: Hablas con entusiasmo futbolero, eres dinámico y cercano. Puedes usar términos amigables comunes en el fútbol.
- Objective yet Patriotic: Tu análisis técnico es 100% objetivo y basado en hechos. Sin embargo, tu corazón y lealtad están firmemente inclinados hacia la Selección Colombia. Siempre muestras apoyo, optimismo y orgullo por la "Tricolor".

# SAFETY AND GUARDRAILS
1. Zero Tolerance for Profanity: Prohibido el uso de groserías o insultos. Mantén siempre el profesionalismo.
2. Moderate User Input: Si hay faltas de respeto, redirige diciendo: "Mantengamos el juego limpio en la cancha...".

# DOMAIN KNOWLEDGE BASE (CONTEXT: CRITICAL YEAR 2026)
- Año Actual: Estás parado en el año 2026. No hables en futuro sobre eventos que ya pasaron.
- Campeón Defensor Real: El campeón defensor actual de la Copa Mundial de la FIFA es ARGENTINA (ganador de Qatar 2022). Francia fue el campeón del 2018, pero YA NO es el defensor.
- Copa Mundial 2026: Celebrada del 11 de junio al 19 de julio de 2026. 48 selecciones, 12 grupos de 4 equipos (Grupo K: Colombia, Uzbekistán y RD Congo).
- Selección Colombia: Plantilla del año 2026 con James Rodríguez, Luis Díaz, Daniel Muñoz, Jhon Arias, Richard Ríos, etc. Clasificó en tercer lugar.

# CRITICAL ANCHORING
Si el usuario hace preguntas en tiempo presente como "¿Quién es el campeón?", "¿Quién juega hoy?" o "¿Cuál es el ranking?", asume SIEMPRE de forma obligatoria que se refiere al contexto del año 2026.

# RESPONSE FORMATTING
- Usa viñetas o listas para desglosar datos o estadísticas.
- Utiliza texto en **negrita** para resaltar nombres de equipos, jugadores clave o fechas.

# INSTRUCCIONES DE VERIFICACIÓN CRUZADA
Cuando recibas información de la herramienta, actúa como un periodista: compara Wikipedia y la Web viva. Si hay contradicciones, busca términos como 'cantera' o 'debut profesional' para desempatar."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt  = HumanMessagePromptTemplate.from_template("{input}")

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    MessagesPlaceholder(variable_name="history"),
    human_message_prompt
])

chain = chat_prompt | llm_con_herramientas

# =====================================================================
# GESTIÓN DEL HISTORIAL (sin cambios en lógica)
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant", avatar="⚽"):
            st.markdown(message.content)

# =====================================================================
# FLUJO DE INTERACCIÓN (sin cambios en lógica)
# =====================================================================
if user_input := st.chat_input("Escribe tu consulta futbolera aquí... ⚽"):

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))

    try:
        respuesta_modelo = chain.invoke({
            "input": user_input,
            "history": st.session_state.messages
        })

        if respuesta_modelo.tool_calls:
            llamada_herramienta = respuesta_modelo.tool_calls[0]
            id_llamada          = llamada_herramienta["id"]
            nombre_funcion      = llamada_herramienta["name"]
            argumentos          = llamada_herramienta["args"]

            if nombre_funcion == "consultar_fuentes_internet":
                st.session_state.messages.append(respuesta_modelo)

                with st.status("🌐 AlonBot está consultando múltiples fuentes de internet...", expanded=True) as status:
                    query_busqueda = argumentos.get("jugador_o_tema", user_input)
                    st.write(f"🔍 Buscando: `{query_busqueda}`...")
                    reporte_de_internet = consultar_fuentes_internet.invoke(query_busqueda)
                    st.write("✍️ Cruzando datos y analizando contradicciones históricas...")
                    status.update(label="✅ Datos de internet recuperados", state="complete", expanded=False)

                mensaje_herramienta = ToolMessage(content=reporte_de_internet, tool_call_id=id_llamada)
                st.session_state.messages.append(mensaje_herramienta)

                respuesta_final = chain.invoke({
                    "input": "Por favor, analiza el reporte de la herramienta que te acabo de entregar y dale la respuesta final al usuario basándote en esos datos.",
                    "history": st.session_state.messages
                })

                st.session_state.messages.append(respuesta_final)
                with st.chat_message("assistant", avatar="⚽"):
                    st.markdown(respuesta_final.content)
        else:
            st.session_state.messages.append(respuesta_modelo)
            with st.chat_message("assistant", avatar="⚽"):
                st.markdown(respuesta_modelo.content)

    except Exception as e:
        st.error(f"❌ Ocurrió un error en la jugada del agente: {e}")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<p style="text-align:center; color:rgba(255,255,255,0.2); font-size:11px; margin-top:2rem; font-family:'DM Sans',sans-serif;">
  AlonBot v2.0 · GPT-4o Mini + LangChain + RAG · Hecho con 🇨🇴 pasión tricolor
</p>
""", unsafe_allow_html=True)