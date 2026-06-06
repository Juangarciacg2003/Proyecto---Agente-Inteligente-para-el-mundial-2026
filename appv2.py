import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Librerías de LangChain y OpenAI/OpenRouter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# CORRECCIÓN DE IMPORTS: Añadimos HumanMessage y AIMessage para evitar caídas en el Session State
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

if not api_key:
    st.error("❌ ¡Falta la API Key! Revisa que el archivo .env tenga la variable OPENROUTE_API_KEY.")
    st.stop()

# Sincronización para el cliente interno de OpenAI
os.environ["OPENAI_API_KEY"] = api_key

# =====================================================================
# CONFIGURACIÓN DE LA INTERFAZ DE STREAMLIT (UI)
# =====================================================================
st.set_page_config(
    page_title="AlonBot - Selección Colombia 2026",
    page_icon="⚽",
    layout="centered"
)

# MEJORA DE RUTA: Cambiamos la ruta absoluta rígida por una ruta relativa dinámica y limpia.
# Así tu código funcionará en cualquier computador o servidor sin romperse.
ruta_logo = Path(__file__).parent / "Assets" / "logo.png"

col1, col2 = st.columns([1, 4])
with col1:
    if ruta_logo.exists():
        st.image(str(ruta_logo), use_container_width=True)
    else:
        st.title("🤖") # Avatar temporal si no encuentra el archivo en la carpeta Assets
with col2:
    st.title("AlonBot")
    st.caption("El agente experto en la Tricolor y el Mundial 2026 | Conectado a Internet 🌐")

# =====================================================================
# INICIALIZACIÓN DE COMPONENTES CORE
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
    
    wikipedia = WikipediaAPIWrapper(lang="es", top_k_results=1, doc_content_chars_max=4000)
    buscador_web = DuckDuckGoSearchRun()
    
    return llm, wikipedia, buscador_web

llm, wikipedia, buscador_web = inicializar_agente()

# Definición de la Herramienta integrada
@tool
def consultar_fuentes_internet(jugador_o_tema: str) -> str:
    """Utiliza esta herramienta cuando necesites verificar datos biográficos, de debut, 
    fichajes o estadísticas. Consulta múltiples fuentes en tiempo real."""
    try:
        res_wiki = wikipedia.run(jugador_o_tema)
    except Exception:
        res_wiki = "No disponible en Wikipedia."
        
    try:
        res_web = buscador_web.run(f"{jugador_o_tema} debut equipo profesional")
    except Exception:
        res_web = "No disponible en la búsqueda web."
        
    return f"-- FUENTE 1: WIKIPEDIA --\n{res_wiki}\n\n-- FUENTE 2: WEB LIVE --\n{res_web}"

herramientas = [consultar_fuentes_internet]
llm_con_herramientas = llm.bind_tools(herramientas)

# =====================================================================
# PROMPTS DE CONFIGURACIÓN
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
Si el usuario hace preguntas en tiempo presente como "¿Quién es el campeón?", "¿Quién juega hoy?" o "¿Cuál es el ranking?", asume SIEMPRE de forma obligatoria que se refiere al contexto del año 2026 y ajusta tus respuestas o activa la herramienta de búsqueda para no usar datos desactualizados.

# RESPONSE FORMATTING
- Usa viñetas o listas para desglosar datos o estadísticas.
- Utiliza texto en **negrita** para resaltar nombres de equipos, jugadores clave o fechas.

# INSTRUCCIONES DE VERIFICACIÓN CRUZADA
Cuando recibas información de la herramienta, actúa como un periodista: compara Wikipedia y la Web viva. Si hay contradicciones sobre el debut o carrera de un jugador, busca términos como 'cantera' o 'debut profesional' para desempatar con ojo crítico."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    MessagesPlaceholder(variable_name="history"), 
    human_message_prompt
])

chain = chat_prompt | llm_con_herramientas

# =====================================================================
# GESTIÓN DEL HISTORIAL DE CONVERSACIÓN (Streamlit Session State)
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dibujar los mensajes históricos en la pantalla cada vez que se recarga la app
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and message.content: # Solo pintar si tiene texto
        with st.chat_message("assistant", avatar="⚽"):
            st.markdown(message.content)

# =====================================================================
# FLUJO DE INTERACCIÓN EN TIEMPO REAL
# =====================================================================
if user_input := st.chat_input("Escribe tu consulta futbolera aquí..."):
    
    # 1. Mostrar la pregunta del usuario en la pantalla inmediatamente
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Guardar la pregunta en el estado de la sesión
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    try:
        # 2. Primera llamada al modelo pasando el historial acumulado de la web
        respuesta_modelo = chain.invoke({
            "input": user_input,
            "history": st.session_state.messages
        })
        
        # 3. Validar Routing (¿El modelo necesita usar herramientas?)
        if respuesta_modelo.tool_calls:
            llamada_herramienta = respuesta_modelo.tool_calls[0]
            id_llamada = llamada_herramienta["id"]
            nombre_funcion = llamada_herramienta["name"]
            argumentos = llamada_herramienta["args"]
            
            if nombre_funcion == "consultar_fuentes_internet":
                st.session_state.messages.append(respuesta_modelo)
                
                # Desplegar un componente animado en Streamlit mientras busca en internet
                with st.status("🌐 AlonBot está consultando múltiples fuentes de internet...", expanded=True) as status:
                    query_busqueda = argumentos.get("jugador_o_tema", user_input)
                    st.write(f"🔍 Buscando información en Wikipedia y DuckDuckGo para: `{query_busqueda}`...")
                    
                    reporte_de_internet = consultar_fuentes_internet.invoke(query_busqueda)
                    
                    st.write("✍️ Cruzando datos y analizando posibles contradicciones históricas...")
                    status.update(label="✅ Datos de internet recuperados", state="complete", expanded=False)
                
                # Crear el ToolMessage formal requerido por el proveedor
                mensaje_herramienta = ToolMessage(
                    content=reporte_de_internet, 
                    tool_call_id=id_llamada
                )
                st.session_state.messages.append(mensaje_herramienta)
                
                # Segunda llamada al modelo con los datos inyectados
                respuesta_final = chain.invoke({
                    "input": "Por favor, analiza el reporte de la herramienta que te acabo de entregar y dale la respuesta final al usuario basándote en esos datos.",
                    "history": st.session_state.messages
                })
                
                st.session_state.messages.append(respuesta_final)
                with st.chat_message("assistant", avatar="⚽"):
                    st.markdown(respuesta_final.content)
        else:
            # Respuesta directa (charla casual o memoria interna)
            st.session_state.messages.append(respuesta_modelo)
            with st.chat_message("assistant", avatar="⚽"):
                st.markdown(respuesta_modelo.content)
                
    except Exception as e:
        st.error(f"❌ Ocurrió un error en la jugada del agente: {e}")