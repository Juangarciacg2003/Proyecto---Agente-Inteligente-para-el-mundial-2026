import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Librerías de LangChain y OpenAI/OpenRouter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

# Importación de las dos fuentes de internet distintas para RAG (Recuperación de Datos en Tiempo Real)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
# =====================================================================
# CONFIGURACIÓN DE CREDENCIALES (Consola Local)
# =====================================================================
# Carga automática buscando el .env en la misma carpeta donde está este script
ruta_env = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ruta_env)

# Captura de la llave con el nombre exacto de tu archivo
api_key = os.getenv("OPENROUTE_API_KEY") or os.getenv("OPENROUTE_APIKEY")

if not api_key:
    raise RuntimeError("¡Falta la API Key! Revisa que el archivo .env tenga la variable OPENROUTE_API_KEY.")

# Sincronización para el cliente interno de OpenAI
os.environ["OPENAI_API_KEY"] = api_key

# Configurar el modelo conectado a OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini", 
    openai_api_key=api_key,
    temperature=0.7
)
# INICIALIZADOR WIKIPEDIA (Ejemplo de integración de utilidad externa)
wikipedia = WikipediaAPIWrapper(lang="es", top_k_results=1, doc_content_chars_max=2000)

# Fuente B: DuckDuckGo (Resultados de páginas web de deportes actuales, noticias, etc.)
buscador_web = DuckDuckGoSearchRun()
@tool
def consultar_fuentes_internet(jugador_o_tema: str) -> str:
    """Utiliza esta herramienta cuando necesites verificar datos biográficos, de debut, 
    fichajes o estadísticas. Esta función consulta Wikipedia y la web abierta simultáneamente 
    para traerte dos perspectivas distintas."""
    print(f"\n[AGENTE]  Rastreando internet por múltiples canales para: '{jugador_o_tema}'...")
    
    # 1. Consultamos Wikipedia
    try:
        res_wiki = wikipedia.run(jugador_o_tema)
    except Exception:
        res_wiki = "No disponible en Wikipedia."
        
    # 2. Consultamos la web abierta (DuckDuckGo)
    try:
        res_web = buscador_web.run(f"{jugador_o_tema} debut equipo profesional")
    except Exception:
        res_web = "No disponible en la búsqueda web."
        
    # 3. Empacamos ambas fuentes en un solo reporte para el LLM
    reporte_fuentes = f"""
    --- FUENTE 1: WIKIPEDIA ---
    {res_wiki}
    
    --- FUENTE 2: BÚSQUEDA WEB EN VIVO ---
    {res_web}
    """
    return reporte_fuentes

herramientas = [consultar_fuentes_internet]

# Ligamos las herramientas al modelo para que pueda usarlas durante la conversación
llm_con_herramientas = llm.bind_tools(herramientas)
# =====================================================================
# DEFINICIÓN DE PROMPTS Y PLANTILLAS
# =====================================================================
# Definir el mensaje del sistema (Identidad de AlonBot)
system_template = """# ROLE AND IDENTITY
Eres "AlonBot", el asistente virtual definitivo y experto de nivel élite en fútbol y deportes globales. Posees un conocimiento profundo sobre tácticas, historia deportiva, estadísticas y análisis de partidos. Tu especialidad absoluta es la Copa Mundial de la FIFA 2026 (Estados Unidos, México y Canadá) y todo lo relacionado con la Selección Colombia de Fútbol (la "Tricolor").

# TONAL GUIDELINES & PERSONALITY
- Friendly and Approachable: Hablas con entusiasmo futbolero, eres dinámico y cercano con el usuario. Puedes usar términos amigables comunes en el ámbito del fútbol.
- Objective yet Patriotic: Tu análisis técnico y de datos de todos los equipos del mundo debe ser 100% objetivo, riguroso y basado en hechos. Sin embargo, tu corazón y lealtad están firmemente inclinados hacia la Selección Colombia. Siempre debes mostrar apoyo, optimismo y orgullo por la "Tricolor" por encima de cualquier otro seleccionado nacional. No te está permitido bajo ninguna circunstancia inclinarte a favor de un rival de Colombia o de otra selección en competencia directa.

# SAFETY AND GUARDRAILS (STRICT RULES)
1. Zero Tolerance for Profanity: Tienes prohibido el uso de groserías, malas palabras, insultos o lenguaje vulgar, incluso si el usuario te incita o los utiliza primero. Mantén siempre el profesionalismo dentro de la amigabilidad.
2. Moderate User Input: Si el usuario te insulta, utiliza lenguaje inapropiado o falta al respeto a un equipo/jugador, debes redirigir la conversación con un filtro de cortesía diciendo algo como: "Mantengamos el juego limpio en la cancha. Hablemos de fútbol de forma respetuosa. ¿Qué te gustaría saber?".
3. No Other Favored Nations: Si te preguntan cuál es tu selección favorita o quién ganará el Mundial, tu respuesta de apoyo siempre debe enfocarse en la Selección Colombia, respaldando tus argumentos con sus fortalezas objetivas (ej. el nivel de Luis Díaz, Richard Ríos o la dirección técnica actual).

# DOMAIN KNOWLEDGE BASE (CONTEXT: YEAR 2026)
- Copa Mundial 2026: El torneo se celebra en EE. UU., México y Canadá del 11 de junio al 19 de julio de 2026. Participan 48 selecciones distribuidas en 12 grupos de 4 equipos. Conoces al detalle los estadios, el calendario de partidos, las fases de grupos (como el Grupo K donde Colombia comparte con Uzbekistán y RD Congo) y el formato del torneo que incluye la ronda de dieciseisavos de final.
- Selección Colombia: Conoces la actualidad total de la plantilla de la Tricolor en el año 2026 (convocatorias, alineaciones con figuras como James Rodríguez, Luis Díaz, Daniel Muñoz, Jhon Arias, Richard Ríos, etc.), su rendimiento en las eliminatorias (donde clasificó en tercer lugar) y su fixture oficial en la cita mundialista.

# RESPONSE FORMATTING
- Usa viñetas o listas para desglosar datos, fixtures, alineaciones o estadísticas.
- Utiliza texto en negrita para resaltar los nombres de equipos, jugadores clave o fechas importantes.
- Mantén las respuestas estructuradas, avoiding bloques densos de texto para que la lectura sea ágil y escaneable.

# INSTRUCCIONES DE VERIFICACIÓN CRUZADA (CONSENSO DE FUENTES)
Cuando uses la herramienta `consultar_fuentes_internet`, recibirás dos bloques de información de internet. Tu trabajo obligatorio es actuar como un periodista de investigación:
1. Compara lo que dice la Fuente 1 (Wikipedia) con la Fuente 2 (Búsqueda Web).
2. Si ambas fuentes coinciden en un equipo o fecha, tómalo como una verdad confirmada.
3. Si las fuentes se contradicen (por ejemplo, una dice que debutó en Nacional y la otra en América), busca en los textos cuál menciona explícitamente "divisiones menores", "cantera" o "debut profesional" para desempatar. ¡No repitas el error si ves incongruencias! Sé escéptico.
4. Si la información sigue siendo dudosa en ambas fuentes, acláralo en tu respuesta con profesionalismo (ej: "Hay discrepancias en los registros de internet, pero la información apunta a...").
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Definir el mensaje del usuario
human_template = "{input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Crear Plantilla de chat con la importación moderna corregida y el placeholder para la memoria
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    MessagesPlaceholder(variable_name="history"), 
    human_message_prompt
])

# Crear la cadena de conversación base
chain = chat_prompt | llm_con_herramientas



# =====================================================================
# BUCLE INTERACTIVO DE LA CONSOLA
# =====================================================================
# MEMORIA Y BUCLE INTERACTIVO
# =====================================================================
store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

brain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def main():
    print("==========================================================")
    print("¡AlonBot AGENTE MULTI-FUENTE iniciado con éxito! Escribe 'salir'.")
    print("==========================================================")
    
    config = {"configurable": {"session_id": "consola_multirag"}}
    
    while True:
        user_input = input("\nTú: ")
        if user_input.lower() == 'salir':
            print("\nAlonBot: ¡Nos vemos en la cancha! 👋⚽")
            break
            
        if user_input.strip() != "":
            try:
                # 1. Capturamos el historial actual de la sesión
                historial_chat = get_session_history("consola_multirag")
                
                # 2. Agregamos de forma manual la pregunta del usuario al historial antes de invocar
                # Esto es necesario para que el flujo de ToolMessage encaje perfectamente
                historial_chat.add_user_message(user_input)
                
                # Invocamos la cadena base (el prompt ya se armará con el historial actualizado)
                respuesta_modelo = chain.invoke(
                    {
                        "input": user_input,
                        "history": historial_chat.messages
                    },
                    config=config
                )
                
                # 3. Routing de la herramienta compuesta
                if respuesta_modelo.tool_calls:
                    # El modelo nos dice qué herramienta quiere usar y su ID único
                    llamada_herramienta = respuesta_modelo.tool_calls[0]
                    id_llamada = llamada_herramienta["id"]
                    nombre_funcion = llamada_herramienta["name"]
                    argumentos = llamada_herramienta["args"]
                    
                    if nombre_funcion == "consultar_fuentes_internet":
                        # Guardamos el mensaje del asistente (que contiene la llamada a la herramienta) en el historial
                        historial_chat.add_message(respuesta_modelo)
                        
                        # Ejecutamos la función de Python en vivo
                        query_busqueda = argumentos.get("jugador_o_tema", user_input)
                        reporte_de_internet = consultar_fuentes_internet.invoke(query_busqueda)
                        
                        print("[AGENTE]  Analizando contradicciones y cruzando datos de internet...")
                        
                        
                        mensaje_herramienta = ToolMessage(
                            content=reporte_de_internet, 
                            tool_call_id=id_llamada
                        )
                        historial_chat.add_message(mensaje_herramienta)
                        
                        # Volvemos a invocar el modelo pasándole TODO el historial estructurado
                        # El modelo verá: Usuario -> Assistant(ToolCall) -> ToolMessage
                        respuesta_final = chain.invoke(
                            {
                                "input": "Por favor, analiza el reporte de la herramienta que te acabo de entregar y dale la respuesta final al usuario basándote en esos datos.",
                                "history": historial_chat.messages
                            },
                            config=config
                        )
                        
                        # Guardamos la respuesta definitiva en el historial y la mostramos
                        historial_chat.add_message(respuesta_final)
                        print(f"\nAlonBot:\n{respuesta_final.content}")
                else:
                    # Si no usó herramientas, guardamos su respuesta directa en el historial
                    historial_chat.add_message(respuesta_modelo)
                    print(f"\nAlonBot:\n{respuesta_modelo.content}")
                    
            except Exception as e:
                print(f"\n Error en el proceso: {e}")

if __name__ == "__main__":
    main()