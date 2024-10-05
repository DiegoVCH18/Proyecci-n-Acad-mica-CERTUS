import gradio as gr
import joblib
import pandas as pd

# Cargar los modelos previamente entrenados
rf_model = joblib.load('modelo_RandomForestClassifier.pkl')
xgb_model_softmax = joblib.load('modelo_XGBClassifier.pkl')
modelo = joblib.load('modelo_xgb.pkl')
modelo_catboost = joblib.load('modelo_catboost.pkl')

# Cargar los encoders
label_encoder_sede = joblib.load('label_encoder_sede_estudiante.pkl')
label_encoder_programa = joblib.load('label_encoder_programa.pkl')
label_encoder_area = joblib.load('label_encoder_area.pkl')
label_encoder_turno = joblib.load('label_encoder_turno.pkl')

# Función de predicción con múltiples modelos
def predict(sede, programa, area, turno, evaluacion1):
    try:
        # Validación de entradas
        if evaluacion1 < 0 or evaluacion1 > 20:
            return "Error: La evaluación debe estar entre 0 y 20"

        # Transformar las entradas usando los encoders
        sede_encoded = label_encoder_sede.transform([sede])[0]
        programa_encoded = label_encoder_programa.transform([programa])[0]
        area_encoded = label_encoder_area.transform([area])[0]
        turno_encoded = label_encoder_turno.transform([turno])[0]

        # Preparar los datos para la predicción
        input_data = [[sede_encoded, programa_encoded, area_encoded, turno_encoded, evaluacion1]]

        # Realizar la predicción usando todos los modelos (aprobación, desaprobación e inasistencia)
        rf_prob_approval = rf_model.predict_proba(input_data)[0, 1] * 100
        rf_prob_inasistencia = rf_model.predict_proba(input_data)[0, 2] * 100
        rf_prob_desaprobacion = rf_model.predict_proba(input_data)[0, 0] * 100

        xgb_prob_approval = modelo.predict_proba(input_data)[0, 1] * 100
        xgb_prob_inasistencia = modelo.predict_proba(input_data)[0, 2] * 100
        xgb_prob_desaprobacion = modelo.predict_proba(input_data)[0, 0] * 100

        xgb_softmax_approval = xgb_model_softmax.predict_proba(input_data)[0, 1] * 100
        xgb_softmax_inasistencia = xgb_model_softmax.predict_proba(input_data)[0, 2] * 100
        xgb_softmax_desaprobacion = xgb_model_softmax.predict_proba(input_data)[0, 0] * 100

        catboost_prob_approval = modelo_catboost.predict_proba(input_data)[0, 1] * 100
        catboost_prob_inasistencia = modelo_catboost.predict_proba(input_data)[0, 2] * 100
        catboost_prob_desaprobacion = modelo_catboost.predict_proba(input_data)[0, 0] * 100

        # Crear el resultado agrupado en cuadros separados para aprobación, desaprobación e inasistencia
        approval_result = (f'<div style="border:1px solid #233269; padding:10px; margin-bottom:10px;">'
                           f'<h2 style="color: #233269;">Probabilidades de Aprobación</h2>'
                           f'<ul>'
                           f'<li>Probabilidad de aprobación (Random Forest): {rf_prob_approval:.2f}%</li>'
                           f'<li style="background-color: #feddb5; padding:5px;">Probabilidad de aprobación (XGBoost - Binary): {xgb_prob_approval:.2f}%</li>'  # Sombreado aquí
                           f'<li>Probabilidad de aprobación (XGBoost - Softmax): {xgb_softmax_approval:.2f}%</li>'
                           f'<li>Probabilidad de aprobación (CatBoost): {catboost_prob_approval:.2f}%</span></li>'
                           f'</ul>'
                           f'</div>')

        desaprobacion_result = (f'<div style="border:1px solid #ed4255; padding:10px; margin-bottom:10px;">'
                                f'<h2 style="color: #ed4255;">Probabilidades de Desaprobación</h2>'
                                f'<ul>'
                                f'<li>Probabilidad de desaprobación (Random Forest): {rf_prob_desaprobacion:.2f}%</li>'
                                f'<li style="background-color: #feddb5; padding:5px;">Probabilidad de desaprobación (XGBoost - Binary): {xgb_prob_desaprobacion:.2f}%</li>'  # Sombreado aquí
                                f'<li>Probabilidad de desaprobación (XGBoost - Softmax): {xgb_softmax_desaprobacion:.2f}%</li>'
                                f'<li>Probabilidad de desaprobación (CatBoost): {catboost_prob_desaprobacion:.2f}%</span></li>'
                                f'</ul>'
                                f'</div>')

        inasistencia_result = (f'<div style="border:1px solid #ad4992; padding:10px; margin-bottom:10px;">'
                               f'<h2 style="color: #ad4992;">Probabilidades de Inasistencia</h2>'
                               f'<ul>'
                               f'<li>Probabilidad de inasistencia (Random Forest): {rf_prob_inasistencia:.2f}%</li>'
                               f'<li style="background-color: #feddb5; padding:5px;">Probabilidad de inasistencia (XGBoost - Binary): {xgb_prob_inasistencia:.2f}%</li>'  # Sombreado aquí
                               f'<li>Probabilidad de inasistencia (XGBoost - Softmax): {xgb_softmax_inasistencia:.2f}%</li>'
                               f'<li>Probabilidad de inasistencia (CatBoost): {catboost_prob_inasistencia:.2f}%</span></li>'
                               f'</ul>'
                               f'</div>')

        # Combinar los resultados en el orden solicitado: aprobación, desaprobación, inasistencia
        return approval_result + desaprobacion_result + inasistencia_result

    except Exception as e:
        return f"Ocurrió un error: {str(e)}"

# Encabezado HTML y CSS
header = """
<div style="display: flex; align-items: center;">
    <img src="https://is2-ssl.mzstatic.com/image/thumb/Purple122/v4/8b/1b/81/8b1b81e6-93b2-d199-43af-939c64bd5238/source/512x512bb.jpg" alt="Certus Logo" style="width: 100px; margin-right: 20px;">
    <div>
        <h1 style="color: #16265a; margin: 0;">PROYECTO PYTHON PARA LA CIENCIA DE DATOS</h1>
        <h3 style="color: #16265a; margin: 0;">Programa: Data Science & Machine Learning  - Toulouse Lautrec</h3>
        <h3 style="color: #16265a; margin: 0;">Elaborado por: Ing. Diego Armando Vasquez Chavez</h3>
        <p style="color: #16265a; margin: 0;">CIP 337613</p>
    </div>
</div>


"""

# Crear la interfaz Gradio
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=[
            "AREQUIPA", "ATE", "CALLAO", "CHICLAYO", "NORTE",
            "SAN JUAN DE LURIGANCHO", "SURCO", "VILLA EL SALVADOR", "VIRTUAL"
        ], label="SEDE ESTUDIANTE"),
        gr.Dropdown(choices=[
            "Administración de Empresas",
            "Administración de Empresas y Gestión de Recursos Humanos",
            "Administración de Negocios Bancarios y Financieros",
            "Administración de Negocios Bancarios, Financieros y Banca Digital",
            "Administración de Negocios Internacionales",
            "Administración de Sistemas para la Transformación Digital",
            "Administración y Gestión Comercial",
            "Contabilidad",
            "Contabilidad y Tributación",
            "Diseño Gráfico",
            "Diseño y Desarrollo de Software",
            "Marketing",
            "Marketing y Gestión de medios digitales",
            "Publicidad"
        ], label="PROGRAMA"),
        gr.Dropdown(choices=[
            "CICLO 01", "CICLO 02", "CICLO 03", "CICLO 04", "CICLO 05", "CICLO 06"
        ], label="CICLO"),
        gr.Dropdown(choices=["MAÑANA", "DIURNO", "TARDE", "NOCHE"], label="TURNO"),
        gr.Slider(minimum=0, maximum=20, step=1, label="Nota de la Evaluación 1")
    ],
    outputs=gr.HTML(label="Probabilidades de Aprobación, Desaprobación e Inasistencia"),
    title="PREDICCIÓN DE APROBACIÓN, DESAPROBACIÓN E INASISTENCIA",
    description=header,
    css="""
        .gradio-container h1 {
            color: #16265a !important;
        }
        button[type="submit"] {
            background-color: #16265a !important;
            color: white !important;
        }
        .output-html div {
            font-size: 18px !important;
            font-weight: bold !important;
            color: #16265a !important;
        }
        .output-html ul {
            font-size: 16px !important;
            font-weight: bold !important;
            color: #16265a !important;
        }
        button[type="submit"]::after {
            content: 'Enviar';
        }
        .gradio-container button {
            text-transform: none !important;
        }
        .output-html div span {
            font-weight: bold !important;
        }
    """
)

# Ejecutar la interfaz
if __name__ == "__main__":
    iface.launch()
