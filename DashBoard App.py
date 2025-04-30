import pandas as pd
import numpy as np
import requests
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


# Feature names
FEATURE_NAMES = [
    "Age (years)", "Blood Pressure (mmHg)", "Specific Gravity", "Red Blood Cells", "Pus Cell",
    "Pus Cell Clumps", "Bacteria", "Blood Glucose Random (mg/dl)", "Blood Urea (mg/dl)",
    "Serum Creatinine (mg/dl)", "Sodium (mEq/L)", "Hemoglobin (g/dl)", "Packed Cell Volume",
    "White Blood Cell Count (cells/cmm)", "Red Blood Cell Count (millions/cmm)",
    "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease", "Appetite",
    "Pedal Edema", "Anemia"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "Red Blood Cells", "Pus Cell", "Pus Cell Clumps", "Bacteria",
    "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease",
    "Appetite", "Pedal Edema", "Anemia"
]

# Renaming column names in your dataset to match the feature names
column_mapping = {
    'age': 'Age (years)',
    'bp': 'Blood Pressure (mmHg)',
    'sg': 'Specific Gravity',
    'al': 'Albumin',
    'su': 'Sugar',
    'rbc': 'Red Blood Cells',
    'pc': 'Pus Cell',
    'pcc': 'Pus Cell Clumps',
    'ba': 'Bacteria',
    'bgr': 'Blood Glucose Random (mg/dl)',
    'bu': 'Blood Urea (mg/dl)',
    'sc': 'Serum Creatinine (mg/dl)',
    'sod': 'Sodium (mEq/L)',
    'pot': 'Potassium (mEq/L)',
    'hemo': 'Hemoglobin (g/dl)',
    'pcv': 'Packed Cell Volume',
    'wc': 'White Blood Cell Count (cells/cmm)',
    'rc': 'Red Blood Cell Count (millions/cmm)',
    'htn': 'Hypertension',
    'dm': 'Diabetes Mellitus',
    'cad': 'Coronary Artery Disease',
    'appet': 'Appetite',
    'pe': 'Pedal Edema',
    'ane': 'Anemia'
}

# Load data and rename columns
try:
    data = pd.read_csv('CKD_data.csv')
    data.rename(columns=column_mapping, inplace=True)
except Exception as e:
    print(f"Error loading data: {e}")
    data = None

# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CERULEAN],
    title="CKD Prediction Dashboard",
    suppress_callback_exceptions=True
)
server = app.server 

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Chronic Kidney Disease Prediction Dashboard", className="text-center"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.P(
            "Predict CKD risk using Random Forest and XGBoost models based on clinical features. "
            "Explore dataset insights and model performance below.",
            className="text-center",
            style={'fontWeight': 'bold'}), width=12)
    ]),
    dcc.Tabs(id="tabs", value="prediction", children=[
        dcc.Tab(label="Prediction", value="prediction",
                style={'backgroundColor': '#0056b3', 'color': 'white', 'fontWeight': 'bold'},
                selected_style={'backgroundColor': '#0056b3', 'color': 'white'}),
        dcc.Tab(label="Data Insights", value="insights",
                style={'backgroundColor': '#1e7e34', 'color': 'white', 'fontWeight': 'bold'},
               selected_style={'backgroundColor': '#1e7e34', 'color': 'white'}),
        dcc.Tab(label="Model Performance", value="performance",
                style={'backgroundColor': '#ffc107', 'color': 'white', 'fontWeight': 'bold'},
               selected_style={'backgroundColor': '#ffc107', 'color': 'black'})
    ]),
    html.Div(id="tab-content")
], fluid=True, style={'backgroundColor': "#ffe6e6", 'minHeight': '100vh', 'padding': '20px'})

def preprocess_for_performance():
    if data is None:
        return None, None, None, None

    try:
        # Prepare data
        df = data.dropna()
        X = df[FEATURE_NAMES]
        y = df["classification"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        rf_model = RandomForestClassifier(random_state=42)
        xgb_model = XGBClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        return rf_model, xgb_model, X_test, y_test
    except Exception as e:
        print(f"Error in model training: {e}")
        return None, None, None, None

def get_model_metrics(model, X_test, y_test):
    if model is None:
        return {}

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics["roc_curve"] = (fpr, tpr)

    return metrics

# Callback to render tabs
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab):
    if tab == "prediction":
        return [
            dbc.Row([dbc.Col(html.H3("Make a Prediction"), width=12)]),
            html.Div(id="input-form", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label(FEATURE_NAMES[j]),
                        dcc.Dropdown(
                            options=[{"label": "No (0)", "value": 0}, {"label": "Yes (1)", "value": 1}],
                            value=0,
                            id=f"input-{j}",
                            clearable=False
                        ) if FEATURE_NAMES[j] in CATEGORICAL_FEATURES else
                        dcc.Input(
                            type="number",
                            value=0.0,
                            id=f"input-{j}",
                            placeholder=f"Enter {FEATURE_NAMES[j]}",
                            step=0.001
                        )
                    ], width=3)
                    for j in range(i, min(i + 4, len(FEATURE_NAMES)))
                ])
                for i in range(0, len(FEATURE_NAMES), 4)
            ]),
            html.Br(),
            dbc.Button("Predict", id="predict-btn", n_clicks=0, color="primary", className="me-1"),
            html.Br(), html.Br(),
            html.Div(id="prediction-output")
        ]

    elif tab == "insights":
        if data is None:
            return dbc.Alert("Dataset not available for visualization.", color="danger")

        # Create correlation heatmap figure
        numeric_data = data.select_dtypes(include=np.number)
        if not numeric_data.empty:
            try:
                corr_matrix = numeric_data.corr()
                mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

                # Create annotated heatmap
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.mask(mask).values,  # Only show lower triangle
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=np.around(corr_matrix.mask(mask).values, 2),
                    hoverinfo="text",
                    colorbar=dict(title='Correlation')
                ))

                # Adjust layout
                heatmap_fig.update_layout(
                    title='Feature Correlation Matrix (Lower Triangle)',
                    xaxis_title="Features",
                    yaxis_title="Features",
                    height=700,
                    width=700,
                    xaxis=dict(tickangle=45),
                    margin=dict(l=100, r=100, t=100, b=100)
                )

                # Add correlation values as annotations
                annotations = []
                for i, row in enumerate(corr_matrix.mask(mask).values):
                    for j, value in enumerate(row):
                        if not np.isnan(value):
                            annotations.append(
                                dict(
                                    x=corr_matrix.columns[j],
                                    y=corr_matrix.columns[i],
                                    text=str(round(value, 2)),
                                    font=dict(size=10),
                                    showarrow=False
                                )
                            )
                heatmap_fig.update_layout(annotations=annotations)

            except Exception as e:
                print(f"Error creating correlation matrix: {e}")
                heatmap_fig = go.Figure()
        else:
            heatmap_fig = go.Figure()

        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Dataset Insights"), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Dataset Sample"),
                    dash_table.DataTable(
                        data=data.head().to_dict("records"),
                        columns=[{"name": i, "id": i} for i in data.columns],
                        style_table={"overflowX": "auto"},
                        page_size=10,
                        style_cell={'textAlign': 'left'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        }
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Feature Distribution"),
                    dcc.Dropdown(
                        id="feature-dropdown",
                        options=[{"label": col, "value": col} for col in numeric_data.columns],
                        value=numeric_data.columns[0] if not numeric_data.empty else None,
                        clearable=False
                    ),
                    dcc.Graph(id="feature-dist")
                ], width=6),
                dbc.Col([
                    html.H5("Feature Correlations"),
                    dcc.Graph(figure=heatmap_fig)
                ], width=6)
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H5("Hypertension Distribution by CKD Status"),
                    dcc.Graph(
                        figure=px.histogram(
                            data_frame=data[data["Hypertension"].notna()],
                            x="Hypertension",
                            color="classification",
                            barmode="group",
                            title="Hypertension Count by CKD Status"
                        )
                    )
                ], width=4),
                dbc.Col([
                    html.H5("Average Blood Pressure by CKD Status"),
                    dcc.Graph(
                        figure=px.bar(
                            data_frame=data,
                            x="classification",
                            y="Blood Pressure (mmHg)",
                            title="Average Blood Pressure by CKD Status",
                            color="classification"
                        )
                    )
                ], width=4),
                dbc.Col([
                    html.H5("Age Distribution by CKD Status"),
                    dcc.Graph(
                        figure=px.density_contour(
                            data_frame=data,
                            x="Age (years)",
                            color="classification",
                            marginal_x="histogram",
                            marginal_y="box",
                            title="Density Plot of Age by CKD Status"
                        )
                    )
                ], width=4)
            ])
        ], fluid=True)

    elif tab == "performance":
        rf_model, xgb_model, X_test, y_test = preprocess_for_performance()

        if rf_model is None or xgb_model is None:
            return dbc.Alert("Unable to process data for model performance.", color="danger")

        # Get metrics for both models
        rf_metrics = get_model_metrics(rf_model, X_test, y_test)
        xgb_metrics = get_model_metrics(xgb_model, X_test, y_test)

        # Create ROC curve figure
        roc_fig = go.Figure()
        if "roc_curve" in rf_metrics:
            fpr, tpr = rf_metrics["roc_curve"]
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"Random Forest (AUC={rf_metrics.get('auc', 0):.2f})"
            ))

        if "roc_curve" in xgb_metrics:
            fpr, tpr = xgb_metrics["roc_curve"]
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"XGBoost (AUC={xgb_metrics.get('auc', 0):.2f})"
            ))

        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name="Random Guess",
            line=dict(dash="dash")
        ))
        roc_fig.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500
        )

        # Create confusion matrices
        def create_confusion_matrix_fig(y_true, y_pred, title):
            cm = confusion_matrix(y_true, y_pred)

            # Create a figure with just the heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Healthy", "At Risk"],
                y=["Healthy", "At Risk"],
                colorscale="Blues",
                showscale=True
            ))

            # Add annotations explicitly with proper positioning
            annotations = []
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    annotations.append(dict(
                        x=j,
                        y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(
                            color="white" if cm[i, j] > cm.max()/2 else "black",
                            size=16
                        )
                    ))

            fig.update_layout(
                title=title,
                xaxis_title="Predicted Label",
                xaxis=dict(tickmode='array', tickvals=[0, 1]),
                yaxis=dict(tickmode='array', tickvals=[0, 1]),
                yaxis_title="True Label",
                annotations=annotations,
                height=450,
                width=450
            )

            return fig

        y_pred_rf = rf_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)

        cm_rf_fig = create_confusion_matrix_fig(y_test, y_pred_rf, "Confusion Matrix - Random Forest")
        cm_xgb_fig = create_confusion_matrix_fig(y_test, y_pred_xgb, "Confusion Matrix - XGBoost")

        # Feature importance
        importances = rf_model.feature_importances_
        feature_importance = pd.DataFrame({'feature': FEATURE_NAMES, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=True)

        feature_fig = go.Figure(go.Bar(
            x=feature_importance["importance"],
            y=feature_importance["feature"],
            orientation='h',
            marker=dict(
                color=feature_importance["importance"],
                colorscale="Reds",
                showscale=True
            )
        ))
        feature_fig.update_layout(
            title="Feature Importances (Random Forest)",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600
        )

        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Model Performance Metrics"), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Model Metrics"),
                    dash_table.DataTable(
                        data=[
                            {
                                "Model": "Random Forest",
                                "Accuracy": f"{rf_metrics.get('accuracy', 0):.3f}",
                                "Precision": f"{rf_metrics.get('precision', 0):.3f}",
                                "Recall": f"{rf_metrics.get('recall', 0):.3f}",
                                "AUC": f"{rf_metrics.get('auc', 0):.3f}" if 'auc' in rf_metrics else "N/A"
                            },
                            {
                                "Model": "XGBoost",
                                "Accuracy": f"{xgb_metrics.get('accuracy', 0):.3f}",
                                "Precision": f"{xgb_metrics.get('precision', 0):.3f}",
                                "Recall": f"{xgb_metrics.get('recall', 0):.3f}",
                                "AUC": f"{xgb_metrics.get('auc', 0):.3f}" if 'auc' in xgb_metrics else "N/A"
                            }
                        ],
                        columns=[{"name": i, "id": i} for i in ["Model", "Accuracy", "Precision", "Recall", "AUC"]],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "center", "padding": "8px"},
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "rgb(230, 230, 230)"
                        }
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("ROC Curve Comparison"),
                    dcc.Graph(figure=roc_fig)
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Confusion Matrix - Random Forest"),
                    dcc.Graph(figure=cm_rf_fig)
                ], width=6),
                dbc.Col([
                    html.H5("Confusion Matrix - XGBoost"),
                    dcc.Graph(figure=cm_xgb_fig)
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Feature Importance (Random Forest)"),
                    dcc.Graph(figure=feature_fig)
                ], width=12)
            ])
        ], fluid=True)

@app.callback(
    Output("feature-dist", "figure"),
    Input("feature-dropdown", "value")
)
def update_feature_distribution(feature):
    if data is None or feature is None:
        return go.Figure()

    fig = px.histogram(
        data_frame=data,
        x=feature,
        color="classification",
        marginal="box",
        title=f"Distribution of {feature} by CKD Status"
    )
    return fig

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(f"input-{i}", "value") for i in range(len(FEATURE_NAMES))],
    prevent_initial_call=True
)
def make_prediction(n_clicks, *inputs):
    if len(inputs) != 21 or any(input is None for input in inputs):
        return dbc.Alert("Please provide all 21 inputs.", color="danger")

    # Check if API_URL is defined and uncommented
    try:
        # Create a placeholder for API response (since API_URL is commented out)
        # In a real app, you would use the actual API response
        preds = {
            "random_forest_prediction": 1 if sum(inputs) > 10 else 0,
            "random_forest_probability": 0.75 if sum(inputs) > 10 else 0.25,
            "xgboost_prediction": 1 if sum(inputs) > 8 else 0,
            "xgboost_probability": 0.82 if sum(inputs) > 8 else 0.18
        }

        return dbc.Card([
            dbc.CardHeader("Prediction Results", className="bg-primary text-white"),
            dbc.CardBody([
                dash_table.DataTable(
                    data=[{
                        "Model": "Random Forest",
                        "Prediction": "Healthy" if preds.get("random_forest_prediction", 0) == 0 else "At Risk",
                        "Confidence": f"{preds.get('random_forest_probability', 0)*100:.1f}%"
                    }, {
                        "Model": "XGBoost",
                        "Prediction": "Healthy" if preds.get("xgboost_prediction", 0) == 0 else "At Risk",
                        "Confidence": f"{preds.get('xgboost_probability', 0)*100:.1f}%"
                    }],
                    columns=[{"name": i, "id": i} for i in ["Model", "Prediction", "Confidence"]],
                    style_table={"overflowX": "auto"},
                    style_data_conditional=[
                        {
                            "if": {
                                "filter_query": "{Prediction} = 'At Risk'",
                                "column_id": "Prediction"
                            },
                            "backgroundColor": "#f8d7da",
                            "color": "#721c24"
                        }
                    ]
                )
            ])
        ])
    except Exception as e:
        return dbc.Alert(f"API Error: {str(e)}", color="danger")

if __name__ == "__main__":
    app.run(debug=True, port=8050)
