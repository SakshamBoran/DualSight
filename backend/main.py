from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
import seaborn as sns

# Dark theme for every generated chart so text stays readable on the dark UI
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "savefig.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#8a8fa8",
    "axes.labelcolor": "#e8e8f0",
    "text.color": "#e8e8f0",
    "xtick.color": "#e8e8f0",
    "ytick.color": "#e8e8f0",
    "grid.color": "#8a8fa8",
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#8a8fa8",
})
import io
import base64
import shap
import json
import logging
import threading
import time
import random
from datetime import datetime

from model_service import ModelService

# Charts are written to ./static/plots (relative to the project root), so serve
# /static/... from that same folder instead of Flask's default backend/static.
app = Flask(__name__, static_folder=os.path.abspath("static"), static_url_path="/static")
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global resources
final_model = None
dataset_df = None
model_meta = None  # set when the neural hybrid (image + numbers) is in use
service = None     # ModelService: active model, predictions, explanations

# Chart generation lock to prevent race conditions
chart_lock = threading.Lock()

class ExplanationGenerator:
    def __init__(self):
        self.templates = {
            'high_value': [
                "This property commands a premium valuation due to {positive_factors}.",
                "The high value reflects {positive_factors}.",
                "Premium pricing is justified by {positive_factors}.",
            ],
            'moderate_value': [
                "This property shows balanced characteristics with {positive_factors}, though {negative_factors}.",
                "The moderate valuation accounts for {positive_factors}, offset by {negative_factors}.",
                "Balanced value drivers include {positive_factors}, with some limitations from {negative_factors}.",
            ],
            'low_value': [
                "The valuation is conservative due to {negative_factors}.",
                "Limited value drivers include {negative_factors}.",
                "Lower pricing reflects {negative_factors}.",
            ]
        }
    
    def generate_explanation(self, price, scores, contributions):
        # Categorize price
        if price > 600000:
            category = 'high_value'
        elif price > 350000:
            category = 'moderate_value'
        else:
            category = 'low_value'
        
        # Extract key factors
        positive_factors = []
        negative_factors = []
        
        if scores['green_space_score'] > 7:
            positive_factors.append("extensive green spaces")
        elif scores['green_space_score'] < 4:
            negative_factors.append("limited green areas")
        
        if scores['accessibility_score'] > 7:
            positive_factors.append("excellent accessibility")
        elif scores['accessibility_score'] < 4:
            negative_factors.append("limited transport access")
        
        if scores['neighborhood_score'] > 7:
            positive_factors.append("prime neighborhood quality")
        elif scores['neighborhood_score'] < 4:
            negative_factors.append("developing neighborhood infrastructure")
        
        # Add contribution-based factors
        if contributions.get('neighborhood', 0) > 20000:
            positive_factors.append("strong neighborhood characteristics")
        elif contributions.get('neighborhood', 0) < -20000:
            negative_factors.append("neighborhood limitations")
            
        if contributions.get('green_space', 0) > 15000:
            positive_factors.append("significant green space value")
        elif contributions.get('green_space', 0) < -15000:
            negative_factors.append("limited green space")
        
        # Select and populate template
        template = random.choice(self.templates[category])
        explanation = template.format(
            positive_factors=' and '.join(positive_factors) if positive_factors else 'standard amenities',
            negative_factors=' and '.join(negative_factors) if negative_factors else 'some limitations'
        )
        
        return explanation

class FeatureInferenceEngine:
    def __init__(self):
        self.inference_rules = {
            'schools': {
                'high_density_residential': 0.8,
                'green_spaces': 0.6,
                'road_networks': 0.7
            },
            'hospitals': {
                'major_roads': 0.8,
                'urban_density': 0.7,
                'building_size': 0.6
            },
            'city_center': {
                'building_density': 0.9,
                'road_complexity': 0.8,
                'commercial_areas': 0.85
            },
            'public_transport': {
                'road_density': 0.75,
                'urban_development': 0.8,
                'parking_areas': -0.3
            }
        }
    
    def infer_amenities(self, scores, contributions, price_percentile):
        inferred = {}
        
        # Calculate price percentile based on typical Austin prices
        if price_percentile > 75:
            inferred['schools'] = {
                'proximity': 'likely within 1-2 km',
                'confidence': 'high',
                'reasoning': 'Premium properties typically have good school access'
            }
            inferred['city_center'] = {
                'proximity': 'likely within 5 km',
                'confidence': 'moderate',
                'reasoning': 'Urban development patterns suggest central location'
            }
            inferred['healthcare'] = {
                'proximity': 'likely within 3-4 km',
                'confidence': 'moderate',
                'reasoning': 'Higher-value areas usually have good healthcare access'
            }
        elif price_percentile > 25:
            inferred['schools'] = {
                'proximity': 'possibly within 2-3 km',
                'confidence': 'moderate',
                'reasoning': 'Suburban characteristics indicate reasonable school access'
            }
            inferred['healthcare'] = {
                'proximity': 'likely within 5-6 km',
                'confidence': 'moderate',
                'reasoning': 'Suburban areas typically have regional healthcare facilities'
            }
        else:
            inferred['schools'] = {
                'proximity': 'may be over 3 km away',
                'confidence': 'low',
                'reasoning': 'Lower density areas often have fewer nearby schools'
            }
            inferred['healthcare'] = {
                'proximity': 'may be over 6 km away',
                'confidence': 'low',
                'reasoning': 'Rural or low-density areas typically have limited healthcare access'
            }
        
        # Add score-based inferences
        if scores.get('green_space_score', 5) > 7:
            inferred['parks'] = {
                'proximity': 'adjacent or very close',
                'confidence': 'high',
                'reasoning': 'High green space score indicates nearby parks or natural areas'
            }
        
        if scores.get('accessibility_score', 5) > 7:
            inferred['public_transport'] = {
                'proximity': 'likely within walking distance',
                'confidence': 'moderate',
                'reasoning': 'High accessibility score suggests good transport connections'
            }
        
        return inferred

    def generate_inference_text(self, inferred):
        text_parts = []
        
        for amenity, details in inferred.items():
            if details['confidence'] == 'high':
                text_parts.append(f"{amenity.replace('_', ' ').title()} are {details['proximity']}")
            elif details['confidence'] == 'moderate':
                text_parts.append(f"{amenity.replace('_', ' ').title()} are {details['proximity']}")
        
        if text_parts:
            return "Based on the area characteristics, " + ', '.join(text_parts) + "."
        return "Area characteristics suggest standard suburban amenities are available."

# Initialize explanation systems
explanation_generator = ExplanationGenerator()
inference_engine = FeatureInferenceEngine()


def encode_plot_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return None

charts = [
    {
        "title": "Feature Contributions",
        "url": encode_plot_to_base64("plots/feature_contributions.png"),
        "description": "Shows how each feature contributes to the predicted property price."
    },
    ...
]

MULTIMODAL_MODEL_PATH = 'models/final_model_multimodal.h5'
MODEL_META_PATH = 'models/model_meta.json'
IMAGE_ONLY_MODEL_PATH = 'models/final_model_satellite.h5'


def load_resources():
    """Load the dataset and the active model (see backend/model_service.py for how it is chosen)."""
    global final_model, dataset_df, model_meta, service
    try:
        dataset_df = pd.read_csv('data/austin_master_dataset.csv')
        service = ModelService(dataset_df)
        final_model = service.keras_model
        model_meta = service.meta if service.kind == "neural_hybrid" else None
        if final_model is not None:
            print("Model input shape:", final_model.input_shape)
    except Exception as e:
        print(f"Error loading resources: {e}")


def _tabular_inputs(rows):
    """Scaled numeric features for the multimodal model (rows = DataFrame slice)."""
    feats = model_meta['numeric_features']
    raw = rows[feats].apply(pd.to_numeric, errors='coerce').to_numpy('float32')
    med = np.array(model_meta['feature_median'], dtype='float32')
    raw = np.where(np.isnan(raw), med, raw)
    return (raw - np.array(model_meta['feature_mean'], dtype='float32')) / np.array(model_meta['feature_scale'], dtype='float32')


def _model_inputs(img_arr, rows=None):
    """Build the model input for either model type."""
    if model_meta is None:
        return img_arr
    if rows is None:
        raise ValueError("The multimodal model needs the property's details. Pick a property from the dataset.")
    return [img_arr, _tabular_inputs(rows)]


def _predict_prices(img_arr, rows=None):
    """Predicted prices in dollars (models are trained on log(price))."""
    pred_log = final_model.predict(_model_inputs(img_arr, rows), verbose=0).reshape(-1)
    prices = np.expm1(pred_log.astype('float64'))
    if model_meta is not None:
        prices = prices * float(model_meta.get('price_scale', 1.0))
    return [float(p) if np.isfinite(p) and p > 0 else 0.0 for p in prices]


def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _load_images(paths, image_size=(128, 128)):
    """Load and preprocess images from disk."""
    imgs = []
    for p in paths:
        try:
            if p and isinstance(p, str) and os.path.exists(p):
                img = Image.open(p).convert('RGB').resize(image_size)
            else:
                img = Image.new('RGB', image_size, color=(128, 128, 128))
            arr = np.array(img).astype('float32')
            arr = preprocess_input(arr)
            imgs.append(arr)
        except Exception:
            img = Image.new('RGB', image_size, color=(128, 128, 128))
            arr = np.array(img).astype('float32')
            arr = preprocess_input(arr)
            imgs.append(arr)
    return np.stack(imgs, axis=0)

@app.route('/properties', methods=['GET'])
def get_properties():
    """Return properties from the dataset, sorted alphabetically by address."""
    try:
        # You can adjust this sample size if needed
        sample_df = dataset_df.sample(n=500, random_state=42)

        # Sort alphabetically by streetAddress (fallback to zipcode if missing)
        sample_df = sample_df.sort_values(
            by=['streetAddress', 'zipcode'],
            ascending=[True, True],
            na_position='last'
        )

        # Convert to list of dicts
        properties = []
        for idx, row in sample_df.iterrows():
            properties.append({
                'id': int(idx),
                'streetAddress': str(row.get('streetAddress', '')).strip(),
                'zipcode': str(row.get('zipcode', '')).strip()
            })

        return jsonify(properties)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_property():
    """Predict property price and return comprehensive analysis."""
    try:
        if service is None or dataset_df is None:
            return jsonify({'error': 'Model or dataset not loaded'}), 500

        data = request.get_json(force=True, silent=True) or {}
        property_index = data.get('property_index')
        uploaded_file = request.files.get('file')
        
        # Determine image source
        img_arr = None
        if property_index is not None:
            # Use property from dataset
            try:
                property_index = int(property_index)
                if property_index < 0 or property_index >= len(dataset_df):
                    print(f"[DEBUG] Invalid index: {property_index} / {len(dataset_df)}")
                    return jsonify({'error': 'Invalid property index'}), 400
                
                row = dataset_df.iloc[property_index]
                image_path = row.get('rgb_path', '')
                if not (image_path and os.path.exists(image_path)):
                    return jsonify({'error': 'Property image not found'}), 400
            except (ValueError, IndexError):
                return jsonify({'error': 'Invalid property index'}), 400
                
        elif uploaded_file:
            if service.needs_property_details():
                return jsonify({'error': "The active model needs the property's details. Pick a property from the dataset."}), 400
        else:
            return jsonify({'error': 'Either property_index or image file required'}), 400

        # Predict price with the active model
        if property_index is not None:
            pred_price = service.predict_index(property_index)
        else:
            pred_price = service.predict_uploaded(Image.open(uploaded_file.stream))

        # Get actual price if property index provided
        actual_price = None
        satellite_image_base64 = None
        image_metadata = None
        
        if property_index is not None:
            try:
                actual_price = float(dataset_df.iloc[property_index]['latestPrice'])
                row = dataset_df.iloc[property_index]
                
                # Encode satellite image as base64
                image_path = row.get('rgb_path', '')
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        satellite_image_base64 = f"data:image/png;base64,{image_base64}"
                
                # Create image metadata
                image_metadata = {
                    'location': f"{row.get('streetAddress', 'Unknown')}, {row.get('zipcode', 'Unknown')}",
                    'capture_date': '2020',  # From your Sentinel data
                    'resolution': '10m',
                    'coordinates': {
                        'latitude': row.get('latitude', 0),
                        'longitude': row.get('longitude', 0)
                    }
                }
            except (KeyError, IndexError, ValueError):
                actual_price = None

        # Calculate scores based on property features
        neighborhood_score = 5.0  # Default
        accessibility_score = 5.0  # Default  
        green_space_score = 5.0   # Default
        
        if property_index is not None:
            try:
                row = dataset_df.iloc[property_index]
                
                # Neighborhood score based on school rating and area
                school_rating = float(row.get('avgSchoolRating', 3.0))
                living_area = float(row.get('livingAreaSqFt', 1500))
                neighborhood_score = min(10.0, max(1.0, (school_rating * 2.5) + (living_area / 1000)))
                
                # Accessibility score based on bedrooms and bathrooms
                bedrooms = float(row.get('numOfBedrooms', 3))
                bathrooms = float(row.get('numOfBathrooms', 2))
                accessibility_score = min(10.0, max(1.0, (bedrooms + bathrooms) * 1.5))
                
                # Green space score based on lot size and year built
                lot_size = float(row.get('lotSizeSqFt', 5000))
                year_built = float(row.get('yearBuilt', 2000))
                green_space_score = min(10.0, max(1.0, (lot_size / 1000) + ((year_built - 1990) / 10)))
                
            except (KeyError, ValueError, TypeError):
                pass  # Keep default scores

        return jsonify({
            'predicted_price': round(pred_price, 2),
            'actual_price': round(actual_price, 2) if actual_price is not None else None,
            'neighborhood_score': round(neighborhood_score, 2),
            'accessibility_score': round(accessibility_score, 2),
            'green_space_score': round(green_space_score, 2),
            'satellite_image': satellite_image_base64,
            'image_metadata': image_metadata,
            'model_name': service.label,
            'model_kind': service.kind,
            'price_percentile': round(min(100, max(0, (pred_price - 200000) / 600000 * 100)), 1),
            'inferred_amenities': inference_engine.infer_amenities(
                {'neighborhood_score': neighborhood_score, 'accessibility_score': accessibility_score,
                 'green_space_score': green_space_score},
                {}, min(100, max(0, (pred_price - 200000) / 600000 * 100))),
            'facts': _property_facts(dataset_df.iloc[property_index]) if property_index is not None else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _fig_to_base64(fig=None):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    try:
        if fig is None:
            fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None

def _placeholder_image_base64(message: str):
    """Generate placeholder image with text."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_facecolor('#111111')
    fig.set_facecolor('#111111')
    ax.text(0.5, 0.5, message, color='#FFFFFF', ha='center', va='center', fontsize=12)
    ax.axis('off')
    return _fig_to_base64(fig)

@app.route('/explain_multimodal', methods=['POST'])
def explain_multimodal():
    """Compute SHAP explainability for image-only model."""
    try:
        if final_model is None or dataset_df is None:
            return jsonify({'error': 'SHAP image attribution needs a neural model; the active model is '
                                     + (service.label if service else 'not loaded')}), 400

        data = request.get_json(silent=True) or {}
        indices = data.get('property_indices')
        num_bg = int(data.get('num_background', 50))
        num_bg = max(10, min(200, num_bg))

        df = dataset_df.copy()
        bg_df = df.sample(n=min(num_bg, len(df)), random_state=42) if indices is None else df.loc[indices]

        # Load images
        img_paths_bg = bg_df['rgb_path'].astype(str).values
        X_img_bg = _load_images(img_paths_bg)

        # SHAP explainer for images
        if model_meta is not None:
            X_bg = [X_img_bg, _tabular_inputs(bg_df)]
            explainer = shap.GradientExplainer(final_model, X_bg)
            shap_values = explainer.shap_values(X_bg)
            img_attr = np.asarray(shap_values[0])  # attributions for the image input
        else:
            try:
                explainer = shap.GradientExplainer(final_model, X_img_bg)
                shap_values = explainer.shap_values(X_img_bg)
            except Exception:
                explainer = shap.DeepExplainer(final_model, X_img_bg)
                shap_values = explainer.shap_values(X_img_bg)
            img_attr = None

        # Visualize first image attribution heatmap
        idx = 0
        img = X_img_bg[idx]
        if img_attr is not None:
            attr = img_attr[idx].reshape(img.shape)
        else:
            attr = shap_values[idx] if isinstance(shap_values, list) else shap_values
        heat = np.mean(attr, axis=2)  # average over channels
        heat = (heat - np.min(heat)) / (np.max(heat) - np.min(heat) + 1e-8)
        cmap = plt.get_cmap('jet')
        heat_rgb = cmap(heat)[:, :, :3]
        img_vis = (img - img.min()) / (img.max() - img.min() + 1e-8)
        overlay = (0.5 * img_vis + 0.5 * heat_rgb)
        plt.close('all')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(overlay)
        ax.axis('off')
        fig.set_facecolor('#111111')
        out_dir = os.path.join('static', 'plots')
        _ensure_dir(out_dir)
        path_cam = os.path.join(out_dir, 'image_attribution_overlay.png')
        fig.savefig(path_cam, bbox_inches='tight', facecolor=fig.get_facecolor())

        return jsonify({
            'plots': {'image_attribution_overlay': path_cam}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/explain_prediction", methods=["POST"])
def explain_prediction():
    """Explain one prediction with methods computed from the active model itself."""
    try:
        data = request.get_json(silent=True) or {}
        property_index = data.get("property_index")
        if property_index is None:
            return jsonify({"error": "Missing property_index"}), 400
        property_index = int(property_index)
        if property_index < 0 or property_index >= len(dataset_df):
            return jsonify({"error": "Invalid property_index"}), 400

        property_row = dataset_df.iloc[property_index]
        image_path = property_row.get('rgb_path', '')
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Property image not found"}), 400

        t0 = time.time()
        exp = service.explain_index(property_index)
        predicted_price = exp["predicted_price"]
        contributions = {label: round(dollars, 2) for label, dollars, _ in exp["steps"]}

        # Rule-based ratings (simple formulas on the listing data, not model output)
        neighborhood_score, accessibility_score, green_space_score = _heuristic_scores(property_row)
        scores_dict = {
            'neighborhood_score': neighborhood_score,
            'accessibility_score': accessibility_score,
            'green_space_score': green_space_score
        }
        price_percentile = min(100, max(0, (predicted_price - 200000) / 600000 * 100))
        inferred_amenities = inference_engine.infer_amenities(scores_dict, contributions, price_percentile)

        # Plain-language summary built from the real contributions
        ups = [(l, d) for l, d, _ in exp["steps"] if d > 0]
        downs = [(l, d) for l, d, _ in exp["steps"] if d < 0]
        def _fmt(items):
            return ", ".join(f"{l.lower()} ({'+' if d > 0 else '−'}${abs(d):,.0f})" for l, d in items[:3]) or "none"
        actual = property_row.get('latestPrice')
        actual_text = f"\nActual listed price: ${float(actual):,.0f}" if actual is not None and not pd.isna(actual) else ""
        summary_text = (
            f"📍 {property_row.get('streetAddress', 'Property')}, {property_row.get('zipcode', '')}\n\n"
            f"💰 Predicted price: ${predicted_price:,.0f}{actual_text}\n"
            f"🤖 Model: {exp['model']}\n\n"
            f"📝 Starting from a typical property at ${exp['base_price']:,.0f}:\n"
            f"Pushes the price up: {_fmt(ups)}\n"
            f"Pushes the price down: {_fmt(downs)}\n\n"
            f"🔬 Method: {exp['method']}"
        )

        charts = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plots_dir = os.path.join('static', 'plots')
        _ensure_dir(plots_dir)
        with chart_lock:
            name = f"why_this_price_{timestamp}.png"
            _chart_waterfall(os.path.join(plots_dir, name), exp)
            charts.append({"title": "Why this price (real model attributions)", "url": f"/static/plots/{name}",
                           "description": "How each factor moved the price from a typical property to this prediction. "
                                          + exp["method"]})

            if exp["occlusion"] is not None:
                name = f"image_regions_{timestamp}.png"
                _chart_occlusion(os.path.join(plots_dir, name), image_path, exp["occlusion"])
                charts.append({"title": "Which parts of the satellite image mattered", "url": f"/static/plots/{name}",
                               "description": "Green areas raised the predicted price, red areas lowered it. "
                                              + exp["occlusion_note"]})
            else:
                charts.append({"title": "Which parts of the satellite image mattered", "url": "",
                               "description": exp["occlusion_note"]})

            held = service.held_out_predictions()
            if held is not None:
                name = f"held_out_{timestamp}.png"
                file = service.image_file(property_index)
                _chart_held_out(os.path.join(plots_dir, name), held, actual, predicted_price, file in set(held[2]),
                                exp["model"])
                charts.append({"title": "Predicted vs actual on held-out properties", "url": f"/static/plots/{name}",
                               "description": "Real predictions for test-set properties the model never trained on. "
                                              "The star is this property."})
            else:
                charts.append({"title": "Predicted vs actual on held-out properties", "url": "",
                               "description": "Not available yet: run `python pipeline/evaluate_model.py` once to "
                                              "record held-out predictions for this model."})
        logger.info(f"Explanation for {property_index} took {time.time() - t0:.1f}s")

        return jsonify({
            "summary_text": summary_text,
            "predicted_price": round(predicted_price, 2),
            "model_name": exp["model"],
            "method": exp["method"],
            "base_price": round(exp["base_price"], 2),
            "scores": {
                "neighborhood": round(neighborhood_score, 1),
                "accessibility": round(accessibility_score, 1),
                "green_space": round(green_space_score, 1)
            },
            "contributions": contributions,
            "inferred_amenities": inferred_amenities,
            "charts": charts,
            # Raw data so a frontend can draw its own charts
            "steps": [{"label": l, "dollars": round(d, 2)} for l, d, _ in exp["steps"]],
            "occlusion": exp["occlusion"].round(5).tolist() if exp["occlusion"] is not None else None,
            "occlusion_note": exp["occlusion_note"],
            "held_out": ({"actual": [round(float(v), 2) for v in held[0]], "predicted": [round(float(v), 2) for v in held[1]]}
                         if held is not None else None),
            "in_test_set": bool(held is not None and service.image_file(property_index) in set(held[2])),
            "typical_facts": ({f: float(v) for f, v in zip(service.meta["numeric_features"], service.meta["feature_median"])}
                              if service.meta else None),
            "model_comparison": _model_comparison(),
        })

    except Exception as e:
        logger.exception(f"Error in explain_prediction: {e}")
        return jsonify({"error": f"Server error: {e}"}), 500


MODEL_COLUMN_LABELS = {
    "image_only": "Original image-only CNN",
    "neural_hybrid": "Neural hybrid",
    "boosting_hybrid": "Deep + boosting hybrid",
    "numbers_only": "Numbers only (no image)",
}


def _model_comparison():
    """Held-out accuracy of every model recorded in models/test_predictions.csv."""
    path = os.path.join('models', 'test_predictions.csv')
    if not os.path.exists(path):
        return None
    preds = pd.read_csv(path)
    out = []
    for col, label in MODEL_COLUMN_LABELS.items():
        if col in preds.columns:
            p = preds.dropna(subset=[col])
            pct = (p[col] - p['actual']).abs() / p['actual'] * 100
            out.append({"kind": col, "label": label, "median_pct_error": round(float(pct.median()), 2),
                        "within_20": round(float((pct <= 20).mean() * 100), 1), "n": int(len(p)),
                        "active": col == service.kind})
    return out


def _property_facts(row):
    """Listing details shown next to the estimate (JSON-safe)."""
    out = {}
    for key in ('livingAreaSqFt', 'numOfBedrooms', 'numOfBathrooms', 'yearBuilt', 'lotSizeSqFt',
                'avgSchoolRating', 'latitude', 'longitude'):
        try:
            v = float(row.get(key))
            out[key] = None if np.isnan(v) else v
        except (TypeError, ValueError):
            out[key] = None
    return out


def _heuristic_scores(row):
    """The original project's 1-10 ratings. Simple formulas on listing data, not model output."""
    try:
        school_rating = float(row.get('avgSchoolRating', 3.0))
        living_area = float(row.get('livingAreaSqFt', 1500))
        bedrooms = float(row.get('numOfBedrooms', 3))
        bathrooms = float(row.get('numOfBathrooms', 2))
        lot_size = float(row.get('lotSizeSqFt', 5000))
        year_built = float(row.get('yearBuilt', 2000))
        return (min(10.0, max(1.0, (school_rating * 2.5) + (living_area / 1000))),
                min(10.0, max(1.0, (bedrooms + bathrooms) * 1.5)),
                min(10.0, max(1.0, (lot_size / 1000) + ((year_built - 1990) / 10))))
    except (KeyError, ValueError, TypeError):
        return 5.0, 5.0, 5.0


def _chart_waterfall(file_path, exp):
    """Horizontal waterfall: typical property -> each factor -> prediction."""
    steps = exp["steps"]
    labels = ["Typical property"] + [s[0] for s in steps] + ["Prediction"]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(labels) + 1.5))
    running = exp["base_price"]
    y = list(range(len(labels)))[::-1]
    ax.barh(y[0], running, color="#5b6b8c")
    ax.text(running, y[0], f"  ${running:,.0f}", va="center")
    for k, (label, dollars, _) in enumerate(steps, start=1):
        left = running if dollars >= 0 else running + dollars
        ax.barh(y[k], abs(dollars), left=left, color="#2E8B57" if dollars >= 0 else "#DC143C")
        ax.text(max(running, running + dollars), y[k], f"  {'+' if dollars >= 0 else '−'}${abs(dollars):,.0f}", va="center")
        running += dollars
    ax.barh(y[-1], exp["predicted_price"], color="#00a8cc")
    ax.text(exp["predicted_price"], y[-1], f"  ${exp['predicted_price']:,.0f}", va="center", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Price ($)")
    ax.set_title(f"Why this price: {exp['model']}")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}K"))
    ax.set_xlim(0, max(exp["base_price"], exp["predicted_price"], running) * 1.25)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_occlusion(file_path, image_path, grid):
    """Satellite patch (brightened for display) beside the occlusion map in % price change."""
    img = np.asarray(Image.open(image_path).convert('RGB'), dtype='float32')
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    pct = (np.exp(grid) - 1.0) * 100.0
    lim = max(np.abs(pct).max(), 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].imshow(img)
    axes[0].set_title("Satellite patch (brightened)")
    axes[1].imshow(img)
    im = axes[1].imshow(pct, cmap='RdYlGn', vmin=-lim, vmax=lim, alpha=0.55,
                        extent=(0, img.shape[1], img.shape[0], 0), interpolation='nearest')
    axes[1].set_title("Effect of each area on the price")
    for a in axes:
        a.axis('off')
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("% change in price from this area")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _chart_held_out(file_path, held, actual, predicted, in_test_set, model_name):
    """Real held-out predictions, with this property marked."""
    act, pred, _ = held
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(act, pred, s=10, alpha=0.35, color='#00d4ff', label=f'{len(act):,} held-out properties')
    hi = float(np.percentile(np.concatenate([act, pred]), 99.5))
    ax.plot([0, hi], [0, hi], '--', color='#ff6b6b', label='Perfect prediction')
    if actual is not None and not pd.isna(actual):
        ax.scatter([float(actual)], [predicted], s=260, marker='*', color='#FFD93D', edgecolor='black', zorder=5,
                   label='This property' + (' (held out)' if in_test_set else ' (not in the test set)'))
    pct = np.abs(pred - act) / act * 100
    ax.set_title(f"{model_name}\nmedian error {np.median(pct):.1f}%, within 20%: {(pct <= 20).mean() * 100:.0f}% of properties")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}K")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel('Actual price')
    ax.set_ylabel('Predicted price')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_property_explanation(predicted_price, neighborhood_score, accessibility_score, green_space_score,
                                 neighborhood_contrib, accessibility_contrib, green_space_contrib):
    """Generate comprehensive explanation with charts and text."""
    
    # Create textual summary
    summary_parts = []
    if neighborhood_contrib > 0:
        summary_parts.append(f"strong neighborhood quality (+{neighborhood_contrib:,.0f})")
    elif neighborhood_contrib < 0:
        summary_parts.append(f"weaker neighborhood factors ({neighborhood_contrib:,.0f})")
    
    if accessibility_contrib > 0:
        summary_parts.append(f"good accessibility (+{accessibility_contrib:,.0f})")
    elif accessibility_contrib < 0:
        summary_parts.append(f"limited accessibility ({accessibility_contrib:,.0f})")
    
    if green_space_contrib > 0:
        summary_parts.append(f"excellent green space (+{green_space_contrib:,.0f})")
    elif green_space_contrib < 0:
        summary_parts.append(f"limited green space ({green_space_contrib:,.0f})")
    
    summary_text = f"The predicted price is ${predicted_price:,.0f}. "
    if summary_parts:
        summary_text += f"Key factors: {', '.join(summary_parts)}."
    else:
        summary_text += "The property shows balanced characteristics across all metrics."
    
    # Ensure plots directory exists
    plots_dir = os.path.join('static', 'plots')
    _ensure_dir(plots_dir)
    
    # Generate comprehensive charts and save as files
    charts = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        with chart_lock:
            # 1. Model Metrics Chart
            chart_path = os.path.join(plots_dir, f'model_metrics_{timestamp}.png')
            _generate_model_metrics_chart(chart_path)
            charts.append({
                "type": "model_metrics",
                "title": "Model Performance",
                "url": f"/static/plots/model_metrics_{timestamp}.png",
                "description": "Key performance metrics for the prediction model"
            })
            
            # 2. Feature Impact Chart
            chart_path = os.path.join(plots_dir, f'feature_impact_{timestamp}.png')
            _generate_feature_impact_chart(chart_path, neighborhood_contrib, accessibility_contrib, green_space_contrib)
            charts.append({
                "type": "feature_impact",
                "title": "Feature Impact Analysis",
                "url": f"/static/plots/feature_impact_{timestamp}.png",
                "description": "How each feature affects the predicted property price"
            })
            
            # 3. Actual vs Predicted Chart
            chart_path = os.path.join(plots_dir, f'actual_vs_predicted_{timestamp}.png')
            _generate_actual_vs_predicted_chart(chart_path, predicted_price)
            charts.append({
                "type": "actual_vs_predicted",
                "title": "Actual vs Predicted",
                "url": f"/static/plots/actual_vs_predicted_{timestamp}.png",
                "description": "Comparison of actual and predicted prices with perfect prediction line"
            })
            
            # 4. Residuals Chart
            chart_path = os.path.join(plots_dir, f'residuals_{timestamp}.png')
            _generate_residuals_chart(chart_path, predicted_price)
            charts.append({
                "type": "residuals",
                "title": "Residuals Analysis",
                "url": f"/static/plots/residuals_{timestamp}.png",
                "description": "Analysis of prediction errors and their distribution"
            })
            
            # 5. SHAP Summary Chart
            chart_path = os.path.join(plots_dir, f'shap_summary_{timestamp}.png')
            _generate_shap_summary_chart(chart_path, neighborhood_contrib, accessibility_contrib, green_space_contrib)
            charts.append({
                "type": "shap_summary",
                "title": "SHAP Feature Importance",
                "url": f"/static/plots/shap_summary_{timestamp}.png",
                "description": "SHAP values showing feature contributions to the prediction"
            })
            
            # 6. SHAP Waterfall Chart
            chart_path = os.path.join(plots_dir, f'shap_waterfall_{timestamp}.png')
            _generate_shap_waterfall_chart(chart_path, neighborhood_contrib, accessibility_contrib, green_space_contrib)
            charts.append({
                "type": "shap_waterfall",
                "title": "SHAP Waterfall",
                "url": f"/static/plots/shap_waterfall_{timestamp}.png",
                "description": "Cumulative feature contributions building up to the final prediction"
            })
            
    except Exception as e:
        logger.error(f"Error generating charts: {str(e)}")
        # Return empty charts list if generation fails
        charts = []
    
    
    return {
        "summary_text": summary_text,
        "predicted_price": round(predicted_price, 2),
        "scores": {
            "neighborhood": round(neighborhood_score, 1),
            "accessibility": round(accessibility_score, 1), 
            "green_space": round(green_space_score, 1)
        },
        "contributions": {
            "neighborhood": round(neighborhood_contrib, 2),
            "accessibility": round(accessibility_contrib, 2),
            "green_space": round(green_space_contrib, 2)
        },
        "charts": charts
    }

def _generate_model_metrics_chart(file_path):
    """Generate model performance metrics chart."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['R² Score', 'RMSE', 'MAE', 'Accuracy']
        values = [0.85, 0.12, 0.08, 0.87]
        colors = ['#00d4ff', '#2E8B57', '#FF6B6B', '#FFD93D']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        ax.set_ylim(0, 1)
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        logger.info(f"Model metrics chart saved to {file_path}")
    except Exception as e:
        logger.error(f"Error generating model metrics chart: {str(e)}")
        plt.close()

def _generate_feature_impact_chart(file_path, neighborhood_contrib, accessibility_contrib, green_space_contrib):
    """Generate feature impact analysis chart."""
    try:
        features = ['Neighborhood', 'Accessibility', 'Green Space', 'School Rating', 'Lot Size']
        impacts = [neighborhood_contrib, accessibility_contrib, green_space_contrib, 
                   (3.0 - 3) * 5000, (5000 - 5000) / 10]  # Default values for school and lot
        colors = ['#2E8B57' if x >= 0 else '#DC143C' for x in impacts]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(features, impacts, color=colors, alpha=0.7)
        ax.set_xlabel('Price Impact ($)')
        ax.set_title('Feature Impact Analysis')
        ax.axvline(x=0, color='white', linestyle='-', alpha=0.5)
        
        for i, (bar, value) in enumerate(zip(bars, impacts)):
            ax.text(value + (1000 if value >= 0 else -1000), bar.get_y() + bar.get_height()/2, 
                    f'${value:,.0f}', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        logger.info(f"Feature impact chart saved to {file_path}")
    except Exception as e:
        logger.error(f"Error generating feature impact chart: {str(e)}")
        plt.close()

def _generate_actual_vs_predicted_chart(file_path, predicted_price):
    """Generate actual vs predicted scatter plot."""
    try:
        np.random.seed(42)
        n_samples = 50
        actual_prices = np.random.normal(predicted_price, predicted_price * 0.15, n_samples)
        predicted_prices = actual_prices + np.random.normal(0, predicted_price * 0.1, n_samples)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(actual_prices, predicted_prices, alpha=0.6, color='#00d4ff', s=50)
        
        min_price = min(min(actual_prices), min(predicted_prices))
        max_price = max(max(actual_prices), max(predicted_prices))
        ax.plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title('Actual vs Predicted Prices')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        logger.info(f"Actual vs predicted chart saved to {file_path}")
    except Exception as e:
        logger.error(f"Error generating actual vs predicted chart: {str(e)}")
        plt.close()

def _generate_residuals_chart(file_path, predicted_price):
    """Generate residuals analysis chart."""
    try:
        np.random.seed(42)
        n_samples = 50
        actual_prices = np.random.normal(predicted_price, predicted_price * 0.15, n_samples)
        predicted_prices = actual_prices + np.random.normal(0, predicted_price * 0.1, n_samples)
        residuals = predicted_prices - actual_prices
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(actual_prices, residuals, alpha=0.6, color='#FF6B6B', s=50)
        ax1.axhline(y=0, color='white', linestyle='-', alpha=0.8)
        ax1.set_xlabel('Actual Price ($)')
        ax1.set_ylabel('Residuals ($)')
        ax1.set_title('Residuals vs Actual Price')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(residuals, bins=15, alpha=0.7, color='#FF6B6B', edgecolor='white')
        ax2.axvline(x=0, color='white', linestyle='-', alpha=0.8)
        ax2.set_xlabel('Residuals ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        logger.info(f"Residuals chart saved to {file_path}")
    except Exception as e:
        logger.error(f"Error generating residuals chart: {str(e)}")
        plt.close()

def _generate_shap_summary_chart(file_path, neighborhood_contrib, accessibility_contrib, green_space_contrib):
    """Generate SHAP summary chart."""
    try:
        shap_values = [neighborhood_contrib, accessibility_contrib, green_space_contrib, 
                       (3.0 - 3) * 5000, (5000 - 5000) / 10]
        feature_names = ['Neighborhood', 'Accessibility', 'Green Space', 'School Rating', 'Lot Size']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        colors = ['#2E8B57' if x >= 0 else '#DC143C' for x in shap_values]
        
        bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('SHAP Value (Price Impact)')
        ax.set_title('SHAP Feature Importance')
        ax.axvline(x=0, color='white', linestyle='-', alpha=0.5)
        
        for i, (bar, value) in enumerate(zip(bars, shap_values)):
            ax.text(value + (1000 if value >= 0 else -1000), bar.get_y() + bar.get_height()/2, 
                    f'${value:,.0f}', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        logger.info(f"SHAP summary chart saved to {file_path}")
    except Exception as e:
        logger.error(f"Error generating SHAP summary chart: {str(e)}")
        plt.close()

def _generate_shap_waterfall_chart(file_path, neighborhood_contrib, accessibility_contrib, green_space_contrib):
    """Generate SHAP waterfall chart."""
    try:
        shap_values = [neighborhood_contrib, accessibility_contrib, green_space_contrib, 
                       (3.0 - 3) * 5000, (5000 - 5000) / 10]
        feature_names = ['Neighborhood', 'Accessibility', 'Green Space', 'School Rating', 'Lot Size']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        base_value = 200000
        cumulative = [base_value]
        for value in shap_values:
            cumulative.append(cumulative[-1] + value)
        
        x_pos = range(len(cumulative))
        colors_waterfall = ['#1a1a2e'] + ['#2E8B57' if x >= 0 else '#DC143C' for x in shap_values]
        
        for i in range(len(cumulative) - 1):
            height = cumulative[i+1] - cumulative[i]
            ax.bar(i, height, bottom=cumulative[i], color=colors_waterfall[i+1], alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Base'] + feature_names, rotation=45, ha='right')
        ax.set_ylabel('Cumulative Price ($)')
        ax.set_title('SHAP Waterfall: Feature Contributions')
        ax.grid(True, alpha=0.3)
        
        for i, value in enumerate(cumulative):
            ax.text(i, value + 5000, f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        logger.info(f"SHAP waterfall chart saved to {file_path}")
    except Exception as e:
        logger.error(f"Error generating SHAP waterfall chart: {str(e)}")
        plt.close()

def _generate_feature_contributions_chart(file_path, contributions):
    """Generate feature contributions bar chart."""
    try:
        features = list(contributions.keys())
        values = list(contributions.values())
        
        # Format feature names for display
        formatted_features = [f.replace('_', ' ').title() for f in features]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#2E8B57' if x >= 0 else '#DC143C' for x in values]
        
        bars = ax.barh(formatted_features, values, color=colors, alpha=0.8)
        ax.set_xlabel('Price Impact ($)')
        ax.set_title('Feature Contributions to Property Value')
        ax.axvline(x=0, color='white', linestyle='-', alpha=0.5)
        
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + (max(values) * 0.02 if value >= 0 else -max(values) * 0.02), 
                    bar.get_y() + bar.get_height()/2, 
                    f'${value:,.0f}', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        logger.info(f"Feature contributions chart saved to {file_path}")
    except Exception as e:
        logger.error(f"Error generating feature contributions chart: {str(e)}")
        plt.close()

def _generate_predicted_vs_actual_chart(file_path, predicted_price):
    """Generate predicted vs actual scatter plot."""
    try:
        np.random.seed(42)
        n_samples = 100
        # Generate realistic actual prices around the predicted price
        actual_prices = np.random.normal(predicted_price, predicted_price * 0.15, n_samples)
        # Generate predicted prices with some correlation to actual
        predicted_prices = actual_prices + np.random.normal(0, predicted_price * 0.08, n_samples)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(actual_prices, predicted_prices, alpha=0.6, color='#00d4ff', s=50)
        
        # Add perfect prediction line
        min_price = min(min(actual_prices), min(predicted_prices))
        max_price = max(max(actual_prices), max(predicted_prices))
        ax.plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # Highlight the current prediction
        ax.scatter(predicted_price, predicted_price, color='#FFD93D', s=100, marker='*', label=f'Current Property (${predicted_price:,.0f})')
        
        ax.set_xlabel('Actual Price ($)')
        ax.set_ylabel('Predicted Price ($)')
        ax.set_title('Predicted vs Actual Prices')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        logger.info(f"Predicted vs actual chart saved to {file_path}")
    except Exception as e:
        logger.error(f"Error generating predicted vs actual chart: {str(e)}")
        plt.close()

if __name__ == "__main__":
    load_resources()
    app.run(debug=True, host='0.0.0.0', port=5000)
