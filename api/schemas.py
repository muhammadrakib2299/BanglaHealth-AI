"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Diabetes
# ---------------------------------------------------------------------------

class DiabetesInput(BaseModel):
    """Input schema for diabetes risk prediction."""

    Pregnancies: float = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=300, description="Plasma glucose (mg/dL)")
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic BP (mmHg)")
    SkinThickness: float = Field(..., ge=0, le=100, description="Skin thickness (mm)")
    Insulin: float = Field(..., ge=0, le=900, description="Serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, le=70, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    Age: float = Field(..., ge=1, le=120, description="Age in years")

    model_config = {"json_schema_extra": {
        "examples": [{
            "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
            "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627, "Age": 50,
        }]
    }}


class HeartInput(BaseModel):
    """Input schema for heart disease risk prediction."""

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, le=300, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dL)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dL")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    model_config = {"json_schema_extra": {
        "examples": [{
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
        }]
    }}


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class RiskPrediction(BaseModel):
    """Single patient risk prediction response."""

    risk_level: str = Field(..., description="Predicted risk: Low, Medium, or High")
    risk_class: int = Field(..., description="Risk class (0=Low, 1=Medium, 2=High)")
    confidence: dict[str, float] = Field(..., description="Probability for each class")
    alerts: list[dict] = Field(default_factory=list, description="Clinical alerts")


class ModelInfo(BaseModel):
    """Model metadata response."""

    model_name: str
    dataset: str
    is_loaded: bool


class ComparisonResult(BaseModel):
    """Model comparison response."""

    models: list[dict]
