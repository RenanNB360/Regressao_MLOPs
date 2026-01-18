import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from app.services.model_service import ModelService

def test_model_service_init_success():
    with patch.object(Path, "exists", return_value=True), \
         patch("joblib.load", return_value=MagicMock()) as mock_load:
        
        service = ModelService()
        
        assert service.model is not None
        assert mock_load.called
        mock_load.assert_called_with(Path("models/xgboost_model.joblib"))

def test_model_service_file_not_found():
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError) as excinfo:
            ModelService()
        
        assert "Model not found" in str(excinfo.value)

def test_model_service_predict():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([5.5])
    
    with patch.object(Path, "exists", return_value=True), \
         patch("joblib.load", return_value=mock_model):
        
        service = ModelService()
        
        df_input = pd.DataFrame({"feat1": [1], "feat2": [2]})
        
        result = service.predict(df_input)
        
        assert result[0] == 5.5
        mock_model.predict.assert_called_once_with(df_input)