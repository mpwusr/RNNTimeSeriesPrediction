# PJME Hourly Energy Consumption Forecasting

This repository contains a Python script for forecasting energy consumption using the PJME hourly dataset. It implements Recurrent Neural Networks (RNNs), including vanilla RNN, LSTM, and GRU variants, with extensive feature engineering and performance evaluation.

## Dataset
- **Source**: PJME Hourly Dataset (`PJME_hourly.csv`)
- **Description**: Hourly energy consumption data (in Megawatts) for the PJM East region
- **Link**: Available in the script or from public sources like Kaggle

## Features
1. **Data Preprocessing**
   - Time-lagged observations (100 lags)
   - Cyclical features (sin/cos transformations for hours)
   - One-hot encoding for categorical variables (month, day, etc.)
   - Holiday detection using the `holidays` library
2. **Modeling**
   - Vanilla RNN, LSTM, and GRU implementations using PyTorch
   - Configurable hyperparameters (layers, hidden units, dropout)
3. **Training & Evaluation**
   - Train/validation/test split (80/10/10)
   - MinMax scaling
   - Batch training with DataLoaders
   - Loss plotting and error metrics (MAE, RMSE, R²)
4. **Baseline**
   - Linear Regression model for comparison
5. **Visualization**
   - Interactive Plotly plots for data exploration and predictions

## Requirements
```bash
pip install torch pandas numpy matplotlib plotly sklearn holidays

## Usage
1. Ensure `PJME_hourly.csv` is in the working directory
2. Run the script:
```bash
python energy_forecasting.py
```
3. View outputs:
   - Console: Loss values, metrics
   - Plots: Dataset visualization, loss curves, predictions vs. actuals

## Script Structure
- **Data Loading**: Loads and preprocesses PJME hourly data
- **Feature Engineering**: Adds lags, cyclical features, holidays
- **Model Definition**: Implements RNN, LSTM, GRU classes
- **Training**: Optimizes model with Adam and MSE loss
- **Evaluation**: Computes predictions and metrics
- **Visualization**: Plots results using Plotly

## Configuration
Key parameters in the script:
- `input_dim`: Number of time lags (default: 100)
- `hidden_dim`: Hidden layer size (default: 64)
- `layer_dim`: Number of RNN layers (default: 3)
- `batch_size`: Training batch size (default: 64)
- `n_epochs`: Training epochs (default: 20)
- `learning_rate`: Optimizer learning rate (default: 1e-3)

## Output
- **Console**:
  - Training/validation loss per epoch
  - MAE, RMSE, R² for RNN and baseline models
- **Plots**:
  - Initial dataset visualization
  - Training/validation loss curves
  - Predictions vs. actual values (RNN and Linear Regression)

## Example Output
```
cuda is available.
Mean Absolute Error:       1234.56
Root Mean Squared Error:   1678.90
R^2 Score:                 0.85
```
- Interactive Plotly graphs showing energy consumption trends and predictions

## Notes
- Updated by Sen Wang (Mar 13, 2024) for compatibility with newer Pandas versions
- Original source: Google Colab notebook (see code comments)
- Requires GPU for faster training if available (`cuda` support)
- Install `holidays` separately if not already present (`pip install holidays`)
- Model selection (RNN/LSTM/GRU) can be switched via `get_model()`

## License
This project is open-source and available under the MIT License.
```
