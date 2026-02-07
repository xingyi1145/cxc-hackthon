import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BayesianLinear(nn.Module):
    """Bayesian Linear Layer with weight uncertainty"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-5, 0.1))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).normal_(-5, 0.1))
    
    def forward(self, x):
        # Sample weights and biases
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return nn.functional.linear(x, weight, bias)
    
    def kl_divergence(self):
        """Calculate KL divergence for regularization"""
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        
        kl_weight = 0.5 * torch.sum(self.weight_mu**2 + weight_sigma**2 - 2*self.weight_log_sigma - 1)
        kl_bias = 0.5 * torch.sum(self.bias_mu**2 + bias_sigma**2 - 2*self.bias_log_sigma - 1)
        
        return kl_weight + kl_bias


class BayesianRegressionModel(nn.Module):
    """Bayesian Neural Network for Regression"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(BayesianLinear(prev_dim, 1))
        
        self.network = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x
    
    def kl_divergence(self):
        """Total KL divergence from all Bayesian layers"""
        kl_total = 0
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                kl_total += layer.kl_divergence()
        return kl_total
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """Make predictions with uncertainty estimates"""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std


def load_and_preprocess_data(file_path):
    """Load and preprocess the housing data with temporal features"""
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nYear range: {df['Year'].min()} - {df['Year'].max()}")
    
    # Replace Year Built with Renovation Year if available
    df['Effective Year Built'] = df['Renovation Year'].fillna(df['Year Built'])
    
    # === TEMPORAL FEATURES ===
    # Property age at time of sale
    df['Property Age'] = df['Year'] - df['Effective Year Built']
    
    # Years since last renovation (0 if never renovated)
    df['Years Since Renovation'] = df.apply(
        lambda row: row['Year'] - row['Renovation Year'] if pd.notna(row['Renovation Year']) else 0,
        axis=1
    )
    
    # Was renovated flag
    df['Was Renovated'] = df['Renovation Year'].notna().astype(int)
    
    # Price per square foot (for context, not used as feature)
    df['Price Per SqFt'] = df['Market Price'] / df['Square Footage (House)']
    
    # Total square footage
    df['Total SqFt'] = df['Square Footage (House)'] + df['Square Footage (Land)']
    
    # Bedrooms per bathroom ratio
    df['Bed_Bath_Ratio'] = df['Bedrooms'] / (df['Bathrooms'] + 1)  # +1 to avoid division by zero
    
    # === CYCLICAL ENCODING FOR SEASON ===
    season_map = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    df['Season_num'] = df['Season'].map(season_map)
    df['Season_sin'] = np.sin(2 * np.pi * df['Season_num'] / 4)
    df['Season_cos'] = np.cos(2 * np.pi * df['Season_num'] / 4)
    
    # === NEIGHBORHOOD PRICE TRENDS ===
    # Calculate average price per neighborhood per year
    neighborhood_yearly_avg = df.groupby(['Neighborhood', 'Year'])['Market Price'].mean().reset_index()
    neighborhood_yearly_avg.columns = ['Neighborhood', 'Year', 'Neighborhood_Avg_Price']
    df = df.merge(neighborhood_yearly_avg, on=['Neighborhood', 'Year'], how='left')
    
    # Calculate overall neighborhood price tier
    neighborhood_avg = df.groupby('Neighborhood')['Market Price'].mean().reset_index()
    neighborhood_avg.columns = ['Neighborhood', 'Neighborhood_Overall_Avg']
    df = df.merge(neighborhood_avg, on='Neighborhood', how='left')
    
    # === YEAR-OVER-YEAR TRENDS ===
    # Overall market price trend by year
    yearly_avg = df.groupby('Year')['Market Price'].mean().reset_index()
    yearly_avg.columns = ['Year', 'Market_Avg_Price']
    df = df.merge(yearly_avg, on='Year', how='left')
    
    # Create mappings for categorical variables
    categorical_cols = ['Neighborhood', 'Property Type', 'Garage Type', 'Basement']
    
    mappings = {}
    for col in categorical_cols:
        if col in df.columns:
            # Fill NaN with a placeholder for categorical encoding
            df[col] = df[col].fillna('None')
            unique_vals = sorted(df[col].unique())
            mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
            df[col + '_encoded'] = df[col].map(mappings[col])
    
    print(f"\nCategorical mappings created for: {list(mappings.keys())}")
    
    # Select features for the model
    feature_cols = [
        'Year', 'Bedrooms', 'Bathrooms', 'Property Age', 'Years Since Renovation',
        'Was Renovated', 'Square Footage (House)', 'Square Footage (Land)', 
        'Total SqFt', 'Legal Units', 'Bed_Bath_Ratio',
        'Season_sin', 'Season_cos',
        'Neighborhood_Avg_Price', 'Neighborhood_Overall_Avg', 'Market_Avg_Price',
        'Neighborhood_encoded', 'Property Type_encoded',
        'Garage Type_encoded', 'Basement_encoded'
    ]
    
    X = df[feature_cols].values
    y = df['Market Price'].values.reshape(-1, 1)
    years = df['Year'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    return X, y, years, feature_cols, mappings, df


def train_bayesian_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """Train the Bayesian regression model with learning rate scheduling"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    print("\n" + "="*60)
    print("Training Bayesian Regression Model")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_mse = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_batch)
            
            # Loss = MSE + KL divergence (ELBO)
            mse_loss = nn.functional.mse_loss(predictions, y_batch)
            kl_loss = model.kl_divergence() / len(train_loader.dataset)
            loss = mse_loss + 0.01 * kl_loss  # Weight KL term
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += nn.functional.mse_loss(predictions, y_batch).item()
        
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train MSE: {train_mse:.2f}, "
                  f"Val MSE: {val_loss:.2f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("="*60)
    print(f"Training completed! Best Val MSE: {best_val_loss:.2f}")
    print("="*60)
    
    return train_losses, val_losses


def evaluate_model(model, X_test, y_test, years_test=None, dataset_name="Model Evaluation"):
    """Evaluate model with uncertainty quantification and temporal analysis"""
    X_test_tensor = torch.FloatTensor(X_test)
    
    mean_pred, std_pred = model.predict_with_uncertainty(X_test_tensor, n_samples=100)
    
    # Calculate metrics
    mse = np.mean((mean_pred.flatten() - y_test.flatten())**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(mean_pred.flatten() - y_test.flatten()))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test.flatten() - mean_pred.flatten()) / y_test.flatten())) * 100
    
    # Calculate R² score
    ss_res = np.sum((y_test.flatten() - mean_pred.flatten())**2)
    ss_tot = np.sum((y_test.flatten() - y_test.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n{dataset_name}")
    print("="*60)
    print(f"RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    print(f"MAE (Mean Absolute Error):      ${mae:,.2f}")
    print(f"MAPE (Mean Abs Percentage Err):  {mape:.2f}%")
    print(f"R² Score:                        {r2:.4f}")
    print(f"\nUncertainty Statistics:")
    print(f"  Mean Uncertainty (Std Dev):    ${std_pred.mean():,.2f}")
    print(f"  Min Uncertainty:               ${std_pred.min():,.2f}")
    print(f"  Max Uncertainty:               ${std_pred.max():,.2f}")
    print("="*60)
    
    # Per-year analysis if years provided
    if years_test is not None:
        print(f"\nPer-Year Performance Analysis:")
        print("-"*60)
        unique_years = np.unique(years_test)
        for year in sorted(unique_years):
            year_mask = years_test == year
            year_actual = y_test[year_mask].flatten()
            year_pred = mean_pred[year_mask].flatten()
            year_rmse = np.sqrt(np.mean((year_actual - year_pred)**2))
            year_mae = np.mean(np.abs(year_actual - year_pred))
            year_mape = np.mean(np.abs((year_actual - year_pred) / year_actual)) * 100
            print(f"  Year {int(year)}: RMSE=${year_rmse:>10,.0f}, MAE=${year_mae:>10,.0f}, MAPE={year_mape:>5.1f}%")
        print("-"*60)
    
    # Show some example predictions with uncertainty
    print("\nExample Predictions (first 10 samples):")
    print("-" * 60)
    print(f"{'Actual':>15} {'Predicted':>15} {'Uncertainty':>15} {'Error':>15}")
    print("-" * 60)
    
    for i in range(min(10, len(y_test))):
        actual = y_test[i, 0]
        pred = mean_pred[i, 0]
        unc = std_pred[i, 0]
        error = abs(actual - pred)
        print(f"${actual:>13,.2f} ${pred:>13,.2f} ±${unc:>11,.2f} ${error:>13,.2f}")
    
    return mse, mae, r2, mean_pred, std_pred


def main():
    print("\n" + "="*60)
    print("Bayesian Regression for Future House Price Prediction")
    print("="*60)
    
    # Load and preprocess data
    file_path = 'data/synthetic_house_prices_20_years.csv'
    X, y, years, feature_cols, mappings, df = load_and_preprocess_data(file_path)
    
    # === TIME-BASED SPLIT (Critical for future prediction!) ===
    # Train on 2004-2020 (70%), Validate on 2021-2023 (30%)
    train_mask = years <= 2020
    val_mask = years > 2020
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    years_train = years[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    years_val = years[val_mask]
    
    print("\n" + "="*60)
    print("TIME-BASED SPLIT (Simulates Real Future Prediction)")
    print("="*60)
    print(f"Training set:   2004-2020 ({X_train.shape[0]} samples, {X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set: 2021-2023 ({X_val.shape[0]} samples, {X_val.shape[0]/len(X)*100:.1f}%)")
    print("="*60)
    
    # Standardize features (fit only on training data!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Create data loaders
    train_dataset = HousingDataset(X_train, y_train)
    val_dataset = HousingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model with more capacity for additional features
    input_dim = X_train.shape[1]
    model = BayesianRegressionModel(input_dim, hidden_dims=[256, 128, 64])
    
    print(f"\nModel architecture (input dim: {input_dim}):")
    print(model)
    print(f"\nFeatures used: {len(feature_cols)}")
    
    # Train model with more epochs and scheduling
    train_losses, val_losses = train_bayesian_model(
        model, train_loader, val_loader, 
        epochs=100, lr=0.001
    )
    
    # Evaluate on validation set (future years)
    mse, mae, r2, predictions, uncertainties = evaluate_model(
        model, X_val, y_val, years_val, 
        "FUTURE PREDICTION SCORES (2021-2023)"
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"\nKey Improvements for Future Prediction:")
    print(f"  ✓ Time-based train/validation split")
    print(f"  ✓ {len(feature_cols)} temporal and engineered features")
    print(f"  ✓ Cyclical season encoding (sin/cos)")
    print(f"  ✓ Neighborhood price trends and market averages")
    print(f"  ✓ Property age and renovation features")
    print(f"  ✓ Learning rate scheduling + early stopping")
    print(f"  ✓ Uncertainty quantification via Bayesian inference")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
