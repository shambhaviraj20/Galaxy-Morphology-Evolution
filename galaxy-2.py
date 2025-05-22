import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix, accuracy_score
from PIL import Image
import glob
import random
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


df = pd.read_csv("GalaxyZoo1_DR_table2.csv")
feature_cols = ['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK',
                'P_MG', 'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED']
df = df[feature_cols].dropna()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)


class ImprovedGalaxyVAE(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[32, 16], latent_dim=5):
        super(ImprovedGalaxyVAE, self).__init__()
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder layers
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]  # Reverse the hidden dimensions
        prev_dim = latent_dim
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = dim
            
        # Final output layer
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class GalaxyPINN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32):
        super(GalaxyPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x_t = torch.cat([x, t], dim=1)
        return self.net(x_t)


def vae_loss_function(recon_x, x, mu, logvar, beta=0.5):
    """
    Enhanced VAE loss function with beta parameter to balance KL divergence
    """
    # Binary Cross Entropy for better reconstruction of binary/probability features
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Add some feature-specific weighting based on importance
    # This can be customized based on domain knowledge
    importance_weights = torch.ones_like(x)
    importance_weights[:, [0, 7]] = 1.5  # Give more weight to P_EL and P_EL_DEBIASED
    weighted_recon = nn.functional.mse_loss(recon_x * importance_weights, 
                                           x * importance_weights, 
                                           reduction='sum')
    
    return BCE + weighted_recon + beta * KLD


def physics_loss(preds, t, weight=0.1):
    d_preds_dt = torch.autograd.grad(
        outputs=preds, inputs=t,
        grad_outputs=torch.ones_like(preds),
        create_graph=True
    )[0]
    return weight * torch.mean(d_preds_dt**2)


def draw_spiral_galaxy_features(step, features):
    P_EL, P_CW, P_ACW, P_EDGE, P_DK, P_MG, P_CS, P_EL_DEBIASED, P_CS_DEBIASED = features
    num_arms = int(2 + 4 * (P_CW + P_ACW))  # 2–6 arms
    spiral_tightness = 0.5 + 3 * (1 - P_EL)  # tighter for spirals
    brightness = 0.5 + P_CS  # brighter center

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor('black')
    num_stars = 1000
    r = np.linspace(0.05, 1.0, num_stars)
    theta = r * spiral_tightness * np.pi + step * 0.3

    for arm in range(num_arms):
        arm_theta = theta + (2 * np.pi / num_arms) * arm
        x = r * np.cos(arm_theta) + np.random.normal(0, 0.02, num_stars)
        y = r * np.sin(arm_theta) + np.random.normal(0, 0.02, num_stars)
        color_shift = min(1.0, brightness)
        colors = plt.cm.plasma(np.linspace(0.2, color_shift, num_stars))
        ax.scatter(x, y, s=1.5, c=colors, alpha=0.9, edgecolors='none')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    plt.savefig(f"results/evolution_step_{step:02d}.png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


def simulate_evolution(pinn, vae, start_state, steps=15):
    t_vals = torch.linspace(0, 1, steps).unsqueeze(1).to(device)
    pinn.eval(); vae.eval()
    with torch.no_grad():
        for i, t in enumerate(t_vals):
            t_input = t.expand(start_state.size(0), 1)
            evolved = pinn(start_state, t_input)
            draw_spiral_galaxy_features(i, evolved.cpu().numpy()[0])


def create_gif(folder="results", gif_name="galaxy_evolution.gif", duration=200):
    image_files = sorted(glob.glob(f"{folder}/evolution_step_*.png"))
    images = [Image.open(f) for f in image_files]
    if images:
        images[0].save(
            gif_name,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"🎞️ GIF saved as {gif_name}")
    else:
        print("⚠️ No evolution images found.")


def evaluate_vae(vae, data, threshold=0.5):
    """Evaluate VAE performance with multiple metrics"""
    vae.eval()
    with torch.no_grad():
        reconstructed, mu, logvar = vae(data)
        
        # Calculate MSE and RMSE
        mse = nn.MSELoss()(reconstructed, data).item()
        rmse = np.sqrt(mse)
        
        # For classification metrics, binarize the data
        # Detach tensors before converting to numpy
        actual_binary = (data.detach().cpu().numpy() > threshold).astype(int)
        pred_binary = (reconstructed.detach().cpu().numpy() > threshold).astype(int)
        
        # Flatten for classification metrics
        actual_flat = actual_binary.flatten()
        pred_flat = pred_binary.flatten()
        
        # Calculate accuracy
        accuracy = accuracy_score(actual_flat, pred_flat)
        
        # Calculate F1 Score (micro-averaged over all features)
        f1 = f1_score(actual_flat, pred_flat, average='micro')
        
        # Create confusion matrix
        cm = confusion_matrix(actual_flat, pred_flat)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('VAE Reconstruction Confusion Matrix')
        plt.ylabel('True Values')
        plt.xlabel('Predicted Values')
        plt.savefig("results/vae_confusion_matrix.png")
        plt.close()
        
        # Calculate per-feature accuracy
        feature_accuracy = []
        for i in range(data.shape[1]):
            feature_acc = accuracy_score(actual_binary[:, i], pred_binary[:, i])
            feature_accuracy.append(feature_acc)
        
        # Plot feature-wise accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(feature_cols, feature_accuracy)
        plt.title('VAE Accuracy by Feature')
        plt.ylabel('Accuracy')
        plt.xlabel('Feature')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/vae_feature_accuracy.png")
        plt.close()
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Confusion Matrix': cm,
            'Feature Accuracy': feature_accuracy
        }


def evaluate_pinn(pinn, data, t_values=None):
    """Evaluate PINN performance with multiple metrics"""
    pinn.eval()
    
    if t_values is None:
        # Use a range of time values
        t_values = torch.linspace(0, 1, 10).to(device)
    
    results = []
    threshold = 0.5
    
    with torch.no_grad():
        for t in t_values:
            # Create t tensor for batch
            t_tensor = t.expand(data.size(0), 1)
            
            # Get predictions
            predictions = pinn(data, t_tensor)
            
            # Calculate MSE
            mse = nn.MSELoss()(predictions, data).item()
            rmse = np.sqrt(mse)
            
            # For classification metrics, binarize
            # Detach tensors before converting to numpy
            actual_binary = (data.detach().cpu().numpy() > threshold).astype(int)
            pred_binary = (predictions.detach().cpu().numpy() > threshold).astype(int)
            
            # Flatten for classification metrics
            actual_flat = actual_binary.flatten()
            pred_flat = pred_binary.flatten()
            
            # Calculate accuracy
            accuracy = accuracy_score(actual_flat, pred_flat)
            
            # Calculate F1 Score
            f1 = f1_score(actual_flat, pred_flat, average='micro')
            
            results.append({
                't': t.item(),
                'MSE': mse,
                'RMSE': rmse,
                'Accuracy': accuracy,
                'F1 Score': f1
            })
    
    # Calculate average metrics across time steps
    avg_mse = np.mean([r['MSE'] for r in results])
    avg_rmse = np.mean([r['RMSE'] for r in results])
    avg_accuracy = np.mean([r['Accuracy'] for r in results])
    avg_f1 = np.mean([r['F1 Score'] for r in results])
    
    # Create confusion matrix for middle time step for visualization
    mid_t = t_values[len(t_values)//2]
    mid_t_tensor = mid_t.expand(data.size(0), 1)
    mid_predictions = pinn(data, mid_t_tensor)
    
    # Detach tensors before converting to numpy
    mid_actual_binary = (data.detach().cpu().numpy() > threshold).astype(int)
    mid_pred_binary = (mid_predictions.detach().cpu().numpy() > threshold).astype(int)
    
    cm = confusion_matrix(mid_actual_binary.flatten(), mid_pred_binary.flatten())
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'PINN Prediction Confusion Matrix (t={mid_t.item():.2f})')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    plt.savefig("results/pinn_confusion_matrix.png")
    plt.close()
    
    return {
        'Detail': results,
        'Average MSE': avg_mse,
        'Average RMSE': avg_rmse,
        'Average Accuracy': avg_accuracy,
        'Average F1 Score': avg_f1,
        'Mid-t Confusion Matrix': cm
    }


def plot_metrics_over_time(pinn_results):
    """Plot PINN metrics over time"""
    metrics = ['MSE', 'RMSE', 'Accuracy', 'F1 Score']
    t_values = [r['t'] for r in pinn_results['Detail']]
    
    plt.figure(figsize=(12, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = [r[metric] for r in pinn_results['Detail']]
        plt.plot(t_values, values, 'o-')
        plt.title(f'PINN {metric} vs Time')
        plt.xlabel('Time (t)')
        plt.ylabel(metric)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/pinn_metrics_over_time.png")
    plt.close()


def plot_latent_space(vae, data, title="VAE Latent Space"):
    """Visualize the latent space of the VAE"""
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(data)
        z = vae.reparameterize(mu, logvar)
        z = z.cpu().numpy()
    
    # If latent space is 2D, plot directly
    if z.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(z[:, 0], z[:, 1], alpha=0.5, s=5)
        plt.title(title)
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.colorbar()
        plt.grid(True)
        plt.savefig("results/vae_latent_space.png")
        plt.close()
    
    # If latent space is 3D, make a 3D plot
    elif z.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], alpha=0.5, s=5)
        ax.set_title(title)
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        ax.set_zlabel("Latent Dimension 3")
        plt.savefig("results/vae_latent_space_3d.png")
        plt.close()
    
    # If latent space is higher dimensional, use PCA or t-SNE to reduce to 2D for visualization
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=5)
        plt.title(f"{title} (PCA)")
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")
        plt.grid(True)
        plt.savefig("results/vae_latent_space_pca.png")
        plt.close()


if __name__ == "__main__":
    # Data preprocessing (ensure balanced dataset)
    # Split data into train/test sets (80/20)
    train_size = int(0.8 * len(X_tensor))
    indices = torch.randperm(len(X_tensor)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = X_tensor[train_indices]
    test_data = X_tensor[test_indices]
    
    # Create data loaders for mini-batch training
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False
    )
    
    # ==== Train VAE ====
    vae = ImprovedGalaxyVAE(input_dim=9, hidden_dims=[32, 16], latent_dim=5).to(device)
    vae_optim = optim.Adam(vae.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(vae_optim, 'min', patience=10, factor=0.5)
    
    # Track metrics
    best_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 20
    training_losses = []
    validation_losses = []
    
    print("Training Improved VAE...")
    num_epochs = 300
    vae.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            vae_optim.zero_grad()
            x_hat, mu, logvar = vae(batch)
            
            # Use improved loss function with beta annealing
            # Start with a small beta and gradually increase it
            beta = min(1.0, 0.1 + epoch * 0.01)
            loss = vae_loss_function(x_hat, batch, mu, logvar, beta=beta)
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            vae_optim.step()
            epoch_loss += loss.item()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_loss)
        
        # Validate on test set
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                x_hat, mu, logvar = vae(batch)
                val_loss += vae_loss_function(x_hat, batch, mu, logvar, beta=beta).item()
        
        avg_val_loss = val_loss / len(test_loader)
        validation_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save the best model
            torch.save(vae.state_dict(), "models/vae_galaxy.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"VAE Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {vae_optim.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('VAE Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/vae_training_loss.png")
    plt.close()
    
    # Load the best model for evaluation
    vae.load_state_dict(torch.load("models/vae_galaxy.pth"))
    print("VAE model loaded!")
    
    # ==== Train PINN ====
    pinn = GalaxyPINN(input_dim=9, hidden_dim=64).to(device)
    pinn_optim = optim.Adam(pinn.parameters(), lr=0.001, weight_decay=1e-5)
    X_tensor.requires_grad = True
    
    # Create timesteps tensor
    batch_size = X_tensor.size(0)
    timesteps = torch.linspace(0, 1, batch_size).unsqueeze(1).to(device).requires_grad_()
    
    print("Training PINN...")
    for epoch in range(300):
        pinn_optim.zero_grad()
        out = pinn(X_tensor, timesteps)
        loss = nn.MSELoss()(out, X_tensor) + physics_loss(out, timesteps, weight=0.2)
        loss.backward()
        pinn_optim.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"PINN Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    torch.save(pinn.state_dict(), "models/pinn_galaxy.pth")
    print("PINN model saved!")
    
    # ==== Evaluate VAE Performance ====
    print("\nEvaluating VAE Performance...")
    vae_metrics = evaluate_vae(vae, X_tensor)
    print(f"VAE MSE: {vae_metrics['MSE']:.4f}")
    print(f"VAE RMSE: {vae_metrics['RMSE']:.4f}")
    print(f"VAE Accuracy: {vae_metrics['Accuracy']:.4f}")
    print(f"VAE F1 Score: {vae_metrics['F1 Score']:.4f}")
    print("VAE Feature Accuracy:")
    for i, feat in enumerate(feature_cols):
        print(f"  - {feat}: {vae_metrics['Feature Accuracy'][i]:.4f}")
    print(" VAE confusion matrix saved to results/vae_confusion_matrix.png")
    print(" VAE feature accuracy plot saved to results/vae_feature_accuracy.png")
    
    # Visualize VAE latent space
    plot_latent_space(vae, X_tensor, "Galaxy VAE Latent Space")
    print(" VAE latent space visualization saved to results/")
    
    # ==== Evaluate PINN Performance ====
    print("\n Evaluating PINN Performance...")
    test_times = torch.linspace(0, 1, 10).to(device)
    pinn_metrics = evaluate_pinn(pinn, X_tensor, test_times)
    print(f"PINN Average MSE: {pinn_metrics['Average MSE']:.4f}")
    print(f"PINN Average RMSE: {pinn_metrics['Average RMSE']:.4f}")
    print(f"PINN Average Accuracy: {pinn_metrics['Average Accuracy']:.4f}")
    print(f"PINN Average F1 Score: {pinn_metrics['Average F1 Score']:.4f}")
    print(" PINN confusion matrix saved to results/pinn_confusion_matrix.png")
    
    # Plot PINN metrics over time
    plot_metrics_over_time(pinn_metrics)
    print("PINN metrics over time plot saved to results/pinn_metrics_over_time.png")
    
    # ==== Select a Random Galaxy ====
    rand_idx = random.randint(0, len(X_tensor) - 1)
    print(f"\nSimulating galaxy at index {rand_idx}")
    sample = X_tensor[rand_idx].unsqueeze(0)
    
    # ==== Simulate and Generate GIF ====
    simulate_evolution(pinn, vae, sample)
    create_gif(gif_name="galaxy_evolution.gif")
    
    print("\n Done. You can now try different galaxies or experiment with the VAE parameters.")