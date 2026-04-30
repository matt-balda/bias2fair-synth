import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from models.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from models.tab_ddpm.modules import MLPDiffusion
from models.tab_ddpm.utils import index_to_log_onehot

class TabDDPMWrapper:
    def __init__(self, metadata, device='cuda', steps=1000, batch_size=4096):
        self.metadata = metadata
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.steps = steps
        self.batch_size = batch_size
        self.num_scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.model = None
        self.diffusion = None
        
        # Track columns
        self.num_cols = []
        self.cat_cols = []
        self.original_columns = []
        
        # Tracking ranges and types for numerical bounds clipping
        self.num_bounds = {}
        self.num_is_int = {}
        
    def _preprocess(self, df):
        # Identify numerical and categorical based on SDV metadata
        self.cat_cols = []
        self.num_cols = []
        for col in df.columns:
            # Fallback for columns not in metadata (should not happen)
            sdtype = self.metadata.columns.get(col, {}).get('sdtype', 'numerical')
            if sdtype in ['categorical', 'boolean']:
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)
        
        # Track limits
        for col in self.num_cols:
            self.num_bounds[col] = (df[col].min(), df[col].max())
            self.num_is_int[col] = pd.api.types.is_integer_dtype(df[col])

        X_num = self.num_scaler.fit_transform(df[self.num_cols].values) if self.num_cols else np.empty((len(df), 0))

        X_cat = df[self.cat_cols].values if self.cat_cols else np.empty((len(df), 0))
        
        # Categorical to Label (for multinomial diffusion)
        # Actually TabDDPM expects OHE for categories in input but handles multinomial loss
        # We'll use Label Encoding for the multinomial part
        self.label_encoders = [LabelEncoder() for _ in self.cat_cols]
        X_cat_encoded = []
        for i, col in enumerate(self.cat_cols):
            X_cat_encoded.append(self.label_encoders[i].fit_transform(df[col]))
        X_cat_encoded = np.stack(X_cat_encoded, axis=1) if self.cat_cols else np.empty((len(df), 0))
        
        return torch.tensor(X_num, dtype=torch.float32), torch.tensor(X_cat_encoded, dtype=torch.long)

    def fit(self, df):
        self.original_columns = df.columns.tolist()
        X_num, X_cat = self._preprocess(df)
        X_num = X_num.to(self.device).float()
        X_cat = X_cat.to(self.device) # Keep as long for categorical
        
        num_classes = np.array([len(le.classes_) for le in self.label_encoders])
        d_in = X_num.shape[1] + (sum(num_classes) if len(num_classes) > 0 else 0)
        
        # Model architecture
        model_params = {
            'd_layers': [256, 512, 256],
            'dropout': 0.1,
        }
        
        self.model = MLPDiffusion(
            d_in=d_in,
            num_classes=0,
            is_y_cond=False,
            rtdl_params=model_params
        ).to(self.device).float()
        
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=num_classes,
            num_numerical_features=X_num.shape[1],
            denoise_fn=self.model,
            num_timesteps=self.steps,
            device=torch.device(self.device)
        ).to(self.device).float()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Training loop
        self.model.train()
        
        # Ensure sufficient training steps (at least 5000 steps roughly)
        steps_per_epoch = max(1, X_num.shape[0] // self.batch_size)
        epochs = max(200, 5000 // steps_per_epoch)
        
        for epoch in range(epochs):
            idx = torch.randperm(X_num.shape[0])
            for i in range(0, X_num.shape[0], self.batch_size):
                batch_idx = idx[i:i+self.batch_size]
                batch_num = X_num[batch_idx]
                batch_cat = X_cat[batch_idx]
                
                optimizer.zero_grad()
                # Use the provided mixed_loss directly
                loss_multi, loss_gauss = self.diffusion.mixed_loss(
                    torch.cat([batch_num.float(), batch_cat.float()], dim=1),
                    out_dict={}
                )
                loss = loss_multi + loss_gauss
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    def sample(self, num_rows):
        self.model.eval()
        with torch.no_grad():
            b = num_rows
            device = self.device
            
            img_num = torch.randn((b, len(self.num_cols)), device=device).float()
            
            cur_img = img_num
            if len(self.cat_cols) > 0:
                num_classes_expanded = torch.from_numpy(
                    np.concatenate([np.repeat(len(le.classes_), len(le.classes_)) for le in self.label_encoders])
                ).to(device).float()
                # Create a batch of log_half_prob
                log_half_prob = -torch.log(num_classes_expanded).repeat(b, 1)
                img_cat = self.diffusion.log_sample_categorical(log_half_prob).float()
                cur_img = torch.cat([img_num, img_cat], dim=1)

            for i in reversed(range(0, self.steps)):
                t = torch.full((b,), i, device=device, dtype=torch.long)
                model_out = self.model(cur_img.float(), t).float()
                
                out_num = model_out[:, :len(self.num_cols)]
                out_cat = model_out[:, len(self.num_cols):]
                
                res_num = self.diffusion.gaussian_p_sample(out_num.float(), cur_img[:, :len(self.num_cols)].float(), t)
                res_cat = self.diffusion.p_sample(out_cat.float(), cur_img[:, len(self.num_cols):].float(), t, {})
                
                cur_img = torch.cat([res_num['sample'].float(), res_cat.float()], dim=1)

            # Post-process
            res_num = cur_img[:, :len(self.num_cols)].cpu().numpy()
            res_num = self.num_scaler.inverse_transform(res_num)
            
            res_cat_log = cur_img[:, len(self.num_cols):]
            # Convert log-onehot to labels
            res_cat_labels = []
            curr_idx = 0
            for i, le in enumerate(self.label_encoders):
                K = len(le.classes_)
                labels = res_cat_log[:, curr_idx:curr_idx+K].argmax(dim=1).cpu().numpy()
                res_cat_labels.append(le.inverse_transform(labels))
                curr_idx += K
            
            # Build DataFrame
            df_num = pd.DataFrame(res_num, columns=self.num_cols)
            
            # Post-process numerical bounds and types
            for col in self.num_cols:
                min_val, max_val = self.num_bounds[col]
                df_num[col] = df_num[col].clip(lower=min_val, upper=max_val)
                if self.num_is_int[col]:
                    df_num[col] = df_num[col].round().astype(int)

            df_cat = pd.DataFrame(np.stack(res_cat_labels, axis=1), columns=self.cat_cols)
            df_final = pd.concat([df_num, df_cat], axis=1)
            return df_final[self.original_columns]
