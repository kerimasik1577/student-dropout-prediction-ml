import joblib
import pandas as pd
import numpy as np

def create_engineered_features(df, top_cols):
    """
    Top-5 kritik özellik üzerinden etkileşim ve satır bazlı istatistik türetir.
    """
    df_new = df.copy()
    for i in range(len(top_cols)):
        for j in range(i + 1, len(top_cols)):
            col1 = top_cols[i]
            col2 = top_cols[j]
            df_new[f"{col1}_x_{col2}"] = df_new[col1] * df_new[col2]
            df_new[f"{col1}_ratio_{col2}"] = df_new[col1] / (df_new[col2] + 1e-5)
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df_new['row_mean'] = df[num_cols].mean(axis=1)
    df_new['row_std'] = df[num_cols].std(axis=1)
    df_new['row_median'] = df[num_cols].median(axis=1)
    return df_new

def predict_student_risk(student_raw_df, model_path='models/lightgbm_model.pkl', threshold=0.65):
    """
    Öğrenci verisini alır, gerekli öznitelik mühendisliğini uygular ve 0.65 baraj eşiğine göre risk üretir.
    """
    model = joblib.load(model_path)
    top_5_features = joblib.load('models/top_5_features.pkl')
    selected_features = joblib.load('models/selected_features.pkl')
    
    df_eng = create_engineered_features(student_raw_df, top_5_features)
    df_eng = df_eng[selected_features]
    
    grad_idx = np.where(model.classes_ == 'Graduate')[0][0] if 'Graduate' in model.classes_ else 1
    grad_probs = model.predict_proba(df_eng)[:, grad_idx]
    
    results = []
    for prob in grad_probs:
        dropout_risk_pct = (1.0 - prob) * 100
        if prob >= threshold:
            prediction = 'Graduate (Mezun Olabilir)'
            risk_status = 'DÜŞÜK RİSK'
        else:
            prediction = 'Dropout (Terke Yatkın)'
            risk_status = 'YÜKSEK RİSK (Erken Müdahale Gerekli)'
            
        results.append({
            'Prediction': prediction,
            'Risk Status': risk_status,
            'Dropout Risk Pct': f"%{dropout_risk_pct:.2f}",
            'Graduate Probability': f"%{prob*100:.2f}"
        })
        
    return pd.DataFrame(results)
