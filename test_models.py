#!/usr/bin/env python3
"""
Test script to verify model loading and basic inference
"""

import joblib
import numpy as np
import pandas as pd

def test_models():
    """Test that all models load and can make predictions"""

    print("Testing model loading...")

    try:
        # Load models
        exercise_model = joblib.load('exercise_classifier_model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        rep_detector_s1 = joblib.load('rep_detector_model_s1.joblib')
        rep_counter_s2 = joblib.load('rep_counter_model_s2.joblib')
        rep_features = joblib.load('rep_counter_features.joblib')

        print("All models loaded successfully")

        # Test exercise classification with dummy data
        print("\nTesting exercise classification...")

        # Create dummy feature vector (44 features for 4 sensors × 11 features each)
        dummy_features = np.random.rand(1, 44)
        feature_names = [f'feature_{i}' for i in range(44)]
        X_test = pd.DataFrame(dummy_features, columns=feature_names)

        # Make prediction
        pred = exercise_model.predict(X_test)[0]
        proba = exercise_model.predict_proba(X_test)[0]
        exercise = label_encoder.inverse_transform([pred])[0]

        print(f"✓ Exercise prediction: {exercise}")
        print(f"✓ Confidence: {proba[pred]:.3f}")

        # Test rep detection with dummy data
        print("\nTesting rep detection...")

        # Create dummy rep features
        rep_dummy = np.random.rand(1, len(rep_features))
        X_rep = pd.DataFrame(rep_dummy, columns=rep_features)

        s1_pred = rep_detector_s1.predict(X_rep.values)[0]
        s1_proba = rep_detector_s1.predict_proba(X_rep.values)[0]

        print(f"✓ Activity detection: {'Active' if s1_pred == 1 else 'Inactive'}")
        print(f"✓ Activity confidence: {s1_proba[1]:.3f}")

        if s1_pred == 1:
            s2_pred = rep_counter_s2.predict(X_rep.values)[0]
            rep_count = 2 if s2_pred == 1 else 1
            print(f"✓ Rep count prediction: {rep_count}")

        print("\n✓ All model tests passed!")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    return True

if __name__ == "__main__":
    test_models()