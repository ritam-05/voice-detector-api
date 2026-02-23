import json

config = {
    "threshold": 0.20,
    "model_type": "LightGBM",
    "parameters" : "n_estimators=300,max_depth=6,num_leaves=31,min_child_samples=50,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,objective='binary',random_state=42,n_jobs=-1",
    "evaluation": "LOLO",
    "priority": "Minimize AI false negatives"
}

with open("inference_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Inference config saved")
