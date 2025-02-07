import optuna

db_url = "sqlite:///resnet_wafer_v2.db"
study_name = "resnet_wafer"

# Delete the study from the database.
optuna.delete_study(study_name=study_name, storage=db_url)
print(f"Study '{study_name}' deleted successfully.")