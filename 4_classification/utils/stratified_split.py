import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split


# Load the dataset
subject_details = pd.read_csv(
    "/net/mraid20/export/genie/LabData/Analyses/davidkro/full_voice/subject_details_df_new.csv"
)
medical_diagnoses = pd.read_csv(
    "/home/sarahk/PycharmProjects/DeepVoice/data/raw/all_cohorts_diagnoses_df.csv"
)
medical_diagnoses["subject_number"] = (
    medical_diagnoses["RegistrationCode"].str.split("_").str[1].astype(int)
)
medical_diagnoses.set_index("subject_number", inplace=True)
subject_details.set_index("subject_number", inplace=True)
subject_details = subject_details.join(medical_diagnoses, how="inner")
X = subject_details["RegistrationCode"].values.reshape(-1, 1)
y = subject_details[
    [
        "Sleep Apnea",
        "Hyperlipidemia",
        "Haemorrhoids",
        "Allergy",
        "Back Pain",
        "ADHD",
        "Hypertension",
        "Urinary tract infection",
        "Fractures",
        "Prediabetes",
        "Obesity",
        "Fatty Liver Disease",
        "Anal Fissure",
        "Migraine",
        "B12 Deficiency",
        "Anemia",
        "Asthma",
        "Hearing loss",
        "Gallstone Disease",
        "Sinusitis",
        "Atopic Dermatitis",
        "Oral apthae",
        "Depression",
        "Urinary Tract Stones",
        "Anxiety",
        "IBS",
        "Osteoarthritis",
        "gender",
    ]
].values
X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)
# save splits to csv
X_train = pd.DataFrame(X_train, columns=["RegistrationCode"])
X_train.to_csv(
    "/home/sarahk/PycharmProjects/DeepVoice/data/splits/train.csv", index=False
)
X_test = pd.DataFrame(X_test, columns=["RegistrationCode"])
X_test.to_csv(
    "/home/sarahk/PycharmProjects/DeepVoice/data/splits/test.csv", index=False
)
