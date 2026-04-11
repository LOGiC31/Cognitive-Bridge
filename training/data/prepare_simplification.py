"""
Prepare parallel medical simplification training data.

Loads the GEM/cochrane-simplification dataset (~4,500 paragraph-level pairs
of Cochrane systematic review abstracts and their plain-language summaries)
and supplements with hand-curated sentence-level clinical pairs.

The prompt prefix matches the Chrome extension's inference prompt so the
fine-tuned model responds to the same instruction at deployment time.

Usage:
    python training/data/prepare_simplification.py

Output:
    training/data/simplification_pairs/ — HuggingFace Dataset on disk
"""

import json
import os
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simplification_pairs")

PREFIX = "Simplify this medical text for a patient: "

CURATED_PAIRS = [
    {
        "complex": "The patient presents with acute myocardial infarction with ST-segment elevation in leads V1-V4.",
        "simple": "The patient is having a heart attack. The heart test shows a specific pattern (ST elevation) that indicates a blockage in the front part of the heart."
    },
    {
        "complex": "Echocardiogram reveals left ventricular ejection fraction of 35%, consistent with moderate systolic dysfunction.",
        "simple": "A heart ultrasound shows the heart is only pumping out 35% of its blood with each beat, which is lower than normal (55-70%). This means the heart isn't pumping as strongly as it should."
    },
    {
        "complex": "Patient diagnosed with bilateral pulmonary embolism requiring anticoagulation therapy.",
        "simple": "Blood clots were found in both lungs. The patient needs blood-thinning medication to prevent the clots from getting bigger."
    },
    {
        "complex": "Labs reveal elevated creatinine at 2.8 mg/dL suggestive of acute kidney injury.",
        "simple": "A blood test shows high creatinine levels (2.8, normal is about 0.7-1.3), which suggests the kidneys are not working properly right now."
    },
    {
        "complex": "Hemoglobin A1c of 9.2% indicates poorly controlled diabetes mellitus.",
        "simple": "The blood sugar test (A1c) is 9.2%, which is higher than the target of below 7%. This means blood sugar has been too high over the past 2-3 months."
    },
    {
        "complex": "MRI of the lumbar spine demonstrates disc herniation at L4-L5 with moderate foraminal stenosis.",
        "simple": "A scan of the lower back shows a disc (cushion between the bones) is bulging out at the L4-L5 level, and it's narrowing the space where the nerves come out."
    },
    {
        "complex": "The patient has a history of atrial fibrillation managed with rate control using metoprolol and anticoagulation with warfarin.",
        "simple": "The patient has an irregular heartbeat (atrial fibrillation). They take metoprolol to keep the heart rate steady and warfarin (a blood thinner) to prevent blood clots."
    },
    {
        "complex": "Biopsy of the hepatic lesion reveals well-differentiated hepatocellular carcinoma.",
        "simple": "A tissue sample from a growth in the liver shows it is a type of liver cancer. 'Well-differentiated' means the cancer cells still look somewhat like normal liver cells."
    },
    {
        "complex": "Chest X-ray demonstrates bilateral pleural effusions and cardiomegaly.",
        "simple": "A chest X-ray shows fluid has collected around both lungs and the heart appears larger than normal."
    },
    {
        "complex": "Complete blood count shows pancytopenia with WBC 2.1, hemoglobin 8.2, and platelets 45,000.",
        "simple": "A blood test shows all three main blood cell types are low: white blood cells (which fight infection), red blood cells (which carry oxygen), and platelets (which help with clotting)."
    },
    {
        "complex": "Patient presents with diabetic ketoacidosis with pH 7.15 and blood glucose of 520 mg/dL.",
        "simple": "The patient's diabetes has caused a dangerous buildup of acid in the blood. Their blood sugar is extremely high (520, normal is under 140), and the blood is more acidic than it should be."
    },
    {
        "complex": "Colonoscopy reveals multiple diverticula in the sigmoid colon with evidence of prior diverticulitis.",
        "simple": "A camera exam of the large intestine found several small pouches in the lower part of the colon. There are signs that these pouches were previously inflamed or infected."
    },
    {
        "complex": "Thyroid function tests demonstrate elevated TSH with low free T4, consistent with primary hypothyroidism.",
        "simple": "Blood tests show the thyroid gland is underactive. The brain is sending stronger signals (high TSH) to the thyroid, but it's not producing enough hormone (low T4)."
    },
    {
        "complex": "Patient requires emergent cholecystectomy for acute calculous cholecystitis with gallbladder wall thickening.",
        "simple": "The patient needs urgent surgery to remove their gallbladder. Gallstones are causing the gallbladder to become inflamed and its walls are swollen."
    },
    {
        "complex": "CT angiogram demonstrates 70% stenosis of the left anterior descending coronary artery.",
        "simple": "A special heart scan shows a 70% blockage in one of the main blood vessels supplying the heart (the LAD artery). This means blood flow to part of the heart is significantly reduced."
    },
    {
        "complex": "Urinalysis positive for leukocyte esterase and nitrites, consistent with urinary tract infection.",
        "simple": "A urine test shows signs of a bladder or urinary tract infection — there are markers indicating bacteria and white blood cells in the urine."
    },
    {
        "complex": "EEG demonstrates generalized epileptiform discharges consistent with primary generalized epilepsy.",
        "simple": "A brain wave test (EEG) shows abnormal electrical activity throughout the brain, which is typical of a type of epilepsy (seizure disorder) that affects the whole brain."
    },
    {
        "complex": "Spirometry reveals FEV1/FVC ratio of 0.58 consistent with moderate obstructive airway disease.",
        "simple": "A breathing test shows the airways are partially blocked, making it harder to breathe out quickly. This indicates a moderate level of lung disease like COPD or asthma."
    },
    {
        "complex": "Patient initiated on dual antiplatelet therapy with aspirin and clopidogrel post percutaneous coronary intervention.",
        "simple": "After a procedure to open a blocked heart artery (using a catheter and stent), the patient started taking two blood-thinning medications — aspirin and clopidogrel — to prevent new clots."
    },
    {
        "complex": "Lumbar puncture demonstrates elevated opening pressure with lymphocytic pleocytosis suggestive of viral meningitis.",
        "simple": "A spinal tap showed increased pressure and a high number of infection-fighting cells in the spinal fluid, suggesting a viral infection of the membranes covering the brain and spinal cord."
    },
    {
        "complex": "The patient has chronic kidney disease stage 3b with an estimated GFR of 38 mL/min.",
        "simple": "The kidneys are working at about 38% of normal capacity. This is a moderate stage of long-term kidney disease."
    },
    {
        "complex": "Doppler ultrasound of the lower extremities reveals deep vein thrombosis in the left popliteal vein.",
        "simple": "An ultrasound of the legs found a blood clot in a deep vein behind the left knee."
    },
    {
        "complex": "Bone densitometry reveals T-score of -2.8 at the lumbar spine consistent with osteoporosis.",
        "simple": "A bone density scan shows the bones in the lower back are significantly weaker than normal, which means they are more likely to break. This is called osteoporosis."
    },
    {
        "complex": "Patient presents with sepsis secondary to perforated appendicitis with peritonitis.",
        "simple": "The patient's appendix burst, causing infection to spread throughout the belly. This has led to a serious whole-body infection called sepsis."
    },
    {
        "complex": "Cardiac catheterization demonstrates three-vessel coronary artery disease requiring surgical revascularization.",
        "simple": "A heart procedure found that three major blood vessels supplying the heart are significantly blocked. The patient needs bypass surgery to restore blood flow to the heart."
    },
]


def load_cochrane():
    """Load the GEM/cochrane-simplification dataset from JSON files."""
    print("Loading GEM/cochrane-simplification dataset...")
    base = "hf://datasets/GEM/cochrane-simplification"
    ds = load_dataset("json", data_files={
        "train": f"{base}/train.json",
        "validation": f"{base}/validation.json",
        "test": f"{base}/test.json",
    })
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
    return ds


def build_dataset():
    """Create the combined simplification dataset."""
    cochrane = load_cochrane()

    curated_inputs = [PREFIX + p["complex"] for p in CURATED_PAIRS]
    curated_targets = [p["simple"] for p in CURATED_PAIRS]
    curated_ds = Dataset.from_dict({
        "input_text": curated_inputs,
        "target_text": curated_targets,
    })
    print(f"\nCurated pairs: {len(curated_ds)}")

    splits = {}
    for split_name in ["train", "validation", "test"]:
        split_data = cochrane[split_name]
        cochrane_inputs = [PREFIX + row["source"] for row in split_data]
        cochrane_targets = [row["target"] for row in split_data]
        cochrane_ds = Dataset.from_dict({
            "input_text": cochrane_inputs,
            "target_text": cochrane_targets,
        })

        if split_name == "train":
            combined = concatenate_datasets([cochrane_ds, curated_ds])
            print(f"  {split_name}: {len(cochrane_ds)} (Cochrane) + {len(curated_ds)} (curated) = {len(combined)}")
        else:
            combined = cochrane_ds
            print(f"  {split_name}: {len(combined)}")

        splits[split_name] = combined

    dataset = DatasetDict(splits)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset.save_to_disk(OUTPUT_DIR)
    print(f"\nDataset saved to {OUTPUT_DIR}")

    with open(os.path.join(OUTPUT_DIR, "pairs.json"), "w") as f:
        json.dump(CURATED_PAIRS, f, indent=2)

    print("\nSample (Cochrane):")
    s = dataset["train"][0]
    print(f"  Input:  {s['input_text'][:100]}...")
    print(f"  Target: {s['target_text'][:100]}...")

    print("\nSample (curated):")
    s = dataset["train"][-1]
    print(f"  Input:  {s['input_text'][:100]}...")
    print(f"  Target: {s['target_text'][:100]}...")


if __name__ == "__main__":
    if os.path.isfile(os.path.join(OUTPUT_DIR, "dataset_dict.json")):
        print(f"Dataset already exists at {OUTPUT_DIR}, skipping preparation.")
        print("Delete the directory to re-prepare: rm -rf " + OUTPUT_DIR)
    else:
        build_dataset()
