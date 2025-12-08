Evaluation of Graph-Based Methods for Rectangle Detection in Images

This repository contains a set of tools for the extraction, analysis, and detection of rectangular objects in videos. The modules are located in src/video_extractor/, each one providing a specific functionality.

📚 References

This project is partially based on the work published in the following repository:

🔗 Rectangle Detector using Ridge – TFM Project by Neathle
Special thanks to the original author for openly sharing their research and code.

🚀 Usage Guide
🔧 1. Opening and configuring the modules

Before running any module, it is important to open each script (FLD, Airline, ROI, YOLO) and manually adjust the parameters according to the scene:

Modify input paths (videos or images)

Adjust thresholds, ROI settings, sizes, or model parameters as needed

Verify the operating system and Python version (especially for the Airline module)

🖼 2. Image-level evaluation

If you prefer to perform evaluation using images instead of video, ensure that the input paths are updated to point to the correct image directory.

🗂 Project Structure
├── compilation/
├── installation/
├── logs/
└── src/
    └── video_extractor/
        ├── Ridge_FLD/
        ├── Ridge_Airline_FLD/
        ├── Ridge_ROI/
        └── Ridge_YOLO/


Important:
The CRG module is compiled for Python 3.11 on Windows, Linux, and macOS.
Please follow the pybind11 tutorial to compile it.
The .cpp file is located in: CRG/extractC.

🧪 SYNTHETIC DATASET – ISAAC SIM

This project was evaluated using a dataset generated synthetically in NVIDIA Isaac Sim, complemented by real data obtained from the Robot@Home2 environment.

The following links include the results from each module (Airline, FLD, ROI, YOLO) as well as their corresponding Ground Truths, used for benchmarking and performance comparison.

Each folder contains:

Original images (images)

Labels and annotations (labels)

Images with Ground Truth overlaid (outputs)

📥 Synthetic Dataset (Isaac Sim):
🔗 https://drive.google.com/drive/folders/1gZvHERa0tMA8oM3BWoNmzE76GrPac45u?usp=drive_link

🤖 REAL DATASET – ROBOT@HOME2

Additionally, the project uses the real-world Robot@Home2 dataset, organized using the same structure as the synthetic dataset to maintain consistency throughout the evaluation process.

Each folder includes:

Original images (images)

Labels and annotations (labels)

Images with Ground Truth overlaid (outputs)

📥 Robot@Home2 Dataset:
🔗 https://drive.google.com/drive/folders/1dpP8Z00DASQUeKYnOC32W2xYegjifNwW?usp=sharing

📂 Available Sessions

The repository includes evaluation sessions for each module under both datasets:

📂 SessionAIRLINE_Synthetic

📂 SessionAIRLINE_Robot@Home2

📂 SessionFLD_Synthetic

📂 SessionFLD_Robot@Home2

📂 SessionROI_Synthetic

📂 SessionROI_Robot@Home2

📂 SessionROI2_Robot@Home2

📂 SessionYOLO_Synthetic

📂 SessionYOLO_Robot@Home2

