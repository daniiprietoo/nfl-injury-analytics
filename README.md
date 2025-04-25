# NFL Injury Analytics

NFL Injury Analytics is a Python-based project designed to analyze NFL injury data and trends using advanced data science and machine learning techniques. The project leverages multiple datasets from the NFL Verse API, including injury reports, play-by-play data, and game schedules, to uncover insights about player injuries, their causes, and their relationship to game conditions.

## Project Overview

The primary goal of this project is to provide a comprehensive analysis of NFL player injuries, identify patterns and risk factors, and build predictive models to estimate injury likelihood based on game and player features. This analysis can help teams, analysts, and medical staff make data-driven decisions to improve player safety and performance.

## Key Features and Implementation

- **Data Acquisition & Integration:**
  - Automated download and integration of injury, play-by-play, and schedule data from the NFL Verse API using the `nfl_data_py` package.
  - Data cleaning, normalization, and merging to create a unified dataset for analysis.

- **Feature Engineering:**
  - Extraction and transformation of relevant features such as play context, player roles, weather, field surface, and injury specifics.
  - Creation of injury flags and mapping of injury body parts for consistent analysis.

- **Exploratory Data Analysis:**
  - Visualization of injury trends by year, surface type, and body part using `matplotlib` and `seaborn`.
  - Statistical summaries to highlight key findings, such as the prevalence of injuries on different surfaces and during specific play types.

- **Predictive Modeling:**
  - Encoding of categorical and numerical features for machine learning.
  - Feature selection using information gain to identify the most important predictors.
  - Model training and evaluation using scikit-learn, including linear regression and performance metrics (accuracy, precision, recall, F1 score).

- **Reproducibility:**
  - All data processing and analysis steps are documented in Jupyter Notebooks for transparency and reproducibility.
  - Processed datasets and visualizations are saved for future reference and reporting.

## Project Outcomes

- Identified key factors associated with increased injury risk, such as field surface, weather conditions, and play type.
- Developed a predictive model capable of estimating injury likelihood for individual plays.
- Produced visualizations and reports to communicate findings to stakeholders.

## How the Project Works

1. **Setup:**
   - Clone the repository and set up a Python virtual environment.
   - Install dependencies using `pip install -r requirements.txt`.

2. **Data Acquisition:**
   - Download raw data from the NFL Verse API or use pre-processed CSV files provided in the `data/raw` directory.

3. **Data Processing:**
   - Run the Jupyter Notebook in the `notebooks` directory to clean, merge, and engineer features from the datasets.

4. **Analysis & Modeling:**
   - Perform exploratory data analysis and train predictive models as documented in the notebook.
   - Generate visualizations and export processed data for further use.

5. **Results:**
   - Review the analysis outcomes, model performance, and visualizations in the notebook and the `visualizations` directory.

## Contributing

### Prerequisites

- Python 3.13 or higher
- Git

### Installation and Setup for Contributors

1. **Get Repository Access**
   - Send your GitHub username to dani to be added as a collaborator
   - Accept the collaboration invitation sent to your email

2. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/nfl-injury-analytics.git
   cd nfl-injury-analytics
   ```

3. **Set Up Development Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On macOS
    .venv\Scripts\activate  # On Windows
   
   # Install dependencies (choose one method)
   pip install -r requirements.txt  # Using requirements.txt
   ```

### Development Workflow

1. **Before Starting New Work**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create a New Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Making Changes**
   - Write your code
   - Commit frequently with clear messages:
   ```bash
   git add .
   git commit -m "Clear description of your changes"
   ```

4. **Push Your Changes**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to the repository on GitHub
   - Create a new Pull Request from your branch
   - Request review from team members
   - Merge after approval

### Branch Naming Convention

- Features: `feature/description`

## License

This project is licensed under the MIT License.




