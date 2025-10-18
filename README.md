# Systems-analysis
# COVID-19 Global Forecasting — System Design Document

This repository contains the complete system design and implementation analysis developed for the **Kaggle COVID-19 Global Forecasting Week 1** competition.  
It forms part of the **Workshop #2: Systems Engineering Analysis**, where a predictive system is modeled, implemented, and evaluated using real-world epidemiological data.

---

## Project Overview

The project focuses on designing a modular and data-driven forecasting system capable of predicting cumulative COVID-19 cases and fatalities based on historical data.  
The design applies **systems engineering principles**, emphasizing structure, data flow, architecture, and feedback control to ensure robustness, interpretability, and reproducibility.

The analysis explores:
- System elements and their interrelations.  
- Data flow from input to output.  
- Technical stack and implementation details.  
- Sensitivity and chaotic behavior in the system.  
- Boundaries, constraints, and performance requirements.  

---

## Technical Stack

The system was developed in **Python**, using the following main libraries and tools:

- **Pandas** and **NumPy** → data processing and numerical operations  
- **Scikit-learn** and **XGBoost** → model training and evaluation  
- **Matplotlib** and **Seaborn** → data visualization  
- **Jupyter Notebook** → code development and documentation  
- **Kaggle API** → dataset management and submission automation  
- **GitHub** → version control and team collaboration  

This stack ensures flexibility, scalability, and full transparency across the workflow.

---

## System Architecture

The system follows a sequential and modular pipeline:

1. **Data Loading** → read and validate `train.csv` and `test.csv`  
2. **Preprocessing** → clean, normalize, and unify datasets  
3. **Feature Engineering** → add temporal or regional variables  
4. **Model Training** → use machine learning algorithms  
5. **Prediction Generation** → export results to `submission.csv`  
6. **Feedback Loop** → submit via Kaggle API and analyze results  

Each component operates independently, ensuring easy maintenance and reproducibility. Feedback from the Kaggle leaderboard is used to refine model accuracy iteratively.

---

## System Boundaries

**Inside the system:**  
- Data preprocessing, model training, and prediction generation.  

**Outside the system:**  
- External data sources (optional), competition scoring, and Kaggle infrastructure.  

This boundary ensures that the design focuses strictly on the internal data processing and predictive logic while external validation is handled by the platform.

---

## Results and Findings

- The system successfully integrates engineering principles into predictive modeling.  
- Data variability and external interventions introduce sensitivity and chaotic patterns.  
- Validation, control, and iteration are essential for accuracy improvement.  
- The modular structure allows scalability and adaptability for future forecasting tasks.

---

## Authors

**Workshop Group - Systems Engineering Analysis**  
- Santiago Vargas Gomez - 20242020139
- David Esteban Sanchez Torres - 20221020093
- Dilan Guisseppe Triana Jimenez - 20221020100
- Daniel Alejandro Castro - 20242020271
