#  Customer Churn Prediction System
A Machine Learning project that predicts customer churn using Probabilistic Graphical Models (PGMs) — specifically, a Bayesian Network and a Markov Model. The system aims to identify customers likely to churn and help businesses take proactive measures to retain them.

# Objectives
- To develop a probabilistic graphical model-based system to predict customer churn.

- To compare the effectiveness of Bayesian Network vs Markov Model.

- To provide interpretable insights for business decision-making.

# Models Used
### Bayesian Network
- Built using pgmpy.

- Trained on preprocessed customer data with features such as usage behavior, demographics, and account activity.

- Capable of capturing conditional dependencies between features.

- Higher accuracy and better generalization in predictions.

### Markov Model
- Modeled customer state transitions (Active → At-Risk → Churned).

- Used for time-sequenced behavioral modeling.

- Less accurate due to lack of feature richness and conditional dependency capture.

# Tech Stack
- Languages: Python

- Libraries: pgmpy, scikit-learn, pandas, numpy, matplotlib

- Visualization: networkx, matplotlib

- Framework: Flask (for serving the model via web interface)

# Results
| Model            | Accuracy | Observations                                           |
| ---------------- | -------- | ------------------------------------------------------ |
| Bayesian Network | ✅ Higher | Better feature interaction modeling & interpretability |
| Markov Model     | ❌ Lower  | Lacks complex feature relationships                    |

#  How to Run
- 1.Clone the repo:
   ```bash
  git clone https://github.com/yourusername/customer-churn-pgm.git

  cd customer-churn-pgm
  
- 2.Install dependencies:
  ```bash
  pip install -r requirements.txt
  
- 3.Run the Flask app:
   ```bash
  python app/app.py
  
- 4.Open in browser:
  
  Navigate to http://localhost:5000 to use the system.

# Conclusion
- The Bayesian Network model is superior for churn prediction tasks, providing both higher accuracy and clearer insights into variable dependencies.

- The Markov model, while useful for state transition analysis, underperformed in comparison due to its simplicity.

# License
This project is licensed under the MIT License - see the LICENSE file for details.


