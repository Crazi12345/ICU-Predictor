import json
import os

# --- CELL 1: Markdown Intro ---
cell1 = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "# ICU Sepsis Prediction - Full Training Pipeline\n",
  "\n",
  "This notebook implements the complete training pipeline for the ICU Sepsis Prediction project. It includes:\n",
  "1. **Data Loading & Preprocessing**: Handling different imputation strategies (neg1, mean, median, linear).\n",
  "2. **Model Architectures**: RNN (LSTM/GRU), CNN (1D), and LGSTM (LSTM + Graph-like structure).\n",
  "3. **Hyperparameter Tuning**: Grid search for optimal parameters.\n",
  "4. **Evaluation**: Using the official PhysioNet Challenge Utility Score and AUC.\n",
  "\n",
  "The notebook is designed to be reproducible and modular."
 ]
}

# --- CELL 2: Imports ---
cell2 = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import os\n",
  "import sys\n",
  "import logging\n",
  "import pickle\n",
  "import json\n",
  "from datetime import datetime\n",
  "import numpy as np\n",
  "import pandas as pd\n",
  "import matplotlib.pyplot as plt\n",
  "import seaborn as sns\n",
  "import torch\n",
  "import torch.nn as nn\n",
  "import torch.optim as optim\n",
  "from torch.utils.data import DataLoader, TensorDataset\n",
  "from sklearn.model_selection import ParameterGrid\n",
  "from sklearn.preprocessing import StandardScaler\n",
  "from sklearn.metrics import (\n",
  "    roc_auc_score,\n",
  "    average_precision_score,\n",
  "    roc_curve,\n",
  "    precision_recall_curve,\n",
  "    confusion_matrix,\n",
  "    precision_score,\n",
  "    recall_score,\n",
  "    f1_score,\n",
  ")\n",
  "\n",
  "# Configure logging to print to cell output\n",
  "logging.basicConfig(\n",
  "    level=logging.INFO,\n",
  "    format='%(asctime)s - %(levelname)s - %(message)s',\n",
  "    handlers=[logging.StreamHandler(sys.stdout)]\n",
  "    )\n",
  "\n",
  "# Plot settings\n",
  "%matplotlib inline\n",
  "plt.rcParams['figure.figsize'] = (10, 6)"
 ]
}

# --- CELL 3: Config ---
cell3 = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# --- Configuration ---",