from pathlib import Path

# Neural network
HIDDEN_LAYERS = (128, 64)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 500

# ASCII brightness thresholds (pixel values range 0–16)
ASCII_CHARS = ["░", "▒", "▓", "█"]
ASCII_THRESHOLDS = [5, 10, 14]

# ANSI color codes
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_GREY = "\033[90m"
ANSI_YELLOW = "\033[93m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"

# Output directories (tried in order, first writable wins)
OUTPUT_DIRS = [Path("/mnt/user-data/outputs"), Path("outputs")]

# PNG filenames in save order
PLOT_FILENAMES = [
    "01_digit_examples.png",
    "02_confusion_matrix.png",
    "03_accuracy_per_digit.png",
    "04_network_diagram.png",
]
