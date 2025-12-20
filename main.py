from src.data_loader import load_data
from src.config import FIGURES_DIR, RESULTS_DIR

def main():
    print("Hello from ing-task!")
    train_df, test_df = load_data()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    


if __name__ == "__main__":
    main()
