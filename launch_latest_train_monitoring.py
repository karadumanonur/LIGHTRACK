"""Compatibility entry point for existing desktop shortcuts."""
from launch_train_monitoring import main

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error))
        raise SystemExit(1)
