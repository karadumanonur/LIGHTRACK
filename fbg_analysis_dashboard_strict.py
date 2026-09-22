"""Compatibility entry point; the final UI lives in train_monitoring_ui."""
from train_monitoring_ui import (
    main, st,
    calculate_axle_heights as calculate_axle_heights_strict,
    analyse_recording as analyse_recording_strict,
)

if __name__ == "__main__":
    main()
    st.sidebar.divider()
