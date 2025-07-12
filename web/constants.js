export const BUTTON_STYLES = {
    base: "width: 100%; padding: 8px 12px; font-size: 12px; color: white; border: none; border-radius: 4px; cursor: pointer; transition: all 0.2s; font-weight: 500;",
    clearMemory: "background-color: #555;",
    success: "background-color: #3ca03c;",
    error: "background-color: #c04c4c;",
    launch: "background-color: #4a7c4a;",
    stop: "background-color: #7c4a4a;",
    grey: "background-color: #555;",
};

export const STATUS_COLORS = {
    DISABLED_GRAY: "#666",
    OFFLINE_RED: "#c04c4c",
    ONLINE_GREEN: "#3ca03c",
    PROCESSING_YELLOW: "#f0ad4e"
};

export const PULSE_ANIMATION_CSS = `
    @keyframes pulse {
        0% {
            opacity: 1;
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(240, 173, 78, 0.7);
        }
        50% {
            opacity: 0.3;
            transform: scale(1.3);
            box-shadow: 0 0 0 6px rgba(240, 173, 78, 0);
        }
        100% {
            opacity: 1;
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(240, 173, 78, 0);
        }
    }
    .status-pulsing {
        animation: pulse 1.2s ease-in-out infinite;
    }
    
    /* Button hover effects */
    .distributed-button:hover:not(:disabled) {
        filter: brightness(1.2);
        transition: filter 0.2s ease;
    }
    .distributed-button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
    
    /* Settings button animation */
    .settings-btn {
        transition: transform 0.2s ease;
    }
    
    
    /* Expanded settings panel */
    .worker-settings {
        max-height: 0;
        overflow: hidden;
        opacity: 0;
        transition: max-height 0.3s ease, opacity 0.3s ease, padding 0.3s ease, margin 0.3s ease;
    }
    .worker-settings.expanded {
        max-height: 500px;
        opacity: 1;
        padding: 12px 0;
    }
`;

export const UI_STYLES = {
    statusDot: "display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px;",
    controlsDiv: "padding: 0 12px 12px 12px; display: flex; gap: 6px;",
    formGroup: "display: flex; flex-direction: column; gap: 5px;",
    formLabel: "font-size: 12px; color: #ccc; font-weight: 500;",
    formInput: "padding: 6px 10px; background: #2a2a2a; border: 1px solid #444; color: white; font-size: 12px; border-radius: 4px; transition: border-color 0.2s;",
    workerCard: "margin-bottom: 12px; border-radius: 6px; overflow: hidden; display: flex; background: #2a2a2a;",
    checkboxColumn: "flex: 0 0 44px; display: flex; align-items: center; justify-content: center; border-right: 1px solid #3a3a3a; cursor: default; background: rgba(0,0,0,0.1);",
    contentColumn: "flex: 1; display: flex; flex-direction: column; transition: background-color 0.2s ease;",
    infoRow: "display: flex; align-items: center; padding: 12px; cursor: pointer;",
    workerContent: "display: flex; align-items: center; gap: 10px; flex: 1;",
    settingsArrow: "font-size: 12px; color: #888; transition: all 0.2s ease; margin-left: auto; padding: 4px;",
    infoBox: "background-color: #333; color: #999; padding: 5px 14px; border-radius: 4px; font-size: 11px; text-align: center; flex: 1; font-weight: 500;",
    workerSettings: "margin: 0 12px; padding: 0 12px; background: #1e1e1e; border-radius: 4px; border: 1px solid #2a2a2a;"
};

export const TIMEOUTS = {
    DEFAULT_FETCH: 5000, // ms for general API calls
    STATUS_CHECK: 2000, // ms for status checks
    LAUNCH: 90000, // ms for worker launch (longer for model loading)
    RETRY_DELAY: 1000, // initial delay for exponential backoff
    MAX_RETRIES: 3 // max retry attempts
};