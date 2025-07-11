import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

class DistributedExtension {
    // Button styling constants
    static BUTTON_STYLES = {
        base: "width: 100%; padding: 8px; font-size: 12px; color: white; border: none; border-radius: 4px; cursor: pointer; transition: background-color 0.2s;",
        clearMemory: "background-color: #555;",
        success: "background-color: #3ca03c;",
        error: "background-color: #c04c4c;",
    };

    // CSS for pulsing animation and button hover effects
    static PULSE_ANIMATION_CSS = `
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
            max-height: 300px;
            opacity: 1;
        }
    `;

    constructor() {
        this.config = null;
        this.originalQueuePrompt = api.queuePrompt.bind(api);
        this.statusCheckInterval = null;
        this.logAutoRefreshInterval = null;
        
        // Initialize centralized state
        this.state = this._createStateManager();
        
        // Initialize UI component factory
        this.ui = this._createUIFactory();
        
        // Initialize API client
        this.api = this._createApiClient();

        // Inject CSS for pulsing animation
        this.injectStyles();

        this.loadConfig().then(async () => {
            this.registerSidebarTab();
            this.setupInterceptor();
            this.startStatusChecking();
            this.loadManagedWorkers();
            // Detect master IP after everything is set up
            this.detectMasterIP();
        });
    }
    
    // Debug logging helpers
    debugLog(message) {
        if (this.config?.settings?.debug) {
            console.log(`[Distributed] ${message}`);
        }
    }
    
    log(message) {
        console.log(`[Distributed] ${message}`);
    }
    
    // --- UI Component Factory ---
    _createUIFactory() {
        const styles = {
            statusDot: "display: inline-block; width: 8px; height: 8px; border-radius: 50%;",
            controlsDiv: "padding: 0 8px 6px 8px; display: flex; gap: 4px;",
            checkboxColumn: "flex: 0 0 40px; display: flex; align-items: center; justify-content: center; border-right: 1px solid #444;",
            workerCard: "margin-bottom: 12px; background: #2a2a2a; border-radius: 4px; overflow: hidden; display: flex;",
            contentColumn: "flex: 1; display: flex; flex-direction: column; transition: background-color 0.2s ease;",
            infoRow: "display: flex; align-items: center; padding: 8px; cursor: pointer; min-height: 60px;",
            workerContent: "display: flex; align-items: center; gap: 8px; flex: 1;",
            settingsArrow: "font-size: 12px; color: #888; transition: all 0.2s ease; margin-left: auto;",
            formGroup: "display: flex; align-items: center; gap: 8px; margin-bottom: 8px;",
            formLabel: "flex: 0 0 100px; font-size: 12px; color: #ccc;",
            formInput: "flex: 1; padding: 4px 8px; background: #1a1a1a; border: 1px solid #444; border-radius: 4px; color: #fff; font-size: 12px;",
            infoBox: "background-color: #333; color: #888; padding: 4px 12px; border-radius: 4px; font-size: 11px; text-align: center; flex: 1;"
        };
        
        return {
            createStatusDot(id, color = "#666", title = "Status") {
                const dot = document.createElement("span");
                if (id) dot.id = id;
                dot.style.cssText = styles.statusDot + ` background-color: ${color};`;
                dot.title = title;
                return dot;
            },
            
            createButton(text, onClick, customStyle = "") {
                const button = document.createElement("button");
                button.textContent = text;
                button.className = "distributed-button";
                button.style.cssText = DistributedExtension.BUTTON_STYLES.base + customStyle;
                if (onClick) button.onclick = onClick;
                return button;
            },
            
            createButtonGroup(buttons, style = "") {
                const group = document.createElement("div");
                group.style.cssText = "display: flex; gap: 4px; margin-top: 10px;" + style;
                buttons.forEach(button => group.appendChild(button));
                return group;
            },
            
            createWorkerControls(workerId, handlers = {}) {
                const controlsDiv = document.createElement("div");
                controlsDiv.id = `controls-${workerId}`;
                controlsDiv.style.cssText = styles.controlsDiv;
                
                const buttons = [];
                
                if (handlers.launch) {
                    const launchBtn = this.createButton('Launch', handlers.launch);
                    launchBtn.id = `launch-${workerId}`;
                    launchBtn.title = "Launch this worker instance";
                    buttons.push(launchBtn);
                }
                
                if (handlers.stop) {
                    const stopBtn = this.createButton('Stop', handlers.stop);
                    stopBtn.id = `stop-${workerId}`;
                    stopBtn.title = "Stop this worker instance";
                    buttons.push(stopBtn);
                }
                
                if (handlers.viewLog) {
                    const logBtn = this.createButton('View Log', handlers.viewLog);
                    logBtn.id = `log-${workerId}`;
                    logBtn.title = "View worker log file";
                    buttons.push(logBtn);
                }
                
                buttons.forEach(btn => controlsDiv.appendChild(btn));
                return controlsDiv;
            },
            
            createFormGroup(label, value, id, type = "text", placeholder = "") {
                const group = document.createElement("div");
                group.style.cssText = styles.formGroup;
                
                const labelEl = document.createElement("label");
                labelEl.textContent = label;
                labelEl.htmlFor = id;
                labelEl.style.cssText = styles.formLabel;
                
                const input = document.createElement("input");
                input.type = type;
                input.id = id;
                input.value = value;
                input.placeholder = placeholder;
                input.style.cssText = styles.formInput;
                
                group.appendChild(labelEl);
                group.appendChild(input);
                return { group, input };
            },
            
            createWorkerCard() {
                const card = document.createElement("div");
                card.style.cssText = styles.workerCard;
                
                const checkboxColumn = document.createElement("div");
                checkboxColumn.style.cssText = styles.checkboxColumn;
                
                const contentColumn = document.createElement("div");
                contentColumn.style.cssText = styles.contentColumn;
                
                const infoRow = document.createElement("div");
                infoRow.style.cssText = styles.infoRow;
                
                const workerContent = document.createElement("div");
                workerContent.style.cssText = styles.workerContent;
                
                const settingsArrow = document.createElement("span");
                settingsArrow.className = "settings-arrow";
                settingsArrow.innerHTML = "▶";
                settingsArrow.style.cssText = styles.settingsArrow;
                
                const controlsDiv = document.createElement("div");
                controlsDiv.style.cssText = styles.controlsDiv;
                
                return {
                    card,
                    checkboxColumn,
                    contentColumn,
                    infoRow,
                    workerContent,
                    settingsArrow,
                    controlsDiv
                };
            },
            
            createInfoBox(text) {
                const box = document.createElement("div");
                box.style.cssText = styles.infoBox;
                box.textContent = text;
                return box;
            },
            
            addHoverEffect(element, onHover, onLeave) {
                element.onmouseover = onHover;
                element.onmouseout = onLeave;
            },
            
            createCard(type = 'worker', options = {}) {
                const card = document.createElement("div");
                const baseStyle = "margin-bottom: 12px; border-radius: 4px; overflow: hidden; display: flex;";
                
                switch(type) {
                    case 'master':
                    case 'worker':
                        card.style.cssText = baseStyle + "background: #2a2a2a;";
                        break;
                    case 'blueprint':
                        card.style.cssText = baseStyle + "border: 2px solid #666; cursor: pointer; transition: all 0.2s ease; background: rgba(255, 255, 255, 0.02);";
                        if (options.onClick) card.onclick = options.onClick;
                        if (options.title) card.title = options.title;
                        break;
                    case 'add':
                        card.style.cssText = "margin-top: 10px; " + baseStyle + "border: 1px solid #444; cursor: pointer; transition: all 0.2s ease;";
                        if (options.onClick) card.onclick = options.onClick;
                        if (options.title) card.title = options.title;
                        break;
                }
                
                if (options.onMouseEnter) {
                    card.addEventListener('mouseenter', options.onMouseEnter);
                }
                if (options.onMouseLeave) {
                    card.addEventListener('mouseleave', options.onMouseLeave);
                }
                
                return card;
            },
            
            createCardColumn(type = 'checkbox', options = {}) {
                const column = document.createElement("div");
                const baseStyle = "display: flex; align-items: center; justify-content: center;";
                
                switch(type) {
                    case 'checkbox':
                        column.style.cssText = baseStyle + "flex: 0 0 40px; border-right: 1px solid #444; cursor: default;";
                        if (options.title) column.title = options.title;
                        break;
                    case 'icon':
                        column.style.cssText = baseStyle + "width: 50px; flex-shrink: 0; font-size: 20px; color: #666;";
                        break;
                    case 'content':
                        column.style.cssText = "flex: 1; display: flex; flex-direction: column; transition: background-color 0.2s ease;";
                        break;
                }
                
                return column;
            },
            
            createInfoRow(options = {}) {
                const row = document.createElement("div");
                row.style.cssText = "display: flex; align-items: center; padding: 8px; cursor: pointer; min-height: 60px;";
                if (options.onClick) row.onclick = options.onClick;
                return row;
            },
            
            createWorkerContent() {
                const content = document.createElement("div");
                content.style.cssText = "display: flex; align-items: center; gap: 8px; flex: 1;";
                return content;
            },
            
            createSettingsForm(fields = [], options = {}) {
                const form = document.createElement("div");
                form.style.cssText = "display: flex; flex-direction: column; gap: 10px;";
                
                fields.forEach(field => {
                    if (field.type === 'checkbox') {
                        const group = document.createElement("div");
                        group.style.cssText = "display: flex; align-items: center; gap: 8px; margin: 5px 0;";
                        
                        const checkbox = document.createElement("input");
                        checkbox.type = "checkbox";
                        checkbox.id = field.id;
                        checkbox.checked = field.checked || false;
                        if (field.onChange) checkbox.onchange = field.onChange;
                        
                        const label = document.createElement("label");
                        label.htmlFor = field.id;
                        label.textContent = field.label;
                        label.style.cssText = "font-size: 12px; color: #ccc; cursor: pointer;";
                        
                        group.appendChild(checkbox);
                        group.appendChild(label);
                        form.appendChild(group);
                    } else {
                        const result = this.createFormGroup(field.label, field.value, field.id, field.type, field.placeholder);
                        if (field.groupId) result.group.id = field.groupId;
                        if (field.display) result.group.style.display = field.display;
                        form.appendChild(result.group);
                    }
                });
                
                if (options.buttons) {
                    const buttonGroup = this.createButtonGroup(options.buttons, options.buttonStyle || " margin-top: 8px;");
                    form.appendChild(buttonGroup);
                }
                
                return form;
            },
            
            createCompleteCard(type, options = {}) {
                // Create the main card
                const card = this.createCard(type, options);
                
                // Create left column
                let leftColumn;
                if (type === 'master' || type === 'worker') {
                    leftColumn = this.createCardColumn('checkbox', options.checkboxOptions);
                    
                    // Add checkbox
                    const checkbox = document.createElement("input");
                    checkbox.type = "checkbox";
                    if (options.checkboxId) checkbox.id = options.checkboxId;
                    checkbox.checked = options.checked || false;
                    checkbox.disabled = options.disabled || false;
                    checkbox.style.cssText = `cursor: ${options.disabled ? 'default' : 'pointer'}; width: 16px; height: 16px;${options.checkboxStyle || ''}`;
                    if (options.checkboxTitle) checkbox.title = options.checkboxTitle;
                    
                    leftColumn.appendChild(checkbox);
                    
                    if (options.onCheckboxClick && !options.disabled) {
                        checkbox.style.pointerEvents = "none";
                        leftColumn.style.cursor = "pointer";
                        leftColumn.onclick = options.onCheckboxClick;
                    }
                } else {
                    // For blueprint and add cards, create icon column
                    leftColumn = document.createElement("div");
                    const borderColor = type === 'blueprint' ? '#666' : '#444';
                    leftColumn.style.cssText = `flex: 0 0 40px; display: flex; align-items: center; justify-content: center; border-right: ${type === 'blueprint' ? '2px' : '1px'} solid ${borderColor}; color: ${type === 'blueprint' ? '#888' : '#555'}; font-size: ${type === 'blueprint' ? '24px' : '18px'}; font-weight: ${type === 'blueprint' ? 'bold' : 'normal'};`;
                    leftColumn.innerHTML = "+";
                    
                    if (options.onHover) {
                        card.addEventListener('mouseenter', () => {
                            leftColumn.style.color = type === 'blueprint' ? '#aaa' : '#888';
                            leftColumn.style.borderColor = type === 'blueprint' ? '#888' : '#666';
                        });
                        card.addEventListener('mouseleave', () => {
                            leftColumn.style.color = type === 'blueprint' ? '#888' : '#555';
                            leftColumn.style.borderColor = borderColor;
                        });
                    }
                }
                
                // Create right column
                const rightColumn = this.createCardColumn('content');
                
                // Add hover effect for master/worker cards
                if ((type === 'master' || type === 'worker') && options.onHover !== false) {
                    rightColumn.onmouseover = () => rightColumn.style.backgroundColor = "#333";
                    rightColumn.onmouseout = () => rightColumn.style.backgroundColor = "transparent";
                }
                
                // Create info row
                const infoRow = this.createInfoRow(options.infoRowOptions);
                
                // Create worker content
                const workerContent = this.createWorkerContent();
                
                // Build the card
                infoRow.appendChild(workerContent);
                rightColumn.appendChild(infoRow);
                
                // Add controls div if specified
                if (options.includeControls) {
                    const controlsDiv = document.createElement("div");
                    controlsDiv.id = options.controlsId;
                    controlsDiv.style.cssText = styles.controlsDiv;
                    rightColumn.appendChild(controlsDiv);
                }
                
                // Add settings div if specified
                if (options.includeSettings) {
                    const settingsDiv = document.createElement("div");
                    settingsDiv.id = options.settingsId;
                    settingsDiv.className = "worker-settings";
                    settingsDiv.style.cssText = styles.workerSettings;
                    rightColumn.appendChild(settingsDiv);
                }
                
                card.appendChild(leftColumn);
                card.appendChild(rightColumn);
                
                return {
                    card,
                    leftColumn,
                    rightColumn,
                    infoRow,
                    workerContent,
                    checkbox: leftColumn.querySelector('input[type="checkbox"]')
                };
            },
            
            styles // Expose styles for custom usage
        };
    }
    
    // --- State Management ---
    _createStateManager() {
        const state = {
            workers: new Map(), // Unified worker state: { status, managed, launching, expanded, ... }
            masterStatus: 'online',
            domCache: new Map() // Cache DOM elements
        };
        
        return {
            // Worker state management
            getWorker(workerId) {
                return state.workers.get(String(workerId)) || {};
            },
            
            updateWorker(workerId, updates) {
                const id = String(workerId);
                const current = state.workers.get(id) || {};
                state.workers.set(id, { ...current, ...updates });
                return state.workers.get(id);
            },
            
            setWorkerStatus(workerId, status) {
                return this.updateWorker(workerId, { status });
            },
            
            setWorkerManaged(workerId, info) {
                return this.updateWorker(workerId, { managed: info });
            },
            
            setWorkerLaunching(workerId, launching) {
                return this.updateWorker(workerId, { launching });
            },
            
            setWorkerExpanded(workerId, expanded) {
                return this.updateWorker(workerId, { expanded });
            },
            
            isWorkerLaunching(workerId) {
                return this.getWorker(workerId).launching || false;
            },
            
            isWorkerExpanded(workerId) {
                return this.getWorker(workerId).expanded || false;
            },
            
            isWorkerManaged(workerId) {
                return !!this.getWorker(workerId).managed;
            },
            
            getWorkerStatus(workerId) {
                return this.getWorker(workerId).status || {};
            },
            
            // Master state
            setMasterStatus(status) {
                state.masterStatus = status;
            },
            
            getMasterStatus() {
                return state.masterStatus;
            },
            
            // DOM cache
            cacheElement(key, element) {
                state.domCache.set(key, element);
            },
            
            getCachedElement(key) {
                return state.domCache.get(key);
            },
            
            clearCache() {
                state.domCache.clear();
            }
        };
    }
    
    // --- API Client ---
    _createApiClient() {
        const baseUrl = window.location.origin;
        
        const request = async (endpoint, options = {}) => {
            try {
                const response = await fetch(`${baseUrl}${endpoint}`, {
                    headers: { 'Content-Type': 'application/json' },
                    ...options
                });
                
                if (!response.ok) {
                    const error = await response.json().catch(() => ({ message: 'Request failed' }));
                    throw new Error(error.message || `HTTP ${response.status}`);
                }
                
                return await response.json();
            } catch (error) {
                this.log(`API Error: ${endpoint} - ${error.message}`);
                throw error;
            }
        };
        
        return {
            // Config endpoints
            async getConfig() {
                return request('/distributed/config');
            },
            
            async updateWorker(workerId, data) {
                return request('/distributed/config/update_worker', {
                    method: 'POST',
                    body: JSON.stringify({ worker_id: workerId, ...data })
                });
            },
            
            async deleteWorker(workerId) {
                return request('/distributed/config/delete_worker', {
                    method: 'POST',
                    body: JSON.stringify({ worker_id: workerId })
                });
            },
            
            async updateSetting(key, value) {
                return request('/distributed/config/update_setting', {
                    method: 'POST',
                    body: JSON.stringify({ key, value })
                });
            },
            
            async updateMaster(data) {
                return request('/distributed/config/update_master', {
                    method: 'POST',
                    body: JSON.stringify(data)
                });
            },
            
            // Worker management endpoints
            async launchWorker(workerId) {
                return request('/distributed/launch_worker', {
                    method: 'POST',
                    body: JSON.stringify({ worker_id: workerId })
                });
            },
            
            async stopWorker(workerId) {
                return request('/distributed/stop_worker', {
                    method: 'POST',
                    body: JSON.stringify({ worker_id: workerId })
                });
            },
            
            async getManagedWorkers() {
                return request('/distributed/managed_workers');
            },
            
            async getWorkerLog(workerId, lines = 1000) {
                return request(`/distributed/worker_log/${workerId}?lines=${lines}`);
            },
            
            async clearLaunchingFlag(workerId) {
                return request('/distributed/worker/clear_launching', {
                    method: 'POST',
                    body: JSON.stringify({ worker_id: workerId })
                });
            },
            
            // Job preparation
            async prepareJob(multiJobId) {
                return request('/distributed/prepare_job', {
                    method: 'POST',
                    body: JSON.stringify({ multi_job_id: multiJobId })
                });
            },
            
            // Image loading
            async loadImage(imagePath) {
                return request('/distributed/load_image', {
                    method: 'POST',
                    body: JSON.stringify({ image_path: imagePath })
                });
            },
            
            // Network info
            async getNetworkInfo() {
                return request('/distributed/network_info');
            },
            
            // Status checking (with timeout)
            async checkStatus(url, timeout = 900) {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), timeout);
                
                try {
                    const response = await fetch(url, {
                        method: 'GET',
                        mode: 'cors',
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);
                    
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return await response.json();
                } catch (error) {
                    clearTimeout(timeoutId);
                    throw error;
                }
            },
            
            // Batch status checking
            async checkMultipleStatuses(urls) {
                return Promise.allSettled(
                    urls.map(url => this.checkStatus(url))
                );
            }
        };
    }

    injectStyles() {
        const styleId = 'distributed-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = DistributedExtension.PULSE_ANIMATION_CSS;
            document.head.appendChild(style);
        }
    }

    // --- State & Config Management (Single Source of Truth) ---

    get enabledWorkers() {
        return this.config?.workers?.filter(w => w.enabled) || [];
    }

    get isEnabled() {
        return this.enabledWorkers.length > 0;
    }

    async loadConfig() {
        try {
            this.config = await this.api.getConfig();
            this.log("Loaded config: " + JSON.stringify(this.config));
        } catch (error) {
            this.log("Failed to load config: " + error.message);
            this.config = { workers: [], settings: {} };
        }
    }

    async updateWorkerEnabled(workerId, enabled) {
        const worker = this.config.workers.find(w => w.id === workerId);
        if (worker) {
            worker.enabled = enabled;
        }
        
        try {
            await this.api.updateWorker(workerId, { enabled });
        } catch (error) {
            this.log("Error updating worker: " + error.message);
        }
    }
    
    async _updateSetting(key, value) {
        // Update local config
        if (!this.config.settings) {
            this.config.settings = {};
        }
        this.config.settings[key] = value;
        
        try {
            await this.api.updateSetting(key, value);
            
            app.extensionManager.toast.add({
                severity: "success",
                summary: "Setting Updated",
                detail: `${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} ${value ? 'enabled' : 'disabled'}`,
                life: 2000
            });
        } catch (error) {
            this.log(`Error updating setting '${key}': ${error.message}`);
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Setting Update Failed",
                detail: error.message,
                life: 3000
            });
        }
    }

    // --- UI Rendering ---

    registerSidebarTab() {
        app.extensionManager.registerSidebarTab({
            id: "distributed",
            icon: "pi pi-server",
            title: "Distributed",
            tooltip: "Distributed Control Panel",
            type: "custom",
            render: (el) => this.renderSidebarContent(el)
        });
    }

    _createButton(text, onClick, style) {
        return this.ui.createButton(text, onClick, style);
    }
    
    // Generic handler for parallel worker operations
    async _handleWorkerOperation(button, operation, successText, errorText) {
        const originalText = button.textContent;
        const originalStyle = button.style.cssText;
        
        button.textContent = operation.loadingText;
        button.disabled = true;
        
        try {
            const urlsToProcess = this.enabledWorkers.map(w => ({ 
                name: w.name, 
                url: this.getWorkerUrl(w)
            }));
            
            if (urlsToProcess.length === 0) {
                button.textContent = "No Workers";
                button.style.backgroundColor = "#c04c4c";
                setTimeout(() => {
                    button.textContent = originalText;
                    button.style.cssText = originalStyle;
                    button.disabled = false;
                }, 3000);
                return;
            }
            
            const promises = urlsToProcess.map(target =>
                fetch(`${target.url}${operation.endpoint}`, { 
                    method: 'POST', 
                    mode: 'cors'
                })
                    .then(response => ({ ok: response.ok, name: target.name }))
                    .catch(() => ({ ok: false, name: target.name }))
            );
            
            const results = await Promise.all(promises);
            const failures = results.filter(r => !r.ok);
            
            if (failures.length === 0) {
                button.textContent = successText;
                button.style.backgroundColor = DistributedExtension.BUTTON_STYLES.success.split(':')[1].trim().replace(';', '');
                if (operation.onSuccess) operation.onSuccess();
            } else {
                button.textContent = errorText;
                button.style.backgroundColor = DistributedExtension.BUTTON_STYLES.error.split(':')[1].trim().replace(';', '');
                this.log(`${operation.name} failed on: ${failures.map(f => f.name).join(", ")}`);
            }
            
            setTimeout(() => {
                button.textContent = originalText;
                button.style.cssText = originalStyle;
            }, 3000);
        } finally {
            button.disabled = false;
        }
    }

    async _handleInterruptWorkers(button) {
        return this._handleWorkerOperation(button, {
            name: "Interrupt",
            endpoint: "/interrupt",
            loadingText: "Interrupting...",
            onSuccess: () => setTimeout(() => this.checkAllWorkerStatuses(), 500)
        }, "Interrupted!", "Error! See Console");
    }

    async _handleClearMemory(button) {
        return this._handleWorkerOperation(button, {
            name: "Clear memory",
            endpoint: "/distributed/clear_memory",
            loadingText: "Clearing..."
        }, "Success!", "Error! See Console");
    }


    async renderSidebarContent(el) {
        // Panel is being opened/rendered
        this.debugLog("Panel opened");
        
        if (!el) {
            this.debugLog("No element provided to renderSidebarContent");
            return;
        }
        
        // Prevent infinite recursion
        if (this._isRendering) {
            this.debugLog("Already rendering, skipping");
            return;
        }
        this._isRendering = true;
        
        try {
            // Store reference to the panel element
            this.panelElement = el;
            
            // Reload config when panel opens to pick up any external changes
            await this.loadConfig();
        
        // Reload managed workers when panel opens
        this.loadManagedWorkers().then(() => {
            // Update all worker controls after loading
            if (this.config?.workers) {
                this.config.workers.forEach(w => this.updateWorkerControls(w.id));
            }
        });
        
        // Start checking worker statuses immediately in parallel
        // Results will update the UI as they come in
        setTimeout(() => this.checkAllWorkerStatuses(), 0);
        
        el.innerHTML = '';
        
        // Create toolbar header to match ComfyUI style
        const toolbar = document.createElement("div");
        toolbar.className = "p-toolbar p-component border-x-0 border-t-0 rounded-none px-2 py-1 min-h-8";
        toolbar.style.cssText = "border-bottom: 1px solid #444; background: transparent;";
        
        const toolbarStart = document.createElement("div");
        toolbarStart.className = "p-toolbar-start";
        
        const titleSpan = document.createElement("span");
        titleSpan.className = "text-xs 2xl:text-sm truncate";
        titleSpan.textContent = "COMFYUI DISTRIBUTED";
        titleSpan.title = "ComfyUI Distributed";
        
        toolbarStart.appendChild(titleSpan);
        toolbar.appendChild(toolbarStart);
        
        const toolbarCenter = document.createElement("div");
        toolbarCenter.className = "p-toolbar-center";
        toolbar.appendChild(toolbarCenter);
        
        const toolbarEnd = document.createElement("div");
        toolbarEnd.className = "p-toolbar-end";
        toolbar.appendChild(toolbarEnd);
        
        el.appendChild(toolbar);
        
        // Main container with adjusted padding
        const container = document.createElement("div");
        container.style.cssText = "padding: 15px; display: flex; flex-direction: column; height: calc(100% - 32px);";
        
        // Detect master info on panel open (in case CUDA info wasn't available at startup)
        this.log(`Panel opened. CUDA device count: ${this.cudaDeviceCount}, Workers: ${this.config?.workers?.length || 0}`);
        if (!this.cudaDeviceCount) {
            await this.detectMasterIP();
        }
        
        // Check if we should auto-populate workers after detection
        if (this.cudaDeviceCount > 0 && (!this.config.workers || this.config.workers.length === 0)) {
            this.log(`Auto-populating workers based on ${this.cudaDeviceCount} CUDA devices (excluding master on CUDA ${this.masterCudaDevice})`);
            
            const newWorkers = [];
            let workerNum = 1;
            let portOffset = 0;
            
            for (let i = 0; i < this.cudaDeviceCount; i++) {
                // Skip the CUDA device used by master
                if (i === this.masterCudaDevice) {
                    this.log(`Skipping CUDA ${i} (used by master)`);
                    continue;
                }
                
                const worker = {
                    id: Date.now() + i,
                    name: `Worker ${workerNum}`,
                    host: "localhost",
                    port: 8189 + portOffset,
                    cuda_device: i,
                    enabled: true
                };
                newWorkers.push(worker);
                workerNum++;
                portOffset++;
            }
            
            // Only proceed if we have workers to add
            if (newWorkers.length > 0) {
                this.log(`Auto-populating ${newWorkers.length} workers`);
                
                // Add workers to config
                this.config.workers = newWorkers;
                
                // Save each worker using the update endpoint
                for (const worker of newWorkers) {
                    try {
                        await this.api.updateWorker(worker.id, worker);
                    } catch (error) {
                        this.log(`Error saving worker ${worker.name}: ${error.message}`);
                    }
                }
                
                this.log(`Auto-populated ${newWorkers.length} workers and saved config`);
                
                // Reload the config to include the new workers
                await this.loadConfig();
                
                // Continue rendering with the updated config
            } else {
                this.log("No additional CUDA devices available for workers (all used by master)");
            }
        }
        
        // Master Node Section - styled exactly like a worker card
        const masterDiv = this.ui.createCard('master');
        
        // Left column - checkbox area (always checked and disabled)
        const checkboxColumn = this.ui.createCardColumn('checkbox', { title: "Master node is always enabled" });
        
        // Greyed out checkbox that's always checked
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.disabled = true;
        checkbox.style.cssText = "cursor: default; width: 16px; height: 16px; opacity: 0.6;";
        
        checkboxColumn.appendChild(checkbox);
        
        // Right column - content area
        const contentColumn = this.ui.createCardColumn('content');
        
        // First row: master info
        const infoRow = this.ui.createInfoRow({ onClick: null });
        infoRow.title = "Click to expand settings";
        
        // Worker content
        const workerContent = this.ui.createWorkerContent();
        
        const statusDot = this.ui.createStatusDot("master-status", "#4CAF50", "Online");
        
        const gpuInfo = document.createElement("span");
        const cudaInfo = this.masterCudaDevice !== undefined ? `CUDA ${this.masterCudaDevice} • ` : '';
        const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
        gpuInfo.innerHTML = `<strong id="master-name-display">${this.config?.master?.name || "Master"}</strong><br><small style="color: #888;"><span id="master-cuda-info">${cudaInfo}Port ${port}</span></small>`;
        
        workerContent.appendChild(statusDot);
        workerContent.appendChild(gpuInfo);
        
        // Settings arrow
        const settingsArrow = document.createElement("span");
        settingsArrow.className = "settings-arrow";
        settingsArrow.innerHTML = "▶";
        settingsArrow.style.cssText = "font-size: 12px; color: #888; transition: all 0.2s ease; margin-left: auto;";
        
        infoRow.appendChild(workerContent);
        infoRow.appendChild(settingsArrow);
        
        // Click handler for info row
        let masterSettingsExpanded = false;
        infoRow.onclick = () => {
            masterSettingsExpanded = !masterSettingsExpanded;
            if (masterSettingsExpanded) {
                masterSettingsDiv.classList.add("expanded");
                masterSettingsDiv.style.padding = "12px";
                masterSettingsDiv.style.marginTop = "8px";
                masterSettingsDiv.style.marginBottom = "8px";
                settingsArrow.style.transform = "rotate(90deg)";
            } else {
                masterSettingsDiv.classList.remove("expanded");
                masterSettingsDiv.style.padding = "0 12px";
                masterSettingsDiv.style.marginTop = "0";
                masterSettingsDiv.style.marginBottom = "0";
                settingsArrow.style.transform = "rotate(0deg)";
            }
        };
        
        // Hover effect for entire content column
        contentColumn.onmouseover = () => {
            contentColumn.style.backgroundColor = "#333";
            settingsArrow.style.color = "#fff";
        };
        contentColumn.onmouseout = () => {
            contentColumn.style.backgroundColor = "transparent";
            settingsArrow.style.color = "#888";
        };
        
        contentColumn.appendChild(infoRow);
        
        // Second row: master controls (styled like remote worker)
        const controlsDiv = document.createElement("div");
        controlsDiv.style.cssText = "padding: 0 8px 6px 8px; display: flex; gap: 4px;";
        
        // Master info box styled like remote worker
        const masterInfo = document.createElement("div");
        masterInfo.style.cssText = "background-color: #333; color: #888; padding: 4px 12px; border-radius: 4px; font-size: 11px; text-align: center; flex: 1;";
        masterInfo.textContent = "Master";
        controlsDiv.appendChild(masterInfo);
        
        // Third row: expandable settings panel
        const masterSettingsDiv = document.createElement("div");
        masterSettingsDiv.id = "master-settings";
        masterSettingsDiv.className = "worker-settings";
        masterSettingsDiv.style.cssText = "margin: 0 8px; padding: 0 12px; background: #1e1e1e; border-radius: 4px;";
        
        // Create settings form using reusable components
        const settingsForm = document.createElement("div");
        settingsForm.style.cssText = "display: flex; flex-direction: column; gap: 8px;";
        
        // Name field
        const nameResult = this.ui.createFormGroup("Name:", this.config?.master?.name || "Master", "master-name");
        settingsForm.appendChild(nameResult.group);
        
        // Host field
        const hostResult = this.ui.createFormGroup("Host:", this.config?.master?.host || "", "master-host", "text", "Auto-detect if empty");
        settingsForm.appendChild(hostResult.group);
        
        // Buttons row - using same style as workers
        const saveBtn = this._createButton("Save", 
            async () => {
            const nameInput = document.getElementById('master-name');
            const hostInput = document.getElementById('master-host');
            
            // Update config
            if (!this.config.master) this.config.master = {};
            this.config.master.name = nameInput.value.trim() || "Master";
            
            // Save both host and name
            await this.saveMasterConfig({
                host: hostInput.value.trim(),
                name: this.config.master.name
            });
            
            // Update display
            document.getElementById('master-name-display').textContent = this.config.master.name;
            this.updateMasterDisplay();
            
            saveBtn.textContent = "Saved!";
            setTimeout(() => { saveBtn.textContent = "Save"; }, 2000);
        },
        "background-color: #4a7c4a;");
        saveBtn.style.cssText += " padding: 6px 12px; font-size: 12px;";
        
        const cancelBtn = this._createButton("Cancel", 
            () => {
                // Reset inputs to original values
                document.getElementById('master-name').value = this.config?.master?.name || "Master";
                document.getElementById('master-host').value = this.config?.master?.host || "";
            },
            "background-color: #555;");
        cancelBtn.style.cssText += " padding: 6px 12px; font-size: 12px;";
        
        const buttonGroup = this.ui.createButtonGroup([saveBtn, cancelBtn], " margin-top: 8px;");
        settingsForm.appendChild(buttonGroup);
        masterSettingsDiv.appendChild(settingsForm);
        
        contentColumn.appendChild(controlsDiv);
        contentColumn.appendChild(masterSettingsDiv);
        
        masterDiv.appendChild(checkboxColumn);
        masterDiv.appendChild(contentColumn);
        container.appendChild(masterDiv);
        
        // Workers Section (no heading)
        const gpuSection = document.createElement("div");
        gpuSection.style.cssText = "flex: 1; overflow-y: auto; margin-bottom: 15px;";
        
        const gpuList = document.createElement("div");
        const workers = this.config?.workers || [];
        
        // If no workers exist, show a full blueprint placeholder first
        if (workers.length === 0) {
            const blueprintDiv = this.ui.createCard('blueprint', {
                title: "Click to add your first worker",
                onClick: () => this.addNewWorker()
            });
            
            // Left column - plus icon
            const blueprintCheckbox = document.createElement("div");
            blueprintCheckbox.style.cssText = "flex: 0 0 40px; display: flex; align-items: center; justify-content: center; border-right: 2px solid #666; color: #888; font-size: 24px; font-weight: bold;";
            blueprintCheckbox.innerHTML = "+";
            
            // Right column
            const blueprintContent = this.ui.createCardColumn('content');
            
            // Info row with placeholder content
            const blueprintInfo = this.ui.createInfoRow();
            
            const blueprintWorkerContent = this.ui.createWorkerContent();
            
            const blueprintDot = this.ui.createStatusDot(null, "transparent", "");
            blueprintDot.style.border = "1px solid #555";
            
            const blueprintText = document.createElement("span");
            blueprintText.innerHTML = `<strong style="color: #aaa; font-size: 16px;">Add New Worker</strong><br><small style="color: #555;">[CUDA] • [Port]</small>`;
            
            blueprintWorkerContent.appendChild(blueprintDot);
            blueprintWorkerContent.appendChild(blueprintText);
            blueprintInfo.appendChild(blueprintWorkerContent);
            
            // Placeholder button row - single Launch button spanning width
            const blueprintButtons = document.createElement("div");
            blueprintButtons.style.cssText = "padding: 0 8px 6px 8px;";
            
            const ghostLaunch = document.createElement("div");
            ghostLaunch.style.cssText = "width: 100%; padding: 3px 8px; font-size: 11px; border: 1px solid #444; background: transparent; color: #555; border-radius: 4px; text-align: center;";
            ghostLaunch.textContent = "Launch";
            
            blueprintButtons.appendChild(ghostLaunch);
            
            blueprintContent.appendChild(blueprintInfo);
            blueprintContent.appendChild(blueprintButtons);
            
            blueprintDiv.appendChild(blueprintCheckbox);
            blueprintDiv.appendChild(blueprintContent);
            
            // Hover effect
            blueprintDiv.onmouseover = () => {
                blueprintDiv.style.borderColor = "#888";
                blueprintDiv.style.backgroundColor = "rgba(255, 255, 255, 0.05)";
                blueprintCheckbox.style.borderColor = "#888";
                blueprintCheckbox.style.color = "#aaa";
            };
            blueprintDiv.onmouseout = () => {
                blueprintDiv.style.borderColor = "#666";
                blueprintDiv.style.backgroundColor = "rgba(255, 255, 255, 0.02)";
                blueprintCheckbox.style.borderColor = "#666";
                blueprintCheckbox.style.color = "#888";
            };
            
            gpuList.appendChild(blueprintDiv);
        }
        
        // Show existing workers
        workers.forEach(worker => {
            const gpuDiv = this.ui.createCard('worker');
            
            // Left column - checkbox area (extends full height)
            const checkboxColumn = this.ui.createCardColumn('checkbox');
            checkboxColumn.style.cursor = "pointer";
            checkboxColumn.title = "Enable/disable this worker";
            
            // Standard checkbox
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.id = `gpu-${worker.id}`;
            checkbox.checked = worker.enabled;
            checkbox.style.cssText = "cursor: pointer; width: 16px; height: 16px; pointer-events: none;"; // pointer-events: none so the column click handles it
            
            checkboxColumn.appendChild(checkbox);
            
            // Click handler for checkbox column
            checkboxColumn.onclick = async (e) => {
                checkbox.checked = !checkbox.checked;
                await this.updateWorkerEnabled(worker.id, checkbox.checked);
                this.updateSummary();
            };
            
            // Right column - content area
            const contentColumn = this.ui.createCardColumn('content');
            
            // First row: worker info
            const infoRow = this.ui.createInfoRow();
            infoRow.title = "Click to expand settings";
            
            // Worker content
            const workerContent = this.ui.createWorkerContent();
            
            const statusDot = this.ui.createStatusDot(`status-${worker.id}`, "#666", "Checking status...");
            
            const gpuInfo = document.createElement("span");
            const isRemote = this.isRemoteWorker(worker);
            
            if (isRemote) {
                // For remote workers: show host:port
                gpuInfo.innerHTML = `<strong>${worker.name}</strong><br><small style="color: #888;">${worker.host}:${worker.port}</small>`;
            } else {
                // For local workers: show CUDA and port
                const cudaInfo = worker.cuda_device !== undefined ? `CUDA ${worker.cuda_device} • ` : '';
                gpuInfo.innerHTML = `<strong>${worker.name}</strong><br><small style="color: #888;">${cudaInfo}Port ${worker.port}</small>`;
            }
            
            workerContent.appendChild(statusDot);
            workerContent.appendChild(gpuInfo);
            
            // Settings arrow
            const settingsArrow = document.createElement("span");
            settingsArrow.className = "settings-arrow";
            settingsArrow.innerHTML = "▶";
            settingsArrow.style.cssText = "font-size: 12px; color: #888; transition: all 0.2s ease; margin-left: auto;";
            if (this.state.isWorkerExpanded(worker.id)) {
                settingsArrow.style.transform = "rotate(90deg)";
            }
            
            infoRow.appendChild(workerContent);
            infoRow.appendChild(settingsArrow);
            
            
            // Add content and arrow back to info row
            infoRow.appendChild(workerContent);
            infoRow.appendChild(settingsArrow);
            
            // Click handler for info row
            infoRow.onclick = () => {
                this.toggleWorkerExpanded(worker.id);
            };
            
            // Hover effect for entire content column
            contentColumn.onmouseover = () => {
                contentColumn.style.backgroundColor = "#333";
                settingsArrow.style.color = "#fff";
            };
            contentColumn.onmouseout = () => {
                contentColumn.style.backgroundColor = "transparent";
                settingsArrow.style.color = "#888";
            };
            
            contentColumn.appendChild(infoRow);
            
            // Second row: launch controls
            const controlsDiv = document.createElement("div");
            controlsDiv.id = `controls-${worker.id}`;
            controlsDiv.style.cssText = "padding: 0 8px 6px 8px; display: flex; gap: 4px;";
            
            if (this.isRemoteWorker(worker)) {
                // For remote workers, show styled info box
                const remoteInfo = document.createElement("div");
                remoteInfo.style.cssText = "background-color: #333; color: #888; padding: 4px 12px; border-radius: 4px; font-size: 11px; text-align: center; flex: 1;";
                remoteInfo.textContent = "Remote worker";
                controlsDiv.appendChild(remoteInfo);
            } else {
                // Local workers get launch/stop/log buttons
                const controls = this.ui.createWorkerControls(worker.id, {
                    launch: () => this.launchWorker(worker.id),
                    stop: () => this.stopWorker(worker.id),
                    viewLog: () => this.viewWorkerLog(worker.id)
                });
                
                // Apply custom button styles
                const launchBtn = controls.querySelector(`#launch-${worker.id}`);
                const stopBtn = controls.querySelector(`#stop-${worker.id}`);
                const logBtn = controls.querySelector(`#log-${worker.id}`);
                
                launchBtn.style.cssText += ' padding: 3px 8px; font-size: 11px; background-color: #555;';
                launchBtn.title = "Launch worker (runs in background with logging)";
                
                stopBtn.style.cssText += ' padding: 3px 8px; font-size: 11px; background-color: #555;';
                stopBtn.title = "Stop worker";
                
                logBtn.style.cssText += ' padding: 3px 8px; font-size: 11px; display: none; background-color: #685434;';
                
                // Move controls into the controlsDiv
                while (controls.firstChild) {
                    controlsDiv.appendChild(controls.firstChild);
                }
            }
            
            // Third row: expandable settings panel
            const settingsDiv = document.createElement("div");
            settingsDiv.id = `settings-${worker.id}`;
            settingsDiv.className = "worker-settings";
            if (this.state.isWorkerExpanded(worker.id)) {
                settingsDiv.classList.add("expanded");
            }
            settingsDiv.style.cssText = "margin: 0 8px; padding: 0 12px; background: #1e1e1e; border-radius: 4px;";
            if (this.state.isWorkerExpanded(worker.id)) {
                settingsDiv.style.padding = "12px";
                settingsDiv.style.marginTop = "8px";
                settingsDiv.style.marginBottom = "8px";
            }
            
            // Create settings form
            const settingsForm = this.createWorkerSettingsForm(worker);
            settingsDiv.appendChild(settingsForm);
            
            contentColumn.appendChild(controlsDiv);
            contentColumn.appendChild(settingsDiv);
            
            gpuDiv.appendChild(checkboxColumn);
            gpuDiv.appendChild(contentColumn);
            gpuList.appendChild(gpuDiv);
            
            // Update controls based on managed worker status
            this.updateWorkerControls(worker.id);
        });
        gpuSection.appendChild(gpuList);
        
        // Only show the minimal "Add Worker" box if there are existing workers
        if (workers.length > 0) {
            // Add Worker placeholder box
            const addWorkerDiv = this.ui.createCard('add', {
                title: "Click to add a new worker",
                onClick: () => this.addNewWorker()
            });
            
            // Left column - plus icon area
            const plusColumn = document.createElement("div");
            plusColumn.style.cssText = "flex: 0 0 40px; display: flex; align-items: center; justify-content: center; border-right: 1px solid #444; color: #555; font-size: 18px;";
            plusColumn.innerHTML = "+";
            
            // Right column - content area
            const addContentColumn = this.ui.createCardColumn('content');
            
            // Info row
            const addInfoRow = this.ui.createInfoRow();
            addInfoRow.style.minHeight = "42px"; // Reduced from 60px to 42px (30% reduction)
            
            // Placeholder content
            const addContent = this.ui.createWorkerContent();
            
            // Placeholder dot
            const placeholderDot = this.ui.createStatusDot(null, "transparent", "");
            placeholderDot.style.border = "1px solid #555";
            
            // Placeholder text
            const placeholderInfo = document.createElement("span");
            placeholderInfo.innerHTML = `<span style="color: #666; font-weight: normal; font-size: 13px;">Add New Worker</span>`;
            
            addContent.appendChild(placeholderDot);
            addContent.appendChild(placeholderInfo);
            addInfoRow.appendChild(addContent);
            addContentColumn.appendChild(addInfoRow);
            
            addWorkerDiv.appendChild(plusColumn);
            addWorkerDiv.appendChild(addContentColumn);
            
            // Hover effects
            addWorkerDiv.onmouseover = () => {
                addWorkerDiv.style.borderColor = "#666";
                addWorkerDiv.style.backgroundColor = "rgba(255, 255, 255, 0.02)";
                plusColumn.style.color = "#888";
                plusColumn.style.borderColor = "#666";
            };
            
            addWorkerDiv.onmouseout = () => {
                addWorkerDiv.style.borderColor = "#444";
                addWorkerDiv.style.backgroundColor = "transparent";
                plusColumn.style.color = "#555";
                plusColumn.style.borderColor = "#444";
            };
            
            gpuSection.appendChild(addWorkerDiv);
        }
        
        container.appendChild(gpuSection);
        
        const actionsSection = document.createElement("div");
        actionsSection.style.cssText = "padding-top: 10px; margin-bottom: 15px; border-top: 1px solid #444;";
        
        // Create a row for both buttons
        const buttonRow = document.createElement("div");
        buttonRow.style.cssText = "display: flex; gap: 8px;";
        
        const clearMemButton = this._createButton("Clear Worker VRAM", (e) => this._handleClearMemory(e.target), DistributedExtension.BUTTON_STYLES.clearMemory);
        clearMemButton.title = "Clear VRAM on all enabled worker GPUs (not master)";
        clearMemButton.style.cssText += " flex: 1;";
        
        const interruptButton = this._createButton("Interrupt Workers", (e) => this._handleInterruptWorkers(e.target), DistributedExtension.BUTTON_STYLES.clearMemory);
        interruptButton.title = "Cancel/interrupt execution on all enabled worker GPUs";
        interruptButton.style.cssText += " flex: 1;";
        
        buttonRow.appendChild(clearMemButton);
        buttonRow.appendChild(interruptButton);
        actionsSection.appendChild(buttonRow);
        
        container.appendChild(actionsSection);
        
        // Settings section
        const settingsSection = document.createElement("div");
        settingsSection.style.cssText = "border-top: 1px solid #444; padding-top: 10px; margin-bottom: 10px;";
        
        // Settings header with toggle
        const settingsHeader = document.createElement("div");
        settingsHeader.style.cssText = "display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none;";
        
        const workerSettingsTitle = document.createElement("h4");
        workerSettingsTitle.textContent = "Settings";
        workerSettingsTitle.style.cssText = "margin: 0; font-size: 14px;";
        
        const workerSettingsToggle = document.createElement("span");
        workerSettingsToggle.textContent = "▶"; // Right arrow when collapsed
        workerSettingsToggle.style.cssText = "font-size: 12px; color: #888; transition: all 0.2s ease;";
        
        settingsHeader.appendChild(workerSettingsTitle);
        settingsHeader.appendChild(workerSettingsToggle);
        
        // Hover effect for header
        settingsHeader.onmouseover = () => {
            workerSettingsToggle.style.color = "#fff";
        };
        settingsHeader.onmouseout = () => {
            workerSettingsToggle.style.color = "#888";
        };
        
        // Collapsible settings content
        const settingsContent = document.createElement("div");
        settingsContent.style.cssText = "max-height: 0; overflow: hidden; transition: max-height 0.3s ease;";
        
        const settingsDiv = document.createElement("div");
        settingsDiv.style.cssText = "display: flex; flex-direction: column; gap: 8px; padding-top: 10px;";
        
        // Toggle functionality
        let settingsExpanded = false;
        settingsHeader.onclick = () => {
            settingsExpanded = !settingsExpanded;
            if (settingsExpanded) {
                settingsContent.style.maxHeight = "200px";
                workerSettingsToggle.style.transform = "rotate(90deg)";
            } else {
                settingsContent.style.maxHeight = "0";
                workerSettingsToggle.style.transform = "rotate(0deg)";
            }
        };
        
        // Debug mode setting
        const debugGroup = document.createElement("div");
        debugGroup.style.cssText = "display: flex; align-items: center; gap: 8px;";
        
        const debugCheckbox = document.createElement("input");
        debugCheckbox.type = "checkbox";
        debugCheckbox.id = "setting-debug";
        debugCheckbox.checked = this.config?.settings?.debug || false;
        debugCheckbox.onchange = (e) => this._updateSetting('debug', e.target.checked);
        
        const debugLabel = document.createElement("label");
        debugLabel.htmlFor = "setting-debug";
        debugLabel.textContent = "Debug Mode";
        debugLabel.style.cssText = "font-size: 12px; color: #ccc; cursor: pointer;";
        
        debugGroup.appendChild(debugCheckbox);
        debugGroup.appendChild(debugLabel);
        
        // Auto-launch workers setting
        const autoLaunchGroup = document.createElement("div");
        autoLaunchGroup.style.cssText = "display: flex; align-items: center; gap: 8px;";
        
        const autoLaunchCheckbox = document.createElement("input");
        autoLaunchCheckbox.type = "checkbox";
        autoLaunchCheckbox.id = "setting-auto-launch";
        autoLaunchCheckbox.checked = this.config?.settings?.auto_launch_workers || false;
        autoLaunchCheckbox.onchange = (e) => this._updateSetting('auto_launch_workers', e.target.checked);
        
        const autoLaunchLabel = document.createElement("label");
        autoLaunchLabel.htmlFor = "setting-auto-launch";
        autoLaunchLabel.textContent = "Auto-launch Local Workers on Startup";
        autoLaunchLabel.style.cssText = "font-size: 12px; color: #ccc; cursor: pointer;";
        
        autoLaunchGroup.appendChild(autoLaunchCheckbox);
        autoLaunchGroup.appendChild(autoLaunchLabel);
        
        // Stop workers on exit setting
        const stopOnExitGroup = document.createElement("div");
        stopOnExitGroup.style.cssText = "display: flex; align-items: center; gap: 8px;";
        
        const stopOnExitCheckbox = document.createElement("input");
        stopOnExitCheckbox.type = "checkbox";
        stopOnExitCheckbox.id = "setting-stop-on-exit";
        stopOnExitCheckbox.checked = this.config?.settings?.stop_workers_on_master_exit !== false; // Default true
        stopOnExitCheckbox.onchange = (e) => this._updateSetting('stop_workers_on_master_exit', e.target.checked);
        
        const stopOnExitLabel = document.createElement("label");
        stopOnExitLabel.htmlFor = "setting-stop-on-exit";
        stopOnExitLabel.textContent = "Stop Local Workers on Master Exit";
        stopOnExitLabel.style.cssText = "font-size: 12px; color: #ccc; cursor: pointer;";
        
        stopOnExitGroup.appendChild(stopOnExitCheckbox);
        stopOnExitGroup.appendChild(stopOnExitLabel);
        
        settingsDiv.appendChild(debugGroup);
        settingsDiv.appendChild(autoLaunchGroup);
        settingsDiv.appendChild(stopOnExitGroup);
        settingsContent.appendChild(settingsDiv);
        
        settingsSection.appendChild(settingsHeader);
        settingsSection.appendChild(settingsContent);
        container.appendChild(settingsSection);

        const summarySection = document.createElement("div");
        summarySection.style.cssText = "border-top: 1px solid #444; padding-top: 10px;";
        const summary = document.createElement("div");
        summary.id = "distributed-summary";
        summary.style.cssText = "font-size: 11px; color: #888;";
        summarySection.appendChild(summary);
        container.appendChild(summarySection);
        el.appendChild(container);
        this.updateSummary();
        } finally {
            // Always reset the rendering flag
            this._isRendering = false;
        }
    }

    updateMasterDisplay() {
        // Update CUDA info if element exists
        const cudaInfo = document.getElementById('master-cuda-info');
        if (cudaInfo) {
            const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
            if (this.masterCudaDevice !== undefined && this.masterCudaDevice !== null) {
                cudaInfo.textContent = `CUDA ${this.masterCudaDevice} • Port ${port}`;
            } else {
                cudaInfo.textContent = `Port ${port}`;
            }
        }
        
        // Update name if changed
        const nameDisplay = document.getElementById('master-name-display');
        if (nameDisplay && this.config?.master?.name) {
            nameDisplay.textContent = this.config.master.name;
        }
    }

    updateSummary() {
        const summaryEl = document.getElementById('distributed-summary');
        if (summaryEl) {
            const totalGPUs = this.enabledWorkers.length + 1;
            if (this.isEnabled) {
                summaryEl.textContent = `If Collector node is present, total generation = (${totalGPUs} GPUs × Batch Size)`;
            } else {
                summaryEl.textContent = "Only the master GPU will be used.";
            }
        }
    }

    // --- Helper Methods ---

    // --- Core Logic & Execution ---

    setupInterceptor() {
        api.queuePrompt = async (number, prompt) => {
            if (this.isEnabled) {
                const hasCollector = this.findNodesByClass(prompt.output, "DistributedCollector").length > 0;
                const hasDistUpscale = this.findNodesByClass(prompt.output, "UltimateSDUpscaleDistributed").length > 0;
                
                if (hasCollector || hasDistUpscale) {
                    const result = await this.executeParallelDistributed(prompt);
                    // Check status after dispatching jobs
                    setTimeout(() => this.checkAllWorkerStatuses(), 500);
                    return result;
                }
            }
            return this.originalQueuePrompt(number, prompt);
        };
    }

    async executeParallelDistributed(promptWrapper) {
        try {
            const executionPrefix = "exec_" + Date.now(); // Unique ID for this specific execution
            const enabledWorkers = this.enabledWorkers;
            
            // Pre-flight health check on all enabled workers
            const activeWorkers = await this.performPreflightCheck(enabledWorkers);
            
            if (activeWorkers.length === 0 && enabledWorkers.length > 0) {
                this.log("No active workers found. Running on master only.");
                // Fall back to master-only execution
                return this.originalQueuePrompt(0, promptWrapper);
            }
            
            this.debugLog(`Pre-flight check: ${activeWorkers.length} of ${enabledWorkers.length} workers are active`);
            
            // Find all distributed nodes in the workflow
            const collectorNodes = this.findNodesByClass(promptWrapper.output, "DistributedCollector");
            const upscaleNodes = this.findNodesByClass(promptWrapper.output, "UltimateSDUpscaleDistributed");
            const allDistributedNodes = [...collectorNodes, ...upscaleNodes];
            
            // Map original node IDs to truly unique job IDs for this specific run
            const job_id_map = new Map(allDistributedNodes.map(node => [node.id, `${executionPrefix}_${node.id}`]));
            
            // Prepare a separate job queue on the backend for each unique job ID
            const preparePromises = Array.from(job_id_map.values()).map(uniqueId => this._prepareDistributedJob(uniqueId));
            await Promise.all(preparePromises);

            const jobs = [];
            // Use only active workers
            const participants = ['master', ...activeWorkers.map(w => w.id)];

            for (const participantId of participants) {
                const options = { 
                    enabled_worker_ids: activeWorkers.map(w => w.id), 
                    workflow: promptWrapper.workflow,
                    job_id_map: job_id_map // Pass the map of unique IDs
                };
                
                const jobApiPrompt = this._prepareApiPromptForParticipant(
                    promptWrapper.output, participantId, options
                );
                
                if (participantId === 'master') {
                    jobs.push({ type: 'master', promptWrapper: { ...promptWrapper, output: jobApiPrompt } });
                } else {
                    const worker = activeWorkers.find(w => w.id === participantId);
                    if (worker) {
                        const job = {
                            type: 'worker',
                            worker,
                            prompt: jobApiPrompt,
                            workflow: promptWrapper.workflow
                        };
                        
                        // Add image references if found for remote workers
                        if (options._imageReferences) {
                            job.imageReferences = options._imageReferences;
                        }
                        
                        jobs.push(job);
                    }
                }
            }
            
            const result = await this._executeJobs(jobs);
            return result;
        } catch (error) {
            this.log("Parallel execution failed: " + error.message);
            throw error;
        }
    }


    async _executeJobs(jobs) {
        let masterPromptId = null;
        
        // Pre-load all unique images before dispatching to workers
        const allImageReferences = new Map();
        for (const job of jobs) {
            if (job.type === 'worker' && job.imageReferences) {
                for (const [filename, info] of job.imageReferences) {
                    allImageReferences.set(filename, info);
                }
            }
        }
        
        if (allImageReferences.size > 0) {
            this.debugLog(`Pre-loading ${allImageReferences.size} unique image(s) for all workers`);
            await this.loadImagesForWorker(allImageReferences);
        }
        
        // Now dispatch jobs in parallel
        const promises = jobs.map(job => {
            if (job.type === 'master') {
                return this.originalQueuePrompt(0, job.promptWrapper).then(result => {
                    masterPromptId = result;
                    return result;
                });
            } else {
                return this._dispatchToWorker(job.worker, job.prompt, job.workflow, job.imageReferences);
            }
        });
        await Promise.all(promises);
        return masterPromptId || { "prompt_id": "distributed-job-dispatched" };
    }
    
    // --- Helper Methods ---

    findNodesByClass(apiPrompt, className) {
        return Object.entries(apiPrompt)
            .filter(([, nodeData]) => nodeData.class_type === className)
            .map(([nodeId, nodeData]) => ({ id: nodeId, data: nodeData }));
    }
    
    /**
     * Find all image references in the workflow
     * Looks for inputs named "image" that contain filename strings
     */
    findImageReferences(apiPrompt) {
        const images = new Map();
        // Updated regex to handle:
        // - Standard files: "image.png"
        // - Subfolder files: "subfolder/image.png"
        // - ComfyUI special format: "clipspace/file.png [input]"
        const imageExtensions = /\.(png|jpg|jpeg|gif|webp|bmp)(\s*\[\w+\])?$/i;
        
        for (const [nodeId, node] of Object.entries(apiPrompt)) {
            if (node.inputs && node.inputs.image) {
                const imageValue = node.inputs.image;
                // Check if it's a string filename (not an array connection)
                if (typeof imageValue === 'string' && imageExtensions.test(imageValue)) {
                    images.set(imageValue, {
                        nodeId,
                        nodeType: node.class_type,
                        inputName: 'image'
                    });
                    this.debugLog(`Found image reference: ${imageValue} in node ${nodeId} (${node.class_type})`);
                }
            }
        }
        
        return images;
    }

    /**
     * Find all nodes connected to collector nodes (both upstream and downstream)
     * @param {Object} apiPrompt - The workflow API prompt
     * @param {Array<string>} collectorIds - Array of collector node IDs
     * @returns {Set<string>} Set of node IDs connected to any collector
     */
    findCollectorConnectedNodes(apiPrompt, collectorIds) {
        const connected = new Set(collectorIds); // Include all collectors
        const toProcess = [...collectorIds];
        
        // First, build a map of which nodes output to which nodes
        const outputMap = new Map(); // sourceNode -> Set of target nodes
        for (const [nodeId, node] of Object.entries(apiPrompt)) {
            if (node.inputs) {
                for (const [inputName, inputValue] of Object.entries(node.inputs)) {
                    if (Array.isArray(inputValue) && inputValue.length === 2) {
                        const sourceNodeId = String(inputValue[0]);
                        if (!outputMap.has(sourceNodeId)) {
                            outputMap.set(sourceNodeId, new Set());
                        }
                        outputMap.get(sourceNodeId).add(nodeId);
                    }
                }
            }
        }
        
        // Traverse both upstream and downstream
        while (toProcess.length > 0) {
            const nodeId = toProcess.pop();
            const node = apiPrompt[nodeId];
            
            // Traverse upstream (inputs)
            if (node && node.inputs) {
                for (const [inputName, inputValue] of Object.entries(node.inputs)) {
                    if (Array.isArray(inputValue) && inputValue.length === 2) {
                        const sourceNodeId = String(inputValue[0]);
                        if (!connected.has(sourceNodeId)) {
                            connected.add(sourceNodeId);
                            toProcess.push(sourceNodeId);
                        }
                    }
                }
            }
            
            // Traverse downstream (outputs)
            if (outputMap.has(nodeId)) {
                for (const targetNodeId of outputMap.get(nodeId)) {
                    if (!connected.has(targetNodeId)) {
                        connected.add(targetNodeId);
                        toProcess.push(targetNodeId);
                    }
                }
            }
        }
        
        return connected;
    }

    /**
     * Find only upstream nodes (inputs) for distributed collector nodes
     * This is used for workers to avoid executing downstream nodes like SaveImage
     * @param {Object} apiPrompt - The API prompt containing the workflow
     * @param {Array<string>} collectorIds - Array of collector node IDs
     * @returns {Set<string>} Set of node IDs that feed into collectors
     */
    findCollectorUpstreamNodes(apiPrompt, collectorIds) {
        const connected = new Set(collectorIds); // Include all collectors
        const toProcess = [...collectorIds];
        
        // Only traverse upstream (inputs)
        while (toProcess.length > 0) {
            const nodeId = toProcess.pop();
            const node = apiPrompt[nodeId];
            
            // Traverse upstream (inputs) only
            if (node && node.inputs) {
                for (const [inputName, inputValue] of Object.entries(node.inputs)) {
                    if (Array.isArray(inputValue) && inputValue.length === 2) {
                        const sourceNodeId = String(inputValue[0]);
                        if (!connected.has(sourceNodeId)) {
                            connected.add(sourceNodeId);
                            toProcess.push(sourceNodeId);
                        }
                    }
                }
            }
        }
        
        return connected;
    }

    /**
     * Prune workflow to only include nodes connected to distributed nodes
     * @param {Object} apiPrompt - The full workflow API prompt
     * @param {Array} distributedNodes - Array of distributed nodes (optional, will find if not provided)
     * @returns {Object} Pruned API prompt with only required nodes
     */
    pruneWorkflowForWorker(apiPrompt, distributedNodes = null) {
        // Find all distributed nodes if not provided
        if (!distributedNodes) {
            const collectorNodes = this.findNodesByClass(apiPrompt, "DistributedCollector");
            const upscaleNodes = this.findNodesByClass(apiPrompt, "UltimateSDUpscaleDistributed");
            distributedNodes = [...collectorNodes, ...upscaleNodes];
        }
        
        if (distributedNodes.length === 0) {
            // No distributed nodes, return full workflow
            return apiPrompt;
        }
        
        // Get all nodes connected to distributed nodes
        const distributedIds = distributedNodes.map(node => node.id);
        
        // For workers, only include upstream nodes (this removes ALL downstream nodes after collectors)
        const connectedNodes = this.findCollectorUpstreamNodes(apiPrompt, distributedIds);
        
        this.debugLog(`Pruning workflow: keeping ${connectedNodes.size} of ${Object.keys(apiPrompt).length} nodes (removed all downstream nodes)`);
        
        // Create pruned prompt with only required nodes
        const prunedPrompt = {};
        for (const nodeId of connectedNodes) {
            prunedPrompt[nodeId] = JSON.parse(JSON.stringify(apiPrompt[nodeId]));
        }
        
        // Check if any distributed node has downstream SaveImage nodes that were removed
        // If so, add a PreviewImage node after the collector
        for (const distNode of distributedNodes) {
            const distNodeId = distNode.id;
            
            // Check if this distributed node had any downstream nodes in the original workflow
            const originalOutputMap = new Map();
            for (const [nodeId, node] of Object.entries(apiPrompt)) {
                if (node.inputs) {
                    for (const [inputName, inputValue] of Object.entries(node.inputs)) {
                        if (Array.isArray(inputValue) && inputValue.length === 2 && String(inputValue[0]) === distNodeId) {
                            if (!originalOutputMap.has(distNodeId)) {
                                originalOutputMap.set(distNodeId, []);
                            }
                            originalOutputMap.get(distNodeId).push({nodeId, inputName});
                        }
                    }
                }
            }
            
            // If this distributed node had downstream nodes that were removed, add a PreviewImage
            if (originalOutputMap.has(distNodeId) && originalOutputMap.get(distNodeId).length > 0) {
                // Generate a unique ID for the preview node
                const previewNodeId = `preview_${distNodeId}`;
                
                // Add PreviewImage node connected to the distributed node
                prunedPrompt[previewNodeId] = {
                    inputs: {
                        images: [distNodeId, 0]  // Connect to first output of distributed node
                    },
                    class_type: "PreviewImage",
                    _meta: {
                        title: "Preview Image (auto-added)"
                    }
                };
                
                this.debugLog(`Added PreviewImage node ${previewNodeId} after distributed node ${distNodeId} for worker`);
            }
        }
        
        return prunedPrompt;
    }


    _prepareApiPromptForParticipant(baseApiPrompt, participantId, options = {}) {
        let jobApiPrompt = JSON.parse(JSON.stringify(baseApiPrompt));
        const isMaster = participantId === 'master';
        
        // Find all distributed nodes once (before pruning)
        const collectorNodes = this.findNodesByClass(jobApiPrompt, "DistributedCollector");
        const upscaleNodes = this.findNodesByClass(jobApiPrompt, "UltimateSDUpscaleDistributed");
        const allDistributedNodes = [...collectorNodes, ...upscaleNodes];
        
        // For workers, prune the workflow to only include distributed node dependencies
        if (!isMaster && allDistributedNodes.length > 0) {
            jobApiPrompt = this.pruneWorkflowForWorker(jobApiPrompt, allDistributedNodes);
        }
        
        // Handle image references for remote workers
        if (!isMaster && options.enabled_worker_ids) {
            // Check if this is a remote worker
            const workerId = participantId;
            const workerInfo = this.config.workers.find(w => w.id === workerId);
            const isRemote = workerInfo && workerInfo.host;
            
            if (isRemote) {
                // Find all image references in the pruned workflow
                const imageReferences = this.findImageReferences(jobApiPrompt);
                if (imageReferences.size > 0) {
                    this.debugLog(`Found ${imageReferences.size} image references for remote worker ${workerId}`);
                    // Store image references for later processing
                    options._imageReferences = imageReferences;
                }
            }
        }
        
        // Handle Distributed seed nodes
        const distributorNodes = this.findNodesByClass(jobApiPrompt, "DistributedSeed");
        if (distributorNodes.length > 0) {
            this.debugLog(`Found ${distributorNodes.length} seed node(s)`);
        }
        
        for (const seedNode of distributorNodes) {
            const { inputs } = jobApiPrompt[seedNode.id];
            inputs.is_worker = !isMaster;
            if (!isMaster) {
                const workerIndex = options.enabled_worker_ids.indexOf(participantId);
                inputs.worker_id = `worker_${workerIndex}`;
                this.debugLog(`Set seed node ${seedNode.id} for worker ${workerIndex}`);
            }
        }
        
        // Handle Distributed collector nodes (already found above)
        for (const collector of collectorNodes) {
            const { inputs } = jobApiPrompt[collector.id];
            
            // Check if this collector is downstream from a distributed upscaler
            const hasUpstreamDistributedUpscaler = this._hasUpstreamNode(
                jobApiPrompt, 
                collector.id, 
                'UltimateSDUpscaleDistributed'
            );
            
            if (hasUpstreamDistributedUpscaler) {
                // Set pass_through mode for this collector
                inputs.pass_through = true;
                this.debugLog(`Collector ${collector.id} set to pass-through mode (downstream from distributed upscaler)`);
            } else {
                // Normal collector behavior
                // Get the unique job ID from the map created for this execution
                const uniqueJobId = options.job_id_map ? options.job_id_map.get(collector.id) : collector.id;
                
                // Use the truly unique ID for this execution
                inputs.multi_job_id = uniqueJobId;
                inputs.is_worker = !isMaster;
                if (isMaster) {
                    inputs.enabled_worker_ids = JSON.stringify(options.enabled_worker_ids || []);
                } else {
                    inputs.master_url = this.getMasterUrl();
                    // Also make the worker_job_id unique to prevent potential caching issues
                    inputs.worker_job_id = `${uniqueJobId}_worker_${participantId}`;
                    inputs.worker_id = participantId;
                }
            }
        }
        
        // Handle Ultimate SD Upscale Distributed nodes
        for (const upscaleNode of upscaleNodes) {
            const { inputs } = jobApiPrompt[upscaleNode.id];
            
            // Get the unique job ID from the map
            const uniqueJobId = options.job_id_map ? options.job_id_map.get(upscaleNode.id) : upscaleNode.id;
            
            inputs.multi_job_id = uniqueJobId;
            inputs.is_worker = !isMaster;
            
            if (isMaster) {
                inputs.enabled_worker_ids = JSON.stringify(options.enabled_worker_ids || []);
            } else {
                inputs.master_url = this.getMasterUrl();
                inputs.worker_id = participantId;
                // Workers also need the enabled_worker_ids to calculate tile distribution
                inputs.enabled_worker_ids = JSON.stringify(options.enabled_worker_ids || []);
            }
        }
        
        return jobApiPrompt;
    }

    async _prepareDistributedJob(multi_job_id) {
        try {
            await this.api.prepareJob(multi_job_id);
        } catch (error) {
            this.log("Error preparing job: " + error.message);
            throw error;
        }
    }

    /**
     * Check if a node has an upstream node of a specific type
     * @param {Object} apiPrompt - The workflow API prompt
     * @param {string} nodeId - The node to check
     * @param {string} upstreamType - The class_type to look for upstream
     * @returns {boolean} True if an upstream node of the specified type exists
     */
    _hasUpstreamNode(apiPrompt, nodeId, upstreamType) {
        const visited = new Set();
        const toProcess = [nodeId];
        
        while (toProcess.length > 0) {
            const currentId = toProcess.pop();
            if (visited.has(currentId)) continue;
            visited.add(currentId);
            
            const node = apiPrompt[currentId];
            if (!node) continue;
            
            // Check inputs for upstream connections
            if (node.inputs) {
                for (const [inputName, inputValue] of Object.entries(node.inputs)) {
                    if (Array.isArray(inputValue) && inputValue.length === 2) {
                        const sourceNodeId = String(inputValue[0]);
                        const sourceNode = apiPrompt[sourceNodeId];
                        
                        if (sourceNode && sourceNode.class_type === upstreamType) {
                            return true;
                        }
                        
                        if (!visited.has(sourceNodeId)) {
                            toProcess.push(sourceNodeId);
                        }
                    }
                }
            }
        }
        
        return false;
    }

    startStatusChecking() {
        // Start checking every 2 seconds for more responsive updates
        // Don't check immediately since panel might not be open yet
        this.statusCheckInterval = setInterval(() => {
            this.checkAllWorkerStatuses();
        }, 2000);
    }
    
    async checkAllWorkerStatuses() {
        // Check master status
        this.checkMasterStatus();
        
        if (!this.config || !this.config.workers) return;
        
        for (const worker of this.config.workers) {
            // Check status for enabled workers OR workers that are launching
            if (worker.enabled || this.state.isWorkerLaunching(worker.id)) {
                this.checkWorkerStatus(worker);
            }
        }
    }
    
    async checkMasterStatus() {
        try {
            const response = await fetch(`${window.location.origin}/prompt`, {
                method: 'GET',
                signal: AbortSignal.timeout(900)
            });
            
            if (response.ok) {
                const data = await response.json();
                const queueRemaining = data.exec_info?.queue_remaining || 0;
                const isProcessing = queueRemaining > 0;
                
                // Update master status dot
                const statusDot = document.getElementById('master-status');
                if (statusDot) {
                    if (isProcessing) {
                        statusDot.style.backgroundColor = "#f0ad4e";
                        statusDot.title = `Processing (${queueRemaining} in queue)`;
                    } else {
                        statusDot.style.backgroundColor = "#4CAF50";
                        statusDot.title = "Online";
                    }
                }
            }
        } catch (error) {
            // Master is always online (we're running on it), so keep it green
            const statusDot = document.getElementById('master-status');
            if (statusDot) {
                statusDot.style.backgroundColor = "#4CAF50";
                statusDot.title = "Online";
            }
        }
    }
    
    // Helper to build worker URL
    getWorkerUrl(worker, endpoint = '') {
        const host = worker.host || window.location.hostname;
        
        // Simple rule: port 443 = HTTPS, anything else = HTTP
        const useHttps = worker.port === 443;
        const protocol = useHttps ? 'https' : 'http';
        
        // Don't add port for standard HTTPS/HTTP ports
        const defaultPort = useHttps ? 443 : 80;
        const port = worker.port === defaultPort ? '' : `:${worker.port}`;
        
        return `${protocol}://${host}${port}${endpoint}`;
    }

    async checkWorkerStatus(worker) {
        const url = this.getWorkerUrl(worker, '/prompt');
        const statusDot = document.getElementById(`status-${worker.id}`);
        
        try {
            const response = await fetch(url, {
                method: 'GET',
                mode: 'cors',
                signal: AbortSignal.timeout(900) // Use your increased timeout
            });
            
            if (response.ok) {
                const data = await response.json();
                const queueRemaining = data.exec_info?.queue_remaining || 0;
                const isProcessing = queueRemaining > 0;
                
                // Update status
                this.state.setWorkerStatus(worker.id, {
                    online: true,
                    processing: isProcessing,
                    queueCount: queueRemaining
                });
                
                // Update status dot based on processing state
                if (isProcessing) {
                    this.updateStatusDot(worker.id, "#f0ad4e", `Online - Processing (${queueRemaining} in queue)`, false);
                } else {
                    this.updateStatusDot(worker.id, "#3ca03c", "Online - Idle", false);
                }
                
                // Clear launching state since worker is now online
                if (this.state.isWorkerLaunching(worker.id)) {
                    this.state.setWorkerLaunching(worker.id, false);
                    this.clearLaunchingFlag(worker.id);
                }
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            // Worker is offline or unreachable
            this.state.setWorkerStatus(worker.id, {
                online: false,
                processing: false,
                queueCount: 0
            });
            
            // Check if worker is launching
            if (this.state.isWorkerLaunching(worker.id)) {
                this.updateStatusDot(worker.id, "#f0ad4e", "Launching...", true);
            } else {
                // Only update to red if not currently launching
                this.updateStatusDot(worker.id, "#c04c4c", "Offline - Cannot connect", false);
            }
            
            this.debugLog(`Worker ${worker.id} status check failed: ${error.message}`);
        }
        
        // Update control buttons based on new status
        this.updateWorkerControls(worker.id);
    }
    
    async _dispatchToWorker(worker, prompt, workflow, imageReferences) {
        const workerUrl = this.getWorkerUrl(worker);
        
        // Debug logging - always log to console for debugging
        console.log(`[Distributed] === Dispatching to ${worker.name} (${worker.id}) ===`);
        console.log('[Distributed] Worker URL:', workerUrl);
        
        // Handle image uploads for remote workers
        if (imageReferences && imageReferences.size > 0) {
            if (this.debugMode) {
                console.log(`[Distributed] Processing ${imageReferences.size} image(s) for remote worker`);
            }
            
            try {
                // Load images from master
                const images = await this.loadImagesForWorker(imageReferences);
                
                // Upload images to worker
                if (images.length > 0) {
                    await this.uploadImagesToWorker(workerUrl, images);
                    if (this.debugMode) {
                        console.log(`[Distributed] Successfully uploaded ${images.length} image(s) to worker`);
                    }
                }
            } catch (error) {
                this.log(`Failed to process images for worker ${worker.name}: ${error.message}`);
                // Continue with workflow execution even if image upload fails
            }
        }
        
        const promptToSend = {
            prompt,
            extra_data: { extra_pnginfo: { workflow } },
            client_id: api.clientId
        };
        
        console.log('[Distributed] Prompt data:', promptToSend);
        
        try {
            await fetch(`${workerUrl}/prompt`, { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                mode: 'cors', 
                body: JSON.stringify(promptToSend) 
            });
        } catch (e) {
            this.log(`Failed to connect to worker ${worker.name} at ${workerUrl}: ${e.message}`);
        }
    }
    
    async loadImagesForWorker(imageReferences) {
        const images = [];
        
        // Use a cache to avoid loading the same image multiple times
        if (!this._imageCache) {
            this._imageCache = new Map();
        }
        
        for (const [filename, info] of imageReferences) {
            try {
                // Check cache first
                if (this._imageCache.has(filename)) {
                    images.push(this._imageCache.get(filename));
                    this.debugLog(`Using cached image: ${filename}`);
                    continue;
                }
                
                // Load image from master's filesystem via API
                try {
                    const data = await this.api.loadImage(filename);
                    const imageData = {
                        name: filename,
                        image: data.image_data
                    };
                    images.push(imageData);
                    
                    // Cache the image for future use
                    this._imageCache.set(filename, imageData);
                    this.debugLog(`Loaded and cached image: ${filename}`);
                } catch (loadError) {
                    this.log(`Failed to load image ${filename}: ${loadError.message}`);
                    throw loadError;
                }
            } catch (error) {
                this.log(`Error loading image ${filename}: ${error.message}`);
            }
        }
        
        // Clear cache after a reasonable time to avoid memory issues
        setTimeout(() => {
            if (this._imageCache && this._imageCache.size > 0) {
                this.debugLog(`Clearing image cache (${this._imageCache.size} images)`);
                this._imageCache.clear();
            }
        }, 30000); // Clear after 30 seconds
        
        return images;
    }
    
    async uploadImagesToWorker(workerUrl, images) {
        // Upload images to worker's ComfyUI instance
        for (const imageData of images) {
            const formData = new FormData();
            
            // Convert base64 to blob
            const base64Data = imageData.image.replace(/^data:image\/\w+;base64,/, '');
            const byteCharacters = atob(base64Data);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: 'image/png' });
            
            // Clean the filename - remove [input] suffix and handle subfolder
            let cleanName = imageData.name;
            let subfolder = '';
            
            // Remove [input] or other suffixes
            cleanName = cleanName.replace(/\s*\[\w+\]$/, '');
            
            // Extract subfolder if present
            if (cleanName.includes('/')) {
                const parts = cleanName.split('/');
                subfolder = parts.slice(0, -1).join('/');
                cleanName = parts[parts.length - 1];
            }
            
            formData.append('image', blob, cleanName);
            formData.append('type', 'input');
            formData.append('subfolder', subfolder);
            formData.append('overwrite', 'true');
            
            try {
                const response = await fetch(`${workerUrl}/upload/image`, {
                    method: 'POST',
                    mode: 'cors',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`Upload failed: ${response.statusText}`);
                }
                
                this.debugLog(`Uploaded image to worker: ${imageData.name} -> ${subfolder}/${cleanName}`);
            } catch (error) {
                throw new Error(`Failed to upload ${imageData.name}: ${error.message}`);
            }
        }
    }
    
    async performPreflightCheck(workers) {
        if (workers.length === 0) return [];
        
        this.debugLog(`Performing pre-flight health check on ${workers.length} workers...`);
        const startTime = Date.now();
        
        const checkPromises = workers.map(async (worker) => {
            const url = this.getWorkerUrl(worker, '/prompt');
            
            this.debugLog(`Pre-flight checking ${worker.name} at: ${url}`);
            
            try {
                const response = await fetch(url, {
                    method: 'GET',
                    mode: 'cors',
                    signal: AbortSignal.timeout(900) // Use your increased timeout
                });
                
                if (response.ok) {
                    this.debugLog(`Worker ${worker.name} is active`);
                    return { worker, active: true };
                } else {
                    this.debugLog(`Worker ${worker.name} returned ${response.status}`);
                    return { worker, active: false };
                }
            } catch (error) {
                this.debugLog(`Worker ${worker.name} is offline or unreachable: ${error.message}`);
                return { worker, active: false };
            }
        });
        
        const results = await Promise.all(checkPromises);
        const activeWorkers = results.filter(r => r.active).map(r => r.worker);
        
        const elapsed = Date.now() - startTime;
        this.debugLog(`Pre-flight check completed in ${elapsed}ms. Active workers: ${activeWorkers.length}/${workers.length}`);
        
        // Update UI status indicators for inactive workers
        results.filter(r => !r.active).forEach(r => {
            const statusDot = document.getElementById(`status-${r.worker.id}`);
            if (statusDot) {
                // Remove pulsing animation once status is determined
                statusDot.classList.remove('status-pulsing');
                statusDot.style.backgroundColor = "#c04c4c"; // Red for offline
                statusDot.title = "Offline - Cannot connect";
            }
        });
        
        return activeWorkers;
    }
    async launchWorker(workerId) {
        const worker = this.config.workers.find(w => w.id === workerId);
        const launchBtn = document.querySelector(`#controls-${workerId} button`);

        this.updateStatusDot(workerId, "#f0ad4e", "Launching...", true);
        this.state.setWorkerLaunching(workerId, true);

        // Allow 90 seconds for worker to launch (model loading can take time)
        setTimeout(() => {
            this.state.setWorkerLaunching(workerId, false);
        }, 90000);

        if (!launchBtn) return;

        try {
            // Disable button immediately
            launchBtn.disabled = true;
            
            const result = await this.api.launchWorker(workerId);
            if (result) {
                this.log(`Launched ${worker.name} (PID: ${result.pid})`);
                if (result.log_file) {
                    this.debugLog(`Log file: ${result.log_file}`);
                }
                
                this.state.setWorkerManaged(workerId, {
                    pid: result.pid,
                    log_file: result.log_file,
                    started_at: Date.now()
                });
                
                // Update controls immediately to hide launch button and show stop/log buttons
                this.updateWorkerControls(workerId);
                setTimeout(() => this.checkWorkerStatus(worker), 2000);
            }
        } catch (error) {
            // Check if worker was already running
            if (error.message && error.message.includes("already running")) {
                this.log(`Worker ${worker.name} is already running`);
                this.updateWorkerControls(workerId);
                setTimeout(() => this.checkWorkerStatus(worker), 100);
            } else {
                this.log(`Error launching worker: ${error.message || error}`);
                
                // Re-enable button on error
                if (launchBtn) {
                    launchBtn.disabled = false;
                }
            }
        }
    }    
    
    async stopWorker(workerId) {
        const worker = this.config.workers.find(w => w.id === workerId);
        const stopBtn = document.querySelectorAll(`#controls-${workerId} button`)[1];
        
        // Provide immediate feedback
        if (stopBtn) {
            stopBtn.disabled = true;
            stopBtn.textContent = "Stopping...";
            stopBtn.style.backgroundColor = "#666";
        }
        
        try {
            const result = await this.api.stopWorker(workerId);
            if (result) {
                this.log(`Stopped worker: ${result.message}`);
                this.state.setWorkerManaged(workerId, null);
                
                // Flash success feedback
                if (stopBtn) {
                    stopBtn.style.backgroundColor = DistributedExtension.BUTTON_STYLES.success;
                    stopBtn.textContent = "Stopped!";
                    setTimeout(() => {
                        this.updateWorkerControls(workerId);
                    }, 1500);
                }
                
                // Update status
                setTimeout(() => this.checkWorkerStatus(worker), 500);
            } else {
                this.log(`Failed to stop worker: ${result.message}`);
                
                // Flash error feedback
                if (stopBtn) {
                    stopBtn.style.backgroundColor = DistributedExtension.BUTTON_STYLES.error;
                    stopBtn.textContent = result.message.includes("already stopped") ? "Not Running" : "Failed";
                    setTimeout(() => {
                        this.updateWorkerControls(workerId);
                    }, 2000);
                }
            }
        } catch (error) {
            this.log(`Error stopping worker: ${error}`);
            
            // Flash error feedback
            if (stopBtn) {
                stopBtn.style.backgroundColor = DistributedExtension.BUTTON_STYLES.error;
                stopBtn.textContent = "Error";
                setTimeout(() => {
                    this.updateWorkerControls(workerId);
                }, 2000);
            }
        }
    }
    
    async clearLaunchingFlag(workerId) {
        try {
            await this.api.clearLaunchingFlag(workerId);
            this.debugLog(`Cleared launching flag for worker ${workerId}`);
        } catch (error) {
            this.debugLog(`Error clearing launching flag: ${error.message || error}`);
        }
    }
    
    // Helper Functions to reduce redundancy
    
    /**
     * Generic async button action handler
     */
    async handleAsyncButtonAction(button, action, successText, errorText, resetDelay = 3000) {
        const originalText = button.textContent;
        const originalStyle = button.style.cssText;
        button.disabled = true;
        
        try {
            await action();
            button.textContent = successText;
            button.style.cssText = originalStyle;
            button.style.backgroundColor = DistributedExtension.BUTTON_STYLES.success;
            return true;
        } catch (error) {
            button.textContent = errorText || `Error: ${error.message}`;
            button.style.cssText = originalStyle;
            button.style.backgroundColor = DistributedExtension.BUTTON_STYLES.error;
            throw error;
        } finally {
            setTimeout(() => {
                button.textContent = originalText;
                button.style.cssText = originalStyle;
                button.disabled = false;
            }, resetDelay);
        }
    }
    
    /**
     * Show toast notification helper
     */
    showToast(severity, summary, detail, life = 3000) {
        if (app.extensionManager?.toast?.add) {
            app.extensionManager.toast.add({ severity, summary, detail, life });
        }
    }
    
    /**
     * Update status dot helper
     */
    updateStatusDot(workerId, color, title, pulsing = false) {
        const statusDot = document.getElementById(`status-${workerId}`);
        if (!statusDot) return;
        
        statusDot.style.backgroundColor = color;
        statusDot.title = title;
        statusDot.classList.toggle('status-pulsing', pulsing);
    }
    
    /**
     * Cleanup method to stop intervals and listeners
     */
    cleanup() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
        
        if (this.logAutoRefreshInterval) {
            clearInterval(this.logAutoRefreshInterval);
            this.logAutoRefreshInterval = null;
        }
        
        this.debugLog("Cleaned up intervals");
    }
    
    async loadManagedWorkers() {
        try {
            const result = await this.api.getManagedWorkers();
            // Clear all managed worker info
            this.state.clearCache();
            
            // Check for launching workers
            for (const [workerId, info] of Object.entries(result.managed_workers)) {
                this.state.setWorkerManaged(workerId, info);
                
                // If worker is marked as launching, add to launchingWorkers set
                if (info.launching) {
                    this.state.setWorkerLaunching(workerId, true);
                    this.debugLog(`Worker ${workerId} is in launching state`);
                }
            }
            
            // Update UI for all workers
            if (this.config?.workers) {
                this.config.workers.forEach(w => this.updateWorkerControls(w.id));
            }
        } catch (error) {
            this.debugLog(`Error loading managed workers: ${error}`);
        }
    }
    
    updateWorkerControls(workerId) {
        const controlsDiv = document.getElementById(`controls-${workerId}`);
        
        if (!controlsDiv) return;
        
        const worker = this.config.workers.find(w => w.id === workerId);
        if (!worker) return;
        
        // Skip button updates for remote workers
        if (this.isRemoteWorker(worker)) {
            return;
        }
        
        // Ensure we check for string ID
        const managedInfo = this.state.getWorker(workerId).managed;
        const status = this.state.getWorkerStatus(workerId);
        
        // Update button states
        const buttons = controlsDiv.querySelectorAll('button');
        const launchBtn = buttons[0];
        const stopBtn = buttons[1];
        const logBtn = document.getElementById(`log-${workerId}`);
        
        // Show log button immediately if we have log file info (even if worker is still starting)
        if (managedInfo?.log_file && logBtn) {
            logBtn.style.display = 'inline-block';
        } else if (logBtn && !managedInfo) {
            logBtn.style.display = 'none';
        }
        
        if (status?.online || managedInfo) {
            // Worker is running or we just launched it
            launchBtn.style.display = 'none'; // Hide launch button when running
            
            if (managedInfo) {
                // Only show stop button if we manage this worker
                stopBtn.style.display = 'inline-block';
                stopBtn.disabled = false;
                stopBtn.textContent = "Stop";
                stopBtn.style.backgroundColor = "#7c4a4a"; // Red when enabled
            } else {
                // Hide stop button for workers launched outside UI
                stopBtn.style.display = 'none';
            }
        } else {
            // Worker is not running
            launchBtn.style.display = 'inline-block'; // Show launch button
            launchBtn.disabled = false;
            launchBtn.textContent = "Launch";
            launchBtn.style.backgroundColor = "#4a7c4a"; // Green when enabled
            
            stopBtn.style.display = 'none'; // Hide stop button when not running
        }
    }
    
    async viewWorkerLog(workerId) {
        const managedInfo = this.state.getWorker(workerId).managed;
        if (!managedInfo?.log_file) return;
        
        const logBtn = document.getElementById(`log-${workerId}`);
        
        // Provide immediate feedback
        if (logBtn) {
            logBtn.disabled = true;
            logBtn.textContent = "Loading...";
            logBtn.style.backgroundColor = "#666";
        }
        
        try {
            // Fetch log content
            const data = await this.api.getWorkerLog(workerId, 1000);
            
            // Create modal dialog
            this.showLogModal(workerId, data);
            
            // Restore button
            if (logBtn) {
                logBtn.disabled = false;
                logBtn.textContent = "View Log";
                logBtn.style.backgroundColor = "#685434"; // Keep the yellow color
            }
            
        } catch (error) {
            this.log('Error viewing log: ' + error.message);
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Error",
                detail: `Failed to load log: ${error.message}`,
                life: 5000
            });
            
            // Flash error and restore button
            if (logBtn) {
                logBtn.style.backgroundColor = DistributedExtension.BUTTON_STYLES.error;
                logBtn.textContent = "Error";
                setTimeout(() => {
                    logBtn.disabled = false;
                    logBtn.textContent = "View Log";
                    logBtn.style.backgroundColor = "#685434"; // Keep the yellow color
                }, 2000);
            }
        }
    }
    
    showLogModal(workerId, logData) {
        // Remove any existing modal
        const existingModal = document.getElementById('distributed-log-modal');
        if (existingModal) {
            existingModal.remove();
        }
        
        const worker = this.config.workers.find(w => w.id === workerId);
        const workerName = worker?.name || `Worker ${workerId}`;
        
        // Create modal container
        const modal = document.createElement('div');
        modal.id = 'distributed-log-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;
        
        // Create modal content
        const content = document.createElement('div');
        content.style.cssText = `
            background: #1e1e1e;
            border-radius: 8px;
            width: 90%;
            max-width: 1200px;
            height: 80%;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        `;
        
        // Header
        const header = document.createElement('div');
        header.style.cssText = `
            padding: 15px 20px;
            border-bottom: 1px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;
        
        const title = document.createElement('h3');
        title.textContent = `${workerName} - Log Viewer`;
        title.style.cssText = 'margin: 0; color: #fff;';
        
        const headerButtons = document.createElement('div');
        headerButtons.style.cssText = 'display: flex; gap: 20px; align-items: center;';
        
        // Auto-refresh container
        const refreshContainer = document.createElement('div');
        refreshContainer.style.cssText = 'display: flex; align-items: center; gap: 4px;';
        
        // Auto-refresh checkbox
        const refreshCheckbox = document.createElement('input');
        refreshCheckbox.type = 'checkbox';
        refreshCheckbox.id = 'log-auto-refresh';
        refreshCheckbox.checked = true; // Enabled by default
        refreshCheckbox.style.cssText = 'cursor: pointer;';
        refreshCheckbox.onchange = (e) => {
            if (e.target.checked) {
                this.startLogAutoRefresh(workerId);
            } else {
                this.stopLogAutoRefresh();
            }
        };
        
        const refreshLabel = document.createElement('label');
        refreshLabel.htmlFor = 'log-auto-refresh';
        refreshLabel.style.cssText = 'font-size: 12px; color: #ccc; cursor: pointer; white-space: nowrap;';
        refreshLabel.textContent = 'Auto-refresh';
        
        // Add checkbox and label to container
        refreshContainer.appendChild(refreshCheckbox);
        refreshContainer.appendChild(refreshLabel);
        
        // Close button
        const closeBtn = this._createButton('✕', 
            () => {
                this.stopLogAutoRefresh();
                modal.remove();
            }, 
            'background-color: #c04c4c;');
        closeBtn.style.cssText += ' padding: 5px 10px; font-size: 14px; font-weight: bold;';
        
        headerButtons.appendChild(refreshContainer);
        headerButtons.appendChild(closeBtn);
        
        header.appendChild(title);
        header.appendChild(headerButtons);
        
        // Log content area
        const logContainer = document.createElement('div');
        logContainer.style.cssText = `
            flex: 1;
            overflow: auto;
            padding: 15px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            color: #ddd;
            background: #0d0d0d;
            white-space: pre-wrap;
            word-wrap: break-word;
        `;
        logContainer.id = 'distributed-log-content';
        logContainer.textContent = logData.content;
        
        // Auto-scroll to bottom
        setTimeout(() => {
            logContainer.scrollTop = logContainer.scrollHeight;
        }, 0);
        
        // Status bar
        const statusBar = document.createElement('div');
        statusBar.style.cssText = `
            padding: 10px 20px;
            border-top: 1px solid #444;
            font-size: 11px;
            color: #888;
        `;
        statusBar.textContent = `Log file: ${logData.log_file}`;
        if (logData.truncated) {
            statusBar.textContent += ` (showing last ${logData.lines_shown} lines of ${this.formatFileSize(logData.file_size)})`;
        }
        
        // Assemble modal
        content.appendChild(header);
        content.appendChild(logContainer);
        content.appendChild(statusBar);
        modal.appendChild(content);
        
        // Close on background click
        modal.onclick = (e) => {
            if (e.target === modal) {
                this.stopLogAutoRefresh();
                modal.remove();
            }
        };
        
        // Close on Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                this.stopLogAutoRefresh();
                modal.remove();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
        
        document.body.appendChild(modal);
        
        // Start auto-refresh
        this.startLogAutoRefresh(workerId);
    }
    
    async refreshLog(workerId, silent = false) {
        const logContent = document.getElementById('distributed-log-content');
        if (!logContent) return;
        
        try {
            const data = await this.api.getWorkerLog(workerId, 1000);
            
            // Update content
            const shouldAutoScroll = logContent.scrollTop + logContent.clientHeight >= logContent.scrollHeight - 50;
            logContent.textContent = data.content;
            
            // Auto-scroll if was at bottom
            if (shouldAutoScroll) {
                logContent.scrollTop = logContent.scrollHeight;
            }
            
            // Only show toast if not in silent mode (manual refresh)
            if (!silent) {
                app.extensionManager.toast.add({
                    severity: "success",
                    summary: "Log Refreshed",
                    detail: "Log content updated",
                    life: 2000
                });
            }
            
        } catch (error) {
            // Only show error toast if not in silent mode
            if (!silent) {
                app.extensionManager.toast.add({
                    severity: "error",
                    summary: "Refresh Failed",
                    detail: error.message,
                    life: 3000
                });
            }
        }
    }
    
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
    
    isRemoteWorker(worker) {
        const host = worker.host || window.location.hostname;
        return host !== "localhost" && host !== "127.0.0.1" && host !== window.location.hostname;
    }
    
    getMasterUrl() {
        // Always use the detected/configured master IP for consistency
        if (this.config?.master?.host) {
            const configuredHost = this.config.master.host;
            
            // If the configured host already includes protocol, use as-is
            if (configuredHost.startsWith('http://') || configuredHost.startsWith('https://')) {
                return configuredHost;
            }
            
            // For domain names (not IPs), default to HTTPS
            const isIP = /^(\d{1,3}\.){3}\d{1,3}$/.test(configuredHost);
            const isLocalhost = configuredHost === 'localhost' || configuredHost === '127.0.0.1';
            
            if (!isIP && !isLocalhost && configuredHost.includes('.')) {
                // It's a domain name, use HTTPS
                return `https://${configuredHost}`;
            } else {
                // For IPs and localhost, use current access method
                const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
                if ((window.location.protocol === 'https:' && port === '443') || 
                    (window.location.protocol === 'http:' && port === '80')) {
                    return `${window.location.protocol}//${configuredHost}`;
                }
                return `${window.location.protocol}//${configuredHost}:${port}`;
            }
        }
        
        // If no master IP is set but we're on a network address, use it
        const hostname = window.location.hostname;
        if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
            return window.location.origin;
        }
        
        // Fallback warning - this won't work for remote workers
        this.debugLog("No master host configured - remote workers won't be able to connect. " +
                     "Master host should be auto-detected on startup.");
        return window.location.origin;
    }
    
    async detectMasterIP() {
        try {
            const data = await this.api.getNetworkInfo();
            this.log("Network info: " + JSON.stringify(data));
            
            // Store CUDA device info
            if (data.cuda_device !== null && data.cuda_device !== undefined) {
                this.masterCudaDevice = data.cuda_device;
                // Update the master display with CUDA info
                this.updateMasterDisplay();
            }
            
            // Store CUDA device count for auto-population
            if (data.cuda_device_count > 0) {
                this.cudaDeviceCount = data.cuda_device_count;
                this.log(`Detected ${this.cudaDeviceCount} CUDA devices`);
                
                // Auto-populate workers if none exist
                this.log(`Current workers: ${this.config.workers ? this.config.workers.length : 'null'}`);
                if (!this.config.workers || this.config.workers.length === 0) {
                    this.log(`Auto-populating ${this.cudaDeviceCount} workers based on CUDA devices`);
                    
                    const newWorkers = [];
                    for (let i = 0; i < this.cudaDeviceCount; i++) {
                        const worker = {
                            id: Date.now() + i,
                            name: `Worker ${i + 1}`,
                            host: "localhost",
                            port: 8189 + i,
                            cuda_device: i,
                            enabled: true
                        };
                        newWorkers.push(worker);
                    }
                    
                    // Update config and save
                    this.config.workers = newWorkers;
                    
                    // Refresh the panel if it's already open
                    const panel = document.getElementById("distributed-panel");
                    if (panel) {
                        this.renderSidebarContent(panel);
                    }
                }
            }
            
            // Check if we already have a master host configured
            if (this.config?.master?.host) {
                this.debugLog(`Master host already configured: ${this.config.master.host}`);
                return;
            }
            
            // Use the recommended IP from the backend
            if (data.recommended_ip && data.recommended_ip !== '127.0.0.1') {
                this.log(`Auto-detected master IP: ${data.recommended_ip}`);
                
                // Save the detected IP (pass true to suppress notification)
                await this.saveMasterConfig({ host: data.recommended_ip }, true);
                
                // Show a single combined notification
                if (app.extensionManager?.toast) {
                    app.extensionManager.toast.add({
                        severity: "info",
                        summary: "Master IP Auto-Detected",
                        detail: `Master IP set to ${data.recommended_ip} for remote worker communication`,
                        life: 5000
                    });
                }
            }
        } catch (error) {
            this.debugLog("Error detecting master IP: " + error.message);
        }
    }
    
    async saveMasterConfig(updates, suppressNotification = false) {
        try {
            const response = await fetch(`${window.location.origin}/distributed/config/update_master`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates)
            });
            
            if (response.ok) {
                // Update local config
                if (!this.config.master) {
                    this.config.master = {};
                }
                // Update all provided fields
                Object.assign(this.config.master, updates);
                
                // Only show notification if not suppressed
                if (!suppressNotification) {
                    app.extensionManager.toast.add({
                        severity: "success",
                        summary: "Master Host Saved",
                        detail: masterIp.trim() ? `Master host set to ${masterIp.trim()}` : "Master host cleared (auto-detect)",
                        life: 3000
                    });
                }
            } else {
                throw new Error("Failed to save master host");
            }
        } catch (error) {
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Save Failed",
                detail: error.message,
                life: 5000
            });
        }
    }
    
    startLogAutoRefresh(workerId) {
        // Stop any existing auto-refresh
        this.stopLogAutoRefresh();
        
        // Refresh every 2 seconds
        this.logAutoRefreshInterval = setInterval(() => {
            this.refreshLog(workerId, true); // silent mode
        }, 2000);
    }
    
    stopLogAutoRefresh() {
        if (this.logAutoRefreshInterval) {
            clearInterval(this.logAutoRefreshInterval);
            this.logAutoRefreshInterval = null;
        }
    }
    
    toggleWorkerExpanded(workerId) {
        const settingsDiv = document.getElementById(`settings-${workerId}`);
        const gpuDiv = settingsDiv.closest('[style*="margin-bottom: 12px"]');
        const settingsArrow = gpuDiv.querySelector('.settings-arrow');
        
        if (!settingsDiv) return;
        
        if (this.state.isWorkerExpanded(workerId)) {
            this.state.setWorkerExpanded(workerId, false);
            settingsDiv.classList.remove("expanded");
            if (settingsArrow) {
                settingsArrow.style.transform = "rotate(0deg)";
            }
            // Animate padding to 0
            settingsDiv.style.padding = "0 12px";
            settingsDiv.style.marginTop = "0";
            settingsDiv.style.marginBottom = "0";
        } else {
            this.state.setWorkerExpanded(workerId, true);
            settingsDiv.classList.add("expanded");
            if (settingsArrow) {
                settingsArrow.style.transform = "rotate(90deg)";
            }
            // Animate padding to full
            settingsDiv.style.padding = "12px";
            settingsDiv.style.marginTop = "8px";
            settingsDiv.style.marginBottom = "8px";
        }
    }
    
    createWorkerSettingsForm(worker) {
        const form = document.createElement("div");
        form.style.cssText = "display: flex; flex-direction: column; gap: 10px;";
        
        // Name field
        const nameGroup = this.createFormGroup("Name:", worker.name, `name-${worker.id}`);
        form.appendChild(nameGroup);
        
        // Remote checkbox
        const remoteGroup = document.createElement("div");
        remoteGroup.style.cssText = "display: flex; align-items: center; gap: 8px; margin: 5px 0;";
        
        const remoteCheckbox = document.createElement("input");
        remoteCheckbox.type = "checkbox";
        remoteCheckbox.id = `remote-${worker.id}`;
        remoteCheckbox.checked = this.isRemoteWorker(worker);
        remoteCheckbox.onchange = (e) => {
            const isRemote = e.target.checked;
            // Show/hide relevant fields
            const hostGroup = document.getElementById(`host-group-${worker.id}`);
            const cudaGroup = document.getElementById(`cuda-group-${worker.id}`);
            const argsGroup = document.getElementById(`args-group-${worker.id}`);
            
            if (isRemote) {
                hostGroup.style.display = "flex";
                cudaGroup.style.display = "none";
                argsGroup.style.display = "none";
                // If switching to remote and host is localhost, clear it
                const hostInput = document.getElementById(`host-${worker.id}`);
                if (hostInput.value === "localhost" || hostInput.value === "127.0.0.1") {
                    hostInput.value = "";
                }
            } else {
                hostGroup.style.display = "none";
                cudaGroup.style.display = "flex";
                argsGroup.style.display = "flex";
            }
        };
        
        const remoteLabel = document.createElement("label");
        remoteLabel.htmlFor = `remote-${worker.id}`;
        remoteLabel.textContent = "Remote Worker";
        remoteLabel.style.cssText = "font-size: 12px; color: #ccc; cursor: pointer;";
        
        remoteGroup.appendChild(remoteCheckbox);
        remoteGroup.appendChild(remoteLabel);
        form.appendChild(remoteGroup);
        
        // Host field (only for remote workers)
        const hostGroup = this.createFormGroup("Host:", worker.host || "", `host-${worker.id}`, "text", "e.g., 192.168.1.100");
        hostGroup.id = `host-group-${worker.id}`;
        hostGroup.style.display = this.isRemoteWorker(worker) ? "flex" : "none";
        form.appendChild(hostGroup);
        
        // Port field
        const portGroup = this.createFormGroup("Port:", worker.port, `port-${worker.id}`, "number");
        form.appendChild(portGroup);
        
        // CUDA Device field (only for local workers)
        const cudaGroup = this.createFormGroup("CUDA Device:", worker.cuda_device || 0, `cuda-${worker.id}`, "number");
        cudaGroup.id = `cuda-group-${worker.id}`;
        cudaGroup.style.display = this.isRemoteWorker(worker) ? "none" : "flex";
        form.appendChild(cudaGroup);
        
        // Extra Args field (only for local workers)
        const argsGroup = this.createFormGroup("Extra Args:", worker.extra_args || "", `args-${worker.id}`);
        argsGroup.id = `args-group-${worker.id}`;
        argsGroup.style.display = this.isRemoteWorker(worker) ? "none" : "flex";
        form.appendChild(argsGroup);
        
        // Buttons
        const buttonGroup = document.createElement("div");
        buttonGroup.style.cssText = "display: flex; gap: 8px; margin-top: 8px;";
        
        const saveBtn = this._createButton("Save", 
            () => this.saveWorkerSettings(worker.id),
            "background-color: #4a7c4a;");
        saveBtn.style.cssText += " padding: 6px 12px; font-size: 12px;";
        
        const cancelBtn = this._createButton("Cancel", 
            () => this.cancelWorkerSettings(worker.id),
            "background-color: #555;");
        cancelBtn.style.cssText += " padding: 6px 12px; font-size: 12px;";
        
        const deleteBtn = this._createButton("Delete", 
            () => this.deleteWorker(worker.id),
            "background-color: #7c4a4a;");
        deleteBtn.style.cssText += " padding: 6px 12px; font-size: 12px; margin-left: auto;";
        
        buttonGroup.appendChild(saveBtn);
        buttonGroup.appendChild(cancelBtn);
        buttonGroup.appendChild(deleteBtn);
        
        form.appendChild(buttonGroup);
        
        return form;
    }
    
    createSettingsToggle() {
        const settingsRow = document.createElement("div");
        settingsRow.style.cssText = "display: flex; align-items: center; gap: 6px; padding: 4px 0; cursor: pointer; user-select: none;";
        
        const settingsTitle = document.createElement("h4");
        settingsTitle.textContent = "Settings";
        settingsTitle.style.cssText = "margin: 0; font-size: 14px;";
        
        const settingsToggle = document.createElement("span");
        settingsToggle.textContent = "▶"; // Right arrow when collapsed
        settingsToggle.style.cssText = "font-size: 12px; color: #888; transition: all 0.2s ease;";
        
        settingsRow.appendChild(settingsToggle);
        settingsRow.appendChild(settingsTitle);
        
        return { settingsRow, settingsToggle };
    }
    
    createFormGroup(label, value, id, type = "text", placeholder = "") {
        const result = this.ui.createFormGroup(label, value, id, type, placeholder);
        return result.group;
    }
    
    async saveWorkerSettings(workerId) {
        const worker = this.config.workers.find(w => w.id === workerId);
        if (!worker) return;
        
        // Get form values
        const name = document.getElementById(`name-${workerId}`).value;
        const isRemote = document.getElementById(`remote-${workerId}`).checked;
        const host = isRemote ? document.getElementById(`host-${workerId}`).value : window.location.hostname;
        const port = parseInt(document.getElementById(`port-${workerId}`).value);
        const cudaDevice = isRemote ? undefined : parseInt(document.getElementById(`cuda-${workerId}`).value);
        const extraArgs = isRemote ? undefined : document.getElementById(`args-${workerId}`).value;
        
        // Validate
        if (!name.trim()) {
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Validation Error",
                detail: "Worker name is required",
                life: 3000
            });
            return;
        }
        
        if (isRemote && !host.trim()) {
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Validation Error",
                detail: "Host is required for remote workers",
                life: 3000
            });
            return;
        }
        
        if (isNaN(port) || port < 1 || port > 65535) {
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Validation Error",
                detail: "Port must be between 1 and 65535",
                life: 3000
            });
            return;
        }
        
        // Check for port conflicts
        // Remote workers can reuse ports, but local workers cannot share ports with each other or master
        if (!isRemote) {
            // Check if port conflicts with master
            const masterPort = parseInt(window.location.port) || (window.location.protocol === 'https:' ? 443 : 80);
            if (port === masterPort) {
                app.extensionManager.toast.add({
                    severity: "error",
                    summary: "Port Conflict",
                    detail: `Port ${port} is already in use by the master server`,
                    life: 3000
                });
                return;
            }
            
            // Check if port conflicts with other local workers
            const localPortConflict = this.config.workers.some(w => 
                w.id !== workerId && 
                w.port === port && 
                !w.host // local workers have no host or host is null
            );
            
            if (localPortConflict) {
                app.extensionManager.toast.add({
                    severity: "error",
                    summary: "Port Conflict",
                    detail: `Port ${port} is already in use by another local worker`,
                    life: 3000
                });
                return;
            }
        } else {
            // For remote workers, only check conflicts with other workers on the same host
            const sameHostConflict = this.config.workers.some(w => 
                w.id !== workerId && 
                w.port === port && 
                w.host === host.trim()
            );
            
            if (sameHostConflict) {
                app.extensionManager.toast.add({
                    severity: "error",
                    summary: "Port Conflict",
                    detail: `Port ${port} is already in use by another worker on ${host}`,
                    life: 3000
                });
                return;
            }
        }
        
        try {
            const response = await fetch(`${window.location.origin}/distributed/config/update_worker`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    worker_id: workerId,
                    name: name.trim(),
                    host: isRemote ? host.trim() : null,
                    port: port,
                    cuda_device: isRemote ? null : cudaDevice,
                    extra_args: isRemote ? null : (extraArgs ? extraArgs.trim() : "")
                })
            });
            
            if (response.ok) {
                // Update local config
                worker.name = name.trim();
                if (isRemote) {
                    worker.host = host.trim();
                    delete worker.cuda_device;
                    delete worker.extra_args;
                } else {
                    delete worker.host;
                    worker.cuda_device = cudaDevice;
                    worker.extra_args = extraArgs ? extraArgs.trim() : "";
                }
                worker.port = port;
                
                app.extensionManager.toast.add({
                    severity: "success",
                    summary: "Settings Saved",
                    detail: `Worker ${name} settings updated`,
                    life: 3000
                });
                
                // Refresh the UI
                if (this.panelElement) {
                    this.renderSidebarContent(this.panelElement);
                }
            } else {
                const error = await response.json();
                throw new Error(error.message || "Failed to save settings");
            }
        } catch (error) {
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Save Failed",
                detail: error.message,
                life: 5000
            });
        }
    }
    
    cancelWorkerSettings(workerId) {
        // Collapse the settings panel
        this.toggleWorkerExpanded(workerId);
        
        // Reset form values to original
        const worker = this.config.workers.find(w => w.id === workerId);
        if (worker) {
            document.getElementById(`name-${workerId}`).value = worker.name;
            document.getElementById(`host-${workerId}`).value = worker.host || "";
            document.getElementById(`port-${workerId}`).value = worker.port;
            document.getElementById(`cuda-${workerId}`).value = worker.cuda_device || 0;
            document.getElementById(`args-${workerId}`).value = worker.extra_args || "";
            
            // Reset remote checkbox
            const remoteCheckbox = document.getElementById(`remote-${workerId}`);
            if (remoteCheckbox) {
                remoteCheckbox.checked = this.isRemoteWorker(worker);
            }
        }
    }
    
    async deleteWorker(workerId) {
        const worker = this.config.workers.find(w => w.id === workerId);
        if (!worker) return;
        
        // Confirm deletion
        if (!confirm(`Are you sure you want to delete worker "${worker.name}"?`)) {
            return;
        }
        
        try {
            const response = await fetch(`${window.location.origin}/distributed/config/delete_worker`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ worker_id: workerId })
            });
            
            if (response.ok) {
                // Remove from local config
                const index = this.config.workers.findIndex(w => w.id === workerId);
                if (index !== -1) {
                    this.config.workers.splice(index, 1);
                }
                
                app.extensionManager.toast.add({
                    severity: "success",
                    summary: "Worker Deleted",
                    detail: `Worker ${worker.name} has been removed`,
                    life: 3000
                });
                
                // Refresh the UI
                if (this.panelElement) {
                    this.renderSidebarContent(this.panelElement);
                }
            } else {
                const error = await response.json();
                throw new Error(error.message || "Failed to delete worker");
            }
        } catch (error) {
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Delete Failed",
                detail: error.message,
                life: 5000
            });
        }
    }
    
    // Backward compatibility wrapper
    async saveMasterIp(masterIp, suppressNotification = false) {
        return this.saveMasterConfig({ host: masterIp.trim() }, suppressNotification);
    }
    
    async addNewWorker() {
        // Generate new worker ID
        const newId = `worker_${Date.now()}`;
        
        // Find next available port
        const usedPorts = this.config.workers.map(w => w.port);
        let nextPort = 8189;
        while (usedPorts.includes(nextPort)) {
            nextPort++;
        }
        
        // Create new worker object
        const newWorker = {
            id: newId,
            name: `Worker ${this.config.workers.length + 1}`,
            port: nextPort,
            cuda_device: this.config.workers.length,
            enabled: true,  // Default to enabled for convenience
            extra_args: ""
        };
        
        // Add to config
        this.config.workers.push(newWorker);
        
        // Save immediately
        try {
            const response = await fetch(`${window.location.origin}/distributed/config/update_worker`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    worker_id: newId,
                    name: newWorker.name,
                    port: newWorker.port,
                    cuda_device: newWorker.cuda_device,
                    extra_args: newWorker.extra_args,
                    enabled: newWorker.enabled
                })
            });
            
            if (!response.ok) {
                // Remove from local config if save failed
                this.config.workers.pop();
                throw new Error("Failed to save new worker");
            }
            
            app.extensionManager.toast.add({
                severity: "success",
                summary: "Worker Added",
                detail: `New worker created on port ${nextPort}`,
                life: 3000
            });
            
            // Refresh UI and expand the new worker
            this.state.setWorkerExpanded(newId, true);
            if (this.panelElement) {
                this.renderSidebarContent(this.panelElement);
            }
            
        } catch (error) {
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Failed to Add Worker",
                detail: error.message,
                life: 5000
            });
        }
    }
}

app.registerExtension({
    name: "Distributed.Panel",
    async setup() {
        new DistributedExtension();
    }
});