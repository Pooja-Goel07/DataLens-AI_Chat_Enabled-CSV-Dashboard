class DataLensApp {
    constructor() {
        this.activeTable = null
        this.chatMessages = []
        this.tables = []
        this.baseURL = "http://localhost:8000"
        this.historySidebarOpen = false
        this.isLoggedIn = false  // ADD THIS
        this.currentUser = null  // ADD THIS
        this.initializeEventListeners()
        this.loadInitialData()
        this.checkLoginStatus()  // ADD THIS
    }

    // ADD THIS METHOD
    checkLoginStatus() {
        const token = localStorage.getItem("token")
        const username = localStorage.getItem("username")
        if (token && username) {
            this.isLoggedIn = true
            this.currentUser = username
            this.updateHeaderButtons()
        }
    }

    // ADD THIS METHOD
    updateHeaderButtons() {
        const loginBtn = document.getElementById("loginBtn")
        const signupBtn = document.getElementById("signupBtn")
        
        if (this.isLoggedIn) {
            // Hide signup button
            signupBtn.style.display = "none"
            
            // Replace login button with username
            loginBtn.innerHTML = `
                <i class="fas fa-user-check"></i>
                ${this.currentUser}
            `
            loginBtn.onclick = null // Remove click handler
        } else {
            // Show both buttons
            signupBtn.style.display = "flex"
            loginBtn.innerHTML = `
                <i class="fas fa-user"></i>
                Login
            `
        }
    }


    initializeEventListeners() {
        // History button
        document.getElementById("historyBtn").addEventListener("click", () => {
            this.toggleHistorySidebar()
        })

        // Close sidebar
        document.getElementById("closeSidebar").addEventListener("click", () => {
            this.toggleHistorySidebar()
        })

        // Upload button - now resets chat
        document.getElementById("uploadBtn").addEventListener("click", () => {
            this.resetChat()
            this.showUploadModal()
        })

        // Upload form submission
        document.getElementById("uploadForm").addEventListener("submit", (e) => {
            e.preventDefault()
            const fileInput = document.getElementById("fileInput")
            const tableName = document.getElementById("tableName").value.trim()

            if (fileInput.files.length > 0 && tableName) {
                this.handleFileUpload(fileInput.files[0], tableName)
            }
        })

        // File input change to enable submit button
        document.getElementById("fileInput").addEventListener("change", (e) => {
            this.updateUploadButton()
        })

        document.getElementById("tableName").addEventListener("input", () => {
            this.updateUploadButton()
        })

        // Login button
        document.getElementById("loginBtn").addEventListener("click", () => {
            this.showLoginModal()
        })
        // Move this earlier in initializeEventListeners(), right after login button
        document.getElementById("signupBtn").addEventListener("click", () => {
            this.showSignupModal()
        })

        // Chat input
        document.getElementById("chatInput").addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                this.sendMessage()
            }
        })

        document.getElementById("sendBtn").addEventListener("click", () => {
            this.sendMessage()
        })

        // Modal close buttons
        document.getElementById("closeUploadModal").addEventListener("click", () => {
            this.hideUploadModal()
        })

        document.getElementById("closeLoginModal").addEventListener("click", () => {
            this.hideLoginModal()
        })

        // File upload
        const fileInput = document.getElementById("fileInput")
        const uploadArea = document.getElementById("uploadArea")
        const browseBtn = uploadArea.querySelector(".browse-btn")

        browseBtn.addEventListener("click", () => {
            fileInput.click()
        })

        // Drag and drop
        uploadArea.addEventListener("dragover", (e) => {
            e.preventDefault()
            uploadArea.classList.add("dragover")
        })

        uploadArea.addEventListener("dragleave", () => {
            uploadArea.classList.remove("dragover")
        })

        uploadArea.addEventListener("drop", (e) => {
            e.preventDefault()
            uploadArea.classList.remove("dragover")
            if (e.dataTransfer.files.length > 0) {
                //this.handleFileUpload(e.dataTransfer.files[0]) // Modified to use form
            }
        })

        // Login form
        document.getElementById("loginForm").addEventListener("submit", (e) => {
            e.preventDefault()
            this.handleLogin()
        })

        // Modal background click
        document.getElementById("uploadModal").addEventListener("click", (e) => {
            if (e.target.id === "uploadModal") {
                this.hideUploadModal()
            }
        })

        document.getElementById("loginModal").addEventListener("click", (e) => {
            if (e.target.id === "loginModal") {
                this.hideLoginModal()
            }
        })

        // Sidebar toggle for mobile
        document.getElementById("sidebarToggle").addEventListener("click", () => {
            document.getElementById("sidebar").classList.toggle("collapsed")
        })

        // Show sign up modal
        document.getElementById("signupBtn").addEventListener("click", () => {
            document.getElementById("signupModal").classList.add("active")
        })
        // Fix signup modal close button
        document.getElementById("closeSignupModal").addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.hideSignupModal();
        });
        document.getElementById("signupForm").addEventListener("submit", (e) => {
            e.preventDefault()
            // Call your handleSignup() here
            this.handleSignup()
        })
    }

    updateUploadButton() {
        const fileInput = document.getElementById("fileInput")
        const tableName = document.getElementById("tableName").value.trim()
        const submitBtn = document.querySelector(".submit-upload-btn")

        submitBtn.disabled = !(fileInput.files.length > 0 && tableName)
    }

    toggleHistorySidebar() {
        const sidebar = document.getElementById("historySidebar")
        this.historySidebarOpen = !this.historySidebarOpen

        if (this.historySidebarOpen) {
            sidebar.classList.add("active")
            this.loadTables() // Refresh tables when opening
        } else {
            sidebar.classList.remove("active")
        }
    }

    renderInitialReportPanel() {
        const reportContent = document.getElementById("reportContent")
        reportContent.innerHTML = `
      <div class="empty-state">
        <h3>No data uploaded yet</h3>
        <p>Upload a CSV file to get started.</p>
        <button class="upload-btn" id="inlineUploadBtn">
          <i class="fas fa-plus"></i> Upload CSV
        </button>
      </div>
    `
        document.getElementById("inlineUploadBtn").addEventListener("click", () => {
            this.showUploadModal()
        })
    }

    // MODIFY loadInitialData method
    async loadInitialData() {
        await this.loadTables()
        // Always show upload form instead of loading state
        this.renderUploadForm()
        this.resetChat()
        // Hide activeTableName and lastUpdated until a table is selected
        document.getElementById('activeTableName').textContent = ''
        document.getElementById('lastUpdated').style.display = 'none'
    }

    // 1. Load tables
    async loadTables() {
        try {
            const response = await fetch(`${this.baseURL}/api/list_tables/`)
            if (response.ok) {
                const data = await response.json()
                this.tables = data.tables
                this.activeTable = data.active_table
                this.renderTablesList()
            }
        } catch (error) {
            console.error("Error loading tables:", error)
            this.renderTablesList()
        }
    }

    renderTablesList() {
        const tableList = document.getElementById("tableList")
        tableList.innerHTML = ""

        this.tables.forEach((tableName) => {
            const tableItem = document.createElement("div")
            tableItem.className = `table-item ${tableName === this.activeTable ? "active" : ""}`
            tableItem.dataset.table = tableName

            tableItem.innerHTML = `
                <i class="fas fa-table"></i>
                <span>${tableName}</span>
                <div class="active-indicator"></div>
            `

            tableItem.addEventListener("click", () => {
                this.switchTable(tableName)
            })

            tableList.appendChild(tableItem)
        })
    }

    // 2. Switch table (send as form data, not JSON)
    async switchTable(tableName) {
        if (tableName === this.activeTable) return;
        this.activeTable = tableName;
        document.querySelectorAll(".table-item").forEach((item) => {
            item.classList.remove("active");
        });
        document.querySelector(`[data-table="${tableName}"]`).classList.add("active");
        this.setActiveTableName(tableName);
        await this.loadReport(tableName);
        try {
            const formData = new FormData();
            formData.append("table_name", tableName);
            await fetch(`${this.baseURL}/api/switch_table/`, {
                method: "POST",
                body: formData,
            });
        } catch (error) {
            console.error("Error setting active table:", error);
        }
    }

    // 3. Add logic to prevent persistent 'Generating report...' state
    async loadReport(tableName) {
        const reportContent = document.getElementById("reportContent");
        if (!tableName) {
            // No table selected, show empty state or upload prompt
            this.renderUploadForm();
            return;
        }
        reportContent.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Generating report...</p></div>';
        try {
            const response = await fetch(`${this.baseURL}/api/generate_report/?table_name=${encodeURIComponent(tableName)}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
            });
            const reportData = await response.json();
            console.log("Report data from backend:", reportData);
            if (response.ok) {
                this.renderReport(reportData);
            } else {
                throw new Error("Failed to load report");
            }
        } catch (error) {
            console.error("Error loading report:", error);
            this.renderMockReport();
        }
        document.getElementById("lastUpdated").textContent = "Updated just now";
    }

    renderReport(data) {
        const reportContent = document.getElementById("reportContent")

        // Defensive: If columns is missing or not an array, show error
        if (!Array.isArray(data.columns)) {
            reportContent.innerHTML = "<div class='error-state'>Error: No columns found in report data.</div>"
            return
        }

        const html = `
        <div class="data-stats">
            <div class="stat-card">
                <h4>Total Rows</h4>
                <div class="stat-value">${data.total_rows}</div>
            </div>
            <div class="stat-card">
                <h4>Total Columns</h4>
                <div class="stat-value">${data.total_columns}</div>
            </div>
            <div class="stat-card">
                <h4>Table Name</h4>
                <div class="stat-value">${data.table_name}</div>
            </div>
            <div class="stat-card">
                <h4>Generated</h4>
                <div class="stat-value">${new Date(data.generated_at).toLocaleDateString()}</div>
            </div>
        </div>

        <div class="report-summary">
            ${this.markdownToHtml(data.summary)}
        </div>

        <div class="columns-section">
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">Dataset Columns</h3>
            <div class="columns-grid">
                ${data.columns
                .map(
                    (column, index) => `
                    <div class="column-item" style="background: ${this.getColumnColor(index)};">
                        <i class="fas fa-columns"></i>
                        <span>${column}</span>
                    </div>
                `,
                )
                .join("")}
            </div>
        </div>

        ${data.outliers && Object.keys(data.outliers).length > 0
                ? `
            <div class="outliers-section">
                <h3 style="color: var(--danger-color); margin-bottom: 1rem;">Outliers Detected</h3>
                ${Object.entries(data.outliers)
                    .map(
                        ([column, outlierData]) => `
                    <div class="outlier-card">
                        <h4>${column}</h4>
                        <div class="outlier-stats">
                            <div class="outlier-stat">
                                <span class="label">Count:</span>
                                <span class="value">${outlierData.count} (${outlierData.percentage}%)</span>
                            </div>
                            <div class="outlier-stat">
                                <span class="label">Values:</span>
                                <span class="value">${outlierData.values.join(", ")}</span>
                            </div>
                            <div class="outlier-stat">
                                <span class="label">Range:</span>
                                <span class="value">${outlierData.bounds.lower.toFixed(2)} - ${outlierData.bounds.upper.toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                `,
                    )
                    .join("")}
            </div>
        `
                : ""
            }

        ${data.graph_urls && data.graph_urls.length > 0
                ? `
            <div class="graphs-section">
                <h3 style="color: var(--success-color); margin-bottom: 1rem;">Data Visualizations</h3>
                <div class="graphs-grid">
                    ${data.graph_urls
                    .map(
                        (url, index) => `
                        <div class="graph-item" style="border-left: 4px solid ${this.getGraphColor(index)};">
                            <img src="${this.baseURL}${url}" alt="Data visualization" loading="lazy">
                        </div>
                    `,
                    )
                    .join("")}
                </div>
            </div>
        `
                : ""
            }
    `

        reportContent.innerHTML = html
    }

    getColumnColor(index) {
        const colors = [
            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
            "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        ]
        return colors[index % colors.length]
    }

    getGraphColor(index) {
        const colors = ["#667eea", "#f093fb", "#4facfe", "#43e97b", "#fa709a", "#a8edea"]
        return colors[index % colors.length]
    }

    renderMockReport() {
        // Mock data for demonstration
        const mockData = {
            table_name: "cars",
            total_rows: 10,
            total_columns: 5,
            generated_at: new Date().toISOString(),
            summary: `## Executive Summary

The dataset contains information on 10 vehicles, encompassing numeric features like odometer reading (KM) and number of doors, and categorical features such as make, color, and price. The data is clean, with no missing values or duplicates, but outlier detection identified potential anomalies in one numeric column.

## Key Findings

**Data quality assessment:** The dataset is of high quality, exhibiting no missing values or duplicates. This facilitates straightforward analysis.

**Notable patterns or insights:** Visualizations revealed potential relationships between odometer reading and vehicle make, suggesting certain makes may have higher average mileage. The distribution of car makes and colors provides a snapshot of the dataset's composition.`,
            outliers: {
                "Odometer (KM)": {
                    count: 1,
                    percentage: 10,
                    values: [213095],
                },
            },
            graph_urls: [
                "/placeholder.svg?height=300&width=400",
                "/placeholder.svg?height=300&width=400",
                "/placeholder.svg?height=300&width=400",
                "/placeholder.svg?height=300&width=400",
            ],
        }

        this.renderReport(mockData)
    }

    markdownToHtml(markdown) {
        // Simple markdown to HTML conversion
        return markdown
            .replace(/## (.*)/g, "<h2>$1</h2>")
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\n\n/g, "</p><p>")
            .replace(/^/, "<p>")
            .replace(/$/, "</p>")
    }

    resetChat() {
        this.chatMessages = []
        const chatMessages = document.getElementById("chatMessages")
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <p>Hello! I'm your AI assistant. I can help you analyze the "${this.activeTable}" table data. What would you like to know?</p>
                </div>
            </div>
        `
    }

    // 7. Chatbot sendMessage uses /api/ask (no trailing slash)
    async sendMessage() {
    const input = document.getElementById("chatInput");
    const message = input.value.trim();
    if (!message) return;
    
    this.addMessageToChat(message, "user");
    input.value = "";
    this.showTypingIndicator();
    
    try {
        const response = await fetch(`${this.baseURL}/api/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message, table_name: this.activeTable }),
        });
        
        if (response.ok) {
            const data = await response.json();
            this.hideTypingIndicator();
            
            // Handle both response formats
            if (data.answer) {
                // Format 2: Simple answer response
                this.addMessageToChat(data.answer, "bot");
            } else if (data.sql_query && data.analysis) {
                // Format 1: SQL query with analysis response
                let botMessage = "";
                
                // // Add SQL query section
                // if (data.sql_query) {
                //     botMessage += "**SQL Query:**\n```sql\n" + data.sql_query + "\n```\n\n";
                // }
                
                // Add results section if available
                if (data.results) {
                    botMessage += "**Results:**\n" + data.results + "\n\n";
                }
                
                // Add analysis section
                if (data.analysis) {
                    botMessage += "**Analysis:**\n" + data.analysis;
                }
                
                // Update active dataset if provided
                if (data.active_dataset) {
                    this.activeTable = data.active_dataset;
                }
                
                this.addMessageToChat(botMessage, "bot");
            } else {
                // Fallback for unexpected response format
                this.addMessageToChat("I received a response but couldn't parse it properly.", "bot");
                console.warn("Unexpected response format:", data);
            }
        } else {
            throw new Error(`HTTP ${response.status}: Failed to get response`);
        }
    } catch (error) {
        this.hideTypingIndicator();
        console.error("Error in sendMessage:", error);
        this.addMessageToChat("Sorry, I encountered an error. Please try again.", "bot");
    }
}
    addMessageToChat(message, sender) {
        const chatMessages = document.getElementById("chatMessages")
        const messageDiv = document.createElement("div")
        messageDiv.className = `message ${sender}-message`

        const avatar = sender === "user" ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>'

        messageDiv.innerHTML = `
            <div class="message-avatar">
                ${avatar}
            </div>
            <div class="message-content">
                <p>${message}</p>
            </div>
        `

        chatMessages.appendChild(messageDiv)
        chatMessages.scrollTop = chatMessages.scrollHeight
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById("chatMessages")
        const typingDiv = document.createElement("div")
        typingDiv.className = "message bot-message typing-indicator"
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <p>Typing...</p>
            </div>
        `
        chatMessages.appendChild(typingDiv)
        chatMessages.scrollTop = chatMessages.scrollHeight
    }

    hideTypingIndicator() {
        const typingIndicator = document.querySelector(".typing-indicator")
        if (typingIndicator) {
            typingIndicator.remove()
        }
    }

    // 4. Upload CSV
    async handleFileUpload(file, tableName) {
        if (!file.name.endsWith(".csv")) {
            this.showError("Please select a CSV file")
            return
        }
        if (!tableName) {
            this.showError("Please enter a table name")
            return
        }

        const uploadProgress = document.getElementById("uploadProgress")
        const uploadForm = document.getElementById("uploadForm")
        const progressFill = document.getElementById("progressFill")
        const progressText = document.getElementById("progressText")

        uploadForm.style.display = "none"
        uploadProgress.style.display = "block"

        try {
            const formData = new FormData()
            formData.append("file", file)
            formData.append("table_name", tableName)

            const response = await fetch(`${this.baseURL}/api/upload_csv/`, {
                method: "POST",
                body: formData,
            })

            if (response.ok) {
                const data = await response.json()
                if (!this.tables.includes(data.table_name)) {
                    this.tables.push(data.table_name)
                    this.renderTablesList()
                }
                await this.switchTable(data.table_name) // after upload
                this.resetChat()
                this.hideUploadModal()
                this.showSuccess("CSV uploaded successfully!")
            }
        } catch (error) {
            console.error("Error uploading file:", error)
            this.showError(error.message || "Failed to upload file. Please try again.")
        } finally {
            uploadForm.style.display = "block"
            uploadProgress.style.display = "none"
            progressFill.style.width = "0%"
            document.getElementById("fileInput").value = ""
            document.getElementById("tableName").value = ""
            this.updateUploadButton()
        }
    }

    // 5. Login (use /api/token for OAuth2)
    // MODIFY handleLogin method
    async handleLogin() {
        const username = document.getElementById("username").value
        const password = document.getElementById("password").value

        try {
            const params = new URLSearchParams()
            params.append("grant_type", "password")
            params.append("username", username)
            params.append("password", password)

            const response = await fetch(`${this.baseURL}/api/token`, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: params,
            })

            if (response.ok) {
                const data = await response.json()
                localStorage.setItem("token", data.access_token)
                localStorage.setItem("username", username)  // ADD THIS
                
                this.isLoggedIn = true  // ADD THIS
                this.currentUser = username  // ADD THIS
                this.updateHeaderButtons()  // ADD THIS
                
                this.hideLoginModal()
                this.showSuccess("Login successful!")
            } else {
                throw new Error("Login failed")
            }
        } catch (error) {
            this.showError("Login failed. Please check your credentials.")
        }
    }
    // ADD THIS METHOD - Replace loading state with upload form
    renderUploadForm() {
        const reportContent = document.getElementById("reportContent")
        reportContent.innerHTML = `
            <div class="upload-form-container">
                <div class="upload-form-header">
                    <div class="upload-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    <h2>Welcome to DataLens</h2>
                    <p>Upload your CSV files and let AI help you discover insights in your data</p>
                </div>
                
                <div class="inline-upload-form">
                    <h3>Get Started</h3>
                    <div class="form-group">
                        <label for="inlineTableName">Table Name</label>
                        <input type="text" id="inlineTableName" placeholder="Enter table name">
                    </div>
                    <div class="inline-upload-area" id="inlineUploadArea">
                        <i class="fas fa-cloud-upload-alt"></i>
                        <p>Click to upload CSV file</p>
                        <button type="button" class="inline-browse-btn">Browse Files</button>
                        <input type="file" id="inlineFileInput" accept=".csv" hidden>
                    </div>
                    <button class="inline-submit-btn" id="inlineSubmitBtn" disabled>Upload & Analyze</button>
                </div>
                
                <div class="upload-features">
                    <div class="feature-item">
                        <i class="fas fa-file-csv"></i>
                        <p>CSV Upload</p>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-chart-bar"></i>
                        <p>Auto Analysis</p>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-brain"></i>
                        <p>AI Insights</p>
                    </div>
                </div>
            </div>
        `

        // Add event listeners for inline upload form
        this.setupInlineUploadListeners()
    }

    // ADD THIS METHOD
    setupInlineUploadListeners() {
        const inlineFileInput = document.getElementById("inlineFileInput")
        const inlineUploadArea = document.getElementById("inlineUploadArea")
        const inlineTableName = document.getElementById("inlineTableName")
        const inlineSubmitBtn = document.getElementById("inlineSubmitBtn")
        const inlineBrowseBtn = inlineUploadArea.querySelector(".inline-browse-btn")

        inlineBrowseBtn.addEventListener("click", () => {
            inlineFileInput.click()
        })

        inlineUploadArea.addEventListener("click", () => {
            inlineFileInput.click()
        })

        inlineFileInput.addEventListener("change", () => {
            this.updateInlineUploadButton()
            if (inlineFileInput.files.length > 0) {
                inlineUploadArea.querySelector("p").textContent = inlineFileInput.files[0].name
            }
        })

        inlineTableName.addEventListener("input", () => {
            this.updateInlineUploadButton()
        })

        inlineSubmitBtn.addEventListener("click", () => {
            const file = inlineFileInput.files[0]
            const tableName = inlineTableName.value.trim()
            if (file && tableName) {
                this.handleFileUpload(file, tableName)
            }
        })
    }

    // ADD THIS METHOD
    updateInlineUploadButton() {
        const inlineFileInput = document.getElementById("inlineFileInput")
        const inlineTableName = document.getElementById("inlineTableName")
        const inlineSubmitBtn = document.getElementById("inlineSubmitBtn")

        if (inlineFileInput && inlineTableName && inlineSubmitBtn) {
            inlineSubmitBtn.disabled = !(inlineFileInput.files.length > 0 && inlineTableName.value.trim())
        }
    }

    // 6. Signup (use /api/signup/)
    async handleSignup() {
        const username = document.getElementById("signupUsername").value;
        const password = document.getElementById("signupPassword").value;

        try {
            const response = await fetch(`${this.baseURL}/api/signup/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password }),
            });

            if (response.ok) {
                localStorage.setItem("username", username);
                this.isLoggedIn = true;
                this.currentUser = username;
                this.updateHeaderButtons();
                this.hideSignupModal();
                this.showSuccess("Sign up successful! You are now logged in.");
            } else {
                const errorData = await response.json();
                if (response.status === 400 && errorData.detail === "Username already registered") {
                    this.showSignupError("User already exists. Please choose a different username.");
                } else {
                    this.showSignupError(errorData.detail || "Sign up failed. Please try again.");
                }
            }
        } catch (error) {
            this.showSignupError("Sign up failed. Please try again.");
        }
    }

    showUploadModal() {
        document.getElementById("uploadModal").classList.add("active")
    }

    hideUploadModal() {
        document.getElementById("uploadModal").classList.remove("active")
        // Reset upload UI
        document.getElementById("uploadArea").style.display = "block"
        document.getElementById("uploadProgress").style.display = "none"
        document.getElementById("fileInput").value = ""
    }

    showLoginModal() {
        document.getElementById("loginModal").classList.add("active")
    }

    hideLoginModal() {
        document.getElementById("loginModal").classList.remove("active")
        document.getElementById("loginForm").reset()
    }

    showSignupModal() {
        document.getElementById("signupModal").classList.add("active")
    }

// ADD THIS METHOD
    showSignupError(message) {
        // Remove existing error message
        const existingError = document.querySelector(".signup-error")
        if (existingError) {
            existingError.remove()
        }

        // Add error message to signup modal
        const modalBody = document.querySelector("#signupModal .modal-body")
        const errorDiv = document.createElement("div")
        errorDiv.className = "signup-error"
        errorDiv.style.cssText = `
            background: #fee2e2;
            color: #dc2626;
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            font-size: 0.875rem;
            border: 1px solid #fecaca;
        `
        errorDiv.textContent = message
        modalBody.insertBefore(errorDiv, modalBody.firstChild)
    }

    hideSignupModal() {
        document.getElementById("signupModal").classList.remove("active")
        const signupForm = document.getElementById("signupForm")
        if (signupForm) signupForm.reset()
        
        // Remove error message
        const existingError = document.querySelector(".signup-error")
        if (existingError) {
            existingError.remove()
        }
    }

    showError(message) {
        // Simple error notification - you can enhance this
        alert("Error: " + message)
    }

    showSuccess(message) {
        // Simple success notification - you can enhance this
        alert("Success: " + message)
    }

    // When a table is selected or uploaded, set activeTableName and show lastUpdated
    setActiveTableName(tableName) {
        document.getElementById('activeTableName').textContent = tableName
        document.getElementById('lastUpdated').style.display = ''
    }
}

document.addEventListener("DOMContentLoaded", () => {
    new DataLensApp()
})

const token = localStorage.getItem("token")
const headers = {
    "Content-Type": "application/json",
    ...(token && { "Authorization": `Bearer ${token}` })
}
