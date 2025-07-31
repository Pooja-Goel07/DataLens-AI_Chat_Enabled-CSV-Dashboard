class DataLensApp {
  constructor() {
    this.activeTable = null
    this.chatMessages = []
    this.tables = []
    this.baseURL = "http://localhost:8000"
    this.historySidebarOpen = false
    this.isLoggedIn = false
    this.currentUser = null
    this.authToken = null
    this.initializeEventListeners()
    this.loadInitialData()
    this.checkLoginStatus()
  }

  checkLoginStatus() {
    const token = localStorage.getItem("token")
    const username = localStorage.getItem("username")
    if (token && username) {
      this.isLoggedIn = true
      this.currentUser = username
      this.authToken = token
      this.updateHeaderButtons()
    }
  }

  getAuthHeaders() {
    if (this.authToken) {
      return {
        Authorization: `Bearer ${this.authToken}`,
        "Content-Type": "application/json",
      }
    }
    return { "Content-Type": "application/json" }
  }

  updateHeaderButtons() {
    const loginBtn = document.getElementById("loginBtn")
    const signupBtn = document.getElementById("signupBtn")
    const logoutBtnLeft = document.getElementById("logoutBtnLeft")

    if (this.isLoggedIn) {
      signupBtn.style.display = "none"
      logoutBtnLeft.style.display = "flex"
      loginBtn.innerHTML = `<i class="fas fa-user-check"></i>${this.currentUser}`
      loginBtn.onclick = null
      loginBtn.style.cursor = "default"
    } else {
      signupBtn.style.display = "flex"
      logoutBtnLeft.style.display = "none"
      loginBtn.innerHTML = `<i class="fas fa-user"></i>Login`
      loginBtn.onclick = () => this.showLoginModal()
      loginBtn.style.cursor = "pointer"
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

    // Upload button
    document.getElementById("uploadBtn").addEventListener("click", () => {
      if (!this.isLoggedIn) {
        this.showError("Please login to upload files")
        this.showLoginModal()
        return
      }
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

    // File input change
    document.getElementById("fileInput").addEventListener("change", () => {
      this.updateUploadButton()
    })
    document.getElementById("tableName").addEventListener("input", () => {
      this.updateUploadButton()
    })

    // Login and signup buttons
    document.getElementById("loginBtn").addEventListener("click", () => {
      if (!this.isLoggedIn) {
        this.showLoginModal()
      }
    })

    document.getElementById("signupBtn").addEventListener("click", () => {
      this.showSignupModal()
    })

    // Left logout button
    document.getElementById("logoutBtnLeft").addEventListener("click", () => {
      this.handleLogout()
    })

    // Chat functionality
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

    document.getElementById("closeSignupModal").addEventListener("click", () => {
      this.hideSignupModal()
    })

    // File upload drag and drop
    const fileInput = document.getElementById("fileInput")
    const uploadArea = document.getElementById("uploadArea")
    const browseBtn = uploadArea.querySelector(".browse-btn")

    browseBtn.addEventListener("click", () => {
      fileInput.click()
    })

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
        fileInput.files = e.dataTransfer.files
        this.updateUploadButton()
      }
    })

    // Form submissions
    document.getElementById("loginForm").addEventListener("submit", (e) => {
      e.preventDefault()
      this.handleLogin()
    })

    document.getElementById("signupForm").addEventListener("submit", (e) => {
      e.preventDefault()
      this.handleSignup()
    })

    // Modal background clicks
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

    document.getElementById("signupModal").addEventListener("click", (e) => {
      if (e.target.id === "signupModal") {
        this.hideSignupModal()
      }
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
      if (this.isLoggedIn) {
        this.loadTables()
      } else {
        this.renderAuthPrompt()
      }
    } else {
      sidebar.classList.remove("active")
    }
  }

  renderAuthPrompt() {
    const tableList = document.getElementById("tableList")
    tableList.innerHTML = `
      <div class="auth-prompt">
        <div class="auth-prompt-content">
          <i class="fas fa-lock"></i>
          <h3>Authentication Required</h3>
          <p>Please login or sign up to save and view your table history.</p>
          <div class="auth-prompt-buttons">
            <button class="auth-prompt-btn login-btn-prompt" onclick="app.showLoginModal(); app.toggleHistorySidebar()">
              <i class="fas fa-sign-in-alt"></i>
              Login
            </button>
            <button class="auth-prompt-btn signup-btn-prompt" onclick="app.showSignupModal(); app.toggleHistorySidebar()">
              <i class="fas fa-user-plus"></i>
              Sign Up
            </button>
          </div>
        </div>
      </div>
    `
  }

  async loadInitialData() {
    if (this.isLoggedIn) {
      await this.loadTables()
    }
    this.renderUploadForm()
    this.resetChat()
    document.getElementById("activeTableName").textContent = ""
    document.getElementById("lastUpdated").style.display = "none"
  }

  async loadTables() {
    if (!this.isLoggedIn) {
      this.tables = []
      this.renderTablesList()
      return
    }

    try {
      const response = await fetch(`${this.baseURL}/api/list_tables/`, {
        headers: this.getAuthHeaders(),
      })

      if (response.status === 401) {
        this.handleAuthError()
        return
      }

      if (response.ok) {
        const data = await response.json()
        this.tables = data.table_details || []
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

    if (!this.isLoggedIn) {
      this.renderAuthPrompt()
      return
    }

    if (this.tables.length === 0) {
      tableList.innerHTML = `
        <div class="empty-tables">
          <i class="fas fa-table"></i>
          <p>No tables uploaded yet</p>
          <button class="upload-table-btn" onclick="app.showUploadModal(); app.toggleHistorySidebar()">
            <i class="fas fa-plus"></i>
            Upload First Table
          </button>
        </div>
      `
      return
    }

    tableList.innerHTML = ""
    this.tables.forEach((table) => {
      const tableItem = document.createElement("div")
      tableItem.className = `table-item ${table.table_name === this.activeTable ? "active" : ""}`
      tableItem.dataset.table = table.table_name
      tableItem.innerHTML = `
        <i class="fas fa-table"></i>
        <div class="table-info">
          <span class="table-name">${table.original_name}</span>
          <span class="table-meta">${table.rows} rows • ${table.columns} cols</span>
        </div>
        <div class="active-indicator"></div>
      `
      tableItem.addEventListener("click", () => {
        this.switchTable(table.table_name)
      })
      tableList.appendChild(tableItem)
    })
  }

  async switchTable(tableName) {
    console.log(`Switching to table: ${tableName}`)

    if (tableName === this.activeTable) {
      console.log("Table already active, skipping switch")
      return
    }

    this.activeTable = tableName
    document.querySelectorAll(".table-item").forEach((item) => {
      item.classList.remove("active")
    })
    document.querySelector(`[data-table="${tableName}"]`)?.classList.add("active")

    // Find the original name for display
    const tableInfo = this.tables.find((t) => t.table_name === tableName)
    this.setActiveTableName(tableInfo?.original_name || tableName)

    // Close sidebar after selection
    if (this.historySidebarOpen) {
      this.toggleHistorySidebar()
    }

    try {
      console.log("Calling switch_table API...")
      // First switch the table on the backend
      const switchResponse = await fetch(`${this.baseURL}/api/switch_table/`, {
        method: "POST",
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ table_name: tableName }),
      })

      if (switchResponse.status === 401) {
        this.handleAuthError()
        return
      }

      if (switchResponse.ok) {
        const switchData = await switchResponse.json()
        console.log("Switch successful:", switchData)

        // Now load the report for the switched table
        console.log("Loading report...")
        await this.loadReport(tableName)

        // Reset chat for the new table
        this.resetChat()

        this.showSuccess(`Switched to table '${tableInfo?.original_name || tableName}'`)
      } else {
        const errorData = await switchResponse.json()
        console.error("Switch failed:", errorData)
        throw new Error(errorData.detail || "Failed to switch table")
      }
    } catch (error) {
      console.error("Error switching table:", error)
      this.showError("Failed to switch table. Please try again.")
    }
  }

  async loadReport(tableName) {
    console.log(`Loading report for table: ${tableName}`)

    const reportContent = document.getElementById("reportContent")
    if (!tableName) {
      console.log("No table name provided, rendering upload form")
      this.renderUploadForm()
      return
    }

    reportContent.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Generating report...</p></div>'

    try {
      console.log("Calling generate_report API...")
      // Use the generate_report endpoint with table_name parameter
      const response = await fetch(`${this.baseURL}/api/generate_report/?table_name=${encodeURIComponent(tableName)}`, {
        method: "POST",
        headers: this.getAuthHeaders(),
      })

      console.log("Generate report response status:", response.status)

      if (response.status === 401) {
        this.handleAuthError()
        return
      }

      if (response.ok) {
        const reportData = await response.json()
        console.log("Report data received:", reportData)
        this.renderReport(reportData)
        document.getElementById("lastUpdated").textContent = "Updated just now"
        document.getElementById("lastUpdated").style.display = "block"
      } else {
        const errorData = await response.json()
        console.error("Report generation failed:", errorData)
        throw new Error(errorData.detail || "Failed to load report")
      }
    } catch (error) {
      console.error("Error loading report:", error)
      this.showError("Failed to generate report. Please try again.")
      this.renderMockReport()
    }
  }

  handleAuthError() {
    this.isLoggedIn = false
    this.currentUser = null
    this.authToken = null
    localStorage.removeItem("token")
    localStorage.removeItem("username")
    this.updateHeaderButtons()
    this.showError("Session expired. Please login again.")
    this.showLoginModal()
  }

  async sendMessage() {
    if (!this.isLoggedIn) {
      this.showError("Please login to use the chat feature")
      this.showLoginModal()
      return
    }

    const input = document.getElementById("chatInput")
    const message = input.value.trim()
    if (!message) return

    this.addMessageToChat(message, "user")
    input.value = ""
    this.showTypingIndicator()

    try {
      const response = await fetch(`${this.baseURL}/api/ask`, {
        method: "POST",
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ message: message, table_name: this.activeTable }),
      })

      if (response.status === 401) {
        this.handleAuthError()
        return
      }

      if (response.ok) {
        const data = await response.json()
        this.hideTypingIndicator()
        if (data.answer) {
          this.addMessageToChat(data.answer, "bot")
        } else if (data.sql_query && data.analysis) {
          let botMessage = ""
          if (data.results) {
            botMessage += "**Results:**\n" + data.results + "\n\n"
          }
          if (data.analysis) {
            botMessage += "**Analysis:**\n" + data.analysis
          }
          if (data.active_dataset) {
            this.activeTable = data.active_dataset
          }
          this.addMessageToChat(botMessage, "bot")
        } else {
          this.addMessageToChat("I received a response but couldn't parse it properly.", "bot")
        }
      } else {
        throw new Error(`HTTP ${response.status}: Failed to get response`)
      }
    } catch (error) {
      this.hideTypingIndicator()
      console.error("Error in sendMessage:", error)
      this.addMessageToChat("Sorry, I encountered an error. Please try again.", "bot")
    }
  }

  async handleFileUpload(file, tableName) {
    if (!this.isLoggedIn) {
      this.showError("Please login to upload files")
      this.showLoginModal()
      return
    }

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

    uploadForm.style.display = "none"
    uploadProgress.style.display = "block"

    try {
      const formData = new FormData()
      formData.append("file", file)
      formData.append("table_name", tableName)

      console.log("Starting file upload...")
      const response = await fetch(`${this.baseURL}/api/upload_csv/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.authToken}`,
        },
        body: formData,
      })

      if (response.status === 401) {
        this.handleAuthError()
        return
      }

      if (response.ok) {
        const data = await response.json()
        console.log("Upload successful:", data)

        // Set the active table immediately
        this.activeTable = data.table_name
        this.setActiveTableName(data.original_name)

        // Hide upload modal first
        this.hideUploadModal()

        // Show generating report message
        const reportContent = document.getElementById("reportContent")
        reportContent.innerHTML =
          '<div class="loading-state"><div class="spinner"></div><p>Generating report for uploaded data...</p></div>'

        // Refresh table list
        await this.loadTables()

        // Generate report for the uploaded table
        console.log("Generating report for uploaded table...")
        await this.loadReport(data.table_name)

        // Reset chat for the new table
        this.resetChat()

        this.showSuccess("CSV uploaded and report generated successfully!")
      } else {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Upload failed")
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
        localStorage.setItem("username", username)
        this.isLoggedIn = true
        this.currentUser = username
        this.authToken = data.access_token
        this.updateHeaderButtons()
        this.hideLoginModal()
        this.showSuccess("Login successful!")
        await this.loadTables() // Load user's tables
      } else {
        throw new Error("Login failed")
      }
    } catch (error) {
      this.showError("Login failed. Please check your credentials.")
    }
  }

  async handleSignup() {
    const username = document.getElementById("signupUsername").value
    const password = document.getElementById("signupPassword").value

    try {
      const response = await fetch(`${this.baseURL}/api/signup/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      })

      if (response.ok) {
        this.hideSignupModal()
        this.showSuccess("Sign up successful! Please login with your credentials.")
        this.showLoginModal()
      } else {
        const errorData = await response.json()
        if (response.status === 400 && errorData.detail === "Username already registered") {
          this.showSignupError("User already exists. Please choose a different username.")
        } else {
          this.showSignupError(errorData.detail || "Sign up failed. Please try again.")
        }
      }
    } catch (error) {
      this.showSignupError("Sign up failed. Please try again.")
    }
  }

  handleLogout() {
    localStorage.removeItem("token")
    localStorage.removeItem("username")
    this.isLoggedIn = false
    this.currentUser = null
    this.authToken = null
    this.activeTable = null
    this.tables = []
    this.updateHeaderButtons()
    this.showSuccess("Logged out successfully!")
    this.loadInitialData()
    if (this.historySidebarOpen) {
      this.renderAuthPrompt()
    }
  }

  // Keep all other existing methods (renderReport, addMessageToChat, etc.)
  renderReport(data) {
    const reportContent = document.getElementById("reportContent")
    if (!Array.isArray(data.columns)) {
      reportContent.innerHTML = "<div class='error-state'>Error: No columns found in report data.</div>"
      return
    }

    const generatedDate = new Date(data.generated_at)
    const formattedDate = generatedDate.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })

    const html = `
      <div class="report-container">
        <!-- Report Overview -->
        <div class="report-overview">
          <h1>${data.table_name || data.original_name}</h1>
          <div class="report-subtitle">Comprehensive Data Analysis Report</div>
          <div class="report-meta-info">
            <div class="meta-item">
              <div class="meta-label">Original Name</div>
              <div class="meta-value">${data.original_name}</div>
            </div>
            <div class="meta-item">
              <div class="meta-label">File Name</div>
              <div class="meta-value">${data.file_name || "N/A"}</div>
            </div>
            <div class="meta-item">
              <div class="meta-label">Generated</div>
              <div class="meta-value">${formattedDate}</div>
            </div>
          </div>
        </div>

        <!-- Data Statistics -->
        <div class="data-stats">
          <div class="stat-card">
            <h4>Total Rows</h4>
            <div class="stat-value">${data.total_rows.toLocaleString()}</div>
          </div>
          <div class="stat-card">
            <h4>Total Columns</h4>
            <div class="stat-value">${data.total_columns}</div>
          </div>
          <div class="stat-card">
            <h4>Data Quality</h4>
            <div class="stat-value">${this.calculateDataQuality(data)}%</div>
          </div>
          <div class="stat-card">
            <h4>Outliers Found</h4>
            <div class="stat-value">${Object.keys(data.outliers || {}).length}</div>
          </div>
        </div>

        <!-- Executive Summary -->
        <div class="report-section summary-section">
          <div class="section-header">
            <div class="section-icon">
              <i class="fas fa-file-alt"></i>
            </div>
            <h2>Executive Summary</h2>
          </div>
          <div class="summary-content">
            ${this.markdownToHtml(data.summary)}
          </div>
        </div>

        <!-- Dataset Columns -->
        <div class="report-section columns-section">
          <div class="section-header">
            <div class="section-icon">
              <i class="fas fa-columns"></i>
            </div>
            <h2>Dataset Columns (${data.columns.length})</h2>
          </div>
          <div class="columns-grid">
            ${data.columns
              .map(
                (column, index) => `
                <div class="column-item" style="background: ${this.getColumnColor(index)};">
                  <i class="fas fa-database"></i>
                  <span>${column}</span>
                </div>
              `,
              )
              .join("")}
          </div>
        </div>

        ${
          Object.keys(data.outliers || {}).length > 0
            ? `
            <!-- Outliers Analysis -->
            <div class="report-section outliers-section">
              <div class="section-header">
                <div class="section-icon">
                  <i class="fas fa-exclamation-triangle"></i>
                </div>
                <h2>Outliers Analysis</h2>
              </div>
              ${Object.entries(data.outliers)
                .map(
                  ([column, outlierData]) => `
                  <div class="outlier-card">
                    <h4>
                      <i class="fas fa-chart-line"></i>
                      ${column}
                    </h4>
                    <div class="outlier-stats">
                      <div class="outlier-stat">
                        <span class="label">Outlier Count</span>
                        <span class="value">${outlierData.count} (${outlierData.percentage}%)</span>
                      </div>
                      <div class="outlier-stat">
                        <span class="label">Outlier Values</span>
                        <span class="value">${outlierData.values.slice(0, 5).join(", ")}${outlierData.values.length > 5 ? "..." : ""}</span>
                      </div>
                      <div class="outlier-stat">
                        <span class="label">Normal Range</span>
                        <span class="value">${outlierData.bounds.lower.toFixed(2)} - ${outlierData.bounds.upper.toFixed(2)}</span>
                      </div>
                      ${
                        outlierData.extreme_values
                          ? `
                          <div class="outlier-stat">
                            <span class="label">Extreme Values</span>
                            <span class="value">Min: ${outlierData.extreme_values.min_outlier}, Max: ${outlierData.extreme_values.max_outlier}</span>
                          </div>
                        `
                          : ""
                      }
                    </div>
                  </div>
                `,
                )
                .join("")}
            </div>
          `
            : ""
        }

        ${
          data.graph_urls && data.graph_urls.length > 0
            ? `
            <!-- Data Visualizations -->
            <div class="report-section visualizations-section">
              <div class="section-header">
                <div class="section-icon">
                  <i class="fas fa-chart-bar"></i>
                </div>
                <h2>Data Visualizations (${data.graph_urls.length})</h2>
              </div>
              <div class="graphs-container">
                ${data.graph_urls
                  .map(
                    (url, index) => `
                    <div class="graph-item">
                      <div class="graph-title">
                        ${this.getGraphTitle(url, index)}
                      </div>
                      <img src="${this.baseURL}${url}" alt="Data visualization ${index + 1}" loading="lazy" onerror="this.style.display='none'">
                    </div>
                  `,
                  )
                  .join("")}
              </div>
            </div>
          `
            : ""
        }
      </div>
    `
    reportContent.innerHTML = html
  }

  calculateDataQuality(data) {
    let qualityScore = 100
    const outlierCount = Object.keys(data.outliers || {}).length
    if (outlierCount > 0) {
      qualityScore -= Math.min(outlierCount * 10, 30)
    }
    return Math.max(qualityScore, 70)
  }

  getGraphTitle(url, index) {
    const filename = url.split("/").pop()
    const parts = filename.split("_")
    if (parts.length > 1) {
      const titlePart = parts.slice(1).join("_").replace(".png", "")
      return titlePart.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
    }
    return `Visualization ${index + 1}`
  }

  getColumnColor(index) {
    const colors = [
      "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
      "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
      "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
      "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
      "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
      "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)",
      "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
    ]
    return colors[index % colors.length]
  }

  renderMockReport() {
    const mockData = {
      table_name: "sample_data",
      original_name: "sample_data",
      file_name: "sample.csv",
      total_rows: 1000,
      total_columns: 8,
      generated_at: new Date().toISOString(),
      summary: `## Executive Summary\n\nThis dataset contains comprehensive information with high data quality and interesting patterns.\n\n## Key Findings\n\n**Data Quality:** The dataset shows excellent quality with minimal missing values.\n\n**Notable Patterns:** Several interesting correlations were discovered in the analysis.`,
      outliers: {
        "Sample Column": {
          count: 5,
          percentage: 0.5,
          values: [100, 200, 300],
          bounds: { lower: 0, upper: 50 },
        },
      },
      columns: ["ID", "Name", "Value", "Category", "Date", "Status", "Amount", "Type"],
      graph_urls: [
        "/placeholder.svg?height=300&width=400&text=Overview",
        "/placeholder.svg?height=300&width=400&text=Distribution",
        "/placeholder.svg?height=300&width=400&text=Correlation",
      ],
    }
    this.renderReport(mockData)
  }

  markdownToHtml(markdown) {
    if (!markdown) return ""

    return markdown
      .replace(/### (.*)/g, "<h4>$1</h4>")
      .replace(/## (.*)/g, "<h3>$1</h3>")
      .replace(/# (.*)/g, "<h2>$1</h2>")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/\$\$(.*?)\$\$/g, "<em>$1</em>")
      .replace(/\$(.*?)\$/g, "<em>$1</em>")
      .replace(/^\* (.*)/gm, "<li>$1</li>")
      .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/\n/g, "<br>")
      .replace(/^(?!<h|<ul|<li)(.+)$/gm, "<p>$1</p>")
      .replace(/<p><\/p>/g, "")
      .replace(/<p>(<h[1-6]>.*<\/h[1-6]>)<\/p>/g, "$1")
      .replace(/<p>(<ul>.*<\/ul>)<\/p>/g, "$1")
  }

  resetChat() {
    this.chatMessages = []
    const chatMessages = document.getElementById("chatMessages")
    const tableName = this.activeTable || "your data"
    chatMessages.innerHTML = `
      <div class="message bot-message">
        <div class="message-avatar">
          <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
          <p>Hello! I'm your AI assistant. I can help you analyze ${tableName}. What would you like to know?</p>
        </div>
      </div>
    `
  }

  addMessageToChat(message, sender) {
    const chatMessages = document.getElementById("chatMessages")
    const messageDiv = document.createElement("div")
    messageDiv.className = `message ${sender}-message`
    const avatar = sender === "user" ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>'

    const formattedMessage = sender === "bot" ? this.markdownToHtml(message) : message

    messageDiv.innerHTML = `
      <div class="message-avatar">
        ${avatar}
      </div>
      <div class="message-content">
        <div>${formattedMessage}</div>
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

  renderUploadForm() {
    const reportContent = document.getElementById("reportContent")
    reportContent.innerHTML = `
      <div class="upload-form-container">
        <div class="upload-form-wrapper">
          <div class="upload-form-left">
            <div class="upload-form-header">
              <div class="upload-icon">
                <i class="fas fa-chart-line"></i>
              </div>
              <h2>Welcome to DataLens</h2>
              <p>Upload your CSV files and let AI help you discover insights in your data</p>
            </div>
            
            <div class="upload-features">
              <div class="feature-item">
                <i class="fas fa-file-csv"></i>
                <p>CSV Upload & Processing</p>
              </div>
              <div class="feature-item">
                <i class="fas fa-chart-bar"></i>
                <p>Automatic Data Analysis</p>
              </div>
              <div class="feature-item">
                <i class="fas fa-brain"></i>
                <p>AI-Powered Insights</p>
              </div>
            </div>
          </div>
          
          <div class="upload-form-right">
            <div class="inline-upload-form">
              <h3>Get Started</h3>
              ${
                !this.isLoggedIn
                  ? `
                <div class="auth-required-notice">
                  <i class="fas fa-info-circle"></i>
                  <p>Please <a href="#" onclick="app.showLoginModal()">login</a> or <a href="#" onclick="app.showSignupModal()">sign up</a> to upload files</p>
                </div>
              `
                  : `
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
              `
              }
            </div>
          </div>
        </div>
      </div>
    `

    if (this.isLoggedIn) {
      this.setupInlineUploadListeners()
    }
  }

  setupInlineUploadListeners() {
    const inlineFileInput = document.getElementById("inlineFileInput")
    const inlineUploadArea = document.getElementById("inlineUploadArea")
    const inlineTableName = document.getElementById("inlineTableName")
    const inlineSubmitBtn = document.getElementById("inlineSubmitBtn")
    const inlineBrowseBtn = inlineUploadArea?.querySelector(".inline-browse-btn")

    if (!inlineFileInput || !inlineUploadArea || !inlineTableName || !inlineSubmitBtn) return

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

    inlineSubmitBtn.addEventListener("click", async () => {
      const file = inlineFileInput.files[0]
      const tableName = inlineTableName.value.trim()
      if (file && tableName) {
        // Show loading state immediately
        const reportContent = document.getElementById("reportContent")
        reportContent.innerHTML =
          '<div class="loading-state"><div class="spinner"></div><p>Uploading and processing your data...</p></div>'

        await this.handleFileUpload(file, tableName)
      }
    })
  }

  updateInlineUploadButton() {
    const inlineFileInput = document.getElementById("inlineFileInput")
    const inlineTableName = document.getElementById("inlineTableName")
    const inlineSubmitBtn = document.getElementById("inlineSubmitBtn")
    if (inlineFileInput && inlineTableName && inlineSubmitBtn) {
      inlineSubmitBtn.disabled = !(inlineFileInput.files.length > 0 && inlineTableName.value.trim())
    }
  }

  setActiveTableName(tableName) {
    document.getElementById("activeTableName").textContent = tableName
    document.getElementById("lastUpdated").style.display = "block"
  }

  // Modal functions
  showUploadModal() {
    if (!this.isLoggedIn) {
      this.showError("Please login to upload files")
      this.showLoginModal()
      return
    }
    document.getElementById("uploadModal").classList.add("active")
  }

  hideUploadModal() {
    document.getElementById("uploadModal").classList.remove("active")
    document.getElementById("uploadForm").reset()
    this.updateUploadButton()
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

  hideSignupModal() {
    document.getElementById("signupModal").classList.remove("active")
    const signupForm = document.getElementById("signupForm")
    if (signupForm) signupForm.reset()
    const existingError = document.querySelector(".signup-error")
    if (existingError) {
      existingError.remove()
    }
  }

  showSignupError(message) {
    const existingError = document.querySelector(".signup-error")
    if (existingError) {
      existingError.remove()
    }
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

  showError(message) {
    alert("Error: " + message)
  }

  showSuccess(message) {
    alert("Success: " + message)
  }
}

// Make app globally available
let app

document.addEventListener("DOMContentLoaded", () => {
  app = new DataLensApp()
})
