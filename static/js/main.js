// Main JavaScript for Sentiment Analysis App

document.addEventListener("DOMContentLoaded", function () {
  // Initialize the application
  initializeApp();
});

function initializeApp() {
  // Character counter for text input
  initializeCharacterCounter();

  // Form validation and submission handling
  initializeFormHandling();

  // Smooth scrolling for anchor links
  initializeSmoothScrolling();

  // Tooltip initialization
  initializeTooltips();

  // Animation on scroll
  initializeScrollAnimations();

  // Auto-set current day
  setCurrentDay();

  // Initialize copy to clipboard functionality
  initializeCopyToClipboard();
}

function initializeCharacterCounter() {
  const textArea = document.getElementById("text_content");
  const charCountElement = document.getElementById("charCount");

  if (textArea && charCountElement) {
    textArea.addEventListener("input", function () {
      const charCount = this.value.length;
      const maxLength = this.getAttribute("maxlength") || 1000;

      charCountElement.textContent = charCount;

      // Update color based on character count
      charCountElement.className = getCharCountClass(charCount, maxLength);

      // Provide visual feedback
      if (charCount > maxLength * 0.9) {
        textArea.classList.add("border-warning");
        textArea.classList.remove("border-success");
      } else if (charCount > 0) {
        textArea.classList.add("border-success");
        textArea.classList.remove("border-warning");
      } else {
        textArea.classList.remove("border-warning", "border-success");
      }
    });
  }
}

function getCharCountClass(charCount, maxLength) {
  if (charCount > maxLength * 0.95) {
    return "text-danger fw-bold";
  } else if (charCount > maxLength * 0.8) {
    return "text-warning fw-semibold";
  } else if (charCount > 0) {
    return "text-success";
  }
  return "text-muted";
}

function initializeFormHandling() {
  const form = document.getElementById("sentimentForm");

  if (form) {
    form.addEventListener("submit", function (e) {
      const textContent = document.getElementById("text_content").value.trim();

      // Basic validation
      if (!textContent) {
        e.preventDefault();
        showAlert("Silakan masukkan teks untuk dianalisis", "error");
        return false;
      }

      if (textContent.length < 5) {
        e.preventDefault();
        showAlert("Teks harus memiliki minimal 5 karakter", "error");
        return false;
      }

      // Show loading state
      showLoadingState();
    });
  }
}

function showLoadingState() {
  const btn = document.getElementById("analyzeBtn");
  const btnText = btn.querySelector(".btn-text");
  const spinner = document.getElementById("loadingSpinner");

  if (btn && btnText && spinner) {
    btn.disabled = true;
    btn.classList.add("disabled");
    btnText.textContent = "Menganalisis...";
    spinner.classList.remove("d-none");

    // Add pulse animation to button
    btn.classList.add("pulse");
  }
}

function hideLoadingState() {
  const btn = document.getElementById("analyzeBtn");
  const btnText = btn.querySelector(".btn-text");
  const spinner = document.getElementById("loadingSpinner");

  if (btn && btnText && spinner) {
    btn.disabled = false;
    btn.classList.remove("disabled", "pulse");
    btnText.textContent = "Analisis Sentimen";
    spinner.classList.add("d-none");
  }
}

function initializeSmoothScrolling() {
  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);

      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
}

function initializeTooltips() {
  // Initialize Bootstrap tooltips
  const tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
}

function initializeScrollAnimations() {
  // Intersection Observer for scroll animations
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("fade-in-up");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px",
      }
    );

    // Observe elements that should animate
    document
      .querySelectorAll(".card, .feature-item, .example-card")
      .forEach((el) => {
        observer.observe(el);
      });
  }
}

function setCurrentDay() {
  const daySelect = document.getElementById("day_of_week");
  if (daySelect) {
    const days = [
      "Sunday",
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
    ];
    const today = days[new Date().getDay()];

    // Set current day as selected
    const todayOption = daySelect.querySelector(`option[value="${today}"]`);
    if (todayOption) {
      todayOption.selected = true;
    }
  }
}

function initializeCopyToClipboard() {
  // Add copy functionality to result page
  const copyButtons = document.querySelectorAll("[data-copy]");

  copyButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const textToCopy = this.getAttribute("data-copy");
      copyToClipboard(textToCopy, this);
    });
  });
}

function copyToClipboard(text, button) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        showCopySuccess(button);
      })
      .catch(() => {
        fallbackCopyToClipboard(text, button);
      });
  } else {
    fallbackCopyToClipboard(text, button);
  }
}

function fallbackCopyToClipboard(text, button) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.left = "-999999px";
  textArea.style.top = "-999999px";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    document.execCommand("copy");
    showCopySuccess(button);
  } catch (err) {
    console.error("Unable to copy to clipboard:", err);
    showAlert("Gagal menyalin ke clipboard", "error");
  }

  document.body.removeChild(textArea);
}

function showCopySuccess(button) {
  const originalText = button.innerHTML;
  button.innerHTML = '<i class="fas fa-check me-2"></i>Disalin!';
  button.classList.add("btn-success");
  button.classList.remove("btn-outline-secondary");

  setTimeout(() => {
    button.innerHTML = originalText;
    button.classList.remove("btn-success");
    button.classList.add("btn-outline-secondary");
  }, 2000);
}

function showAlert(message, type = "info") {
  const alertContainer = document.createElement("div");
  alertContainer.className = `alert alert-${
    type === "error" ? "danger" : type
  } alert-dismissible fade show position-fixed`;
  alertContainer.style.cssText =
    "top: 20px; right: 20px; z-index: 9999; min-width: 300px;";

  alertContainer.innerHTML = `
        <i class="fas fa-${
          type === "error" ? "exclamation-triangle" : "info-circle"
        } me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

  document.body.appendChild(alertContainer);

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (alertContainer.parentNode) {
      alertContainer.remove();
    }
  }, 5000);
}

// API Functions
function makePredictionAPI(data) {
  return fetch("/api/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => response.json())
    .catch((error) => {
      console.error("API Error:", error);
      throw error;
    });
}

// Utility Functions
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function throttle(func, limit) {
  let inThrottle;
  return function () {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

// Form auto-save (optional feature)
function initializeAutoSave() {
  const form = document.getElementById("sentimentForm");
  if (!form) return;

  const formData = {};
  const inputs = form.querySelectorAll("input, textarea, select");

  inputs.forEach((input) => {
    // Load saved data
    const savedValue = localStorage.getItem(`sentiment_form_${input.name}`);
    if (savedValue && input.value === "") {
      input.value = savedValue;
    }

    // Save on change
    input.addEventListener(
      "change",
      debounce(() => {
        localStorage.setItem(`sentiment_form_${input.name}`, input.value);
      }, 500)
    );
  });
}

// Performance monitoring
function logPerformance() {
  if ("performance" in window) {
    window.addEventListener("load", () => {
      const loadTime =
        performance.timing.loadEventEnd - performance.timing.navigationStart;
      console.log(`Page loaded in ${loadTime}ms`);
    });
  }
}

// Dark mode toggle (future feature)
function initializeDarkMode() {
  const darkModeToggle = document.getElementById("darkModeToggle");
  if (!darkModeToggle) return;

  const currentTheme = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", currentTheme);

  darkModeToggle.addEventListener("click", () => {
    const theme = document.documentElement.getAttribute("data-theme");
    const newTheme = theme === "light" ? "dark" : "light";

    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
  });
}

// Export functions for global use
window.SentimentApp = {
  showAlert,
  makePredictionAPI,
  copyToClipboard,
  showLoadingState,
  hideLoadingState,
};

// Initialize performance logging
logPerformance();
