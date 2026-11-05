(function () {
    const supportedLanguages = ["es", "eu", "en"];
    const flags = {
        es: '<svg class="language-flag" viewBox="0 0 24 16" aria-hidden="true"><rect width="24" height="16" fill="#aa151b"/><rect y="4" width="24" height="8" fill="#f1bf00"/></svg>',
        eu: '<svg class="language-flag" viewBox="0 0 24 16" aria-hidden="true"><rect width="24" height="16" fill="#d52b1e"/><path d="M0 0l24 16M24 0L0 16" stroke="#007934" stroke-width="3.5"/><path d="M12 0v16M0 8h24" stroke="#fff" stroke-width="3.5"/></svg>',        en: '<svg class="language-flag" viewBox="0 0 24 16" aria-hidden="true"><rect width="24" height="16" fill="#012169"/><path d="m0 0 24 16M24 0 0 16" stroke="#fff" stroke-width="4"/><path d="m0 0 24 16M24 0 0 16" stroke="#c8102e" stroke-width="1.7"/><path d="M12 0v16M0 8h24" stroke="#fff" stroke-width="6"/><path d="M12 0v16M0 8h24" stroke="#c8102e" stroke-width="3"/></svg>'
    };

    function readSavedLanguage() {
        try {
            const saved = localStorage.getItem("app_lang");
            return supportedLanguages.includes(saved) ? saved : "es";
        } catch (error) {
            console.warn("Could not read the saved language preference.", error);
            return "es";
        }
    }

    function getTranslation(key, lang) {
        return (window.TRANSLATIONS[lang] && window.TRANSLATIONS[lang][key])
            || window.TRANSLATIONS.en[key]
            || key;
    }

    function renderLanguagePicker(lang) {
        const container = document.getElementById("language-picker");
        if (!container) return;

        const selectedLabel = getTranslation(`language.name.${lang}`, lang);
        const menuOptions = supportedLanguages.map(function (option) {
            return `<button class="lang-picker-option${option === lang ? " active" : ""}" type="button" role="menuitem" data-lang="${option}">${flags[option]}<span>${getTranslation(`language.name.${option}`, lang)}</span></button>`;
        }).join("");

        container.innerHTML = `
            <button class="lang-picker-btn" id="language-select-button" type="button"
                    aria-haspopup="menu" aria-expanded="false" aria-label="${getTranslation("language.label", lang)}">
                ${flags[lang]}<span>${selectedLabel}</span><span class="lang-picker-chevron" aria-hidden="true"></span>
            </button>
            <div class="lang-picker-menu" id="language-select-menu" role="menu" hidden>${menuOptions}</div>`;

        const button = document.getElementById("language-select-button");
        const menu = document.getElementById("language-select-menu");
        button.addEventListener("click", function () {
            const isOpen = button.getAttribute("aria-expanded") === "true";
            button.setAttribute("aria-expanded", String(!isOpen));
            menu.hidden = isOpen;
        });

        menu.querySelectorAll("[data-lang]").forEach(function (option) {
            option.addEventListener("click", function () {
                window.applyLanguage(option.dataset.lang);
                document.getElementById("language-select-button").focus();
            });
        });
    }

    window.t = function (key, values) {
        const lang = supportedLanguages.includes(document.documentElement.lang)
            ? document.documentElement.lang
            : "es";
        let text = getTranslation(key, lang);
        if (values) {
            text = text.replace(/\{(\w+)\}/g, function (match, name) {
                return Object.prototype.hasOwnProperty.call(values, name) ? values[name] : match;
            });
        }
        return text;
    };

    window.applyLanguage = function (lang) {
        const activeLanguage = supportedLanguages.includes(lang) ? lang : "es";
        document.documentElement.lang = activeLanguage;

        try {
            localStorage.setItem("app_lang", activeLanguage);
        } catch (error) {
            console.warn("Could not save the language preference.", error);
        }

        document.querySelectorAll("[data-i18n]").forEach(function (element) {
            element.textContent = getTranslation(element.dataset.i18n, activeLanguage);
        });

        document.querySelectorAll("[data-i18n-attr]").forEach(function (element) {
            element.dataset.i18nAttr.split(",").forEach(function (mapping) {
                const parts = mapping.trim().split(":");
                if (parts.length === 2) {
                    element.setAttribute(parts[0], getTranslation(parts[1], activeLanguage));
                }
            });
        });

        renderLanguagePicker(activeLanguage);
        window.dispatchEvent(new CustomEvent("app-language-changed", {
            detail: { lang: activeLanguage }
        }));
    };

    document.addEventListener("click", function (event) {
        const container = document.getElementById("language-picker");
        const button = document.getElementById("language-select-button");
        const menu = document.getElementById("language-select-menu");
        if (container && button && menu && !container.contains(event.target)) {
            button.setAttribute("aria-expanded", "false");
            menu.hidden = true;
        }
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            const button = document.getElementById("language-select-button");
            const menu = document.getElementById("language-select-menu");
            if (button && menu) {
                button.setAttribute("aria-expanded", "false");
                menu.hidden = true;
                button.focus();
            }
        }
    });

    document.addEventListener("DOMContentLoaded", function () {
        window.applyLanguage(readSavedLanguage());
    });
})();
