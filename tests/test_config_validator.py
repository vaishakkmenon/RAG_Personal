"""
Tests for environment variable validation (Phase 1.3).

Tests the config_validator module to ensure:
- Missing required variables are detected
- Insecure default values trigger warnings
- Provider-specific validation works
- Production-specific checks function correctly
"""

import pytest

from app.config_validator import validate_config, REQUIRED_VARS


class TestConfigValidator:
    """Test suite for configuration validation."""

    def test_missing_api_key(self, monkeypatch, caplog):
        """Test that missing API_KEY causes startup failure."""
        # Clear all required env vars
        for var in REQUIRED_VARS:
            monkeypatch.delenv(var, raising=False)

        # Set only REDIS_PASSWORD (so only API_KEY is missing)
        monkeypatch.setenv("REDIS_PASSWORD", "test-password")

        with pytest.raises(SystemExit) as exc_info:
            validate_config()

        assert exc_info.value.code == 1
        assert "API_KEY" in caplog.text
        assert "Missing required environment variables" in caplog.text

    def test_missing_redis_password(self, monkeypatch, caplog):
        """Test that missing REDIS_PASSWORD causes startup failure."""
        # Clear all required env vars
        for var in REQUIRED_VARS:
            monkeypatch.delenv(var, raising=False)

        # Set only API_KEY (so only REDIS_PASSWORD is missing)
        monkeypatch.setenv("API_KEY", "test-api-key-12345")

        with pytest.raises(SystemExit) as exc_info:
            validate_config()

        assert exc_info.value.code == 1
        assert "REDIS_PASSWORD" in caplog.text

    def test_missing_groq_api_key_when_using_groq(self, monkeypatch, caplog):
        """Test that missing Groq API key fails when LLM_PROVIDER=groq."""
        # Set required vars
        monkeypatch.setenv("API_KEY", "test-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "test-redis-password")

        # Set provider to groq but don't provide key
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.delenv("LLM_GROQ_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            validate_config()

        assert exc_info.value.code == 1
        assert "LLM_GROQ_API_KEY" in caplog.text

    def test_insecure_api_key_default(self, monkeypatch, caplog):
        """Test that insecure default API_KEY triggers warning."""
        # Set required vars with an insecure default
        monkeypatch.setenv("API_KEY", "change-me")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-password-12345")

        # Should not exit, but should warn
        validate_config()

        assert "SECURITY WARNING" in caplog.text
        assert "API_KEY" in caplog.text
        assert "insecure default" in caplog.text.lower()

    def test_insecure_redis_password_default(self, monkeypatch, caplog):
        """Test that insecure default REDIS_PASSWORD triggers warning."""
        # Set required vars with an insecure default for Redis
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "change-me-devpassword123")

        # Should not exit, but should warn
        validate_config()

        assert "SECURITY WARNING" in caplog.text
        assert "REDIS_PASSWORD" in caplog.text
        # Note: The actual password is redacted in logs, so we check for the warning presence
        assert "default placeholder" in caplog.text

    def test_insecure_grafana_password_default(self, monkeypatch, caplog):
        """Test that insecure default GRAFANA_ADMIN_PASSWORD triggers warning."""
        # Set required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")

        # Set insecure Grafana password
        monkeypatch.setenv("GRAFANA_ADMIN_PASSWORD", "admin123")

        # Should not exit, but should warn
        validate_config()

        assert "SECURITY WARNING" in caplog.text
        assert "GRAFANA_ADMIN_PASSWORD" in caplog.text

    def test_secure_configuration_passes(self, monkeypatch, caplog):
        """Test that secure configuration passes without warnings."""
        import logging

        caplog.set_level(logging.INFO)

        # Set all required vars with secure values
        monkeypatch.setenv(
            "API_KEY",
            "d5160646e4199a5d88ea3626a3795e4139eef33adb29a56568b4b52bcbe703d5",
        )
        monkeypatch.setenv("REDIS_PASSWORD", "very-secure-redis-password-12345")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_GROQ_API_KEY", "test-groq-key")

        # Set secure Grafana password (to avoid warning from .env default)
        monkeypatch.setenv("GRAFANA_ADMIN_PASSWORD", "secure-grafana-password-xyz123")

        # Should pass without warnings
        validate_config()

        assert "Environment configuration validated successfully" in caplog.text
        assert "SECURITY WARNING" not in caplog.text

    def test_production_missing_allowed_origins_warning(self, monkeypatch, caplog):
        """Test that missing ALLOWED_ORIGINS in production triggers warning."""
        # Set required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")

        # Set ENV to production
        monkeypatch.setenv("ENV", "production")

        # Don't set ALLOWED_ORIGINS
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

        # Should not exit, but should warn
        validate_config()

        assert "PRODUCTION WARNING" in caplog.text
        assert "ALLOWED_ORIGINS" in caplog.text

    def test_production_missing_session_require_https_warning(
        self, monkeypatch, caplog
    ):
        """Test that missing SESSION_REQUIRE_HTTPS in production triggers warning."""
        # Set required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://vaishakmenon.com")

        # Set ENV to production
        monkeypatch.setenv("ENV", "production")

        # Don't set SESSION_REQUIRE_HTTPS
        monkeypatch.delenv("SESSION_REQUIRE_HTTPS", raising=False)

        # Should not exit, but should warn
        validate_config()

        assert "PRODUCTION WARNING" in caplog.text
        assert "SESSION_REQUIRE_HTTPS" in caplog.text

    def test_production_https_false_warning(self, monkeypatch, caplog):
        """Test that SESSION_REQUIRE_HTTPS=false in production triggers warning."""
        # Set required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://vaishakmenon.com")

        # Set ENV to production
        monkeypatch.setenv("ENV", "production")

        # Set HTTPS to false (insecure for production)
        monkeypatch.setenv("SESSION_REQUIRE_HTTPS", "false")

        # Should warn about insecure cookies
        validate_config()

        assert "SESSION_REQUIRE_HTTPS=false" in caplog.text
        assert "cookies may be insecure" in caplog.text.lower()

    def test_production_localhost_in_origins_warning(self, monkeypatch, caplog):
        """Test that localhost in ALLOWED_ORIGINS triggers production warning."""
        # Set required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")

        # Set ENV to production
        monkeypatch.setenv("ENV", "production")

        # Include localhost in origins (bad for production)
        monkeypatch.setenv(
            "ALLOWED_ORIGINS", "https://vaishakmenon.com,http://localhost:3000"
        )
        monkeypatch.setenv("SESSION_REQUIRE_HTTPS", "true")

        # Should warn about localhost
        validate_config()

        assert "localhost" in caplog.text
        assert "security risk" in caplog.text.lower()

    def test_development_environment_no_production_warnings(self, monkeypatch, caplog):
        """Test that development environment doesn't trigger production warnings."""
        import logging

        caplog.set_level(logging.INFO)

        # Set required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")

        # Set ENV to development (or don't set it)
        monkeypatch.setenv("ENV", "development")

        # Don't set production vars (should be fine for dev)
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        monkeypatch.delenv("SESSION_REQUIRE_HTTPS", raising=False)

        # Avoid Grafana password warning from .env
        monkeypatch.setenv("GRAFANA_ADMIN_PASSWORD", "secure-grafana-password-xyz123")

        # Should not warn about production settings
        validate_config()

        assert "PRODUCTION WARNING" not in caplog.text
        assert "Environment configuration validated successfully" in caplog.text

    def test_multiple_missing_variables(self, monkeypatch, caplog):
        """Test that multiple missing variables are all reported."""
        # Clear all required vars
        for var in REQUIRED_VARS:
            monkeypatch.delenv(var, raising=False)

        # Also set Groq provider without key
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.delenv("LLM_GROQ_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            validate_config()

        assert exc_info.value.code == 1
        # Should report all missing vars
        assert "API_KEY" in caplog.text
        assert "REDIS_PASSWORD" in caplog.text
        assert "LLM_GROQ_API_KEY" in caplog.text

    def test_placeholder_groq_key_triggers_warning(self, monkeypatch, caplog):
        """Test that placeholder Groq API key triggers warning."""
        # Set required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")

        # Set Groq provider with placeholder key
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_GROQ_API_KEY", "your-groq-api-key-here")

        # Should not exit, but should warn
        validate_config()

        assert "SECURITY WARNING" in caplog.text
        assert "LLM_GROQ_API_KEY" in caplog.text
        # Note: The actual API key is redacted in logs, so we check for the warning presence
        assert "default placeholder" in caplog.text

    def test_config_validation_logs_environment_summary(self, monkeypatch, caplog):
        """Test that successful validation logs environment summary."""
        import logging

        caplog.set_level(logging.INFO)

        # Set required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_GROQ_API_KEY", "real-groq-key-12345")
        monkeypatch.setenv("SESSION_STORAGE_BACKEND", "redis")
        monkeypatch.setenv("RESPONSE_CACHE_ENABLED", "true")
        monkeypatch.setenv("PROMPT_GUARD_ENABLED", "false")

        # Avoid Grafana password warning from .env
        monkeypatch.setenv("GRAFANA_ADMIN_PASSWORD", "secure-grafana-password-xyz123")

        validate_config()

        # Should log environment details
        assert "Environment: development" in caplog.text  # Default
        assert "LLM Provider: groq" in caplog.text
        assert "Session Backend: redis" in caplog.text
        assert "Response Caching: enabled" in caplog.text
        assert "Prompt Guard: disabled" in caplog.text


class TestConfigValidatorIntegration:
    """Integration tests for config validation with actual startup."""

    def test_app_startup_with_valid_config(self, monkeypatch):
        """Test that app can start with valid configuration."""
        # Set all required vars
        monkeypatch.setenv("API_KEY", "secure-api-key-12345")
        monkeypatch.setenv("REDIS_PASSWORD", "secure-redis-password")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_GROQ_API_KEY", "test-groq-key")

        # Should not raise
        from app.config_validator import validate_config

        validate_config()

    def test_app_startup_fails_with_missing_config(self, monkeypatch):
        """Test that app startup fails with missing critical config."""
        # Clear required vars
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)

        # Should raise SystemExit
        from app.config_validator import validate_config

        with pytest.raises(SystemExit) as exc_info:
            validate_config()

        assert exc_info.value.code == 1
