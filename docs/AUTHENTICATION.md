# Authentication Guide

This document describes the authentication system used in the Personal RAG System.

## Overview

The system uses two authentication methods:

| Method | Header | Use Case | Protected Endpoints |
|--------|--------|----------|---------------------|
| **API Key** | `X-API-Key` | Chat, feedback, public access | `/chat`, `/feedback` |
| **JWT Token** | `Authorization: Bearer <token>` | Admin operations | `/admin/*`, `/ingest`, `/debug/*` |

---

## API Key Authentication

### Configuration

API keys are configured via environment variables:

```bash
# Primary API key
API_KEY=your-secure-api-key

# Additional keys for rotation (comma-separated)
API_KEYS=key1,key2,key3
```

### Usage

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-api-key" \
  -d '{"question": "What certifications do I hold?"}'
```

### Generating Secure API Keys

```bash
# Generate a secure 32-byte hex key
openssl rand -hex 32
```

### Implementation Details

- Location: `app/middleware/api_key.py`
- Validates against `settings.api_key` and `settings.api_keys` list
- Failed attempts are logged with masked key values
- Origin validation for additional security

---

## JWT Authentication

### Architecture

```
┌─────────────┐     POST /auth/token      ┌─────────────┐
│   Client    │ ────────────────────────► │   FastAPI   │
│             │ ◄──────────────────────── │             │
└─────────────┘   { access_token: ... }   └─────────────┘
       │                                         │
       │  Authorization: Bearer <token>          │
       ▼                                         ▼
┌─────────────┐                           ┌─────────────┐
│  Protected  │ ◄─────────────────────────│    Auth     │
│  Endpoint   │     Validated User        │  Middleware │
└─────────────┘                           └─────────────┘
```

### Components

| File | Purpose |
|------|---------|
| `app/core/auth.py` | JWT token creation, password hashing, user validation |
| `app/core/security_config.py` | Security constants (algorithm, expiration) |
| `app/models/users.py` | SQLAlchemy User model |
| `app/api/routers/auth.py` | `/auth/token` and `/auth/users/me` endpoints |

### Configuration

Environment variables:

```bash
# Required: Secret key for JWT signing (use strong random value)
# Generate with: openssl rand -hex 32
SECRET_KEY=your-256-bit-secret-key
```

The token expiration is configured in `app/core/security_config.py` (default: 30 minutes).

---

## User Management

### Creating Admin Users

Use the provided script to create admin users:

```bash
# Set password via environment variable
export ADMIN_PASSWORD="your-secure-password"

# Run in Docker
docker compose -f docker-compose.prod.yml run --rm api python scripts/create_admin.py

# Or locally
python scripts/create_admin.py
```

The script creates a user with:
- **Username**: `admin`
- **Email**: `admin@example.com` (placeholder)
- **is_active**: `True`
- **is_superuser**: `True`

### User Model

```python
class User(Base):
    __tablename__ = "users"

    id: int              # Primary key
    username: str        # Unique, indexed
    email: str           # Optional
    hashed_password: str # bcrypt hash
    is_active: bool      # Account status
    is_superuser: bool   # Admin privileges
```

### Password Storage

Passwords are hashed using bcrypt:

```python
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hash password
hashed = pwd_context.hash(plain_password)

# Verify password
pwd_context.verify(plain_password, hashed_password)
```

---

## Token Flow

### 1. Request Token

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your-password"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. Use Token

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  http://localhost:8000/admin/stats
```

### 3. Token Structure

The JWT payload contains:
```json
{
  "sub": "admin",           // Username
  "exp": 1703443200         // Expiration timestamp
}
```

### Token Validation

1. Extract token from `Authorization: Bearer <token>` header
2. Decode and verify signature using `JWT_SECRET_KEY`
3. Check expiration (`exp` claim)
4. Look up user by `sub` claim
5. Verify user is active

---

## Protected Endpoints

### Admin Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/admin/stats` | GET | System statistics |
| `/admin/sessions` | GET | Active sessions |
| `/admin/sessions/{id}` | DELETE | Remove session |

### Ingest Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ingest` | POST | Trigger document ingestion |
| `/ingest/status` | GET | Check ingestion status |

### Debug Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/debug/search` | POST | Raw vector search |
| `/debug/sample` | GET | Sample chunks |

### Using Dependencies

To protect a new endpoint:

```python
from app.core.auth import get_current_active_user, get_current_admin_user
from app.models.users import User

# Require any authenticated user
@router.get("/protected")
async def protected_route(current_user: User = Depends(get_current_active_user)):
    return {"user": current_user.username}

# Require admin user
@router.get("/admin-only")
async def admin_route(current_user: User = Depends(get_current_admin_user)):
    return {"admin": current_user.username}
```

---

## Security Best Practices

### Secret Key Management

1. **Never commit secrets to git**
   - Use `.env` files (gitignored)
   - Use Docker secrets in production

2. **Use strong keys**
   ```bash
   openssl rand -hex 32
   ```

3. **Rotate keys periodically**
   - Update `SECRET_KEY` in .env
   - Restart services
   - Users will need to re-authenticate

### Token Security

1. **Use HTTPS in production**
   - Tokens are transmitted in headers
   - Caddy handles TLS automatically

2. **Keep tokens short-lived**
   - Default: 30 minutes
   - Adjust via `ACCESS_TOKEN_EXPIRE_MINUTES`

3. **Don't store tokens in localStorage**
   - For browser clients, use httpOnly cookies
   - This API is primarily for server-to-server or CLI use

### Password Security

1. **Enforce strong passwords**
   - Minimum 12 characters recommended
   - Mix of letters, numbers, symbols

2. **bcrypt provides protection**
   - Automatic salting
   - Configurable work factor
   - Resistant to rainbow tables

---

## Troubleshooting

### "Could not validate credentials"

**Causes:**
- Token expired
- Invalid token format
- Wrong secret key
- User deleted or deactivated

**Solution:**
1. Request a new token
2. Verify `SECRET_KEY` matches between token generation and validation
3. Check user exists: `docker compose exec postgres psql -U rag -d rag -c "SELECT * FROM users;"`

### "Incorrect username or password"

**Causes:**
- Wrong username or password
- User doesn't exist
- Password hash mismatch

**Solution:**
1. Verify credentials
2. Reset password by running `scripts/create_admin.py` again (updates existing user)

### "The user doesn't have enough privileges"

**Causes:**
- User is not a superuser
- Endpoint requires admin access

**Solution:**
1. Create user with superuser flag
2. Update existing user in database:
   ```sql
   UPDATE users SET is_superuser = true WHERE username = 'admin';
   ```

---

## API Reference

### POST /auth/token

Request OAuth2 access token.

**Request:**
```
Content-Type: application/x-www-form-urlencoded

username=admin&password=secret
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

**Errors:**
- `401 Unauthorized`: Invalid credentials

### GET /auth/users/me

Get current user information.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "username": "admin",
  "email": "admin@example.com",
  "is_active": true,
  "is_superuser": true
}
```

**Errors:**
- `401 Unauthorized`: Missing or invalid token
- `400 Bad Request`: User is inactive

---

**Last Updated:** 2025-12-24
