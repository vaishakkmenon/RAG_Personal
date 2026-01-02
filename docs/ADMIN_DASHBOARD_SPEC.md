# Admin Dashboard Frontend Specification

## Overview

Build a Next.js admin dashboard for the Personal RAG System. The dashboard should provide a clean, professional interface for managing documents, monitoring system health, and controlling the knowledge base.

## Tech Stack

- **Framework**: Next.js 14+ (App Router)
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (recommended) or similar
- **State Management**: React Query or SWR for API calls
- **Authentication**: JWT tokens stored in httpOnly cookies or localStorage

## API Base URL

Production: `https://api.vaishakmenon.com`

## Authentication

All admin endpoints require JWT authentication.

### Login Flow

1. POST to `/auth/token` with form data:
   ```
   Content-Type: application/x-www-form-urlencoded
   Body: username=admin&password=<password>
   ```

2. Response:
   ```json
   {
     "access_token": "eyJhbGc...",
     "token_type": "bearer"
   }
   ```

3. Store the token and include in all subsequent requests:
   ```
   Authorization: Bearer <access_token>
   ```

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/token` | Get JWT token (form data: username, password) |
| GET | `/auth/users/me` | Get current user info |

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/documents` | List all documents |
| GET | `/admin/documents/{filename}` | Get document content |
| POST | `/admin/documents/upload` | Upload document (multipart form) |
| DELETE | `/admin/documents/{filename}` | Delete document |

### Knowledge Base Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Ingest documents (add to existing) |
| POST | `/admin/reingest` | Clear and rebuild entire knowledge base |
| DELETE | `/admin/chromadb` | Clear ChromaDB collection only |
| GET | `/admin/chromadb/status` | Get ChromaDB storage stats |

### Cache Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/fallback-cache/stats` | Get cache statistics |
| DELETE | `/admin/fallback-cache` | Clear fallback cache |
| POST | `/admin/fallback-cache/cleanup` | Clean expired entries |

### Health & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check (no auth required) |
| GET | `/metrics` | Prometheus metrics (no auth required) |

## Response Examples

### List Documents Response
```json
{
  "status": "success",
  "documents": [
    {
      "name": "master_profile.md",
      "path": "master_profile.md",
      "size_bytes": 15234,
      "size": "14.88 KB",
      "modified": "2026-01-02T18:30:00",
      "extension": ".md"
    }
  ],
  "total_count": 5,
  "docs_dir": "./data/mds"
}
```

### Get Document Response
```json
{
  "status": "success",
  "name": "master_profile.md",
  "content": "---\ntitle: Master Resume...\n---\n# Header\n...",
  "size_bytes": 15234,
  "modified": "2026-01-02T18:30:00"
}
```

### Upload Document Response
```json
{
  "status": "success",
  "message": "Document 'new-doc.md' uploaded successfully",
  "name": "new-doc.md",
  "size_bytes": 1024,
  "size": "1.00 KB",
  "path": "new-doc.md",
  "overwritten": false,
  "next_steps": [
    "Call POST /ingest to add this document to the knowledge base",
    "Or call POST /admin/reingest to clear and rebuild the entire knowledge base"
  ]
}
```

### ChromaDB Status Response
```json
{
  "status": "populated",
  "message": "ChromaDB directory contains data",
  "path": "/workspace/data/chroma",
  "exists": true,
  "files_count": 45,
  "dirs_count": 3,
  "total_size_bytes": 52428800,
  "total_size": "50.00 MB"
}
```

### Reingest Response
```json
{
  "status": "success",
  "message": "Knowledge base rebuilt successfully",
  "ingested_chunks": 92,
  "bm25_stats": {
    "status": "rebuilt",
    "doc_count": 92
  }
}
```

### Health Response
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "chromadb": "connected",
    "redis": "connected",
    "llm": "available"
  }
}
```

## Dashboard Pages

### 1. Login Page (`/login`)
- Username and password form
- Error handling for invalid credentials
- Redirect to dashboard on success

### 2. Dashboard Home (`/`)
- System health status cards
- Quick stats: document count, chunk count, cache stats
- Recent activity or alerts

### 3. Documents Page (`/documents`)
- Table/list of all documents with columns:
  - Name
  - Size
  - Last Modified
  - Actions (View, Edit, Delete)
- Upload button (opens modal or dedicated page)
- Search/filter functionality
- Bulk actions (optional)

### 4. Document Editor (`/documents/[filename]`)
- Markdown editor with preview
- Save button (uses upload with overwrite=true)
- Metadata display (size, last modified)
- Delete button with confirmation

### 5. Knowledge Base Page (`/knowledge-base`)
- ChromaDB status card
- Actions:
  - "Ingest New Documents" button
  - "Full Rebuild" button (with confirmation modal - this is destructive)
  - "Clear Database" button (with confirmation)
- BM25 index status
- Last ingest timestamp

### 6. Cache Management (`/cache`)
- Cache statistics display
- "Clear Cache" button
- "Cleanup Expired" button
- Hit/miss ratio visualization (optional)

### 7. Monitoring Page (`/monitoring`)
- Embed Grafana dashboards via iframe, OR
- Display key metrics from `/metrics` endpoint
- Links to external Grafana (https://grafana.vaishakmenon.com or localhost:3000)

## UI Components Needed

### Core Components
- `LoginForm` - Username/password with validation
- `Sidebar` - Navigation menu
- `Header` - Top bar with user info and logout
- `StatusCard` - For displaying health/stats
- `DataTable` - For document list
- `ConfirmModal` - For destructive actions
- `FileUpload` - Drag-and-drop or click to upload
- `MarkdownEditor` - For editing documents
- `Toast/Notification` - For success/error messages

### Layout
- Responsive sidebar layout
- Mobile-friendly navigation
- Dark mode support (optional but nice)

## Error Handling

- Show toast notifications for API errors
- Handle 401 (unauthorized) by redirecting to login
- Handle 409 (conflict) for document upload - offer overwrite option
- Handle 413 (file too large) with clear message
- Handle network errors gracefully

## File Upload Details

The upload endpoint expects `multipart/form-data`:
```javascript
const formData = new FormData();
formData.append('file', fileObject);
formData.append('overwrite', 'true'); // or 'false'

fetch('/admin/documents/upload', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: formData
});
```

**Allowed file types**: `.md`, `.txt`, `.yaml`, `.yml`
**Max file size**: 1 MB

## Security Considerations

- Store JWT securely (httpOnly cookie preferred, or localStorage with XSS protection)
- Include CSRF protection if using cookies
- Validate file types on frontend before upload
- Sanitize any user input displayed in the UI
- Add confirmation dialogs for destructive actions (delete, reingest, clear)

## Optional Enhancements

1. **Real-time updates**: WebSocket or polling for long-running operations
2. **Activity log**: Track who did what and when
3. **Diff viewer**: Show changes before saving document edits
4. **Syntax highlighting**: For markdown preview
5. **Keyboard shortcuts**: For power users
6. **Export/backup**: Download all documents as zip

## Development Notes

- Use environment variables for API base URL
- Implement request interceptors for adding auth headers
- Add loading states for all async operations
- Implement proper error boundaries
- Consider optimistic updates for better UX
