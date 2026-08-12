Create a new FastAPI router in the GSF project.

Router name and URL prefix: $ARGUMENTS
(e.g. `users /api/users` — name first, then the prefix. If no prefix given, default to `/api/<name>`)

## Instructions

1. Parse the router name and prefix from $ARGUMENTS.

2. Create `gsf/app/routers/<name>.py` following this exact pattern:
   ```python
   from __future__ import annotations

   from fastapi import APIRouter, HTTPException

   router = APIRouter()


   @router.get("/")
   def list_<name>() -> dict:
       return {"data": [], "count": 0}


   @router.get("/{item_id}")
   def get_<singular>(item_id: str) -> dict:
       raise HTTPException(status_code=404, detail="Not found")
   ```
   - All route handlers must be plain `def` (not `async def`). The DAL is synchronous; there is no async I/O in route handlers.
   - All route handlers must have return type annotations.
   - Helper/private functions are prefixed with `_`.
   - Raise `HTTPException` for error responses.

3. Register the router in `gsf/app/main.py`:
   ```python
   from app.routers import <name>
   app.include_router(<name>.router, prefix="<prefix>", tags=["<name>"])
   ```
   Add the import and `include_router` call in the correct location, following the existing pattern.

4. Run `uv run ruff check gsf/` from the repo root and fix any issues.

5. Report the files created and modified.
