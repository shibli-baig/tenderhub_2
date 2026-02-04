# Memory analysis: 512MB Render instance and Analytics OOM

## Summary

The **Analytics page** (`/analytics`) is the main trigger for OOM. It loads large amounts of data into memory in a single request, uses heavy N+1 query patterns, and runs in the same process as background workers. With 512MB total and 2 Gunicorn workers, each process has ~256MB; the analytics handler alone can exceed that when favorites/shortlists/projects/employees are numerous.

---

## 1. Analytics page: why it blows memory

**Route:** `app.py` ~2376–3065 (`analytics_page`).

### 1.1 Full-table loads (all in one request)

| Data | Code | Risk |
|------|------|------|
| All favorites | `FavoriteDB.filter(user_id=...).all()` | Can be hundreds/thousands |
| All projects | `ProjectDB.filter(user_id=...).all()` | Same |
| All shortlisted tenders | `ShortlistedTenderDB.filter(user_id=...).all()` | Same |
| All calendar activities | `CalendarActivityDB.filter(...).all()` | Grows with favorites + projects + reminders |
| All reminders (twice) | `ReminderDB.filter(...).all()` | Once for calendar sync, once for section 1 |
| All employees | `EmployeeDB.filter(company_code_id=...).all()` | For workload section |

Each `.all()` pulls full ORM objects (with relationships and any lazy-loaded fields) into memory. No pagination or limit.

### 1.2 N+1 queries (per-item DB hits)

- **Shortlisted tenders:** For each `st` in `shortlisted_tenders`:
  - `db.query(TenderDB).filter(TenderDB.id == st.tender_id).first()`
  - `db.query(TenderAssignmentDB).filter(tender_id=...).all()`
  - `db.query(EmployeeDB).filter(EmployeeDB.id.in_(emp_ids)).all()`
- **Favourites (section 1):** For each `fav` in `favourited`:
  - `db.query(TenderDB).filter(TenderDB.id == fav.tender_id).first()`
- **Reminders:** For each `r` in `reminders`:
  - `db.query(TenderDB).filter(TenderDB.id == r.tender_id).first()`
- **Section 3 (stage analytics):** Same shortlisted loop again: for each `st`, again query TenderDB, TenderAssignmentDB, EmployeeDB.

So with 200 shortlisted + 300 favorites + 50 reminders, you get hundreds of extra queries and many duplicate TenderDB/EmployeeDB objects in memory.

### 1.3 Calendar sync on every page load

Before rendering, the handler:

- Loops **all favorites** and, for each with a deadline, calls `save_calendar_activity()` (DB read + maybe insert).
- Loops **all projects** and, for each start/end date, calls `save_calendar_activity()` twice.
- Loops **all reminders** and calls `save_calendar_activity()` for each.

So the analytics request also does a large number of writes and object creations, and keeps all favorites/projects/reminders in memory while doing it.

### 1.4 Duplicated and heavy Python structures

- `deadlines`: one dict per deadline (shortlisted + favourited), each with `tender_title`, `assigned_employees` (list of dicts), etc.
- `stage_tenders`: for each step (1–6), a list of tender info + `assigned_employees` + `step_employees`.
- `employee_workload`: for each employee, a dict with `stage_distribution` (6 steps), built by iterating again over **all** `shortlisted_tenders`.

So the same shortlisted tenders and assignments are traversed many times and copied into multiple big lists/dicts.

### 1.5 Template context size

The handler passes to `analytics.html` in one go:

- All deadline lists (overdue, today, this_week, next_two_weeks, later)
- All reminders
- All timeline data (30 days × 5 series)
- Full `stage_breakdown` and `stage_tenders` (per-step lists of tenders)
- Full `employee_workload`
- Full `calendar_events`
- Counts and aggregates

Jinja2 then holds this entire context while rendering a 2000+ line template (with ApexCharts). So peak memory = big Python structures + big template context + rendered output.

---

## 2. Other memory factors (same 512MB)

### 2.1 Gunicorn workers

From `render.yaml`:

```yaml
startCommand: gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT --timeout 120
```

- 2 worker processes.
- 512MB is shared across both → **~256MB per process** (before any request).

### 2.2 Background workers in the same process

- `certificate_queue` is imported from `app` (e.g. `enqueue_certificate`, `get_queue_status`). At import time, `certificate_queue.start_workers()` runs and starts **3 certificate worker threads** in the same process.
- If `tender_eligibility_queue` is also imported, **2 tender analysis worker threads** start in the same process.

So one “web” process is actually: 1 Uvicorn worker + 3 cert threads + (optionally) 2 tender threads. They all share the same 256MB. A single heavy request (e.g. analytics) can spike that process over 256MB and trigger OOM.

### 2.3 In-process caches

- **Tender recommendation / embeddings:** `tender_recommendation.py` keeps an in-memory cache (`MEMORY_CACHE_SIZE = 1000`). Each entry is an embedding vector (e.g. 3072 floats for text-embedding-3-large) → on the order of tens of MB.
- **Core cache:** `core/cache.py` can fall back to an in-memory dict (`_memory_cache`) if Redis is unavailable.
- **Sessions:** `core/security.py` can fall back to in-memory session storage.

These add to baseline memory per process.

### 2.4 No streaming or chunking

- Analytics does not stream the response.
- No pagination or “load section by section” for the three big sections (calendar/deadlines, tenders analytics, stage/employee analytics). Everything is loaded and rendered in one shot.

---

## 3. Why “only” opening Analytics is enough

- One GET to `/analytics` runs the whole `analytics_page` function.
- It loads all favorites, all projects, all shortlisted tenders, all calendar activities, all reminders, all employees.
- It does hundreds of extra queries (N+1) and builds many large lists/dicts.
- It runs calendar sync (more DB and objects).
- It passes a huge context to Jinja2 and renders a large HTML page.

So a single request can push one worker process from ~100–150MB to 300MB+; with 256MB effective per process, that process goes OOM. S3 and Postgres don’t need to “hold” the data; the problem is that the app **loads** too much of that data into the **same process** at once.

---

## 4. Recommendations (in order of impact)

### 4.1 Analytics: reduce data loaded per request (high impact)

- **Limit or paginate:**
  - Cap shortlisted tenders (e.g. last 200 or 500 by `created_at`) for section 1 and section 3.
  - Cap favorites and projects used for calendar and deadlines (e.g. last N by date).
- **Use aggregation in the DB instead of Python:**
  - Section 2 (counts, sums, timelines): already uses aggregates; keep and ensure no extra `.all()` for that section.
  - Section 3 (stage breakdown): compute counts per step/status with `GROUP BY` (and optionally one query per step) instead of loading all shortlisted tenders and looping in Python.
  - Employee workload: one query that returns counts per (employee_id, step) (or similar), then build the list in Python from that small result.
- **Remove or defer calendar sync from the analytics request:**
  - Option A: Do not call `save_calendar_activity` on every analytics load. Run calendar sync in a scheduled job or a separate, lighter endpoint (e.g. “sync calendar” button or cron).
  - Option B: If you keep it on the page, at least limit to “recent” favorites/projects/reminders and batch DB writes (e.g. bulk insert/upsert) instead of one query per activity.
- **Eager loading to remove N+1:**
  - Use `joinedload`/`selectinload` for ShortlistedTenderDB → TenderDB, and for TenderAssignmentDB → EmployeeDB, so one (or few) queries replace per-item queries. This reduces query count and can reduce duplicate object graphs in memory.
- **Lazy-load sections (medium effort, high impact):**
  - Serve the analytics page shell with minimal data (e.g. only header and section titles).
  - Load Section 1 (deadlines/calendar), Section 2 (tenders analytics), Section 3 (stage/employees) via separate API calls (e.g. `/api/analytics/deadlines`, `/api/analytics/tenders`, `/api/analytics/stages`). Each endpoint loads only what that section needs and returns JSON.
  - Frontend fetches these in parallel or on demand (e.g. when the section is visible). This caps peak memory per request (one section at a time) and avoids building one giant template context.

### 4.2 Analytics: keep response smaller (medium impact)

- Don’t pass full tender/project objects to the template. Pass only what the template needs (ids, titles, dates, counts). For example, use list of dicts with a fixed set of keys instead of ORM objects so Jinja2 doesn’t touch large object graphs.
- If you keep calendar events in the same response, limit to the current month (or current + next month) and load other months via a small API when the user changes month.

### 4.3 Process and deployment (quick wins)

- **Reduce Gunicorn workers to 1** in `render.yaml` (e.g. `-w 1`). That gives the single process the full 512MB for that one web worker + certificate (and tender) threads. Fewer concurrent requests, but avoids splitting 512MB into two smaller chunks. Revert to 2 when you have more RAM.
- **Move certificate (and tender) workers out of the web process:** Run them in a separate Render “worker” service (or another process) so they don’t share the 512MB with the web app. Then the web process’ memory is dominated by request handling (and analytics) only.
- **Upgrade plan:** If you stay on 512MB, the above changes are almost mandatory. Moving to 1GB (or more) would give more headroom for current code and for multiple workers.

### 4.4 Caches and fallbacks (lower impact, still useful)

- Ensure Redis is used for sessions and for cache so that in-memory fallbacks are not used under normal conditions (no unbounded in-memory session/cache growth).
- If you keep the recommendation in-memory cache, consider lowering `MEMORY_CACHE_SIZE` (e.g. 200–500) on the 512MB instance to reduce baseline memory.

---

## 5. Quick wins you can do first

1. **Limit analytics data:** In `analytics_page`, add `.limit(500)` (or similar) to the shortlisted and favorites queries used for deadlines and stage analytics, and add `.order_by(ShortlistedTenderDB.created_at.desc())` so you get the most recent. Same idea for projects if they’re large.
2. **Stop calendar sync on every analytics load:** Comment out or remove the blocks that call `save_calendar_activity` inside `analytics_page`; run that logic in a background job or a separate endpoint called less often.
3. **Use one worker:** In `render.yaml`, set `-w 1` so the single web process has 512MB to itself (plus any in-process worker threads).
4. **Eager load in analytics:** For `shortlisted_tenders`, use something like:
   - `ShortlistedTenderDB.query.filter(...).options(joinedload(ShortlistedTenderDB.tender)).limit(500).all()`
   and for assignments/employees, either joinedload or one bulk query per section (e.g. all assignment counts for the relevant tender_ids), then build the per-tender lists in Python. That removes the worst N+1 and reduces memory churn.

After that, the next step is to split the analytics page into section-level API calls and optionally add DB-side aggregation for section 3 (stage breakdown and employee workload). That will make the Analytics page safe for a 512MB instance even with many shortlisted tenders and favorites.
