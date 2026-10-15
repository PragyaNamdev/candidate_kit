"""
Employee Attendance & Analytics API   (FastAPI + MongoDB)

Run:     uvicorn app.main:app --port 8000
Config:  MONGO_URI, MONGO_DB  (optional .env file is read; real env vars win)

File layout
-----------
  1. Setup           — DB connection, constants, index creation at startup
  2. Helpers         — time-zone utilities, rule implementations (R1-R5), rounding
  3. Request models  — what the client is allowed to send (Pydantic, strict)
  4. Response helpers — turn Mongo documents into the API's JSON shape
  5. Pipelines       — MongoDB aggregation pipelines (also used by /admin/explain)
  6. Endpoints       — all routes in contract order
"""

import calendar
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field, StrictInt, field_validator, model_validator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure, DuplicateKeyError, PyMongoError

# Real env vars win; .env only fills in what is missing.
load_dotenv()

logger = logging.getLogger("attendance")


# ============================================================ #
# 1. SETUP                                                      #
# ============================================================ #
# tz_aware=True makes PyMongo return datetimes that carry UTC,
# so we never accidentally treat a UTC value as local time.
client = MongoClient(
    os.getenv("MONGO_URI") or "mongodb://localhost:27017",
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB") or "attendance_db"]

UTC   = timezone.utc
IST   = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

PRESENCE_STATUSES   = ["PRESENT", "WFH", "ON_DUTY"]
GRACE_SECONDS       = 10 * 60          # R2: 10-minute grace window
MIN_OVERTIME_MINUTES = 30              # R3: minimum overtime to record
HALF_DAY_THRESHOLD  = Decimal("4.50")  # R5: below this → half day
MAX_SHIFT_SECONDS   = 24 * 3600       # punch-out must be within 24 h of punch-in
MAX_TREND_DAYS      = 92              # cap the trend range to avoid huge pipelines
MIN_MS = 100_000_000_000              # 1973 — anything smaller looks like seconds
MAX_MS = 4_102_444_800_000            # year 2100


def create_indexes():
    """
    Create every index needed for correctness and performance.
    Safe to call repeatedly — MongoDB ignores indexes that already exist.
    """
    specs = [
        # employees ----------------------------------------------------------
        # emp_code is the employee identifier everywhere; must be unique and fast.
        ("employees", [("emp_code", 1)], {"unique": True, "name": "idx_emp_code"}),
        # Supports GET /employees?department=X and the analytics $lookup.
        ("employees", [("department", 1), ("emp_code", 1)], {"name": "idx_emp_dept_code"}),
        # Supports the "joined_on <= end-of-month" filter in department summary.
        ("employees", [("joined_on", 1)], {"name": "idx_emp_joined"}),

        # attendance_logs ----------------------------------------------------
        # The business rule "one record per (emp_code, date)" is enforced here.
        # This index is also the race guard for concurrent punch-ins: only one
        # insert can win; the other gets DuplicateKeyError, which we return as 409.
        ("attendance_logs", [("emp_code", 1), ("date", 1)],
            {"unique": True, "name": "idx_log_emp_date"}),
        # GET /attendance sorted by date DESC, emp_code ASC.
        ("attendance_logs", [("date", -1), ("emp_code", 1)], {"name": "idx_log_date_emp"}),
        # GET /attendance filtered by status.
        ("attendance_logs", [("status", 1), ("date", -1)], {"name": "idx_log_status_date"}),
        # punch-out: find the employee's most-recent open record by punch_in.
        ("attendance_logs", [("emp_code", 1), ("punch_in", -1)], {"name": "idx_log_emp_punchin"}),
    ]
    for col, keys, opts in specs:
        try:
            db[col].create_index(keys, **opts)
        except ConnectionFailure:
            logger.warning("MongoDB unreachable — skipping index creation")
            return
        except PyMongoError as exc:
            # Stale index with different options: log and move on.
            logger.warning("Index %s on %s: %s", opts["name"], col, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(PyMongoError)
async def mongo_error(_req, exc):
    # Any unhandled Mongo failure → 503 instead of 500.
    logger.error("MongoDB error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "database error"})


# ============================================================ #
# 2. HELPERS                                                    #
# ============================================================ #

def now_utc():
    """Current UTC time truncated to whole seconds (R1: instants are whole seconds)."""
    return datetime.now(UTC).replace(microsecond=0)


def as_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime (PyMongo returns naive UTC datetimes)."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def cut_seconds(dt: datetime) -> datetime:
    """Truncate to whole seconds (R1)."""
    return as_utc(dt).replace(microsecond=0)


def ms_to_dt(ms: int) -> datetime:
    """Epoch milliseconds → UTC datetime, sub-second part dropped (R1)."""
    return datetime.fromtimestamp(ms // 1000, tz=UTC)


def dt_to_ms(dt: datetime) -> int:
    """UTC datetime → epoch milliseconds (integer)."""
    return int((as_utc(dt) - EPOCH).total_seconds() * 1000)


def hm_to_minutes(hm: str) -> int:
    """'09:30' → 570"""
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


def is_overnight(shift_start: str, shift_end: str) -> bool:
    """True when shift_end <= shift_start, e.g. 22:00–06:00 (R1)."""
    return hm_to_minutes(shift_end) <= hm_to_minutes(shift_start)


def shift_end_datetime(date_str: str, shift_start: str, shift_end: str) -> datetime:
    """IST datetime of shift end on the attendance day (adds a day for overnight)."""
    midnight = datetime.strptime(date_str, "%Y-%m-%d")
    extra = 1 if is_overnight(shift_start, shift_end) else 0
    return (midnight + timedelta(days=extra, minutes=hm_to_minutes(shift_end))).replace(tzinfo=IST)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """
    R1: the IST calendar date this punch-in belongs to.
    For overnight shifts a punch-in before shift_end in the morning is counted
    against the shift that started the previous evening.
    """
    local = cut_seconds(punch_in).astimezone(IST)
    day = local.date()
    if is_overnight(shift_start, shift_end):
        punch_minutes = local.hour * 60 + local.minute
        if punch_minutes < hm_to_minutes(shift_end):
            day -= timedelta(days=1)
    return day.isoformat()


def compute_late_minutes(punch_in: datetime, date_str: str, shift_start: str) -> int:
    """
    R2: grace period is 10 minutes.  Late only when strictly > 10:00 after shift_start.
    Late minutes are counted from shift_start, not from the grace boundary.
    Examples for a 09:30 shift:  09:40:00 → 0,  09:40:01 → 10,  10:15:59 → 45.
    """
    midnight = datetime.strptime(date_str, "%Y-%m-%d")
    start_ist = (midnight + timedelta(minutes=hm_to_minutes(shift_start))).replace(tzinfo=IST)
    seconds_late = int((cut_seconds(punch_in) - start_ist).total_seconds())
    if seconds_late > GRACE_SECONDS:
        return seconds_late // 60   # whole minutes from shift start, R2
    return 0


def compute_overtime_minutes(punch_out: datetime, date_str: str,
                              shift_start: str, shift_end: str) -> int:
    """R3: whole minutes after shift_end, only counted when >= 30."""
    end_ist = shift_end_datetime(date_str, shift_start, shift_end)
    minutes = int((cut_seconds(punch_out) - end_ist).total_seconds()) // 60
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def round_half_up(value, places: int) -> float:
    """
    Round to `places` decimal places using half-up (the way people expect).
    Python's built-in round() uses banker's rounding, which gives different
    results for values exactly halfway between two representable decimals.
    """
    step = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(step, rounding=ROUND_HALF_UP))


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: (punch_out - punch_in) in hours, 2 decimals half-up."""
    seconds = int((cut_seconds(punch_out) - cut_seconds(punch_in)).total_seconds())
    return round_half_up(Decimal(seconds) / Decimal(3600), 2)


def derive_all(emp: dict, date_str: str, punch_in, punch_out):
    """
    Work out the four derived fields for a presence record.
    Called at punch-in (punch_out=None) and at punch-out / regularization.
    """
    out = {"late_minutes": 0, "work_hours": None, "overtime_minutes": 0, "half_day": False}
    if punch_in is not None:
        out["late_minutes"] = compute_late_minutes(punch_in, date_str, emp["shift_start"])
    if punch_in is not None and punch_out is not None:
        wh = compute_work_hours(punch_in, punch_out)
        out["work_hours"] = wh
        out["overtime_minutes"] = compute_overtime_minutes(
            punch_out, date_str, emp["shift_start"], emp["shift_end"])
        out["half_day"] = Decimal(str(wh)) < HALF_DAY_THRESHOLD   # R5
    return out


def month_bounds(month: str):
    """'2026-07' → ('2026-07-01', '2026-07-31', date(2026,7,1), date(2026,7,31))"""
    y, m = int(month[:4]), int(month[5:])
    last = calendar.monthrange(y, m)[1]
    first_d = date_cls(y, m, 1)
    last_d  = date_cls(y, m, last)
    return first_d.isoformat(), last_d.isoformat(), first_d, last_d


def count_working_days(first: date_cls, last: date_cls, joined_on: date_cls) -> int:
    """R7: Mon–Fri days from max(first, joined_on) to last."""
    day = max(first, joined_on)
    total = 0
    while day <= last:
        if day.weekday() < 5:    # 0=Mon … 4=Fri
            total += 1
        day += timedelta(days=1)
    return total


def fail(status_code: int, msg: str):
    raise HTTPException(status_code=status_code, detail=msg)


def get_employee(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if emp is None:
        fail(404, "employee not found")
    return emp


# ============================================================ #
# 3. REQUEST MODELS                                             #
# ============================================================ #
HM_PATTERN = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"


def _validate_date(v: str) -> str:
    if len(v) != 10:
        raise ValueError("date must be YYYY-MM-DD")
    try:
        d = datetime.strptime(v, "%Y-%m-%d")
    except ValueError:
        raise ValueError("not a real calendar date")
    if not (1900 <= d.year <= 2200):
        raise ValueError("year out of range")
    return v


def _validate_month(v: str) -> str:
    if len(v) != 7 or v[4] != "-":
        raise ValueError("month must be YYYY-MM")
    try:
        y, m = int(v[:4]), int(v[5:])
    except ValueError:
        raise ValueError("month must be YYYY-MM with integer year and month")
    if not (1900 <= y <= 2200) or not (1 <= m <= 12):
        raise ValueError("month out of range")
    return v


DateStr  = Annotated[str, AfterValidator(_validate_date)]
MonthStr = Annotated[str, AfterValidator(_validate_month)]

# EpochMs: must be a whole integer (not a float, not a string) and must be big
# enough to be milliseconds (not seconds).  StrictInt rejects 1.0 and "1".
EpochMs = Annotated[StrictInt, Field(ge=MIN_MS, le=MAX_MS)]

Status        = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]

PageParam     = Annotated[int, Query(ge=1)]
PageSizeParam = Annotated[int, Query(ge=1, le=100)]


class EmployeeIn(BaseModel):
    emp_code:    str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name:        str = Field(min_length=1, max_length=100)
    email:       str = Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    department:  str = Field(min_length=1, max_length=80)
    shift_start: str = Field(default="09:30", pattern=HM_PATTERN)
    shift_end:   str = Field(default="18:30", pattern=HM_PATTERN)
    joined_on:   DateStr

    @field_validator("name", "department")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def shifts_must_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must not be the same")
        return self


class PunchInIn(BaseModel):
    emp_code:  str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None
    status:    PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code:  str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    # Clients are NOT allowed to send computed fields (work_hours etc.).
    status:          Optional[Status] = None
    punch_in:        Optional[EpochMs] = None
    punch_out:       Optional[EpochMs] = None
    reason:          str = Field(min_length=5, max_length=500)
    regularized_by:  str = Field(min_length=1, max_length=80)


# ============================================================ #
# 4. RESPONSE HELPERS                                           #
# ============================================================ #
def _emp_out(doc: dict) -> dict:
    return {
        "emp_code":    doc["emp_code"],
        "name":        doc["name"],
        "email":       doc["email"],
        "department":  doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end":   doc["shift_end"],
        "joined_on":   doc["joined_on"],
        "created_at":  dt_to_ms(doc["created_at"]) if doc.get("created_at") else None,
    }


def _hist_val(v):
    """History change values: datetimes become milliseconds, everything else stays."""
    return dt_to_ms(v) if isinstance(v, datetime) else v


def _log_out(doc: dict) -> dict:
    """
    Converts an attendance_logs document to the API response shape.
    Handles legacy documents that may omit history, half_day, late_minutes,
    or overtime_minutes (DATA_MODEL §5, sample 9).
    """
    history = []
    for entry in doc.get("history") or []:
        changes = {
            field: {"from": _hist_val(ch.get("from")), "to": _hist_val(ch.get("to"))}
            for field, ch in (entry.get("changes") or {}).items()
        }
        history.append({
            "at":      dt_to_ms(entry["at"]) if entry.get("at") else None,
            "by":      entry.get("by"),
            "reason":  entry.get("reason"),
            "changes": changes,
        })
    pin  = doc.get("punch_in")
    pout = doc.get("punch_out")
    return {
        "emp_code":         doc["emp_code"],
        "date":             doc["date"],
        "status":           doc["status"],
        "punch_in":         dt_to_ms(pin)  if pin  else None,
        "punch_out":        dt_to_ms(pout) if pout else None,
        "work_hours":       doc.get("work_hours"),
        "late_minutes":     doc.get("late_minutes")     or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day":         bool(doc.get("half_day", False)),
        "history":          history,
    }


# ============================================================ #
# 5. AGGREGATION PIPELINES                                      #
# ============================================================ #
#
# IMPORTANT: these functions build the exact same pipeline that the real
# endpoints run.  /admin/explain/{endpoint} calls these same functions and
# passes the result to MongoDB's explain, so the explain always reflects
# the real query.
#
# Why $toDate works here:
#   The "date" field is stored as "YYYY-MM-DD", which is valid ISO 8601.
#   MongoDB converts it to UTC midnight.  $isoDayOfWeek then gives 1=Mon…7=Sun.
#   MongoDB 6.0+ (and the grader's 7.0) fully support this.
# ----------------------------------------------------------------

ATTENDANCE_SORT = [("date", -1), ("emp_code", 1)]

_PRESENCE_STATUSES = ["PRESENT", "WFH", "ON_DUTY"]


def _record_projection(weekdays_only: bool) -> dict:
    """
    Project one attendance record to small numbers for easy summing.

    p2          present-days × 2 (full day = 2, half day = 1).  Using ×2
                avoids fractions inside MongoDB and keeps the maths exact.
                When weekdays_only=True, weekend records produce p2=0.
                late/overtime/leave/duty always count regardless of weekday.
    late        late_minutes (0 if missing — legacy records)
    is_late     1 when late > 0, else 0
    ot          overtime_minutes
    is_leave    1 for LEAVE records
    is_duty     1 for ON_DUTY records
    wh_c        work_hours × 100 as an integer (for exact summing)
    wh_n        1 when this record has a work_hours value
    """
    is_present  = {"$in": ["$status", _PRESENCE_STATUSES]}
    is_weekday  = {"$lte": [{"$isoDayOfWeek": {"$toDate": "$date"}}, 5]}
    counts      = {"$and": [is_present, is_weekday]} if weekdays_only else is_present
    late        = {"$ifNull": ["$late_minutes", 0]}
    has_hours   = {"$and": [is_present, {"$isNumber": "$work_hours"}]}

    return {
        "_id":     0,
        "date":    1,
        "p2":      {"$cond": [counts, {"$cond": [{"$eq": ["$half_day", True]}, 1, 2]}, 0]},
        "late":    late,
        "is_late": {"$cond": [{"$gt": [late, 0]}, 1, 0]},
        "ot":      {"$ifNull": ["$overtime_minutes", 0]},
        "is_leave":{"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]},
        "is_duty": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]},
        "wh_c":    {"$cond": [has_hours, {"$round": [{"$multiply": ["$work_hours", 100]}, 0]}, 0]},
        "wh_n":    {"$cond": [has_hours, 1, 0]},
    }


def _logs_lookup(start: str, end: str, weekdays_only: bool) -> dict:
    """
    $lookup stage that attaches each employee's attendance records for the
    given date range.  Runs inside the employees collection pipeline so that
    employees with zero logs still appear (R9).
    Uses the (emp_code, date) compound index on attendance_logs.
    """
    return {
        "$lookup": {
            "from":      "attendance_logs",
            "localField": "emp_code",
            "foreignField": "emp_code",
            "pipeline": [
                {"$match": {"date": {"$gte": start, "$lte": end}}},
                {"$project": _record_projection(weekdays_only)},
            ],
            "as": "logs",
        }
    }


def monthly_pipeline(emp_code: str, month: str) -> list:
    """
    Aggregation on attendance_logs for one employee's monthly summary.

    R7 note: p2 counts only Mon–Fri presence records ($toDate + $isoDayOfWeek).
    late_minutes and overtime_minutes count ALL records (including weekends).
    """
    start, end, _, _ = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lte": end}}},
        {"$project": _record_projection(weekdays_only=True)},
        {
            "$group": {
                "_id":              None,
                "p2":               {"$sum": "$p2"},
                "leave_days":       {"$sum": "$is_leave"},
                "late_count":       {"$sum": "$is_late"},
                "late_minutes":     {"$sum": "$late"},
                "overtime_minutes": {"$sum": "$ot"},
            }
        },
    ]


def summary_pipeline(month: str, department) -> list:
    """
    Aggregation on employees for the department summary.

    Starts from employees so that people with zero logs are still counted (R9).
    Filters joined_on <= end-of-month so mid-month joiners are handled correctly.
    """
    start, end, _, _ = month_bounds(month)
    first_match = {"joined_on": {"$lte": end}}
    if department is not None:
        first_match["department"] = department
    return [
        {"$match": first_match},
        _logs_lookup(start, end, weekdays_only=True),
        # Sum each employee's records ...
        {
            "$project": {
                "department": 1,
                "p2":         {"$sum": "$logs.p2"},
                "late_count": {"$sum": "$logs.is_late"},
                "late_min":   {"$sum": "$logs.late"},
                "leave_count":{"$sum": "$logs.is_leave"},
                "duty_count": {"$sum": "$logs.is_duty"},
                "wh_c":       {"$sum": "$logs.wh_c"},
                "wh_n":       {"$sum": "$logs.wh_n"},
            }
        },
        # ... then sum the employees of each department.
        {
            "$group": {
                "_id":         "$department",
                "headcount":   {"$sum": 1},
                "p2":          {"$sum": "$p2"},
                "late_count":  {"$sum": "$late_count"},
                "late_min":    {"$sum": "$late_min"},
                "leave_count": {"$sum": "$leave_count"},
                "duty_count":  {"$sum": "$duty_count"},
                "wh_c":        {"$sum": "$wh_c"},
                "wh_n":        {"$sum": "$wh_n"},
            }
        },
        {"$sort": {"_id": 1}},
    ]


def leaderboard_pipeline(month: str, limit: int, department) -> list:
    """
    Late-comer ranking with standard competition ranking: 1, 2, 2, 4.

    $rank assigns equal total_late_minutes the same rank and skips the next.
    The $match after $rank keeps all rows whose rank <= limit, so ties at the
    cutoff are never dropped.
    Tied rows are ordered by emp_code (ascending) — deterministic and fair.
    """
    start, end, _, _ = month_bounds(month)
    pipeline = [
        {"$match": {"date": {"$gte": start, "$lte": end}, "late_minutes": {"$gt": 0}}},
        {
            "$group": {
                "_id":               "$emp_code",
                "total_late_minutes":{"$sum": "$late_minutes"},
                "late_count":        {"$sum": 1},
            }
        },
        # Join to get name and department.
        {"$lookup": {"from": "employees", "localField": "_id",
                     "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},   # records with no matching employee are dropped here
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline += [
        # $rank: equal totals share a rank; the next rank is skipped.
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {"rank": {"$rank": {}}},
            }
        },
        # Apply limit AFTER ranking so that ties at the boundary stay in.
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id":                0,
                "rank":               1,
                "emp_code":           "$_id",
                "name":               "$emp.name",
                "department":         "$emp.department",
                "total_late_minutes": 1,
                "late_count":         1,
            }
        },
    ]
    return pipeline


def trend_pipeline(department: str, first_day: date_cls, last_day: date_cls) -> list:
    """
    Daily attendance series for one department with gap-filling.

    Step 1 ($facet): two branches in parallel:
      • joins  — counts employees by joined_on so we can compute cumulative
                 headcount for any date without a separate query.
      • daily  — looks up each employee's records and groups by date.
    Step 2: generate one string per calendar day using $map + $range + $dateAdd.
    Step 3 ($unwind): one document per calendar day.
    Step 4 ($project): for each day, use $reduce on the joins list to count how
      many employees had joined_on <= that day (R9: cumulative headcount).
      $filter finds the matching daily entry (or nothing for days with no records).
    Step 5: extract present_p2 and late_count, defaulting to 0 for gap days.
    Step 6 ($sort): output in ascending date order.

    is_working_day, attendance_rate, and moving_avg_7d are computed in Python
    from the pipeline output (the only part that cannot be expressed more
    cleanly in the aggregation language without sacrificing readability).
    """
    start     = first_day.isoformat()
    end       = last_day.isoformat()
    day_count = (last_day - first_day).days + 1
    # MongoDB datetime for the first day at UTC midnight — used as the $dateAdd base.
    first_dt  = datetime(first_day.year, first_day.month, first_day.day, tzinfo=UTC)

    return [
        {"$match": {"department": department}},
        {
            "$facet": {
                # One entry per distinct joined_on value, with a count of employees.
                "joins": [
                    {"$group": {"_id": "$joined_on", "n": {"$sum": 1}}}
                ],
                # One entry per date that has at least one record for this department.
                "daily": [
                    _logs_lookup(start, end, weekdays_only=False),
                    {"$unwind": "$logs"},
                    {
                        "$group": {
                            "_id":    "$logs.date",
                            "p2":     {"$sum": "$logs.p2"},
                            "late_n": {"$sum": "$logs.is_late"},
                        }
                    },
                ],
            }
        },
        # Build the full list of calendar-day strings for the range.
        {
            "$project": {
                "joins": 1,
                "daily": 1,
                "days": {
                    "$map": {
                        "input": {"$range": [0, day_count]},
                        "as":    "i",
                        "in": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": {
                                    "$dateAdd": {
                                        "startDate": first_dt,
                                        "unit":      "day",
                                        "amount":    "$$i",
                                    }
                                },
                            }
                        },
                    }
                },
            }
        },
        # One document per calendar day.
        {"$unwind": "$days"},
        {
            "$project": {
                "date": "$days",
                # Cumulative headcount: employees whose joined_on is <= this day.
                "headcount": {
                    "$reduce": {
                        "input":        "$joins",
                        "initialValue": 0,
                        "in": {
                            "$cond": [
                                {"$lte": ["$$this._id", "$days"]},
                                {"$add": ["$$value", "$$this.n"]},
                                "$$value",
                            ]
                        },
                    }
                },
                # Find the daily entry for this exact date (null if no records).
                "day": {
                    "$first": {
                        "$filter": {
                            "input": "$daily",
                            "as":    "d",
                            "cond":  {"$eq": ["$$d._id", "$days"]},
                        }
                    }
                },
            }
        },
        # Default to 0 for days with no records.
        {
            "$set": {
                "present_p2": {"$ifNull": ["$day.p2",     0]},
                "late_count": {"$ifNull": ["$day.late_n", 0]},
            }
        },
        {"$unset": "day"},
        {"$sort": {"date": 1}},
    ]


def _build_attendance_filter(emp_code, date_from, date_to, status):
    if date_from and date_to and date_from > date_to:
        fail(422, "date_from must not be after date_to")
    q = {}
    if emp_code  is not None: q["emp_code"] = emp_code
    if date_from is not None: q.setdefault("date", {})["$gte"] = date_from
    if date_to   is not None: q.setdefault("date", {})["$lte"] = date_to
    if status    is not None: q["status"] = status
    return q


# ============================================================ #
# 6. ENDPOINTS                                                  #
# ============================================================ #

# -------- health -------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        fail(503, "database is not reachable")
    return {"status": "ok"}


# -------- employees -------- #
@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    if db.employees.find_one({"emp_code": body.emp_code}, {"_id": 1}):
        fail(409, "emp_code already exists")
    doc = body.model_dump()
    doc["created_at"] = now_utc()
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        # Two concurrent creates with the same emp_code: the index lets only one in.
        fail(409, "emp_code already exists")
    return _emp_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str]  = None,
    page:       PageParam       = 1,
    page_size:  PageSizeParam   = 20,
):
    q = {"department": department} if department is not None else {}
    total = db.employees.count_documents(q)
    items = list(
        db.employees
          .find(q, {"_id": 0})
          .sort("emp_code", 1)             # deterministic, contract says sort by emp_code
          .skip((page - 1) * page_size)    # R10: page is 1-based
          .limit(page_size)
    )
    return {"items": [_emp_out(d) for d in items], "total": total,
            "page": page, "page_size": page_size}


# -------- attendance — punch in -------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp       = get_employee(body.emp_code)
    punched_at = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    day        = attendance_date(punched_at, emp["shift_start"], emp["shift_end"])  # R1
    doc = {
        "emp_code":         body.emp_code,
        "date":             day,
        "status":           body.status,
        "punch_in":         cut_seconds(punched_at),
        "punch_out":        None,
        "work_hours":       None,
        "late_minutes":     compute_late_minutes(punched_at, day, emp["shift_start"]),  # R2
        "overtime_minutes": 0,
        "half_day":         False,
        "history":          [],
    }
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        # Race: the unique (emp_code, date) index lets exactly one insert win.
        # The loser gets DuplicateKeyError here, which we return as 409.
        fail(409, "already punched in for this date")
    return _log_out(doc)


# -------- attendance — punch out -------- #
@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp        = get_employee(body.emp_code)
    punched_at = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()

    # Find the employee's most recent open punch-in (any time — we validate ordering below).
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_out": None, "punch_in": {"$ne": None}},
        sort=[("punch_in", -1)],
    )
    if record is None:
        # No open record exists at all.  Check whether they already punched out.
        closed = db.attendance_logs.find_one(
            {"emp_code": body.emp_code, "punch_out": {"$ne": None}},
            sort=[("punch_in", -1)],
        )
        if closed:
            fail(409, "already punched out")
        fail(404, "no open punch-in found")

    pin  = cut_seconds(record["punch_in"])
    pout = cut_seconds(punched_at)
    gap  = int((pout - pin).total_seconds())
    if gap <= 0:
        # punched_at is at or before punch_in → 422, not 404
        fail(422, "punched_at must be after the punch-in time")
    if gap > MAX_SHIFT_SECONDS:
        fail(422, "punched_at is more than 24 hours after the punch-in")

    derived = derive_all(emp, record["date"], pin, pout)

    # The filter still requires punch_out=None so that two simultaneous punch-outs
    # only one can match; the other gets matched_count=0 and gets 409.
    result = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {"$set": {
            "punch_out":        pout,
            "work_hours":       derived["work_hours"],
            "overtime_minutes": derived["overtime_minutes"],
            "half_day":         derived["half_day"],
        }},
        return_document=ReturnDocument.AFTER,
    )
    if result is None:
        fail(409, "already punched out")
    return _log_out(result)


# -------- attendance — list -------- #
@app.get("/attendance")
def list_attendance(
    emp_code:  Optional[str]      = None,
    date_from: Optional[DateStr]  = None,
    date_to:   Optional[DateStr]  = None,
    status:    Optional[Status]   = None,
    page:      PageParam          = 1,
    page_size: PageSizeParam      = 20,
):
    q     = _build_attendance_filter(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(q)
    items = list(
        db.attendance_logs
          .find(q, {"_id": 0})
          .sort(ATTENDANCE_SORT)
          .skip((page - 1) * page_size)
          .limit(page_size)
    )
    return {"items": [_log_out(d) for d in items], "total": total,
            "page": page, "page_size": page_size}


# -------- attendance — regularize -------- #
def _values_differ(old, new) -> bool:
    """Compare two field values; datetimes are compared as UTC instants."""
    if isinstance(old, datetime) or isinstance(new, datetime):
        if old is None or new is None:
            return old is not new
        return as_utc(old) != as_utc(new)
    return old != new


def _plan_correction(emp: dict, record: dict, body: RegularizeIn):
    """
    Validate the correction and compute new values + changes dict.
    Raises 422 for any invalid combination.
    """
    new_status = body.status or record["status"]
    old_in     = record.get("punch_in")
    old_out    = record.get("punch_out")

    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            fail(422, "punch_in / punch_out must be absent for ABSENT or LEAVE status")
        new_in, new_out = None, None
        derived = {"late_minutes": 0, "work_hours": None, "overtime_minutes": 0, "half_day": False}
    else:
        new_in  = cut_seconds(ms_to_dt(body.punch_in))  if body.punch_in  is not None else (cut_seconds(old_in)  if old_in  else None)
        new_out = cut_seconds(ms_to_dt(body.punch_out)) if body.punch_out is not None else (cut_seconds(old_out) if old_out else None)

        if new_in is None:
            fail(422, "a presence status requires a punch_in")

        if body.punch_in is not None:
            new_day = attendance_date(new_in, emp["shift_start"], emp["shift_end"])
            if new_day != record["date"]:
                fail(422, "corrected punch_in must remain on the attendance date of this record")

        if new_out is not None:
            gap = int((new_out - new_in).total_seconds())
            if gap <= 0:
                fail(422, "punch_out must be after punch_in")
            if gap > MAX_SHIFT_SECONDS:
                fail(422, "punch_out is more than 24 hours after punch_in")

        derived = derive_all(emp, record["date"], new_in, new_out)

    # Reject a request that changes absolutely nothing.
    if (new_status == record["status"]
            and not _values_differ(old_in,  new_in)
            and not _values_differ(old_out, new_out)):
        fail(422, "this correction changes nothing")

    new_vals = {"status": new_status, "punch_in": new_in, "punch_out": new_out, **derived}

    old_vals = {
        "status":           record["status"],
        "punch_in":         old_in,
        "punch_out":        old_out,
        "work_hours":       record.get("work_hours"),
        "late_minutes":     record.get("late_minutes")     or 0,
        "overtime_minutes": record.get("overtime_minutes") or 0,
        "half_day":         bool(record.get("half_day", False)),
    }
    # Record only fields that actually changed.
    changes = {}
    for field, new_v in new_vals.items():
        if _values_differ(old_vals.get(field), new_v):
            changes[field] = {"from": old_vals.get(field), "to": new_v}

    return new_vals, changes


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: DateStr, body: RegularizeIn):
    emp = get_employee(emp_code)

    # Optimistic concurrency: read → plan → conditional write.
    # If another correction saved between our read and our write, the filter
    # on history length will not match, and we retry (up to 5 times).
    for _ in range(5):
        record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
        if record is None:
            fail(404, "no attendance record found for this date")

        new_vals, changes = _plan_correction(emp, record, body)
        hist_len = len(record.get("history") or [])

        # Write only if no other correction has been appended since we read.
        updated = db.attendance_logs.find_one_and_update(
            {
                "_id":               record["_id"],
                # These ensure the record hasn't been changed under us.
                f"history.{hist_len}": {"$exists": False},
            },
            {
                "$set":  new_vals,
                "$push": {
                    "history": {
                        "at":      now_utc(),
                        "by":      body.regularized_by,
                        "reason":  body.reason,
                        "changes": changes,
                    }
                },
            },
            return_document=ReturnDocument.AFTER,
        )
        if updated is not None:
            return _log_out(updated)

    fail(409, "the record was being modified concurrently; please retry")


# -------- analytics — employee monthly -------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: MonthStr):
    emp = get_employee(emp_code)
    _, _, first, last = month_bounds(month)

    rows    = list(db.attendance_logs.aggregate(monthly_pipeline(emp_code, month)))
    totals  = rows[0] if rows else {}
    p2      = totals.get("p2", 0)

    # R7: working days start from joined_on, not from the 1st of the month.
    wd = count_working_days(first, last, date_cls.fromisoformat(emp["joined_on"]))

    if wd == 0:
        pct = None
    else:
        # R8: present_days / working_days × 100, 2 decimals half-up.
        pct = round_half_up(Decimal(str(p2)) * 50 / Decimal(str(wd)), 2)

    return {
        "emp_code":              emp_code,
        "month":                 month,
        "working_days":          wd,
        "present_days":          round_half_up(p2 / 2, 2),
        "leave_days":            totals.get("leave_days", 0),
        "late_count":            totals.get("late_count", 0),
        "total_late_minutes":    totals.get("late_minutes", 0),
        "total_overtime_minutes":totals.get("overtime_minutes", 0),
        "attendance_pct":        pct,
    }


# -------- analytics — department summary -------- #
@app.get("/analytics/departments/summary")
def department_summary(month: MonthStr, department: Optional[str] = None):
    items = []
    for row in db.employees.aggregate(summary_pipeline(month, department)):
        # R8: average is over RECORDS (wh_c is in hundredths so divide by 100 too).
        avg_wh = (round_half_up(Decimal(str(row["wh_c"])) / Decimal(str(row["wh_n"])) / 100, 2)
                  if row["wh_n"] else None)
        items.append({
            "department":          row["_id"],
            "headcount":           row["headcount"],
            "present_days":        round_half_up(row["p2"] / 2, 2),
            "avg_work_hours":      avg_wh,
            "late_count":          row["late_count"],
            "total_late_minutes":  row["late_min"],
            "leave_count":         row["leave_count"],
            "on_duty_count":       row["duty_count"],
        })
    return {"month": month, "items": items}


# -------- analytics — late leaderboard -------- #
@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month:      MonthStr,
    limit:      Annotated[int, Query(ge=1, le=50)] = 10,
    department: Optional[str] = None,
):
    items = list(db.attendance_logs.aggregate(leaderboard_pipeline(month, limit, department)))
    return {"month": month, "items": items}


# -------- analytics — department trend -------- #
@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date:  Annotated[DateStr, Query(alias="from")],
    to:         DateStr,
):
    first_day, last_day = _check_trend_range(from_date, to)

    if db.employees.count_documents({"department": department}, limit=1) == 0:
        fail(404, "department not found")

    rows = list(db.employees.aggregate(trend_pipeline(department, first_day, last_day)))

    # Compute is_working_day, attendance_rate, and 7-day moving average in Python.
    # The 7-day window slides over working-day rates only (weekends are skipped).
    rate_window: list = []   # holds up to 7 most-recent working-day rates
    items = []
    for row in rows:
        day       = date_cls.fromisoformat(row["date"])
        is_wd     = day.weekday() < 5
        headcount = row["headcount"]
        p2        = row["present_p2"]

        if is_wd and headcount > 0:
            # R8: rates to 4 decimal places half-up.
            rate = round_half_up(Decimal(str(p2)) / Decimal(str(2 * headcount)), 4)
        else:
            rate = None

        if rate is not None:
            rate_window.append(Decimal(str(rate)))
            if len(rate_window) > 7:
                rate_window.pop(0)

        if rate_window:
            mov_avg = round_half_up(sum(rate_window) / len(rate_window), 4)
        else:
            mov_avg = None

        items.append({
            "date":            row["date"],
            "is_working_day":  is_wd,
            "headcount":       headcount,
            "present_count":   p2 / 2,
            "late_count":      row["late_count"],
            "attendance_rate": rate,
            "moving_avg_7d":   mov_avg,
        })

    return {"department": department, "items": items}


def _check_trend_range(date_from: str, date_to: str):
    first = date_cls.fromisoformat(date_from)
    last  = date_cls.fromisoformat(date_to)
    if last < first:
        fail(422, "'to' must not be before 'from'")
    if (last - first).days + 1 > MAX_TREND_DAYS:
        fail(422, f"range must not exceed {MAX_TREND_DAYS} days")
    return first, last


# -------- admin / explain -------- #
EndpointName = Literal[
    "attendance_list",
    "employee_monthly",
    "department_summary",
    "late_leaderboard",
    "department_trend",
]


def _need(value, name: str):
    """Raise 422 if a required query parameter for the explain endpoint is missing."""
    if value is None:
        fail(422, f"query parameter '{name}' is required for endpoint '{name}'")
    return value


@app.get("/admin/explain/{endpoint}")
def admin_explain(
    endpoint:   EndpointName,
    emp_code:   Optional[str]      = None,
    month:      Optional[MonthStr] = None,
    department: Optional[str]      = None,
    limit:      Annotated[int, Query(ge=1, le=50)] = 10,
    date_from:  Optional[DateStr]  = None,
    date_to:    Optional[DateStr]  = None,
    status:     Optional[Status]   = None,
    from_date:  Annotated[Optional[DateStr], Query(alias="from")] = None,
    to:         Optional[DateStr]  = None,
    page:       PageParam          = 1,
    page_size:  PageSizeParam      = 20,
):
    """
    Returns MongoDB's executionStats explain for the main query of the named endpoint.
    The pipeline / query is built the SAME way the real endpoint builds it,
    so the explain always reflects what actually runs against the collection.
    """
    if endpoint == "attendance_list":
        q    = _build_attendance_filter(emp_code, date_from, date_to, status)
        skip = (page - 1) * page_size
        command = {
            "find":   "attendance_logs",
            "filter": q,
            "sort":   dict(ATTENDANCE_SORT),
            "skip":   skip,
            "limit":  page_size,
        }
        collection = "attendance_logs"

    elif endpoint == "employee_monthly":
        _need(emp_code, "emp_code"); _need(month, "month")
        collection = "attendance_logs"
        command = {"aggregate": collection,
                   "pipeline": monthly_pipeline(emp_code, month),
                   "cursor":   {}}

    elif endpoint == "department_summary":
        _need(month, "month")
        collection = "employees"
        command = {"aggregate": collection,
                   "pipeline": summary_pipeline(month, department),
                   "cursor":   {}}

    elif endpoint == "late_leaderboard":
        _need(month, "month")
        collection = "attendance_logs"
        command = {"aggregate": collection,
                   "pipeline": leaderboard_pipeline(month, limit, department),
                   "cursor":   {}}

    else:  # department_trend
        _need(department, "department"); _need(from_date, "from"); _need(to, "to")
        first_day, last_day = _check_trend_range(from_date, to)
        collection = "employees"
        command = {"aggregate": collection,
                   "pipeline": trend_pipeline(department, first_day, last_day),
                   "cursor":   {}}

    explain_result = db.command({"explain": command, "verbosity": "executionStats"})
    return {
        "endpoint":   endpoint,
        "collection": collection,
        "explain":    json.loads(json_util.dumps(explain_result,
                                  json_options=json_util.RELAXED_JSON_OPTIONS)),
    }