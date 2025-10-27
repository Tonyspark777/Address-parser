import re
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UK Address Cleaner", layout="wide")

# Optional logo
try:
    st.image("RAND Main Logo (1).png", width=150)
except Exception:
    pass

st.title("Address Parsing Tool")
st.write(
    "Upload a CSV/Excel containing messy address strings and get clean fields: "
    "**building_number, address line 1–4, town_or_city, postcode, block**."
)

# -------------------------
# Parsing utilities
# -------------------------
POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b", re.I)

# Expanded street types (includes PARK/WOODLANDS/RISE/RYE/DALE, etc.)
STREET_TYPES = (
    "ROAD|STREET|AVENUE|CLOSE|LANE|GARDEN|GARDENS|COURT|DRIVE|WAY|CRESCENT|PLACE|HILL|"
    "TERRACE|MEWS|ROW|SQUARE|GROVE|VILLAS|VIEW|PARADE|ARCADE|WALK|WHARF|QUAY|"
    "HIGHWAY|BYWAY|RIDGE|BANK|GREEN|LINK|END|VALE|SIDE|MARKET|CIRCUS|"
    "PARK|PARKWAY|BOULEVARD|CAUSEWAY|COMMON|FIELDS?|GATE|BRAE|"
    "WOODLANDS|RISE|RYE|DALE"
)

# House number token (optional; supports ranges & joins: 90-92, 24 & 26, 1/3, 5,7, 107A)
HOUSE_TOKEN = r"(?:\d+[A-Z]?)(?:\s*(?:&|AND|/|,|-)\s*\d+[A-Z]?)*"

# Allow punctuation inside street names: apostrophes, dots, hyphens, slashes, ampersands, parentheses
STREET_BODY = r"[A-Z0-9\s'’\.\-/&()]+?"

# House number OPTIONAL; still require a known street type at end.
ADDR_PATTERN = re.compile(
    rf"(?:^|\s)({HOUSE_TOKEN})?\s+({STREET_BODY}(?:{STREET_TYPES}))\b"
)

# Capture blocks like "Flats A-C", "Flats 1 & 2", "Units 3-5", "Block C"
BLOCK_PATTERN = re.compile(
    r"\b(?:FLATS?|APTS?|APARTMENTS?|UNITS?|UNIT|BLOCKS?|BLOCK|BLK)\b[^,)]{0,120}",
    re.I,
)

LONDON_AREAS = {"E", "EC", "N", "NW", "SE", "SW", "W", "WC"}


def normalise_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def titlecase_street(s: str) -> str:
    if not s:
        return s
    s = s.title()
    s = re.sub(r"\bAnd\b", "and", s)
    s = re.sub(r"\bO'(\w)", lambda m: f"O'{m.group(1).upper()}", s)
    return s


def extract_postcode(s: str):
    """Return the LAST postcode in the string, and index span if found."""
    last = None
    if not s:
        return None, None
    for m in POSTCODE_RE.finditer(s):
        last = m
    if last:
        return last.group(0).upper(), (last.start(), last.end())
    return None, None


def postcode_area(pc: str | None) -> str | None:
    if not pc:
        return None
    m = re.match(r"([A-Z]{1,2})\d", pc, re.I)
    return m.group(1).upper() if m else None


def is_london_postcode(pc: str | None) -> bool:
    area = postcode_area(pc)
    return bool(area and area in LONDON_AREAS)


# ---------- Flats / block parsing ----------

def extract_flat_tokens(text: str) -> dict:
    """
    Extract flat/unit tokens from phrases like:
      - 'Flats A to C'  -> {'display':'A - C', 'tokens':['A','C'], 'type':'range_letters'}
      - 'Flats 1 & 2'   -> {'display':'1 & 2', 'tokens':['1','2'], 'type':'list_numbers'}
      - 'Flats 107 & 107A' -> {'display':'107 & 107A', 'tokens':['107','107A'], 'type':'list_mixed'}
    Returns {} if nothing found.
    """
    m = re.search(r"\b(FLATS?|UNITS?|FLAT|UNIT)\b\s*(.+?)\b(?=(?:\d|\(|$))", text, re.I)
    if not m:
        return {}

    tail = m.group(2).strip()

    # Normalise connectors: "to" -> "-", tidy &
    tail_norm = re.sub(r"\s*to\s*", "-", tail, flags=re.I)  # "A to C" -> "A-C"
    tail_norm = re.sub(r"\s*&\s*", " & ", tail_norm)
    tail_norm = re.sub(r"\s*,\s*", ",", tail_norm)

    # Simple range A-C / 1-3
    mrange = re.match(r"^([A-Z]|\d+)\s*-\s*([A-Z]|\d+)$", tail_norm, re.I)
    if mrange:
        a, b = mrange.group(1).upper(), mrange.group(2).upper()
        ttype = "range_letters" if (len(a) == 1 and len(b) == 1 and a.isalpha() and b.isalpha()) else "range_numbers"
        return {"display": f"{a} - {b}", "tokens": [a, b], "type": ttype}

    # Otherwise split on & / , / /
    parts = re.split(r"\s*(?:&|/|,)\s*", tail_norm)
    parts = [p.strip().upper() for p in parts if p.strip()]
    if not parts:
        return {}

    # Classify type
    if all(re.match(r"^\d+$", p) for p in parts):
        ttype = "list_numbers"
    elif all(re.match(r"^[A-Z]\w*$", p) for p in parts):
        ttype = "list_letters"
    else:
        ttype = "list_mixed"

    display = " & ".join(parts)
    return {"display": display, "tokens": parts, "type": ttype}


def normalise_block_and_unit(block_raw: str | None) -> tuple[str | None, dict]:
    """
    Return (block_string, flat_info_dict)
    flat_info_dict = {} if no flats; otherwise dict from extract_flat_tokens()
    """
    if not block_raw:
        return None, {}

    # Keep only the block phrase (trim when street digits start)
    head = re.split(r"(?=\d)", block_raw)[0]
    b = head.strip().strip(") ,;-\"'")
    b = re.sub(r"\s*[:#-]\s*", " ", b)
    b = re.sub(r"\bBlk\b", "Block", b, flags=re.I)

    flat_info = extract_flat_tokens(b)

    # Build a nice block string
    block_str = b
    if flat_info:
        disp = flat_info["display"]
        prefix = "Flats" if ("range" in flat_info["type"] or len(flat_info["tokens"]) > 1) else "Flat"
        block_str = f"{prefix} {disp}"

    return block_str.title(), flat_info


# ---------- locality split ----------

def split_locality_to_lines(locality: str | None) -> tuple[str, str, str]:
    """Split leftover locality into up to 3 lines for address line 2–4."""
    if not locality:
        return "", "", ""
    parts = re.split(r"\s{2,}|,|;|/|\s-\s", locality)
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        parts = [locality.strip()]
    parts = [p.title() for p in parts]
    p1 = parts[0] if len(parts) >= 1 else ""
    p2 = parts[1] if len(parts) >= 2 else ""
    p3 = parts[2] if len(parts) >= 3 else ""
    return p1, p2, p3


# ---------- main parser ----------

def parse_address(full: str):
    s = normalise_spaces(full)

    # 1) Postcode anchor (last one)
    postcode, span = extract_postcode(s)
    s2 = s[: span[0]].strip() if span else s

    # 2) Block info & flat tokens
    block = None
    flat_info = {}
    mblock = BLOCK_PATTERN.search(s)
    if mblock:
        block, flat_info = normalise_block_and_unit(mblock.group(0))

    # 3) House number + street (use LAST match in the string)
    m2 = None
    for m2 in ADDR_PATTERN.finditer(s2.upper()):
        pass

    if m2:
        house = (m2.group(1) or "").strip()
        street_raw = m2.group(2).strip()

        # If the matched street accidentally starts with a preceding flat-clause number
        # (e.g., "1 & 2) 19 BARGERY ROAD"), strip that leading HOUSE_TOKEN + punctuation.
        street_raw = re.sub(rf"^[\s,)\]]*{HOUSE_TOKEN}\s+", "", street_raw)

        # Tidy duplicated numbers like "35 35 ..."
        house = re.sub(r"^(\d+[A-Z]?)\s+\1\b", r"\1", house)

        street_tc = titlecase_street(street_raw)
        address_line_1 = (f"{house} {street_tc}".strip() if house else street_tc)
        remainder = s2.upper()[m2.end():].strip(" ,")
    else:
        house = ""
        address_line_1 = ""
        remainder = s2.upper()

    # 4) Locality leftovers -> address line 2–4
    remainder = re.sub(r"[^A-Z\s'’\.\-/&()]", " ", remainder)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    locality = remainder.title() if remainder else ""
    line2, line3, line4 = split_locality_to_lines(locality)

    # 5) Town/City rule
    town_or_city = "London" if is_london_postcode(postcode) else (line2 or "")

    # 6) Decide building_number using flats vs house heuristics
    def flats_match_house(house_num: str, flats: dict) -> bool:
        if not house_num or not flats:
            return False
        base = re.match(r"^\d+", house_num)
        if not base:
            return False
        root = base.group(0)
        return all(re.match(rf"^{root}\b", t) for t in flats.get("tokens", []))

    if flat_info and not flats_match_house(house, flat_info):
        building_number = flat_info["display"]  # e.g., "A - C" or "1 & 2"
    else:
        building_number = house  # e.g., "107" when flats are "107 & 107A"

    out = {
        "building_number": building_number,
        "address line 1": address_line_1,
        "address line 2": line2,
        "address line 3": line3,
        "address line 4": line4,
        "town_or_city": town_or_city,
        "postcode": (postcode or ""),
        "block": (block or ""),
    }
    return out


# -------------------------
# Sidebar (load options)
# -------------------------
with st.sidebar:
    st.header("Options")
    sample_toggle = st.toggle("Use sample data", value=False)
    st.markdown(
        "**Tip**: If your data also has a postcode column, you can select it — "
        "the parser will cross-check/override from the messy string if needed."
    )

# -------------------------
# Load data
# -------------------------
@st.cache_data(show_spinner=False)
def load_file(upload) -> pd.DataFrame:
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)

if sample_toggle:
    df = pd.DataFrame(
        [
            {"address": "BROCKLEY ROAD(141), BROCKLEY, SE4 Flats A to C 141 Upper Brockley Road Brockley  SE4 1TF"},
            {"address": "Arngask Rd, SE6 (107, Flats 107 & 107A) 107 ARNGASK ROAD  CATFORD  SE6 1XZ"},
            {"address": "Bargery Road, SE6 (19, Flat 1 & 2) 19 BARGERY ROAD  CATFORD  SE6 2LJ"},
            {"address": "Lewisham Park, SE13 (24) 24 LEWISHAM PARK  LEWISHAM  SE13 6QZ"},
        ]
    )
else:
    upload = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if upload is not None:
        df = load_file(upload)
    else:
        df = None

if df is None:
    st.info("Upload a file or turn on 'Use sample data' in the sidebar.")
    st.stop()

st.subheader("1) Choose column(s)")
cols = list(df.columns)
address_col = st.selectbox(
    "Address text column",
    options=cols,
    index=(cols.index("address") if "address" in cols else 0),
)

postcode_col = st.selectbox(
    "(Optional) Separate postcode column",
    options=["<none>"] + cols,
    index=0,
)

run = st.button("Clean addresses", type="primary")

if run:
    work = df.copy()

    # Build robust full string for parsing; append postcode from separate col if needed
    full_strings = work[address_col].astype(str)
    if postcode_col != "<none>":
        pc_series = work[postcode_col].fillna("").astype(str)
        need_pc = ~full_strings.str.contains(POSTCODE_RE)
        full_strings = full_strings.where(~need_pc, (full_strings + " " + pc_series).str.strip())

    parsed = full_strings.apply(parse_address).apply(pd.Series)

    # If explicit postcode col exists, prefer it when parsed is empty
    if postcode_col != "<none>":
        parsed["postcode"] = parsed["postcode"].where(parsed["postcode"].ne(""), work[postcode_col].fillna(""))

    # ---- prevent duplicate columns; keep original messy address for traceability
    original_cols = work.columns.tolist()
    clashes = {c: f"original_{c}" for c in original_cols if c in parsed.columns}
    work_safe = work.rename(columns=clashes)
    if address_col in work_safe.columns:
        work_safe = work_safe.rename(columns={address_col: "original_address"})

    # Final output
    output_cols = [
        "building_number",
        "address line 1",
        "address line 2",
        "address line 3",
        "address line 4",
        "town_or_city",
        "postcode",
        "block",
    ]
    output = pd.concat([work_safe, parsed[output_cols]], axis=1)

    # Defensive: drop any accidental dups
    if output.columns.duplicated().any():
        output = output.loc[:, ~output.columns.duplicated()].copy()

    st.subheader("2) Results")
    st.dataframe(output, use_container_width=True)

    # Downloads: CSV and Excel
    csv_bytes = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="cleaned_addresses.csv",
        mime="text/csv",
    )

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        output.to_excel(writer, sheet_name="Cleaned", index=False)
    st.download_button(
        "Download Excel",
        data=bio.getvalue(),
        file_name="cleaned_addresses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption(
        "Heuristics: last postcode anchor; last 'house number + street type' "
        "(house optional; ranges & joins allowed; punctuation allowed in street names). "
        "Block phrases parsed (e.g., 'Flats A to C' → building_number 'A - C', block 'Flats A - C'; "
        "'Flats 107 & 107A' keeps building_number '107'). "
        "Locality → address line 2–4; town_or_city='London' for London postcodes. "
        "Original input kept as 'original_address'."
    )
else:
    st.info("Set your columns and press **Clean addresses** to parse.")
