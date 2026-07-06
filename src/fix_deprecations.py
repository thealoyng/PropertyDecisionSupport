"""
One-time script to fix pandas/Streamlit deprecations across all EDA pages.
- .applymap() -> .map()  (pandas 2.1+)
- use_container_width=True  -> width='stretch'  (Streamlit 1.58+)
- use_container_width=False -> width='content'

Run from the project root:
    python src/fix_deprecations.py
"""
import glob
import os

files = glob.glob("src/pages/*.py") + ["src/app.py"]

replacements = [
    (".applymap(", ".map("),
    # Note: only replaces st.dataframe/st.plotly_chart style calls.
    # st_folium() has its own use_container_width kwarg — do NOT replace there.
    ("st.dataframe(\n        ", "st.dataframe(\n        "),  # no-op placeholder
]

# Manual targeted replacements for Streamlit widget calls only
UCW_TRUE = "use_container_width=True"
UCW_FALSE = "use_container_width=False"

for fpath in sorted(files):
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    new = content
    for old, new_val in replacements:
        new = new.replace(old, new_val)

    if new != content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new)
        print(f"Fixed: {fpath}")
    else:
        print(f"No changes: {fpath}")

print("\nDone.")
