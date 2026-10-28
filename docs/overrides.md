# Overrides: recording human decisions

The package proposes; a person may decide otherwise on an UNCERTAIN day, or on any day.
The decision is recorded in a table, not applied to the code, so the automatic proposal and
the human choice sit side by side and raw readings are never altered.

## The override table

One row per site-day a reviewer touched; untouched days stay automatic.

| column | type | meaning |
|---|---|---|
| `site` | string | as in the site-day table |
| `date` | string | YYYY-MM-DD |
| `choice` | string | `accept` (apply the proposed window), `keep` (leave the day alone), `manual` (apply the window below) |
| `manual_start` | int | first slot to flip when `choice = manual`, 24 to 71 |
| `manual_end` | int | last slot to flip, inclusive, `manual_start` to 71 |
| `reviewer` | string | who decided |
| `reason` | string | free text |
| `recorded_at` | string | ISO 8601 timestamp of the decision |

A reviewer typically fills it in Excel or Power BI from the site-day table (which carries the
window, the probability, the runner-up and the MWh at stake) and the day's picture.

## Applying it

Building the effective interval series from the site-day table plus this override table is a
small, mechanical step (validate one row per site-day, a manual window inside the scan
range, then flip the chosen slots) that the calling pipeline can implement today. A
packaged `apply_overrides` is planned for a later release once the workflow settles; the
schema above is fixed so that tables written now stay valid.
