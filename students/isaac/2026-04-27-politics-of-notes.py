import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import seaborn as sns

    return Path, mo, pl, sns


@app.cell
def _(Path):
    DATA_DIR = Path("..") / ".." / "data"
    return (DATA_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read
    """)
    return


@app.cell
def _(DATA_DIR, pl):
    raw_notes = pl.read_parquet(DATA_DIR / "intermediate" / "notes_enriched.parquet")
    return (raw_notes,)


@app.cell
def _(pl):
    raw_post_creation_times = (
        pl.read_csv("../../data/metadata/post_created_at.csv", 
                    schema_overrides={"post_id": pl.String, 
                                      "post_created_at": pl.Datetime(time_zone="UTC")})
    )
    return (raw_post_creation_times,)


@app.cell
def _(pl):
    raw_statuses = pl.read_csv(
        "../../data/metadata/note_status_records.csv", 
        schema_overrides={
            "note_id": pl.String, 
            "status_time": pl.Datetime(time_zone="UTC"),
            "status": pl.Enum([
                "CURRENTLY_RATED_NOT_HELPFUL",
                "NEEDS_MORE_RATINGS",
                "CURRENTLY_RATED_HELPFUL",
            ]),
            "event_type": pl.Enum([
              "NoteCreated",
              "MostRecentStatusChange",
              "FirstNonNMRStatus",
              "LatestNonNMRStatus",
              "CurrentStatus",
              "FirstNmrDueToMinStableCrhTime",
              "NmrDueToMinStableCrhTime",
              "RetroLock",
              "StatusLock",
            ]),
        }
    )
    return (raw_statuses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Process
    """)
    return


@app.cell
def _(raw_statuses):
    # Make sure there aren't any events at the same time but different statuses
    n_status_dupes = raw_statuses.select("note_id", "status_time", "status").is_duplicated().len()
    n_time_dupes = raw_statuses.select("note_id", "status_time").is_duplicated().len()

    assert n_status_dupes == n_time_dupes
    return


@app.cell
def _(pl, raw_statuses):
    statuses = (
        raw_statuses
        .sort("note_id", "status_time", "event_type")
        .group_by("note_id").agg(
            min_status=pl.col("status").min(),
            max_status=pl.col("status").max(),
            first_crh=pl.col("status_time").filter(pl.col("status") == "CURRENTLY_RATED_HELPFUL").first(),
            first_crnh=pl.col("status_time").filter(pl.col("status") == "CURRENTLY_RATED_NOT_HELPFUL").first(),
            first_status=pl.col("status").first(),
            first_event=pl.col("event_type").first(),
            final_status=pl.col("status").last(),
            final_event=pl.col("event_type").last(),
            status_count=pl.len(),
        )
    )
    return (statuses,)


@app.cell
def _(raw_notes, statuses):
    notes = (
        raw_notes
        .join(
            statuses.select("note_id", "first_crnh", "first_crh", "final_status", "min_status", "max_status"), 
            left_on="noteId", right_on="note_id", how="left", validate="m:1", coalesce=True)
        .sort("tweet_author_party", "calendarMonth", "max_status")

    )
    return (notes,)


@app.cell
def _(pl, raw_post_creation_times):
    post_creation_times = (
        raw_post_creation_times
        .with_columns(postCreatedAtMonth=pl.col("post_created_at").dt.strftime("%Y-%m"))
    )
    return (post_creation_times,)


@app.cell
def _(notes, pl, post_creation_times):
    # NB: Only calculating statuses amongst notes that classify the posts as misleading
    posts = (
        notes
        .group_by("tweetId", "tweet_author_party", "tweet_lang")
        .agg(
            max_msld_status = pl.col("final_status").filter(
                pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING"
            ).max(),
            min_msld_status = pl.col("final_status").filter(
                pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING"
            ).min(),
        )
        .join(post_creation_times, left_on="tweetId", right_on="post_id", how="left", validate="m:1")

    )

    assert not posts.select("tweetId").is_duplicated().sum()
    return (posts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NNN
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TIL that only 2 NNN notes had a CRH status as of 2026/02/03
    """)
    return


@app.cell
def _(notes):
    notes.group_by("classification", "noteFinalRatingStatus").len().sort("classification", "noteFinalRatingStatus")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is also visible if you look directly at the data downloads.
    """)
    return


@app.cell
def _(DATA_DIR, pl):
    _statuses_downloaded = pl.read_parquet(DATA_DIR / "2026-02-03" / "noteStatusHistory.parquet")


    _notes_downloaded = (
        pl.read_parquet(DATA_DIR / "2026-02-03" / "notes.parquet")
        .join(
            _statuses_downloaded, on="noteId", how="left", validate="1:1"
        )
    )


    _notes_downloaded.group_by("classification", "currentStatus").len().sort("classification", "currentStatus")
    return


@app.cell
def _(notes, pl):
    (
        notes
        .filter(pl.col("summary").str.to_lowercase().str.contains("nnn"))
        .select("summary", "classification")
    )["classification"].value_counts()
    return


@app.cell
def _(notes, pl):
    (
        notes
        .filter(pl.col("summary").str.to_lowercase().str.contains("nnn"))
        .filter(pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING")
        .select("summary", "classification", "noteFinalRatingStatus")
    )["noteFinalRatingStatus"].value_counts()
    return


@app.cell
def _(notes, pl):
    (
        notes
        .filter(pl.col("summary").str.to_lowercase().str.contains("nnn"))
        .filter(pl.col("noteFinalRatingStatus") == "CURRENTLY_RATED_HELPFUL")
        .select("summary", "classification", "tweet_lang")
        ["tweet_lang"]
        .value_counts(normalize=True)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note Level
    """)
    return


@app.cell
def _():
    parties = ["democrat", "republican" ] # "unknown","NULL"]
    status_values = ["NULL", "CURRENTLY_RATED_HELPFUL", "NEEDS_MORE_RATINGS","CURRENTLY_RATED_NOT_HELPFUL"]
    return parties, status_values


@app.cell
def _(notes, pl):
    notes_to_plot = (
        notes
        .fill_null("NULL")
        .filter(pl.col("tweet_lang") == "en")
        .filter(pl.col("tweet_author_party").is_in(["democrat", "republican"]))
        .filter(pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING")
    )
    return (notes_to_plot,)


@app.cell
def _(notes_to_plot, parties, sns, status_values):
    _status_palette = {
        "CURRENTLY_RATED_NOT_HELPFUL": "tab:red",
        "CURRENTLY_RATED_HELPFUL": "tab:green",
        "NEEDS_MORE_RATINGS": "tab:gray",
        "NULL": "lightgray",
    }

    _hist_params = {
        "x": "calendarMonth",
        "hue": "final_status",
        "hue_order": status_values,
        "palette": _status_palette,
        "stat": "count",
        "multiple": "fill",
        "discrete": True,
        "shrink": 0.8,
    }

    _facet = (
        sns.FacetGrid(
            data=notes_to_plot, 
            col="tweet_author_party", 
            col_order=parties,
            height=4, aspect=1.5, col_wrap=2, sharey=True, 
        )
        .map_dataframe(sns.histplot, **_hist_params)
        .add_legend()
    )

    _facet
    return


@app.cell
def _(notes_to_plot, sns):
    _party_palette = {
        "democrat": "tab:red",
        "republican": "tab:blue",
        "unknown": "tab:gray",
        "NULL": "lightgray",
    }

    _hist_params = {
        "x": "calendarMonth",
        "hue": "tweet_author_party",
        "hue_order": ["democrat", "republican", "unknown", "NULL"],
        "palette": _party_palette,
        "stat": "count",
        "multiple": "fill",
        "discrete": True,
        "shrink":1,
        "edgecolor": None,
    }

    _facet = (
        sns.FacetGrid(
            data=notes_to_plot, 
            col="final_status", 
            col_order=[
                "CURRENTLY_RATED_HELPFUL",
                "CURRENTLY_RATED_NOT_HELPFUL",
                "NEEDS_MORE_RATINGS",
                "NULL",
            ],
            height=4,
            aspect=1.5,
            col_wrap=2,
            sharey=True,
        )
        .map_dataframe(sns.histplot, **_hist_params)
    )


    # add vertical lines at each January
    for _ax in _facet.axes.flat:
        _ax.tick_params(axis="x", labelbottom=True)
        _ticks = _ax.get_xticks()
        _labels = [t.get_text() for t in _ax.get_xticklabels()]
        for _x, _lab in zip(_ticks, _labels):
            if _lab.endswith("-01"):
                _ax.axvline(x=_x, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=10)


    # get ordered unique _months
    _months = sorted(notes_to_plot["calendarMonth"].unique())

    # keep only Januarys
    _year_ticks = [m for m in _months if m.endswith("-01")]
    _year_positions = [_months.index(m) for m in _year_ticks]

    for _ax in _facet.axes.flat:
        _ax.set_xticks(_year_positions)
        _ax.set_xticklabels(_year_ticks, rotation=20, ha="right")

    # keep _ticks only on bottom row

    for _i, _ax in enumerate(_facet.axes.flat):
        if _i < len(_facet.axes.flat) - 2:  # not bottom row
            _ax.tick_params(axis="x", labelbottom=False)

    _facet
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Post Level
    """)
    return


@app.cell
def _(pl, posts):
    posts_to_plot = (
        posts
        .fill_null("NULL")
        .filter(pl.col("tweet_lang") == "en")
        .filter(pl.col("tweet_author_party").is_in(["democrat", "republican"]))
        .sort("post_created_at")
        .filter(pl.col("postCreatedAtMonth") >= "2022-01-01")
    )
    return (posts_to_plot,)


@app.cell
def _(parties, posts_to_plot, sns, status_values):
    _status_palette = {
        "CURRENTLY_RATED_NOT_HELPFUL": "tab:red",
        "CURRENTLY_RATED_HELPFUL": "tab:green",
        "NEEDS_MORE_RATINGS": "tab:gray",
        "NULL": "lightgray",
    }

    _hist_params = {
        "x": "postCreatedAtMonth",
        "hue": "max_msld_status",
        "hue_order": status_values,
        "palette": _status_palette,
        "stat": "count",
        "multiple": "stack",
        "discrete": True,
        "shrink": 0.8,
    }

    _facet = (
        sns.FacetGrid(
            data=posts_to_plot, 
            col="tweet_author_party", 
            col_order=parties,
            height=4, aspect=1.5, col_wrap=2, sharey=True, 
        )
        .map_dataframe(sns.histplot, **_hist_params)
        .add_legend()
    )

    _facet
    return


@app.cell
def _(notes_to_plot, posts_to_plot, sns):
    _party_palette = {
        "democrat": "tab:blue",
        "republican": "tab:red",
        "unknown": "tab:gray",
        "NULL": "lightgray",
    }

    _hist_params = {
        "x": "postCreatedAtMonth",
        "hue": "tweet_author_party",
        "hue_order": ["democrat", "republican", "unknown", "NULL"],
        "palette": _party_palette,
        "stat": "count",
        "multiple": "fill",
        "discrete": True,
        "shrink":1,
        "edgecolor": None,
    }

    _facet = (
        sns.FacetGrid(
            data=posts_to_plot, 
            col="max_msld_status",
            col_order=[
                "CURRENTLY_RATED_HELPFUL",
                "CURRENTLY_RATED_NOT_HELPFUL",
                "NEEDS_MORE_RATINGS",
                "NULL",
            ],
            height=4,
            aspect=1.5,
            col_wrap=2,
            sharey=False,
        )
        .map_dataframe(sns.histplot, **_hist_params)
    )


    # add vertical lines at each January
    for _ax in _facet.axes.flat:
        _ax.tick_params(axis="x", labelbottom=True)
        _ticks = _ax.get_xticks()
        _labels = [t.get_text() for t in _ax.get_xticklabels()]
        for _x, _lab in zip(_ticks, _labels):
            if _lab.endswith("-01"):
                _ax.axvline(x=_x, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=10)


    # get ordered unique _months
    _months = sorted(notes_to_plot["calendarMonth"].unique())

    # keep only Januarys
    _year_ticks = [m for m in _months if m.endswith("-01")]
    _year_positions = [_months.index(m) for m in _year_ticks]

    for _ax in _facet.axes.flat:
        _ax.set_xticks(_year_positions)
        _ax.set_xticklabels(_year_ticks, rotation=20, ha="right")

    # keep _ticks only on bottom row

    for _i, _ax in enumerate(_facet.axes.flat):
        if _i < len(_facet.axes.flat) - 2:  # not bottom row
            _ax.tick_params(axis="x", labelbottom=False)

    _facet
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
