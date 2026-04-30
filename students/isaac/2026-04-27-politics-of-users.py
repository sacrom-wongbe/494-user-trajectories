import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import marimo as mo
    import plotly.express as px
    from datetime import date
    import colorsys
    import seaborn as sns
    import altair as alt

    return Path, pl, plt, sns


@app.cell
def _(Path, all_activity_levels, apply_rules, pl):
    # NB: If you run locally, replace "user_...\_traj.parquet" with "sampled_user_...\_traj.parquet"

    # Load data
    data_dir = Path("../../data/") 

    user_months = (
        pl.read_parquet(data_dir / "output" / "user_month_traj.parquet")
        .with_columns(month_role=apply_rules(all_activity_levels))
    )
    return data_dir, user_months


@app.cell
def _(data_dir, pl):
    enriched_notes = (
        pl.read_parquet(data_dir/ "intermediate" / "notes_enriched.parquet")
        .with_columns(createdAtDt=pl.from_epoch(pl.col("createdAtMillis"), "ms").dt.replace_time_zone("UTC"))
        .with_columns(createdAtMonth=pl.col("createdAtDt").dt.strftime("%Y-%m"))
    )
    return


@app.cell
def _(pl):
    # NB: Order matters; first match takes precedence.
    writer_party_levels = [
        ("80-100% conservative", pl.col("pctPoliticalNotesRepAligned") >= 80),
        ("60-80% conservative", pl.col("pctPoliticalNotesRepAligned") >= 60),
        ("40-60% conservative", pl.col("pctPoliticalNotesRepAligned") >= 40),
        ("20-40% conservative", pl.col("pctPoliticalNotesRepAligned") >= 20),
        ("0-20% conservative", pl.col("pctPoliticalNotesRepAligned") >= 0),
    ]

    rater_party_levels = [
        ("80-100% conservative", pl.col("pctPoliticalRatingsRepAligned") >= 80),
        ("60-80% conservative", pl.col("pctPoliticalRatingsRepAligned") >= 60),
        ("40-60% conservative", pl.col("pctPoliticalRatingsRepAligned") >= 40),
        ("20-40% conservative", pl.col("pctPoliticalRatingsRepAligned") >= 20),
        ("0-20% conservative", pl.col("pctPoliticalRatingsRepAligned") >= 0),
    ]
    return rater_party_levels, writer_party_levels


@app.cell
def _(pl):
    # NB: Order matters; first match takes precedence.
    writing_activity_levels = [
        ("triple_digit_writer", pl.col("notesWritten") >= 100),
        ("double_digit_writer", pl.col("notesWritten") >= 10),
        ("single_digit_writer", pl.col("notesWritten") >= 2),
        ("single_note_writer", pl.col("notesWritten") >= 1),
    ]

    rating_activity_levels = [
        ("triple_digit_rater",  pl.col("notesRated") >= 100),
        ("double_digit_rater",  pl.col("notesRated") >= 10),
        ("single_digit_rater",   pl.col("notesRated") >= 2),
        ("single_note_rater",   pl.col("notesRated") >= 1),
    ]

    requesting_activity_levels = [
        ("triple_digit_requestor", pl.col("notesRequested") >= 100),
        ("double_digit_requestor", pl.col("notesRequested") >= 10),
        ("single_digit_requestor", pl.col("notesRequested") >= 2),
        ("single_note_requestor", pl.col("notesRequested") >= 1),
    ]

    all_activity_levels = (
        writing_activity_levels 
        + rating_activity_levels
        + requesting_activity_levels
    )

    return all_activity_levels, rating_activity_levels, writing_activity_levels


@app.cell
def _(pl):
    # Build the classification expression from rules
    def apply_rules(levels) -> pl.Expr:
        levels = levels + [("not_active", pl.lit(True))]
        # Apply rules in reverse order to ensure first match takes precedence
        expr = pl.lit(None, dtype=pl.String)
        for label, condition in reversed(levels):
            expr = pl.when(condition).then(pl.lit(label)).otherwise(expr)

        # Extract ordered labels from rules
        labels = [label for label, _ in levels]

        # Make the column an ordered categorical with the specified levels
        expr = expr.cast(pl.Enum(categories=labels))

        return expr

    return (apply_rules,)


@app.cell
def _(
    all_activity_levels,
    apply_rules,
    pl,
    rater_party_levels,
    rating_activity_levels,
    user_months,
    writer_party_levels,
    writing_activity_levels,
):
    min_month = user_months.select(pl.col("userMonth").min()).item()
    max_month = user_months.select(pl.col("userMonth").max()).item()

    _user_months_wide = (
        user_months
        .pivot(index="participantId", on="userMonth", values="month_role")
        .rename({f"{col}": f"month_{col}_role" for col in range(min_month, max_month + 1)})
        .fill_null("not_active")
    )

    users = (
        user_months
        .group_by("participantId")
        .agg(
            userFirstActiveMonth=pl.col("calendarMonth").min(),
            hitRate = pl.col("hits").sum() / pl.col("notesWritten").sum(),
            correctRatingsRate = 
                (pl.col("correctHelpfuls").sum() + pl.col("correctNotHelpfuls").sum())
                / pl.col("notesRated").sum(),
            notesWritten = pl.col("notesWritten").sum(),
            notesRated = pl.col("notesRated").sum(),
            notesRequested = pl.col("notesRequested").sum(),
            proRepRatings = pl.col("proRepRatings").sum(),
            antiRepRatings = pl.col("antiRepRatings").sum(),
            proDemRatings = pl.col("proDemRatings").sum(),
            antiDemRatings = pl.col("antiDemRatings").sum(),
            notesWrittenOnEnglishPosts = pl.col("notesWrittenOnEnglishPosts").sum(),
            notesRatedOnEnglishPosts = pl.col("notesRatedOnEnglishPosts").sum(),
            notesWrittenOnIdentifiedPosts = pl.col("notesWrittenOnIdentifiedPosts").sum(),
            notesRatedOnIdentifiedPosts = pl.col("notesRatedOnIdentifiedPosts").sum(),
            repAlignedRatings = pl.col("repAlignedRatings").sum(),
            demAlignedRatings = pl.col("demAlignedRatings").sum(),
            repAlignedNotes = pl.col("repAlignedNotes").sum(),
            demAlignedNotes = pl.col("demAlignedNotes").sum(),
            politicalNotesRated = pl.col("repAlignedRatings").sum() + pl.col("demAlignedRatings").sum(),
            politicalNotesWritten = pl.col("repAlignedNotes").sum() + pl.col("demAlignedNotes").sum(),
            pctPoliticalRatingsRepAligned = 
                pl.col("repAlignedRatings").sum() 
                / (pl.col("repAlignedRatings").sum() + pl.col("demAlignedRatings").sum())
                * 100,
            pctPoliticalNotesRepAligned =
                pl.col("repAlignedNotes").sum() 
                / (pl.col("repAlignedNotes").sum() + pl.col("demAlignedNotes").sum())
                * 100,
            pctPoliticalRatingsDemAligned =
                pl.col("demAlignedRatings").sum()
                / (pl.col("repAlignedRatings").sum() + pl.col("demAlignedRatings").sum())
                * 100,
            pctPoliticalNotesDemAligned =
                pl.col("demAlignedNotes").sum()
                / (pl.col("repAlignedNotes").sum() + pl.col("demAlignedNotes").sum())
                * 100,
        )
        .with_columns(
            (pl.selectors.starts_with("nMonths") / pl.col("nActiveMonths") * 100).name.map(lambda s: s.replace("nMonths", "pctActiveMonths")),
            total_role=apply_rules(all_activity_levels),
            writing_role=apply_rules(writing_activity_levels),
            rating_role=apply_rules(rating_activity_levels),
            rating_party=apply_rules(rater_party_levels),
            writing_party=apply_rules(writer_party_levels),
            hit_rate_bin=pl.col("hitRate").cut(breaks=[0.02, 0.04, 0.06, 0.08, 0.1, 0.12], labels=["0-2%", "2-4%", "4-6%", "6-8%", "8-10%", "10-12%", "12%+"])
        )
        .join(_user_months_wide, on="participantId", how="left")
    )
    return (users,)


@app.cell
def _(users):
    (
        users
        .select("notesRated", "notesRatedOnIdentifiedPosts", "notesRatedOnEnglishPosts", "politicalNotesRated",
                "pctPoliticalRatingsDemAligned", "pctPoliticalRatingsRepAligned",
                "notesWritten", "politicalNotesWritten",
                "pctPoliticalNotesDemAligned", "pctPoliticalNotesRepAligned",
                ).with_columns())
    return


@app.cell
def _(pl, sns, users):
    _var = "pctPoliticalRatingsDemAligned"

    _data = (
        users
        .filter(pl.col(_var).is_not_null())
        # .filter(pl.col("politicalNotesRated") >= 5)
    )

    sns.histplot(
        data= _data,
        x=_var,
        binwidth=10,
    )
    return


@app.cell
def _(pl, sns, users):
    _var = "pctPoliticalNotesDemAligned"

    _data = (
        users
        .filter(pl.col(_var).is_not_null())
        .filter(pl.col("politicalNotesWritten") >= 5)
    )

    sns.histplot(
        data= _data,
        x=_var,
        binwidth=10,
    )
    return


@app.cell
def _(pl, sns, users):
    _var = "pctPoliticalNotesDemAligned"

    _data = (
        users
        .filter(pl.col(_var).is_not_null())
        .filter(pl.col("politicalNotesWritten") >= 5)
    )

    sns.regplot(data=_data, x="pctPoliticalNotesDemAligned", y="hitRate", x_bins=range(0, 101, 10), fit_reg=False)
    return


@app.cell
def _(pl, sns, users):
    _var = "pctPoliticalRatingsDemAligned"

    _data = (
        users
        .filter(pl.col(_var).is_not_null())
        .filter(pl.col("politicalNotesRated") >= 10)
    )

    sns.regplot(data=_data, x="pctPoliticalRatingsDemAligned", y="correctRatingsRate", x_bins=range(0, 101, 10), fit_reg=False)
    return


@app.cell
def _(pl, users):
    users.sort("userFirstActiveMonth").filter(pl.col("politicalNotesRated") > 0)["rating_party"].value_counts()
    return


@app.cell
def _(pl, sns, users):
    colors = {
        "80-100% conservative": "#800000",  # Maroon
        "60-80% conservative": "#FF0000",    # Red
        "40-60% conservative": "#D3D3D3", # Light gray
        "20-40% conservative": "#ADD8E6", # Light blue
        "0-20% conservative": "#000080",  # Navy
        "not_active": "#808080",           # Gray"
    }

    sns.histplot(
        x="userFirstActiveMonth", 
        data = users.sort("userFirstActiveMonth").filter(pl.col("politicalNotesRated") > 10),
        hue = "rating_party",
        multiple="fill",
        palette=colors,
    )
    return (colors,)


@app.cell
def _(colors, pl, sns, users):
    sns.histplot(
        x="userFirstActiveMonth", 
        data = users.sort("userFirstActiveMonth").filter(pl.col("politicalNotesWritten") > 5),
        hue = "writing_party",
        multiple="fill",
        palette=colors,
    )
    return


@app.cell
def _(colors, pl, plt, sns, user_months, users):
    _data = (
        user_months
        .filter(pl.col("activeMonth"))
        .select("participantId", "calendarMonth", "notesRated", "notesWritten", "notesRequested")
        .join(
            users.select("participantId", "rating_party", "writing_party", "hitRate"), 
            on="participantId", how="left")
        .sort("calendarMonth")
        .filter(pl.col("hitRate") > 0.08)
        .filter(pl.col("notesWritten") > 0)
    )

    sns.histplot(
        x="calendarMonth", 
        data = _data,
        hue = "writing_party",
        multiple="fill",
        palette=colors,
    )

    for _y in [0.2, 0.4, 0.6, 0.8]:
        plt.axhline(_y, color="black", alpha=0.5)

    plt.show()
    return


@app.cell
def _(colors, pl, plt, sns, user_months, users):
    _data = (
        user_months
        .filter(pl.col("activeMonth"))
        .select("participantId", "calendarMonth", "notesRated", "notesWritten", "notesRequested")
        .join(
            users.select("participantId", "rating_party", "writing_party"), 
            on="participantId", how="left")
        .sort("calendarMonth")
        .filter(pl.col("notesRated") > 0)
    )

    sns.histplot(
        x="calendarMonth", 
        data = _data,
        hue = "rating_party",
        multiple="fill",
        palette=colors,
    )

    for _y in [0.2, 0.4, 0.6, 0.8]:
        plt.axhline(_y, color="black", alpha=0.5)

    plt.show()
    return


@app.cell
def _(pl, users):
    (
        users
        .filter(
            (pl.col("repAlignedRatings") + pl.col("demAlignedRatings"))
            > 5
        )
        .filter(pl.col("proRepRatings") > 10)
        .select("correctRatingsRate", "rating_party","notesWritten", "notesRated", "proRepRatings", "antiRepRatings", "proDemRatings", "antiDemRatings")
    )
    return


if __name__ == "__main__":
    app.run()
