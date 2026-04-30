import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path

    return Path, go, mo, pl, px


@app.cell
def _(Path, pl):
    base_dir_candidates = [
        Path("../Archive-2"),
        Path("Archive-2"),
        Path("../../Archive-2"),
    ]

    archive_dir = next((p for p in base_dir_candidates if p.exists()), base_dir_candidates[0])

    notes = (
        pl.read_parquet(archive_dir / "sample_user_note_traj.parquet")
        .select(["noteAuthorParticipantId", "userMonth", "calendarMonth", "notesCreated"])
        .rename(
            {
                "noteAuthorParticipantId": "participantId",
                "notesCreated": "notesWritten",
            }
        )
    )

    ratings = (
        pl.read_parquet(archive_dir / "sample_user_rating_traj.parquet")
        .select(["raterParticipantId", "userMonth", "calendarMonth", "notesRated"])
        .rename({"raterParticipantId": "participantId"})
    )

    requests = (
        pl.read_parquet(archive_dir / "sample_user_request_traj.parquet")
        .select(["requesterParticipantId", "userMonth", "calendarMonth", "requestsMade"])
        .rename(
            {
                "requesterParticipantId": "participantId",
                "requestsMade": "notesRequested",
            }
        )
    )
    return notes, ratings, requests


@app.cell
def _(pl):
    # Classifier definitions copied.
    total_activity_rules = [
        ("4_digit_writer", pl.col("notesWritten") >= 1000),
        ("triple_digit_writer", pl.col("notesWritten") >= 100),
        ("double_digit_writer", pl.col("notesWritten") >= 10),
        ("single_digit_writer", pl.col("notesWritten") >= 2),
        ("single_note_writer", pl.col("notesWritten") == 1),
        ("4_digit_rater", pl.col("notesRated") >= 1000),
        ("triple_digit_rater", pl.col("notesRated") >= 100),
        ("double_digit_rater", pl.col("notesRated") >= 10),
        ("single_digit_rater", pl.col("notesRated") >= 2),
        ("single_note_rater", pl.col("notesRated") == 1),
        ("4_digit_requestor", pl.col("notesRequested") >= 1000),
        ("triple_digit_requestor", pl.col("notesRequested") >= 100),
        ("double_digit_requestor", pl.col("notesRequested") >= 10),
        ("single_digit_requestor", pl.col("notesRequested") >= 2),
        ("single_post_requestor", pl.col("notesRequested") == 1),
        ("not_active", pl.lit(True)),
    ]

    month_activity_rules = [
        ("double_digit_writer", pl.col("notesWritten") >= 10),
        ("single_digit_writer", pl.col("notesWritten") >= 2),
        ("single_note_writer", pl.col("notesWritten") == 1),
        ("double_digit_rater", pl.col("notesRated") >= 10),
        ("single_digit_rater", pl.col("notesRated") >= 2),
        ("single_note_rater", pl.col("notesRated") == 1),
        ("double_digit_requestor", pl.col("notesRequested") >= 10),
        ("single_digit_requestor", pl.col("notesRequested") >= 2),
        ("single_post_requestor", pl.col("notesRequested") == 1),
        ("not_active", pl.lit(True)),
    ]

    def apply_rules(rules):
        expr = pl.lit(None, dtype=pl.String)
        for label, cond in reversed(rules):
            expr = pl.when(cond).then(pl.lit(label)).otherwise(expr)
        activity_levels = [label for label, _ in total_activity_rules]
        return expr.cast(pl.Enum(activity_levels))

    return apply_rules, month_activity_rules, total_activity_rules


@app.cell
def _(apply_rules, month_activity_rules, notes, pl, ratings, requests):
    # Merge note/rating/request streams into one user-month table and classify each month.
    user_months = (
        notes.join(
            ratings,
            how="full",
            coalesce=True,
            on=["participantId", "userMonth", "calendarMonth"],
        )
        .join(
            requests,
            how="full",
            coalesce=True,
            on=["participantId", "userMonth", "calendarMonth"],
        )
        .with_columns(
            pl.col("notesWritten").fill_null(0),
            pl.col("notesRated").fill_null(0),
            pl.col("notesRequested").fill_null(0),
            calendarDate=pl.col("calendarMonth").str.strptime(pl.Date, "%Y-%m"),
        )
        .with_columns(
            activeMonth=(
                pl.col("notesWritten") + pl.col("notesRated") + pl.col("notesRequested")
            )
            > 0
        )
        .sort(["participantId", "calendarDate"])
        .with_columns(month_role=apply_rules(month_activity_rules))
    )
    return (user_months,)


@app.cell
def _(apply_rules, pl, total_activity_rules, user_months):
    # Aggregate per-user lifetime metrics and assign total + dominant roles.
    users = (
        user_months.group_by("participantId")
        .agg(
            notesWritten=pl.col("notesWritten").sum(),
            notesRated=pl.col("notesRated").sum(),
            notesRequested=pl.col("notesRequested").sum(),
            nActiveMonths=pl.col("activeMonth").sum(),
            firstMonthRole=pl.col("month_role").filter(pl.col("activeMonth")).first(),
        )
        .with_columns(
            total_role=apply_rules(total_activity_rules),
            active_month_bucket=pl.when(pl.col("nActiveMonths") >= 24)
            .then(pl.lit("24+"))
            .otherwise(pl.col("nActiveMonths").cast(pl.String)),
        )
    )

    role_counts = (
        users.group_by("total_role")
        .agg(n_users=pl.len())
        .with_columns(pl.col("total_role").cast(pl.String))
        .sort("n_users", descending=True)
    )

    dominant_role = (
        user_months.filter(pl.col("activeMonth"))
        .group_by(["participantId", "month_role"])
        .agg(n_months=pl.len())
        .sort(["participantId", "n_months"], descending=[False, True])
        .group_by("participantId")
        .agg(pl.col("month_role").first().alias("dominant_month_role"))
    )

    users = users.join(dominant_role, on="participantId", how="left")
    return role_counts, users


@app.cell
def _(mo, px, role_counts):
    # Bar chart: how many users fall into each lifetime role bucket.
    fig_total_role = px.bar(
        role_counts.to_pandas(),
        x="total_role",
        y="n_users",
        title="Number of Users by Lifetime Role",
        labels={"total_role": "Lifetime Role", "n_users": "Num Users"},
    )
    fig_total_role.update_xaxes(tickangle=-45)
    mo.ui.plotly(fig_total_role)
    return


@app.cell
def _(go, pl, users):
    # Sankey chart: role in first active month vs lifetime role.
    sankey_df = (
        users
        .filter(
            pl.col("firstMonthRole").is_not_null() &
            pl.col("total_role").is_not_null()
        )
        .with_columns(
            pl.col("firstMonthRole").cast(pl.String),
            pl.col("total_role").cast(pl.String),
        )
        .group_by(["firstMonthRole", "total_role"])
        .agg(value=pl.len())
    )

    left_nodes = sankey_df["firstMonthRole"].unique().to_list()
    right_nodes = sankey_df["total_role"].unique().to_list()

    labels = [f"first: {x}" for x in left_nodes] + [f"lifetime: {x}" for x in right_nodes]

    left_idx = {k: i for i, k in enumerate(left_nodes)}
    right_idx = {k: i + len(left_nodes) for i, k in enumerate(right_nodes)}

    source = [left_idx[x] for x in sankey_df["firstMonthRole"].to_list()]
    target = [right_idx[x] for x in sankey_df["total_role"].to_list()]
    value = sankey_df["value"].to_list()

    sankey_fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=15,
                    thickness=18,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                ),
            )
        ]
    )

    sankey_fig.update_layout(
        title_text="First Active Role to Lifetime Role",
        font_size=11,
        height=650,
    )

    sankey_fig
    return


@app.cell
def _(mo, px, users):
    # Histogram: distribution of active-month counts by first-month role.
    hist_df = users.to_pandas()
    fig_hist = px.histogram(
        hist_df,
        x="nActiveMonths",
        color="firstMonthRole",
        barmode="overlay",
        nbins=30,
        title="Number of Active Months by First Active Role",
        labels={
            "nActiveMonths": "N Active Months",
            "firstMonthRole": "First Active Month Role",
        },
    )
    mo.ui.plotly(fig_hist)
    return


@app.cell
def _(mo, pl, px, users):
    # Bar chart: dominant monthly role distribution across users.
    dominant_counts = (
        users.group_by("dominant_month_role")
        .agg(n_users=pl.len())
        .with_columns(pl.col("dominant_month_role").cast(pl.String))
        .sort("n_users", descending=True)
    )

    fig_dominant = px.bar(
        dominant_counts.to_pandas(),
        x="dominant_month_role",
        y="n_users",
        title="Number of Users by Dominant Monthly Role",
        labels={"dominant_month_role": "Dominant Monthly Role", "n_users": "Num Users"},
    )
    fig_dominant.update_xaxes(tickangle=-45)
    mo.ui.plotly(fig_dominant)
    return


@app.cell
def _(mo, pl, px, users):
    # Stacked bars: dominant-role mix within each active-month bucket.
    by_month_bucket = (
        users.group_by(["active_month_bucket", "dominant_month_role"])
        .agg(n_users=pl.len())
        .with_columns(pl.col("dominant_month_role").cast(pl.String))
        .sort(["active_month_bucket", "n_users"], descending=[False, True])
    )

    fig_bucket = px.bar(
        by_month_bucket.to_pandas(),
        x="active_month_bucket",
        y="n_users",
        color="dominant_month_role",
        barmode="stack",
        title="Dominant Monthly Role by Number of Active Months",
        labels={
            "active_month_bucket": "Active Months (24+ grouped)",
            "n_users": "Num Users",
            "dominant_month_role": "Dominant Monthly Role",
        },
    )
    mo.ui.plotly(fig_bucket)
    return


@app.cell
def _(mo, month_activity_rules, pl, user_months, users):
    # For each user: role share over active months, then average by dominant role.
    role_labels = [r for r, _ in month_activity_rules if r != "not_active"]

    role_share_exprs = [
        (
            (pl.col("month_role") == role).sum() / pl.len()
        ).alias(f"share_{role}")
        for role in role_labels
    ]

    per_user_role_share = (
        user_months.filter(pl.col("activeMonth"))
        .group_by("participantId")
        .agg(*role_share_exprs)
        .join(
            users.select(["participantId", "dominant_month_role"]),
            on="participantId",
            how="left",
        )
    )

    avg_share_by_dominant = (
        per_user_role_share.group_by("dominant_month_role")
        .agg(
            [
                pl.col(f"share_{role}").mean().alias(f"avg_share_{role}")
                for role in role_labels
            ]
        )
        .sort("dominant_month_role")
    )

    mo.md("### Average role-share profile by dominant monthly role")
    avg_share_by_dominant
    return


if __name__ == "__main__":
    app.run()
