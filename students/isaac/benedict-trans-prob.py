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

    return Path, colorsys, mo, np, pl, plt, px


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

    # Build the classification expression from rules
    def apply_rules(levels) -> pl.Expr:
        levels = levels + [("not_active", pl.lit(True))]
        # Apply rules in reverse order to ensure first match takes precedence
        expr = pl.lit(None, dtype=pl.String)
        for label, condition in reversed(levels):
            expr = pl.when(condition).then(pl.lit(label)).otherwise(expr)

        # Extract ordered labels from rules
        activity_level_labels = [label for label, _ in levels]

        # Make the column an ordered categorical with the specified levels
        expr = expr.cast(pl.Enum(categories=activity_level_labels))

        return expr


    return (
        all_activity_levels,
        apply_rules,
        rating_activity_levels,
        requesting_activity_levels,
        writing_activity_levels,
    )


@app.cell
def _(
    all_activity_levels,
    apply_rules,
    pl,
    rating_activity_levels,
    requesting_activity_levels,
    user_months,
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
            notesWritten = pl.col("notesWritten").sum(),
            notesRated = pl.col("notesRated").sum(),
            notesRequested = pl.col("notesRequested").sum(),
            hits = pl.col("hits").sum(),
            correctHelpfuls = pl.col("correctHelpfuls").sum(),
            correctNotHelpfuls = pl.col("correctNotHelpfuls").sum(),
            numRequestsResultingInCrh = pl.col("numRequestsResultingInCrh").sum(),
            userFirstCalendarMonth = pl.col("calendarMonth").min(),
            # userLastActiveCalendarMonth = pl.col("userLastActiveCalendarMonth").max(),
            nActiveMonths = pl.col("activeMonth").sum(),
            age = pl.col("userMonth").max(),
            activeWindow = pl.col("userMonth").filter(pl.col("activeMonth")).max() - pl.col("userMonth").min() + 1,
            firstMonthRole = pl.col("month_role").filter(pl.col("activeMonth")).first(),
            lastMonthRole = pl.col("month_role").filter(pl.col("activeMonth")).last(),
            max_role=pl.col("month_role").min(),
            *[(pl.col("month_role") == role).sum().alias(f"nMonths{role}") for role, _ in all_activity_levels],
        )
        .with_columns(
            (pl.selectors.starts_with("nMonths") / pl.col("nActiveMonths") * 100).name.map(lambda s: s.replace("nMonths", "pctActiveMonths")),
            total_role=apply_rules(all_activity_levels),
            writing_role=apply_rules(writing_activity_levels),
            rating_role=apply_rules(rating_activity_levels),
            requesting_role=apply_rules(requesting_activity_levels),
        )
        .join(_user_months_wide, on="participantId", how="left")
    )
    return max_month, min_month, users


@app.cell
def _(pl, users):
    total_role_counts = (
        users
        .group_by("total_role")
        .agg(n=pl.len())
    )
    return (total_role_counts,)


@app.cell
def _(pl, users):
    max_role_counts = (
        users
        .group_by("max_role")
        .agg(n=pl.len())
    )
    return (max_role_counts,)


@app.cell
def _(pl, users):
    writing_role_counts = (
        users
        .group_by("writing_role")
        .agg(n=pl.len())
    )
    rating_role_counts = (
        users
        .group_by("rating_role")
        .agg(n=pl.len())
    )
    requesting_role_counts = (
        users
        .group_by("requesting_role")
        .agg(n=pl.len())
    )
    return rating_role_counts, writing_role_counts


@app.cell
def _(max_role_counts, pl, users):
    # Calculate percent of users with each max role who start out in each role
    _fm_per_mr = (
        users
        .group_by("firstMonthRole", "max_role")
        .agg(n=pl.len()).sort("firstMonthRole")
        .pivot(index="max_role", on="firstMonthRole", values="n")
        .fill_null(0)
    )

    (
        _fm_per_mr.join(max_role_counts, on="max_role")
        .with_columns((
            pl.selectors.numeric().exclude("n")
            .truediv(pl.col("n")) * 100).round(1)
        )
        .sort("max_role")
        .select("max_role", "n", pl.selectors.exclude(["max_role", "n"]))
    )
    return


@app.cell
def _(all_activity_levels, max_role_counts, pl, users):
    (
        users.group_by("max_role")
        .agg(*[pl.col(f"pctActiveMonths{role}").mean() for role, _ in all_activity_levels])
        .with_columns(pl.selectors.numeric().round(1))
        .join(max_role_counts, on="max_role")
        .select("max_role", "n", pl.selectors.exclude(["max_role", "n"]))
        .sort("max_role")
    )
    return


@app.cell
def _(pl, total_role_counts, users):
    (
        users
        .group_by("firstMonthRole", "total_role")
        .agg(n=pl.len()).sort("firstMonthRole")
        .pivot(index="total_role", on="firstMonthRole", values="n")
        .fill_null(0)
        .join(total_role_counts, on="total_role")
        .with_columns((
            pl.selectors.numeric().exclude("n")
            .truediv(pl.col("n")) * 100).round(1)
        )
        .sort("total_role")
        .select("total_role", "n", pl.selectors.exclude(["total_role", "n"]))
    )
    return


@app.cell
def _(all_activity_levels, pl, total_role_counts, users):
    (
        users.group_by("total_role")
        .agg(*[pl.col(f"pctActiveMonths{role}").mean() for role, _ in all_activity_levels])
        .with_columns(pl.selectors.numeric().round(1))
        .join(total_role_counts, on="total_role")
        .select("total_role", "n", pl.selectors.exclude(["total_role", "n"]))
        .sort("total_role")
    )
    return


@app.cell
def _(all_activity_levels, pl, total_role_counts, users):
    (
        users.group_by("total_role")
        .agg(*[pl.col(f"nMonths{role}").mean() for role, _ in all_activity_levels])
        .with_columns(pl.selectors.numeric().round(1))
        .join(total_role_counts, on="total_role")
        .select("total_role", "n", pl.selectors.exclude(["total_role", "n"]))
        .sort("total_role")
    )
    return


@app.cell
def _(all_activity_levels, pl, users, writing_role_counts):
    (
        users
        .group_by("writing_role")
        .agg(*[pl.col(f"pctActiveMonths{role}").mean() for role, _ in all_activity_levels])
        .with_columns(pl.selectors.numeric().round(1))
        .join(writing_role_counts, on="writing_role")
        .select("writing_role", "n", pl.selectors.exclude(["writing_role", "n"]))
        .sort("writing_role")
    )
    return


@app.cell
def _(all_activity_levels, pl, rating_role_counts, users):
    (
        users.group_by("rating_role")
        .agg(*[pl.col(f"pctActiveMonths{role}").mean() for role, _ in all_activity_levels])
        .with_columns(pl.selectors.numeric().round(1))
        .join(rating_role_counts, on="rating_role")
        .select("rating_role", "n", pl.selectors.exclude(["rating_role", "n"]))
        .sort("rating_role")
    )
    return


@app.cell
def _(pl, users):
    (
        users
        .group_by("total_role").agg(
            pl.len(), 
            pl.col("notesWritten").sum(), pl.col("hits").sum(),
            ratingsCreated=pl.col("notesRated").sum(), correctRatingsCreated = pl.col("correctHelpfuls").sum() + pl.col("correctNotHelpfuls").sum(),
            requestsCreated=pl.col("notesRequested").sum(), correctRequestsCreated= pl.col("numRequestsResultingInCrh").sum(),
        )
        .sort("total_role")
        .filter(pl.col("len") > 100)
    )
    return


@app.cell
def _(pl, users):
    (
        users
        .group_by("writing_role", "rating_role", "requesting_role").agg(
            pl.len(), 
            pl.col("notesWritten").sum(), pl.col("hits").sum(),
            ratingsCreated=pl.col("notesRated").sum(), correctRatingsCreated = pl.col("correctHelpfuls").sum() + pl.col("correctNotHelpfuls").sum(),
            requestsCreated=pl.col("notesRequested").sum(), correctRequestsCreated= pl.col("numRequestsResultingInCrh").sum(),
        )
        .sort("writing_role", "rating_role", "requesting_role")
        .filter(pl.col("len") > 100)
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    role_colors = {
        # Writers — red
        "single_note_writer": "rgba(252,146,114,0.85)",
        "single_digit_writer": "rgba(251,106,74,0.85)",
        "double_digit_writer": "rgba(222,45,38,0.85)",
        "triple_digit_writer": "rgba(165,15,21,0.85)",

        # Raters — blue
        "single_note_rater": "rgba(158,202,225,0.85)",
        "single_digit_rater": "rgba(107,174,214,0.85)",
        "double_digit_rater": "rgba(49,130,189,0.85)",
        "triple_digit_rater": "rgba(8,81,156,0.85)",

        # Requestors — green
        "single_note_requestor": "rgba(161,217,155,0.85)",
        "single_digit_requestor": "rgba(116,196,118,0.85)",
        "double_digit_requestor": "rgba(49,163,84,0.85)",
        "triple_digit_requestor": "rgba(0,109,44,0.85)",

        # Inactive — gray
        "not_active": "rgba(150,150,150,0.85)",
    }
    return (role_colors,)


@app.cell
def _(pl, users):
    cohort_df = (
        users.group_by("firstMonthRole", "userFirstCalendarMonth")
        .agg(
            n=pl.len(),
            median_lifetime=pl.col("activeWindow").median(),
            median_active_months=pl.col("nActiveMonths").median(),
            median_notes_written=pl.col("notesWritten").median(),
            median_notes_rated=pl.col("notesRated").median(),
            median_notes_requested=pl.col("notesRequested").median(),
        )
        .sort("userFirstCalendarMonth", "firstMonthRole")
    )

    roles = cohort_df["firstMonthRole"].unique().sort().to_list()
    return (cohort_df,)


@app.cell
def _(pl, users):
    roles_by_month = (
        users.group_by("total_role", "userFirstCalendarMonth")
        .agg(
            n=pl.len(),
            median_lifetime=pl.col("activeWindow").median(),
            median_active_months=pl.col("nActiveMonths").median(),
            median_notes_written=pl.col("notesWritten").median(),
            median_notes_rated=pl.col("notesRated").median(),
            median_notes_requested=pl.col("notesRequested").median(),
        )
        .sort("userFirstCalendarMonth", "total_role")
    )
    return (roles_by_month,)


@app.cell
def _(cohort_df, mo, px, role_colors):
    _fig = px.line(
        cohort_df,
        x="userFirstCalendarMonth",
        y="median_active_months",
        color="firstMonthRole",
        markers=True,
        color_discrete_map=role_colors,
        title="Median Number of Active Months by Cohort Month and Role",
        labels={
            "userFirstCalendarMonth": "First Calendar Month",
            "median_active_months": "Median N Active Months",
        },
        height=550,
    )

    _fig.update_yaxes(matches=None)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo, pl, px, role_colors, user_months):
    _variable = "notesRated" 

    _df = (
        user_months.group_by("calendarMonth", "month_role").agg(
            pl.col(_variable).sum(),
        )
        .sort("month_role", "calendarMonth")
    )

    _df = _df.filter(pl.col(_variable) > 0)
    _fig = px.line(
        _df,
        x="calendarMonth",
        y=_variable,
        color="month_role",
        markers=True,
        color_discrete_map=role_colors,
        labels={
            "calendarMonth": "Calendar Month",
            "notesWritten": "Num Notes Produced by Group",
        },
        height=550,
        log_y=True,
    )

    _fig.update_yaxes(matches=None)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo, pl, px, role_colors, user_months, users):
    _variable = "numRequestsResultingInCrh" 

    _df = (
        user_months
        .join(users.select("participantId", "total_role"), on="participantId", how="left")
        .group_by("calendarMonth", "total_role").agg(
            pl.col(_variable).sum(),
        )
        .sort("total_role", "calendarMonth")
    )

    _df = _df.filter(pl.col(_variable) > 0)
    _fig = px.line(
        _df,
        x="calendarMonth",
        y=_variable,
        color="total_role",
        markers=True,
        color_discrete_map=role_colors,
        labels={
            "calendarMonth": "Calendar Month",
            "notesWritten": "Num Notes Produced by Group",
        },
        height=550,
    )

    _fig.update_yaxes(matches=None)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


    mo.ui.plotly(_fig)
    return


@app.cell
def _(pl, user_months, users):
    _role_counts = user_months.group_by("total_role").agg(n=pl.len())

    _df = (
        user_months
        .join(users.select("participantId", "total_role"), on="participantId", how="left")
        .group_by("userMonth", "total_role", "month_role").agg(
            n_in_role=pl.len(),
        )
        .filter(pl.col("month_role") != "not_active")
        .sort("total_role", "month_role", "userMonth")
        .join(_role_counts, on="total_role", how="left")
        .with_columns(pct_in_role=pl.col("n_in_role") / pl.col("n") * 100)
    )



    _df
    return


@app.cell
def _(cohort_df, mo, pl, px, role_colors):
    _fig = px.line(
        cohort_df.filter(
            ~pl.col("firstMonthRole").cast(pl.String).str.contains("requ")
        ),
        x="userFirstCalendarMonth",
        y="n",
        color="firstMonthRole",
        markers=True,
        color_discrete_map=role_colors,
        title="Number of Users Joining in a Month by First Role",
        labels={
            "userFirstCalendarMonth": "First Calendar Month",
            "n": "Num Users",
        },
        height=550,
        # log_y=True,
    )

    _fig.update_yaxes(matches=None)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo, px, role_colors, roles_by_month):
    _fig = px.line(
        roles_by_month,
        x="userFirstCalendarMonth",
        y="n",
        color="total_role",
        markers=True,
        color_discrete_map=role_colors,
        title="Number of Users Joining in a Month by Total Role",
        labels={
            "userFirstCalendarMonth": "First Calendar Month",
            "n": "Num Users",
        },
        height=550,
        log_y=True,
    )

    _fig.update_yaxes(matches=None)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


    mo.ui.plotly(_fig)
    return


@app.cell
def _(colorsys, mo, pl, px, user_months):
    def plot_notes_by_join_month(user_months, metric: str, pct: bool = False):
        label_map = {
            "notesWritten": "Notes Written",
            "notesRated": "Notes Rated",
            "notesRequested": "Notes Requested",
        }
        label = label_map[metric]

        join_month_df = (
            user_months
            .with_columns(
                (pl.col("calendarDate") - pl.duration(days=pl.col("userMonth") * 30))
                .dt.strftime("%Y-%m")
                .alias("joinMonth")
            )
            .group_by(["calendarMonth", "calendarDate", "joinMonth"])
            .agg(pl.col(metric).sum())
            .with_columns(
                (pl.col(metric) / pl.col(metric).sum().over("calendarMonth") * 100)
                .alias("pctMetric")
            )
            .sort(["calendarDate", "joinMonth"])
            .filter(pl.col("calendarMonth") >= "2023-01")
        )

        y_col = "pctMetric" if pct else metric
        y_label = f"% of {label}" if pct else label
        title = f"{'% of' if pct else 'Raw'} {label} by User Join Month"

        join_months = sorted(join_month_df["joinMonth"].unique().to_list())
        join_years  = sorted(set(m[:4] for m in join_months))
        year_hues   = {year: i / len(join_years) for i, year in enumerate(join_years)}

        color_map = {}
        for year in join_years:
            months_in_year = sorted(m for m in join_months if m.startswith(year))
            n = len(months_in_year)
            hue = year_hues[year]
            for i, month in enumerate(months_in_year):
                lightness = 0.25 + 0.45 * (i / max(n - 1, 1))
                r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.75)
                color_map[month] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        fig = px.area(
            join_month_df,
            x="calendarMonth",
            y=y_col,
            color="joinMonth",
            color_discrete_map=color_map,
            title=title,
            labels={"calendarMonth": "Calendar Month", y_col: y_label, "joinMonth": "Join Month"},
            height=500,
            width=600,
        )
        fig.update_traces(line=dict(width=0.4))

        boundary_years = join_years[:-1]
        for year in boundary_years:
            months_up_to = [m for m in join_months if m[:4] <= year]
            boundary_df = (
                join_month_df
                .filter(pl.col("joinMonth").is_in(months_up_to))
                .group_by(["calendarMonth", "calendarDate"])
                .agg(pl.col(y_col).sum().alias("cumVal"))
                .sort("calendarDate")
            )
            fig.add_scatter(
                x=boundary_df["calendarMonth"].to_list(),
                y=boundary_df["cumVal"].to_list(),
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )

        if pct:
            fig.update_yaxes(range=[0, 100])

        return mo.ui.plotly(fig)


    mo.vstack([
        plot_notes_by_join_month(user_months, "notesWritten", pct=False),
        plot_notes_by_join_month(user_months, "notesRated",   pct=False),
        plot_notes_by_join_month(user_months, "notesRequested", pct=False),
    ])
    return


@app.cell
def _(colorsys, mo, pl, px, user_months):
    def plot_active_users_by_join_month(user_months, pct: bool = False, user_type: str = "total"):
        type_map = {
            "total":     (pl.col("activeMonth"),          "Active Users"),
            "writers":   (pl.col("notesWritten") > 0,     "Active Writers"),
            "raters":    (pl.col("notesRated") > 0,       "Active Raters"),
            "requesters":(pl.col("notesRequested") > 0,   "Active Requesters"),
        }
        filter_expr, type_label = type_map[user_type]

        join_month_df = (
            user_months
            .with_columns(
                (pl.col("calendarDate") - pl.duration(days=pl.col("userMonth") * 30))
                .dt.strftime("%Y-%m")
                .alias("joinMonth")
            )
            .filter(filter_expr)
            .group_by(["calendarMonth", "calendarDate", "joinMonth"])
            .agg(pl.len().alias("activeUsers"))
            .with_columns(
                (pl.col("activeUsers") / pl.col("activeUsers").sum().over("calendarMonth") * 100)
                .alias("pctMetric")
            )
            .sort(["calendarDate", "joinMonth"])
            .filter(pl.col("calendarMonth") >= "2023-01")
        )

        y_col = "pctMetric" if pct else "activeUsers"
        y_label = f"% of {type_label}" if pct else type_label
        title = f"{'% of' if pct else ''} {type_label} by Join Month".strip()

        join_months = sorted(join_month_df["joinMonth"].unique().to_list())
        join_years  = sorted(set(m[:4] for m in join_months))
        year_hues   = {year: i / len(join_years) for i, year in enumerate(join_years)}

        color_map = {}
        for year in join_years:
            months_in_year = sorted(m for m in join_months if m.startswith(year))
            n = len(months_in_year)
            hue = year_hues[year]
            for i, month in enumerate(months_in_year):
                lightness = 0.25 + 0.45 * (i / max(n - 1, 1))
                r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.75)
                color_map[month] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        fig = px.area(
            join_month_df,
            x="calendarMonth",
            y=y_col,
            color="joinMonth",
            color_discrete_map=color_map,
            title=title,
            labels={"calendarMonth": "Calendar Month", y_col: y_label, "joinMonth": "Join Month"},
            height=500,
            width=600,
        )
        fig.update_traces(line=dict(width=0.4))

        boundary_years = join_years[:-1]
        for year in boundary_years:
            months_up_to = [m for m in join_months if m[:4] <= year]
            boundary_df = (
                join_month_df
                .filter(pl.col("joinMonth").is_in(months_up_to))
                .group_by(["calendarMonth", "calendarDate"])
                .agg(pl.col(y_col).sum().alias("cumVal"))
                .sort("calendarDate")
            )
            fig.add_scatter(
                x=boundary_df["calendarMonth"].to_list(),
                y=boundary_df["cumVal"].to_list(),
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )

        if pct:
            fig.update_yaxes(range=[0, 100])

        return mo.ui.plotly(fig)


    mo.vstack([
        plot_active_users_by_join_month(user_months, user_type="total"),
        plot_active_users_by_join_month(user_months, user_type="writers"),
        plot_active_users_by_join_month(user_months, user_type="raters"),
        plot_active_users_by_join_month(user_months, user_type="requesters"),
    ])
    return


@app.cell
def _(activity_level_labels, pl, user_months):
    # Build month-to-next-month transitions within each user trajectory
    _transitions = (
        user_months
        .sort(["participantId", "userMonth"])
        .with_columns(
                prev_role=pl.col("month_role").shift(1).over("participantId"),
        )
        .filter(pl.col("prev_role").is_not_null())
        .select(["month_role", "prev_role"])
    )

    # Ensure all state pairs exist
    _month_roles = pl.DataFrame(activity_level_labels, schema={"month_role":pl.Enum(categories=activity_level_labels)})
    _prev_roles  = pl.DataFrame(activity_level_labels, schema={"prev_role": pl.Enum(categories=activity_level_labels)})
    _state_grid  = _month_roles.join(_prev_roles, how="cross")

    # Count transitions
    transition_counts = (
        _transitions
        .group_by(["month_role", "prev_role"])
        .agg(n_transitioning=pl.len())
        .join(_state_grid, on=["month_role", "prev_role"], how="right")
        .with_columns(pl.col("n_transitioning").fill_null(0).cast(pl.Int64))
    )
    return (transition_counts,)


@app.cell
def _(pl, transition_counts):
    _n_per_prev_role = (
        transition_counts
        .group_by("prev_role")
        .agg(n_starting_in_role=pl.col("n_transitioning").sum())
    )

    transition_matrix = (
        transition_counts.join(_n_per_prev_role,on="prev_role", how="left",)
        .with_columns(probability=pl.col("n_transitioning") / pl.col("n_starting_in_role"))
        .select("prev_role", "month_role", "probability")
        .pivot(
            index="month_role",
            on="prev_role",
            values="probability",
            aggregate_function="sum",
        )
    )
    return (transition_matrix,)


@app.cell
def _(activity_level_labels, np, plt, transition_matrix):
    # Plot heatmap
    _heat_values = transition_matrix.select(activity_level_labels).to_numpy().transpose()

    plt.figure(figsize=(12, 8))

    _img = plt.imshow(
        _heat_values,
        cmap="YlOrRd",
        aspect="auto",
        vmin=0,
        vmax=max(1e-9, float(np.max(_heat_values))),
    )

    plt.colorbar(_img, label="Transition Probability")

    plt.xticks(
        ticks=np.arange(len(activity_level_labels)),
        labels=activity_level_labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=np.arange(len(activity_level_labels)),
        labels=activity_level_labels,
    )

    # Annotate cells
    for _i in range(_heat_values.shape[0]):
        for _j in range(_heat_values.shape[1]):
            _value = _heat_values[_i, _j]
            _text_color = "white" if _value > 0.5 else "black"

            plt.text(
                _j,
                _i,
                f"{_value:.3f}",
                ha="center",
                va="center",
                color=_text_color,
                fontsize=8,
            )

    plt.title("Empirical Transition Matrix (Row-Normalized)")
    plt.xlabel("Next State")
    plt.ylabel("Current State")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(pl, user_months):
    transitions_by_month = (
        user_months
        .sort(["participantId", "userMonth"])
        .with_columns(
            next_role=pl.col("month_role").shift(-1).over("participantId"),
            next_userMonth=pl.col("userMonth").shift(-1).over("participantId"),
        )
        .filter(pl.col("next_role").is_not_null())
        .select(
            "participantId",
            "userMonth",
            "month_role",
            "next_userMonth",
            "next_role",
        )
    )
    return


@app.cell
def _(all_activity_levels, max_month, min_month, mo):
    month_slider = mo.ui.slider(
        start=min_month,
        stop=max_month,
        step=1,
        value=min_month,
        label="userMonth",
    )

    role_filter = mo.ui.multiselect(
        options=[role for role, _ in all_activity_levels],
        value=[role for role, _ in all_activity_levels],
        label="role",
    )
    mo.hstack([role_filter, month_slider])
    return (role_filter,)


@app.cell
def _(pl, role_filter, users):
    (
        users
        .with_columns(
            ratings_per_notes_written =  pl.col("notesRated") / pl.col("notesWritten"),
        )
        .select("ratings_per_notes_written", "nActiveMonths", "notesWritten", "notesRated", "notesRequested",  "total_role")
        .filter(pl.col("total_role").is_in(role_filter.value))
    )
    return


if __name__ == "__main__":
    app.run()
