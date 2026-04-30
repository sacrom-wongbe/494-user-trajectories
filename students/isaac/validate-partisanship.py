import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import zipfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    return Path, mo, pl, sns, zipfile


@app.cell
def _(Path):
    DATA_DIR = Path(".").absolute().parent.parent / "data"
    return (DATA_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Can we infer posts' and notes' partisanship?
    """)
    return


@app.cell
def _(DATA_DIR, pl):
    raw_enriched_notes = pl.read_parquet(DATA_DIR / "intermediate" / "notes_enriched.parquet")
    return (raw_enriched_notes,)


@app.cell
def _(DATA_DIR, pl):
    ######### Read data from first email
    user_ideology_barbera = (
        pl.concat([
            pl.read_csv(
                DATA_DIR / "input/2026-04-14-mosleh-partisanship/07-user_ideology-barbera.csv",
                schema_overrides={"id_str": pl.String}, null_values=["NA"],),
            pl.read_csv(
                DATA_DIR / "input/2026-04-14-mosleh-partisanship/07b-user_ideology-barbera-unhelpful.csv",
                schema_overrides={"id_str": pl.String}, null_values=["NA"],),])
        # There are some infinite and null values present, which are not valid
        .filter(pl.col("ideo").is_not_null() & ~pl.col("ideo").is_infinite())
        .drop("")
        .rename({"id_str": "tweet_author_id", "ideo": "partisan_score_barbera"})
    )
    barbera_mosleh_gpt = (
        pl.read_excel(DATA_DIR / "input/2026-04-14-mosleh-partisanship/barbera_mosleh_gpt.xlsx")
        .drop("__UNNAMED__0")
        .rename({"partisan_gpt": "party_gpt_private", "partisan_perplexity": "party_perplexity_private"})
        .drop("barbera_accurate", "mosleh_accurate", "note_published_helpful")
    )
    barbera_mosleh_perplexity = (
        pl.read_excel(DATA_DIR / "input/2026-04-14-mosleh-partisanship/barbera_mosleh_perplexity.xlsx")
        .drop("__UNNAMED__0")
        .rename({"partisan_perplexity": "party_perplexity_private"})
        .drop("barbera_accurate", "mosleh_accurate", "note_published_helpful")
    )
    return barbera_mosleh_gpt, barbera_mosleh_perplexity, user_ideology_barbera


@app.cell
def _(
    barbera_mosleh_gpt,
    barbera_mosleh_perplexity,
    pl,
    user_ideology_barbera,
):
    ######## Concat scores from different data sets in first email, then categorize based on scores
    first_email_scores = (
        pl.concat([
            user_ideology_barbera    .select(["tweet_author_id", "partisan_score_barbera"])
                .with_columns(partisan_score=pl.lit(None).cast(pl.Float64)),
            barbera_mosleh_gpt       .select(["tweet_author_id", "partisan_score_barbera", "partisan_score"]),
            barbera_mosleh_perplexity.select(["tweet_author_id", "partisan_score_barbera", "partisan_score"]),
        ])
        .with_columns(
            party_barbera_private = 
                pl.when(pl.col("partisan_score_barbera") > 1).then(pl.lit("republican"))
                    .when(pl.col("partisan_score_barbera").is_not_null()).then(pl.lit("democrat")),
            party_mosleh_private = 
                    pl.when(pl.col("partisan_score") > 0).then(pl.lit("republican"))
                        .when(pl.col("partisan_score").is_not_null()).then(pl.lit("democrat")))
        .drop("partisan_score_barbera", "partisan_score")
        # Get first non null value, which will be only unique non-null value
        .group_by("tweet_author_id").agg(
            party_barbera_private = pl.col("party_barbera_private").filter(pl.col("party_barbera_private").is_not_null()).first(),
            party_mosleh_private = pl.col("party_mosleh_private").filter(pl.col("party_mosleh_private").is_not_null()).first(),
            nunique_barbera = pl.col("party_barbera_private").filter(pl.col("party_barbera_private").is_not_null()).n_unique(),
            nunique_mosleh = pl.col("party_mosleh_private").filter(pl.col("party_mosleh_private").is_not_null()).n_unique(),
        )
    )


    # Make sure there aren't any contradictory labels for the same author
    assert first_email_scores.filter(pl.col("nunique_barbera") > 1).is_empty(), "There are contradictory Barbera labels for the same author"
    assert first_email_scores.filter(pl.col("nunique_mosleh") > 1).is_empty(), "There are contradictory Mosleh labels for the same author"
    return


@app.cell
def _(barbera_mosleh_gpt, barbera_mosleh_perplexity, pl):
    ######### Concat AI labels from first_score
    first_email_ai = (
        pl.concat([
            barbera_mosleh_perplexity.select(["tweet_author_id", "party_perplexity_private"])
                .with_columns(party_gpt_private=pl.lit(None).cast(pl.String)),
            barbera_mosleh_gpt       .select(["tweet_author_id", "party_perplexity_private", "party_gpt_private"]),
        ])
        .group_by("tweet_author_id").agg(
            party_perplexity_private = pl.col("party_perplexity_private").filter(pl.col("party_perplexity_private").is_not_null()).first(),
            party_gpt_private = pl.col("party_gpt_private").filter(pl.col("party_gpt_private").is_not_null()).first(),
            nunique_perplexity = pl.col("party_perplexity_private").filter(pl.col("party_perplexity_private").is_not_null()).n_unique(),
            nunique_gpt = pl.col("party_gpt_private").filter(pl.col("party_gpt_private").is_not_null()).n_unique(),
        )
    )

    # Make sure there aren't any contradictory labels for the same author
    assert first_email_ai.filter(pl.col("nunique_perplexity") > 1).is_empty(), "There are contradictory Perplexity labels for the same author"
    assert first_email_ai.filter(pl.col("nunique_gpt") > 1).is_empty(), "There are contradictory GPT labels for the same author"

    return


@app.cell
def _(DATA_DIR, pl):
    renault_public_notes = (
        pl.read_csv(DATA_DIR / "input/renault_partisanship_labels.csv", 
                    schema_overrides={"note_id": pl.String, "tweet_author_id": pl.String, "tweet_id": pl.String})
        .select("note_id", "tweet_id", "tweet_author_id")
        .with_columns(note_in_renault_public=pl.lit(True))
    )

    renault_public_posts = (
        renault_public_notes
        .select("tweet_id")
        .unique()
        .with_columns(post_in_renault_public=pl.lit(True))
    )

    renault_public_authors = (
        renault_public_notes
        .select("tweet_author_id")
        .unique()
        .with_columns(author_in_renault_public=pl.lit(True))
    )
    return renault_public_authors, renault_public_notes, renault_public_posts


@app.cell
def _(DATA_DIR, pl, zipfile):
    # Read data they sent most recently
    with zipfile.ZipFile(DATA_DIR / "input/2026-04-29-mosleh-partisanship/isaac-partisan.csv.zip") as _z:
        second_email_scores = (
            pl.read_csv(
                _z.open(_z.namelist()[0]),
                schema_overrides={"userid": pl.String}
            )
            .with_columns(
                mosleh_party_v2 = pl.when(pl.col("partisan_score") > 0).then(pl.lit("republican"))
                        .when(pl.col("partisan_score").is_not_null()).then(pl.lit("democrat")),
                author_in_mosleh_v2 = pl.lit(True),
            )
            .rename({"userid": "tweet_author_id"})
            .filter(pl.col("partisan_score") != -99)
            .select("tweet_author_id", "mosleh_party_v2", "author_in_mosleh_v2")
        )
    return (second_email_scores,)


@app.cell
def _(pl):
    posts_we_scraped = (
        pl.scan_parquet("/data/cn_archive/derivatives/20260227_raw_posts.parquet")
        .select("post_id", "author_id")
        .group_by("post_id").agg(pl.col("author_id").filter(pl.col("author_id").is_not_null()))
        .collect()
        .with_columns(post_in_our_data=pl.lit(True))
    )
    return (posts_we_scraped,)


@app.cell
def _(
    pl,
    posts_we_scraped,
    raw_enriched_notes,
    renault_public_authors,
    renault_public_notes,
    renault_public_posts,
    second_email_scores,
):
    notes = (
        raw_enriched_notes
        .join(
            renault_public_notes.select("note_id", "note_in_renault_public"),
            left_on="noteId", right_on="note_id", how="left", coalesce=True, validate="1:1",
        )
        .join(renault_public_posts, left_on="tweetId", right_on="tweet_id",how="left",coalesce=True)
        .join(renault_public_authors, on="tweet_author_id", how="left", coalesce=True)
        .join(second_email_scores, on="tweet_author_id", how="left", coalesce=True)
        .join(
            posts_we_scraped.select("post_id", "post_in_our_data"),
            left_on="tweetId", right_on="post_id", how="left", coalesce=True, validate="m:1",
        )
        .with_columns(
            tweet_author_party = pl.coalesce([pl.col("tweet_author_party"), pl.col("mosleh_party_v2")]),
            author_only_in_mosleh_v2 = pl.col("author_in_mosleh_v2") & pl.col("tweet_author_party").is_null(),
        )
        .with_columns(
            claimsMisinfo = pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING",
            note_in_renault_public = pl.col("note_in_renault_public").fill_null(False), 
            post_in_renault_public = pl.col("post_in_renault_public").fill_null(False),  
            author_in_renault_public = pl.col("author_in_renault_public").fill_null(False),  
            post_in_our_data = pl.col("post_in_our_data").fill_null(False),
            author_only_in_mosleh_v2 = pl.col("author_only_in_mosleh_v2").fill_null(False),
            weHaveTweetAuthorId = pl.col("tweet_author_id").is_not_null(),
            tweetInEnglish = pl.col("tweet_lang") == "en",
            weInferredParty = pl.col("tweet_author_party").is_not_null(),
            timePeriod = pl.when("calendarMonth" < "2023-01").then(pl.lit("Pre-Renault"))
                            .when("calendarMonth" < "2024-06").then(pl.lit("Renault"))
                            .when("calendarMonth" >= "2024-06").then(pl.lit("Post-Renault"))
                            .otherwise(pl.lit("UNKNOWN"))
        )
    )
    return (notes,)


@app.cell
def _(DATA_DIR, pl):
    enriched_ratings = (
        pl.read_parquet(DATA_DIR / "intermediate" / "ratings_enriched.parquet")
        .join(_mosleh_partisan_v2, on="tweet_author_id", how="left", coalesce=True)
        .with_columns(
            tweet_author_party = pl.coalesce([pl.col("tweet_author_party"), pl.col("mosleh_party_v2")]),
            author_only_in_mosleh_v2 = pl.col("author_in_mosleh_v2") & pl.col("tweet_author_party").is_null(),
        )
        .with_columns(
            author_only_in_mosleh_v2 = pl.col("author_only_in_mosleh_v2").fill_null(False),
            claimsMisinfo = pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING",
            weHaveTweetAuthorId = pl.col("tweet_author_id").is_not_null(),
            tweetInEnglish = pl.col("tweet_lang") == "en",
            weInferredParty = pl.col("tweet_author_party").is_not_null(),
            timePeriod = pl.when("calendarMonth" < "2023-01").then(pl.lit("Pre-Renault"))
                            .when("calendarMonth" < "2024-06").then(pl.lit("Renault"))
                            .when("calendarMonth" >= "2024-06").then(pl.lit("Post-Renault"))
                            .otherwise(pl.lit("UNKNOWN"))
        )
    )
    return (enriched_ratings,)


@app.cell
def _(enriched_ratings, pl):
    (
        enriched_ratings
        .filter(pl.col("claimsMisinfo") | pl.col("claimsMisinfo").is_null())
        .filter(pl.col("tweetInEnglish") | pl.col("tweetInEnglish").is_null())
        .group_by("claimsMisinfo","weHaveTweetAuthorId", "weInferredParty", "author_only_in_mosleh_v2").len()
    )
    return


@app.cell
def _(notes, pl):
    (
        notes
        .filter(pl.col("claimsMisinfo") | pl.col("claimsMisinfo").is_null())
        .filter(pl.col("tweetInEnglish") | pl.col("tweetInEnglish").is_null())
        .group_by("claimsMisinfo","weHaveTweetAuthorId", "weInferredParty", "author_only_in_mosleh_v2").len()
    )
    return


@app.cell
def _(notes, pl):
    (
        notes
        .filter(pl.col("claimsMisinfo"))
        .filter(pl.col("tweetInEnglish"))
        .filter(
            (~pl.col("author_only_in_mosleh_v2"))
            & (~pl.col("weInferredParty"))
        )
        # .filter(
        #     pl.col("author_only_in_mosleh_v2")
        #     & (pl.col("weInferredParty"))
        # )
        .select("tweet_author_id")
        .unique()
    )
    return


@app.cell
def _(enriched_notes, pl):
    (    
        enriched_notes
        .filter(pl.col("claimsMisinfo") | pl.col("claimsMisinfo").is_null())
        .filter(pl.col("tweetInEnglish") | pl.col("tweetInEnglish").is_null())
        .filter(~pl.col("weInferredParty"))
        .filter(pl.col("weHaveTweetAuthorId"))
        .select("tweetId")
        .unique()
    )
    return


@app.cell
def _(enriched_notes):
    triplets = (
        enriched_notes
        .select("noteId", "tweetId", "tweet_author_id")
        .unique()
    )

    triplets
    return


@app.cell
def _(availabilities, pl):
    (
        availabilities
        .group_by("calendarMonth").agg(
            total_notes = pl.count(),
            notes_in_renault_public = pl.col("note_in_renault_public").sum(),
            posts_in_renault_public = pl.col("post_in_renault_public").sum(),
            authors_in_renault_public = pl.col("author_in_renault_public").sum(),
            posts_in_our_data = pl.col("post_in_our_data").sum(),
        )
        .sort("calendarMonth")
    )
    return


@app.cell
def _(availabilities, pl):
    (
        availabilities
        .filter(~pl.col("note_in_renault_public"))
        .filter(~pl.col("post_in_renault_public"))
        .group_by("calendarMonth").agg(
            total_notes = pl.count(),
            posts_in_our_data = pl.col("post_in_our_data").sum(),
        )
        .sort("calendarMonth")
    )
    return


@app.cell
def _(availabilities, pl):
    (
        availabilities
        .filter(~pl.col("post_in_our_data"))
        .group_by("calendarMonth").agg(
            total_notes = pl.count(),
            note_in_renault_public = pl.col("note_in_renault_public").sum(),
            post_in_renault_public = pl.col("post_in_renault_public").sum(),
        )
        .sort("calendarMonth")
    )
    return


@app.cell
def _():
    return


@app.cell
def _(enriched_notes, pl):
    for_plotting = (
        enriched_notes
        .select("noteId", "tweetId", "calendarMonth","tweet_lang", "tweet_author_id", "tweet_author_party")
        .fill_null("NULL")
        .sort("calendarMonth")
        .filter(pl.col("tweet_lang") == "en")
    )
    return (for_plotting,)


@app.cell
def _(for_plotting, pl, sns):
    ax = sns.histplot(
        data=for_plotting,
        x="calendarMonth",
        hue="tweet_author_party",
        multiple="fill",
        edgecolor=None,
    )

    year_starts = (
        for_plotting
        .select("calendarMonth")
        .drop_nulls()
        .unique()
        .sort("calendarMonth")
        .filter(pl.col("calendarMonth").str.ends_with("-01"))
        ["calendarMonth"]
        .to_list()
    )

    ax.set_xticks(year_starts)
    ax.set_xticklabels(year_starts, rotation=45, ha="right")

    for x in year_starts:
        ax.axvline(x, color="black", linewidth=1, alpha=0.6, zorder=5)
    ax
    return


@app.cell
def _(enriched_notes):
    enriched_notes.group_by("calendarMonth")
    return


if __name__ == "__main__":
    app.run()
