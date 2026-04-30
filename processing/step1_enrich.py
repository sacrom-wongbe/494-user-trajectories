# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="columns", auto_download=["html"])


@app.cell(column=0)
def _():
    import zipfile
    import os

    import polars as pl
    from loguru import logger

    from constants import _top_5_topics as top_5_topics

    return logger, os, pl, top_5_topics, zipfile


@app.cell
def _(logger):
    logger.add("logs/step1.log", rotation="10 MB", level="DEBUG", serialize=True)
    return


@app.cell(column=1)
def _(pl):
    def cast_ids(df: pl.DataFrame) -> pl.DataFrame:
        id_columns = [
            column
            for column in df.columns
            if column in {
                "post_id", "tweet_id", "tweet_author_id", "tweet_author_id_renault",
                "noteId", "note_id", "tweetId",
                "author_id", "authorId",
                "userId", "user_id", "userid",
                "participantId", "participant_id",
                "noteAuthorParticipantId", "raterParticipantId", "requesterParticipantId",
            }
        ]
        if not id_columns:
            return df
        return df.with_columns([pl.col(column).cast(pl.String) for column in id_columns])

    return (cast_ids,)


@app.cell
def _(pl):
    # Calculate calendar-based user month (months since first action) and calendar month
    def enrich_with_user_and_calendar_month(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            _actionDt=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"),
            _firstActionDt=pl.from_epoch(pl.col("participantFirstActionMillis"), time_unit="ms"),
        ).with_columns(
            userMonth=(
                (pl.col("_actionDt").dt.year() - pl.col("_firstActionDt").dt.year()) * 12
                + pl.col("_actionDt").dt.month() - pl.col("_firstActionDt").dt.month()
            ).cast(pl.Int32),
            calendarMonth=pl.col("_actionDt").dt.strftime("%Y-%m"),
        ).drop("_actionDt", "_firstActionDt")

    return (enrich_with_user_and_calendar_month,)


@app.cell
def _(cast_ids, logger, pl):
    def enrich_with_scores(
        notes: pl.DataFrame, ratings: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        scores = cast_ids(pl.read_parquet("../data/input/2026-02-03-scored_notes.parquet"))
        I_AND_F_COLUMNS = {
            "CoreModel (v1.1)": ("coreNoteIntercept", "coreNoteFactor1"),
            "ExpansionModel (v1.1)": ("expansionNoteIntercept", "expansionNoteFactor1"),
            "ExpansionPlusModel (v1.1)": ("expansionPlusNoteIntercept", "expansionPlusNoteFactor1"),
            "GroupModel01 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel02 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel03 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel04 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel05 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel06 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel07 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel08 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel09 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel10 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel11 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel12 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel13 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel14 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "MultiGroupModel01 (v1.0)": ("multiGroupNoteIntercept", "multiGroupNoteFactor1"),
            "TopicModel01 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
            "TopicModel02 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
            "TopicModel03 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
            "ScoringDriftGuard (v1.0)": (None, None),
            "NmrDueToMinStableCrhTime (v1.0)": (None, None),
            "InsufficientExplanation (v1.0)": (None, None),
        }

        scores = (
            scores
            .with_columns(scoreCreatedAtDt=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"))
            # Infer the pre drift guard model
            .with_columns(
                preDriftModel=pl.when(pl.col("decidedBy").str.contains("ScoringDriftGuard"))
                                .then(pl.col("metaScorerActiveRules").str.split(",").list[-2])
                                .otherwise(pl.col("decidedBy"))
            )
            # Retrieve the intercept from the inferred model
            .with_columns(
                noteFinalIntercept=pl.coalesce([
                    pl.when(pl.col("preDriftModel").str.starts_with(prefix))
                      .then(pl.col(intercept_col))
                    for prefix, (intercept_col, _) in I_AND_F_COLUMNS.items()
                    if intercept_col is not None
                ]),
                # Retrieve the factor from the inferred model
                noteFinalFactor=pl.coalesce([
                    pl.when(pl.col("preDriftModel").str.starts_with(prefix))
                      .then(pl.col(factor_col))
                    for prefix, (_, factor_col) in I_AND_F_COLUMNS.items()
                    if factor_col is not None
                ]),
            )
            .rename({"finalRatingStatus": "noteFinalRatingStatus"})
            .select("noteId", "noteFinalRatingStatus", "numRatings", "decidedBy", "noteFinalIntercept", "noteFinalFactor")
        )

        notes = notes.join(scores, on="noteId", how="left", coalesce=True, validate="1:1")
        ratings = ratings.join(scores, on="noteId", how="left", coalesce=True, validate="m:1")
        logger.info("Enriched notes and ratings with scores and factors")
        return notes, ratings

    return (enrich_with_scores,)


@app.cell
def _(cast_ids, logger, pl):
    def enrich_with_crh(
        notes: pl.DataFrame, ratings: pl.DataFrame, requests: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        statuses = cast_ids(pl.read_csv("../data/input/2026-02-27-note_status_records.csv"))  # Processed statuses, taken from a scm-prep run on 2/27.
        # Calculate whether a note ever achieved CRH status
        note_ever_crh = (
            statuses
            .with_columns(status_time=pl.col("status_time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z"))
            .filter(pl.col("status_time") <= pl.datetime(2026, 2, 3, time_zone="UTC"))
            .with_columns(Crh = pl.col("status") == "CURRENTLY_RATED_HELPFUL")
            .group_by("note_id")
            .agg(noteEverCrh = pl.col("Crh").any())
        )
        post_ever_crh = (
            notes
            .select("noteId", "tweetId")
            .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", validate="1:1")
            .group_by("tweetId")
            .agg(postEverCrh=pl.col("noteEverCrh").fill_null(False).any())
        )
        logger.info(f"Calculated CRH statuses for {len(note_ever_crh):,} notes")
        logger.info(f"Calculated CRH statuses for {len(post_ever_crh):,} posts")

        notes    = notes   .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", coalesce=True, validate="1:1")
        ratings  = ratings .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", coalesce=True, validate="m:1")
        requests = requests.join(post_ever_crh, on="tweetId", how="left", validate="m:1")
        logger.info("Enriched notes and ratings with CRH statuses")
        return notes, ratings, requests

    return (enrich_with_crh,)


@app.cell
def _(cast_ids, logger, pl, top_5_topics):
    def enrich_with_topics(
        notes: pl.DataFrame, ratings: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        # Join anything outside to top 5 topics into "other"
        topics = cast_ids(pl.read_parquet("../data/input/from-soham-notes_full.parquet"))
        topics = topics.with_columns(condensed_topic=pl.when(pl.col("topic").is_in(top_5_topics)).then(pl.col("topic")).otherwise(pl.lit("other")))
        topics = topics.select("noteId", "topic", "condensed_topic")

        notes   = notes  .join(topics, on="noteId", how="left", validate="1:1")  # TODO: Get more recent data from soham
        ratings = ratings.join(topics, on="noteId", how="left", validate="m:1")
        logger.info("Enriched notes and ratings with topics")
        # TODO: Topics for note requests?
        return notes, ratings

    return (enrich_with_topics,)


@app.cell
def _(logger, pl):
    def enrich_with_first_action(
        notes: pl.DataFrame, ratings: pl.DataFrame, requests: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        first_note_written      = notes     .group_by("noteAuthorParticipantId").agg(createdAtMillis=pl.col("createdAtMillis").min())
        first_note_rated        = ratings   .group_by("raterParticipantId")     .agg(createdAtMillis=pl.col("createdAtMillis").min())
        first_note_requested    = requests  .group_by("requesterParticipantId") .agg(createdAtMillis=pl.col("createdAtMillis").min())
        first_action = (
            pl.concat([
                first_note_written  .rename({"noteAuthorParticipantId": "participantId"}),
                first_note_rated    .rename({"raterParticipantId":      "participantId"}),
                first_note_requested.rename({"requesterParticipantId":  "participantId"}),
            ])
            .group_by("participantId")
            .agg(participantFirstActionMillis=pl.col("createdAtMillis").min())
        )
        logger.info(f"First action calculated for {len(first_action):,} users")


        notes    = notes   .join(first_action.rename({"participantId": "noteAuthorParticipantId"}),on="noteAuthorParticipantId",   how="left", validate="m:1")
        ratings  = ratings .join(first_action.rename({"participantId": "raterParticipantId"}),     on="raterParticipantId",        how="left", validate="m:1")
        requests = requests.join(first_action.rename({"participantId": "requesterParticipantId"}), on="requesterParticipantId",    how="left", validate="m:1")
        logger.info("Enriched notes and ratings with user join dates")
        return notes, ratings, requests

    return (enrich_with_first_action,)


@app.cell
def _(cast_ids, pl):
    _our_post_data = cast_ids(
        pl.scan_parquet("/data/cn_archive/derivatives/20260227_raw_posts.parquet")
        .select("post_id", "author_id", "lang")
        .filter(pl.col("author_id").is_not_null() | pl.col("lang").is_not_null())
        .collect()
        # Select first non null lang/author_id
        .group_by("post_id")
        .agg(
            tweet_author_id = pl.col("author_id").filter(pl.col("author_id").is_not_null()).first(),
            tweet_lang = pl.col("lang").filter(pl.col("lang").is_not_null()).first(),
            n_non_null_author_id = pl.col("author_id").filter(pl.col("author_id").is_not_null()).n_unique(),
            n_non_null_lang = pl.col("lang").filter(pl.col("lang").is_not_null()).n_unique())
    )

    _renault_post_data = cast_ids(
        pl.read_csv(
            "../data/input/renault_partisanship_labels.csv", 
            schema_overrides={"tweet_author_id": pl.String, "tweet_id": pl.String})
        .group_by("tweet_id")
        .agg(
            tweet_author_id = pl.col("tweet_author_id").filter(pl.col("tweet_author_id").is_not_null()).first(),
            n_non_null_author_id = pl.col("tweet_author_id").filter(pl.col("tweet_author_id").is_not_null()).n_unique())
        .with_columns(tweet_lang=pl.lit("en")) # Renault data only contains English tweets (see M&M)
    )

    # Make sure no contradictory values within a dataset
    assert _our_post_data.filter(pl.col("n_non_null_author_id") > 1).is_empty()
    assert _our_post_data.filter(pl.col("n_non_null_lang") > 1).is_empty()
    assert _renault_post_data.filter(pl.col("n_non_null_author_id") > 1).is_empty()

    # Join renault/our data
    post_data = (
        _our_post_data.select("post_id", "tweet_author_id", "tweet_lang")
        .join(_renault_post_data.select("tweet_id", "tweet_author_id", "tweet_lang"), 
            left_on="post_id", right_on="tweet_id",
            how="full", validate="1:1", suffix="_renault")
    )

    # Make sure no contradictory values between datasets
    assert post_data.filter(
        pl.col("tweet_author_id").is_not_null() & pl.col("tweet_author_id_renault").is_not_null() 
        & (pl.col("tweet_author_id") != pl.col("tweet_author_id_renault"))).is_empty()
    assert post_data.filter(
        pl.col("tweet_lang").is_not_null() & pl.col("tweet_lang_renault").is_not_null() 
        & (pl.col("tweet_lang") != pl.col("tweet_lang_renault"))).is_empty()

    # Coalesce 
    post_data = (
        post_data.with_columns(
            post_id = pl.coalesce(["post_id", "tweet_id"]),
            tweet_author_id = pl.coalesce(["tweet_author_id", "tweet_author_id_renault"]),
            tweet_lang = pl.coalesce(["tweet_lang", "tweet_lang_renault"]))
        .rename({"post_id": "tweetId"})
        .select("tweetId", "tweet_author_id", "tweet_lang")
    )
    return (post_data,)


@app.cell
def _(logger, pl, post_data):
    def enrich_with_post_lang_and_author(
        df: pl.DataFrame,
        col: str = "tweetId",
    ) -> pl.DataFrame:
        # Join notes to post data
        df = df.join(post_data, left_on=col, right_on="tweetId", how="left", coalesce=True, validate="m:1")

        logger.info("Enriched data with post language and author id")
        return df

    return (enrich_with_post_lang_and_author,)


@app.cell
def _(cast_ids, logger, pl, zipfile):
    def enrich_with_party(
        notes: pl.DataFrame,
    ) -> pl.DataFrame:
        _renault = cast_ids(
            pl.read_csv(
                "../data/input/renault_partisanship_labels.csv", 
                schema_overrides={"note_id": pl.String, "tweet_author_id": pl.String, "tweet_id": pl.String}))
        _renault_authors = _renault.select(["tweet_author_id", "party"]).unique().rename({"party": "author_party"})
        _renault_posts = _renault.select(["tweet_id", "party"]).unique().rename({"party": "post_party"})
        _renault_notes = _renault.select(["note_id", "party"]).unique().rename({"party": "note_party"})


        with zipfile.ZipFile("../data/input/2026-04-29-mosleh-partisanship/isaac-partisan.csv.zip") as z:
            with z.open(z.namelist()[0]) as f:
                _mohsen_authors = (
                    cast_ids(pl.read_csv(f))
                    .filter(pl.col("partisan_score") != -99)  # Filter out users without a partisan score
                    .with_columns(tweet_author_id = pl.col("user_id"),
                                  mohsen_author_party = pl.when(pl.col("partisan_score") > 0).then("republican")
                                      .when(pl.col("partisan_score") < 0).then("democrat")
                                      .otherwise(pl.lit(None)))
                    .select("tweet_author_id", "mohsen_author_party"))

        notes = (
            notes
            .join(
                _renault_authors, on="tweet_author_id", 
                how="left", coalesce=True, validate="m:1")
            .join(
                _renault_posts, left_on="tweetId", right_on="tweet_id", 
                how="left", coalesce=True, validate="m:1")
            .join(
                _renault_notes, left_on="noteId", right_on="note_id", 
                how="left", coalesce=True, validate="m:1")
            .join(
                _mohsen_authors, on="tweet_author_id", 
                how="left", coalesce=True, validate="m:1"
            )
        )

        # Make sure there aren't contradictory labels for the same note/author/post
        assert notes.filter(
            pl.col("author_party").is_not_null() & pl.col("post_party").is_not_null() 
            & (pl.col("author_party") != pl.col("post_party"))).is_empty()
        assert notes.filter(
            pl.col("author_party").is_not_null() & pl.col("note_party").is_not_null() 
            & (pl.col("author_party") != pl.col("note_party"))).is_empty()
        assert notes.filter(
            pl.col("post_party").is_not_null() & pl.col("note_party").is_not_null() 
            & (pl.col("post_party") != pl.col("note_party"))).is_empty()
        assert notes.filter(
            pl.col("author_party").is_not_null() & pl.col("mohsen_author_party").is_not_null() 
            & (pl.col("author_party") != pl.col("mohsen_author_party"))).is_empty()

        # Coalesce
        notes = notes.with_columns(
            tweet_author_party=pl.coalesce([
                pl.col("author_party"), pl.col("post_party"), pl.col("note_party"), pl.col("mohsen_author_party")])
        ).drop("author_party", "post_party", "note_party", "mohsen_author_party")



        logger.info("Enriched notes with Renault author party labels")
        return notes

    return (enrich_with_party,)


@app.cell
def _(logger, pl):
    def enrich_ratings_with_note_data(
        ratings: pl.DataFrame, notes: pl.DataFrame,
    ) -> pl.DataFrame:
        ratings = ratings.join(
            notes.select("noteId", "classification"),
            on="noteId",
            how="left",
            validate="m:1"
        )
        logger.info("Enriched ratings with note-level data")
        return ratings

    return (enrich_ratings_with_note_data,)


@app.cell
def _(logger, pl):
    def enrich_requests_with_outcomes(
        requests: pl.DataFrame, notes: pl.DataFrame,
    ) -> pl.DataFrame:
        request_outcomes = (
            notes
            .select("tweetId", "noteEverCrh")
            .group_by("tweetId")
            .agg(requestResultedInNote=pl.lit(True), requestResultedInCrh=pl.col("noteEverCrh").any())
        )
        requests = requests.join(request_outcomes, on="tweetId", how="left", coalesce=True, validate="m:1")
        requests = requests.with_columns(
            requestResultedInNote=pl.col("requestResultedInNote").fill_null(False),
            requestResultedInCrh =pl.col("requestResultedInCrh") .fill_null(False)
        )
        logger.info("Enriched requests with outcomes")
        return requests

    return (enrich_requests_with_outcomes,)


@app.cell(column=2)
def _(
    cast_ids,
    enrich_ratings_with_note_data,
    enrich_requests_with_outcomes,
    enrich_with_crh,
    enrich_with_first_action,
    enrich_with_party,
    enrich_with_post_lang_and_author,
    enrich_with_scores,
    enrich_with_topics,
    enrich_with_user_and_calendar_month,
    logger,
    os,
    pl,
):
    # Load data
    users = pl.read_parquet("../data/input/2026-02-03/userEnrollment.parquet")
    notes = cast_ids(pl.read_parquet("../data/input/2026-02-03/notes.parquet"))
    ratings = cast_ids(pl.read_parquet("../data/input/2026-02-03/noteRatings.parquet"))
    requests = cast_ids(pl.read_parquet("../data/input/2026-01-09/noteRequests.parquet")).rename({"userId": "requesterParticipantId"}) # Using the user-level requests, not post-level requests!
    logger.info("Data loaded successfully")

    ratings  = ratings .with_columns(ratingDate= pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms").dt.date())
    requests = requests.with_columns(requestDate=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms").dt.date())

    # Enrich
    notes, ratings = enrich_with_scores(notes, ratings)
    notes, ratings, requests = enrich_with_crh(notes, ratings, requests)
    notes, ratings = enrich_with_topics(notes, ratings)
    notes, ratings, requests = enrich_with_first_action(notes, ratings, requests)

    notes    = enrich_with_user_and_calendar_month(notes)
    ratings  = enrich_with_user_and_calendar_month(ratings)
    requests = enrich_with_user_and_calendar_month(requests)
    logger.info("Calculated user months and calendar months")

    notes = enrich_with_post_lang_and_author(notes)
    notes = enrich_with_party(notes)
    ratings = enrich_ratings_with_note_data(ratings, notes)
    requests = enrich_requests_with_outcomes(requests, notes)

    # Save
    os.makedirs("../data/intermediate", exist_ok=True)
    notes.write_parquet("../data/intermediate/notes_enriched.parquet")
    ratings.write_parquet("../data/intermediate/ratings_enriched.parquet")
    requests.write_parquet("../data/intermediate/requests_enriched.parquet")
    logger.info("Enriched datasets saved successfully")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
