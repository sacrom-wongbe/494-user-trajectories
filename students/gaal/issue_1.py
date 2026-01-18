import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl

    df = pl.read_csv(
        "/Users/gaaljaylaani/494-user-trajectories/local-data/notes-00000.tsv",
        separator="\t"
    )

    df.height
    return


if __name__ == "__main__":
    app.run()
