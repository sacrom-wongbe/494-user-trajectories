import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("data/cn_data/notes-00000.tsv", sep='\t')
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    print("Total number of rows:" , len(df.index))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
