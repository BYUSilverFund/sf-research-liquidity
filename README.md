# Silver Fund Momentum Research Repository

## Set Up

Set up your Python virtual environment using `uv`.
```bash
uv sync
```

Source your Python virtual environment.
```bash
source .venv/bin/activate
```

Set up your environment variables in a `.env` file. You can follow the example found in `.env.example`.
```
ASSETS_TABLE=
EXPOSURES_TABLE=
COVARIANCES_TABLE=
CRSP_DAILY_TABLE=
CRSP_MONTHLY_TABLE=
CRSP_EVENTS_TABLE=
BYU_EMAIL=
PROJECT_ROOT=
```

Set up pre-commit by running:
```bash
prek install
```

Now all of your files will be formatted on commit (you will need to re-commit after the formatting).

## Experiments
1. Standard MVO
2. CAPM MVO
3. Fama 3 MVO
4. Fama 5 MVO
5. Barra MVO
6. Volatility scaling?
