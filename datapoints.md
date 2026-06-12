# Datapoints we want to train on:

- **Self-calculated Elo (1900–present)** — Rolling Elo computed from all
  international results using the eloratings.net formula (K weighted by
  tournament importance, goal-difference multiplier, +100 home advantage).
  Stored as each team's *pre-match* rating plus the signed Elo difference.
  Source: martj42 international results CSV (auto-updated daily).

- **Total squad value (start of each year)** — Aggregate market value of the
  squad, taken as a yearly snapshot to avoid lookahead leakage. Optionally
  split by position group (attack / midfield / defense). One of the strongest
  known predictors of national team strength. Source: Transfermarkt
  (scraped or via Kaggle dumps).

- **Fatigue indicators** — Days since each team's last competitive match,
  number of matches in the last 30 days, and whether the previous match went
  to extra time. Match-day temperature and altitude as environmental
  modifiers if obtainable (note: historical weather per stadium is a separate
  data source and may be a stretch goal). Derived from match dates + venue.

- **Player availability** — Whether key players (e.g. top N by market value
  or minutes played) are missing through injury or suspension at match date.
  High value but hard to source historically; treat as experimental and only
  for recent tournaments where injury lists are documented.

- **Momentum** — Change in Elo over the team's last 10 matches (rising or
  declining trend), plus current unbeaten/winless streak length. Captures
  trajectory that the absolute Elo level smooths out.

- **Head-to-head record** — Win/draw/loss record and average goal difference
  between the two specific teams, restricted to a recent window (e.g. last
  20 years) so ancient results don't dominate. Expected weak signal; kept
  because it's free to compute.

- **Quality-adjusted form** — Points-per-game over the last 5/10 matches,
  weighted by the Elo of each opponent, so beating strong teams counts for
  more than beating minnows. Distinguishes "good run vs good teams" from
  "good run vs weak schedule".

- **Attack vs defense split** — Separate rolling averages of goals scored
  and goals conceded per game (last 5/10 matches). Two equally-rated teams
  can have very different profiles (3–2 team vs 1–0 team); essential for
  predicting the *scoreline*, not just the winner.

- **Squad demographics** — Average squad age (peak performance ~26–28),
  total caps (international experience), number of players with prior
  major-tournament appearances, and share of players at top-5-league clubs.
  Source: FIFA/EA ratings dumps or Transfermarkt squad pages, snapshotted
  per tournament.

- **Match-context flags** — Knockout vs group stage vs friendly, neutral
  venue, and whether either team is the host nation (host advantage is a
  well-documented World Cup effect). Zero-effort: all derivable from the
  tournament, neutral, and country columns already in the results CSV.

- **Country development efficiency** — Measures how much football talent a
  country produces relative to its size and football culture. Built by
  regressing log(talent output) on log(population) and log(popularity)
  across countries; the residual is the feature, so positive values mean a
  country overperforms its resources (e.g. Uruguay, Croatia) and negative
  values mean it underconverts. Output is proxied by the total Transfermarkt
  market value of each country's professional players (yearly snapshots from
  the dcaribou dataset), population comes from World Bank data, and
  popularity starts as a simple dominant/major/minor sport classification.
  The raw log components are fed to the model alongside the residual. The
  regression is fitted on training-era snapshots only to avoid leakage.
  Expected to be a weak-but-unique signal, largely redundant with Elo;
  kept if it survives ablation against the baseline
